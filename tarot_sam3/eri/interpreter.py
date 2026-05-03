"""Expression Reasoning Interpreter for single-image Tarot-SAM3 inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tarot_sam3.models import DinoFeatureExtractor, QwenVLReasoner, Sam3Segmentor
from tarot_sam3.utils.geometry import (
    MaskCandidate,
    box_to_mask,
    clip_box_xyxy,
    mask_box_iou,
    mask_center,
    pick_candidate,
)
from tarot_sam3.utils.json_utils import as_list
from tarot_sam3.utils.prompts import load_prompt
from tarot_sam3.utils.visualization import candidate_panel, save_image


@dataclass
class ERIOutput:
    reasoning: dict[str, Any]
    text_prompts: list[str]
    refer_objects: list[str]
    criteria: list[dict[str, Any]]
    rephrased: dict[str, str]
    boxes: list[list[float]]
    text_candidates: list[MaskCandidate]
    box_candidates: list[MaskCandidate]
    point_candidates: list[MaskCandidate]
    selected_text: MaskCandidate | None
    selected_box: MaskCandidate | None
    selected_point: MaskCandidate | None
    high_confidence: np.ndarray
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    artifacts: dict[str, str] = field(default_factory=dict)

    def best_candidates(self) -> list[MaskCandidate]:
        return [c for c in [self.selected_text, self.selected_box, self.selected_point] if c is not None]


class ExpressionReasoningInterpreter:
    """Implements the ERI phase from the paper for one image/query pair."""

    def __init__(
        self,
        qwen: QwenVLReasoner,
        sam3: Sam3Segmentor,
        dino: DinoFeatureExtractor,
        cfg: dict[str, Any],
        prompts_dir: str | Path = "prompts",
        output_dir: str | Path = "outputs",
    ):
        self.qwen = qwen
        self.sam3 = sam3
        self.dino = dino
        self.cfg = cfg
        self.prompts_dir = prompts_dir
        self.output_dir = Path(output_dir)
        self.tau = float(cfg.get("tau", 0.80))
        self.sneg = float(cfg.get("sneg", 0.30))
        self.text_prompt_topk = int(cfg.get("text_prompt_topk", 3))
        self.mask_score_topk = int(cfg.get("mask_score_topk", 5))
        self.anchor_count = int(cfg.get("point_prompt", {}).get("anchor_count", 5))

    def run(self, image: Image.Image, query: str) -> ERIOutput:
        reasoning = self._parse_expression(image, query)
        target_name = str(reasoning.get("target_name") or query)
        refer_objects = [str(item) for item in as_list(reasoning.get("refer_objects")) if str(item).strip()]

        text_prompts = self._augment_target(image, target_name)
        text_candidates = self._collect_text_candidates(text_prompts)

        criteria = self._build_criteria(image, target_name, refer_objects)
        rephrased = self._rephrase(image, query, target_name, refer_objects, criteria)
        boxes = self._predict_boxes(image, query, rephrased)
        box_candidates = self._collect_box_candidates(boxes, target_name)

        text_candidates = self._prompt_consistency_filter(text_candidates, boxes)
        selected_text = self._select_candidate(image, query, text_candidates, "text")
        selected_box = self._select_candidate(image, query, box_candidates, "box")

        high_confidence = self._high_confidence_region(selected_text, selected_box, image.size)
        positive_points, negative_points = self._generate_point_prompts(high_confidence, selected_text, selected_box)
        point_candidates = self.sam3.predict_points(
            positive_points=positive_points,
            negative_points=negative_points,
            text_hint=target_name,
            limit=self.mask_score_topk,
        )
        selected_point = self._select_candidate(image, query, point_candidates, "point")

        artifacts: dict[str, str] = {}
        if text_candidates:
            path = self.output_dir / "intermediates" / "eri_text_candidates.png"
            save_image(candidate_panel(image, text_candidates, "ERI text candidates"), path)
            artifacts["text_candidates"] = str(path)
        if box_candidates:
            path = self.output_dir / "intermediates" / "eri_box_candidates.png"
            save_image(candidate_panel(image, box_candidates, "ERI box candidates"), path)
            artifacts["box_candidates"] = str(path)
        if point_candidates:
            path = self.output_dir / "intermediates" / "eri_point_candidates.png"
            save_image(candidate_panel(image, point_candidates, "ERI point candidates"), path)
            artifacts["point_candidates"] = str(path)

        return ERIOutput(
            reasoning=reasoning,
            text_prompts=text_prompts,
            refer_objects=refer_objects,
            criteria=criteria,
            rephrased=rephrased,
            boxes=boxes,
            text_candidates=text_candidates,
            box_candidates=box_candidates,
            point_candidates=point_candidates,
            selected_text=selected_text,
            selected_box=selected_box,
            selected_point=selected_point,
            high_confidence=high_confidence,
            positive_points=positive_points,
            negative_points=negative_points,
            artifacts=artifacts,
        )

    def _parse_expression(self, image: Image.Image, query: str) -> dict[str, Any]:
        template = load_prompt("eri_parse_prompt.txt", self.prompts_dir)
        prompt = f"{template}\n\nText query:\n{query}"
        return self.qwen.generate_json(image, prompt)

    def _augment_target(self, image: Image.Image, target_name: str) -> list[str]:
        template = load_prompt("target_augmentation_prompt.txt", self.prompts_dir)
        prompt = f"{template}\n\nTarget object name:\n{target_name}"
        response = self.qwen.generate_json(image, prompt)
        prompts = [str(item).strip() for item in as_list(response.get("prompts")) if str(item).strip()]
        if not prompts:
            prompts = [target_name]
        if prompts[0].lower() != target_name.lower():
            prompts.insert(0, target_name)
        return prompts[: self.text_prompt_topk]

    def _collect_text_candidates(self, text_prompts: list[str]) -> list[MaskCandidate]:
        candidates: list[MaskCandidate] = []
        for prompt in text_prompts:
            candidates.extend(self.sam3.predict_text(prompt, limit=self.mask_score_topk))
        candidates.sort(key=lambda item: (item.score, item.area()), reverse=True)
        return candidates[: self.mask_score_topk * max(1, len(text_prompts))]

    def _build_criteria(self, image: Image.Image, target_name: str, refer_objects: list[str]) -> list[dict[str, Any]]:
        template = load_prompt("criterion_prompt.txt", self.prompts_dir)
        criteria = []
        for refer_object in refer_objects[:3]:
            refer_candidates = self.sam3.predict_text(refer_object, limit=1)
            refer_box = refer_candidates[0].box if refer_candidates else None
            prompt = (
                f"{template}\n\nTarget name: {target_name}\n"
                f"Refer object name: {refer_object}\n"
                f"Refer object box: {refer_box}"
            )
            try:
                item = self.qwen.generate_json(image, prompt)
            except Exception as exc:  # noqa: BLE001 - keep single-image inference moving.
                item = {"relationship": "", "positive_criteria": [], "negative_criteria": [], "error": str(exc)}
            item["refer_object"] = refer_object
            item["refer_box"] = refer_box
            criteria.append(item)
        return criteria

    def _rephrase(
        self,
        image: Image.Image,
        query: str,
        target_name: str,
        refer_objects: list[str],
        criteria: list[dict[str, Any]],
    ) -> dict[str, str]:
        template = load_prompt("rephrase_prompt.txt", self.prompts_dir)
        prompt = (
            f"{template}\n\nOriginal expression: {query}\n"
            f"Target name: {target_name}\n"
            f"Refer objects: {refer_objects}\n"
            f"Evaluation criteria: {criteria}"
        )
        response = self.qwen.generate_json(image, prompt)
        return {
            "short_expression": str(response.get("short_expression") or target_name),
            "long_expression": str(response.get("long_expression") or query),
        }

    def _predict_boxes(self, image: Image.Image, query: str, rephrased: dict[str, str]) -> list[list[float]]:
        template = load_prompt("bbox_prompt.txt", self.prompts_dir)
        expressions = [
            query,
            rephrased.get("short_expression", ""),
            rephrased.get("long_expression", ""),
        ]
        boxes: list[list[float]] = []
        width, height = image.size
        for expression in expressions:
            if not expression:
                continue
            prompt = f"{template}\n\nTarget-centric expression:\n{expression}"
            try:
                response = self.qwen.generate_json(image, prompt)
            except Exception:
                continue
            box = response.get("box")
            if isinstance(box, list) and len(box) == 4:
                boxes.append(clip_box_xyxy([float(v) for v in box], width, height))
        unique_boxes: list[list[float]] = []
        for box in boxes:
            if box not in unique_boxes:
                unique_boxes.append(box)
        return unique_boxes

    def _collect_box_candidates(self, boxes: list[list[float]], target_name: str) -> list[MaskCandidate]:
        candidates: list[MaskCandidate] = []
        for box in boxes:
            candidates.extend(self.sam3.predict_box(box, text_hint=target_name, limit=self.mask_score_topk))
        candidates.sort(key=lambda item: (item.score, item.area()), reverse=True)
        return candidates[: self.mask_score_topk * max(1, len(boxes))]

    def _prompt_consistency_filter(
        self,
        candidates: list[MaskCandidate],
        boxes: list[list[float]],
    ) -> list[MaskCandidate]:
        if not candidates or not boxes:
            return candidates
        kept = [
            candidate
            for candidate in candidates
            if max(mask_box_iou(candidate.mask, box) for box in boxes) > self.tau
        ]
        return kept if kept else candidates

    def _select_candidate(
        self,
        image: Image.Image,
        query: str,
        candidates: list[MaskCandidate],
        prompt_type: str,
    ) -> MaskCandidate | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        panel = candidate_panel(image, candidates, f"Select best {prompt_type} mask")
        template = load_prompt("mask_selection_prompt.txt", self.prompts_dir)
        prompt = (
            f"{template}\n\nReferring expression: {query}\n"
            f"Prompt type: {prompt_type}\n"
            f"Candidate count: {len(candidates)}\n"
            "The panel shows candidates from left to right with index labels. "
            "Choose the single best index."
        )
        try:
            response = self.qwen.generate_json(panel, prompt)
            index = int(response.get("best_index", -1))
        except Exception:
            index = None
        return pick_candidate(candidates, index)

    def _high_confidence_region(
        self,
        selected_text: MaskCandidate | None,
        selected_box: MaskCandidate | None,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        width, height = image_size
        if selected_text is not None and selected_box is not None:
            overlap = np.logical_and(selected_text.mask, selected_box.mask)
            if overlap.any():
                return overlap
        if selected_text is not None:
            return selected_text.mask.astype(bool)
        if selected_box is not None:
            return selected_box.mask.astype(bool)
        return np.zeros((height, width), dtype=bool)

    def _sample_anchor_points(self, mask: np.ndarray) -> list[tuple[float, float]]:
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0:
            return []
        center = (float(xs.mean()), float(ys.mean()))
        points = [center]
        if len(xs) > 1 and self.anchor_count > 1:
            order = np.linspace(0, len(xs) - 1, num=min(self.anchor_count - 1, len(xs)), dtype=int)
            points.extend((float(xs[i]), float(ys[i])) for i in order)
        return points[: self.anchor_count]

    def _generate_point_prompts(
        self,
        high_confidence: np.ndarray,
        selected_text: MaskCandidate | None,
        selected_box: MaskCandidate | None,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        anchors = self._sample_anchor_points(high_confidence)
        if not anchors:
            for candidate in [selected_text, selected_box]:
                if candidate is not None:
                    center = mask_center(candidate.mask)
                    if center is not None:
                        anchors = [center]
                        break
        if not anchors:
            height, width = high_confidence.shape
            anchors = [(width / 2.0, height / 2.0)]

        sim = self.dino.accumulated_similarity(anchors)
        inside = high_confidence.astype(bool)
        if inside.any():
            pos_scores = np.where(inside, sim, -1.0)
            py, px = np.unravel_index(int(np.argmax(pos_scores)), pos_scores.shape)
        else:
            py, px = np.unravel_index(int(np.argmax(sim)), sim.shape)
        positive = [(float(px), float(py))]

        gy, gx = np.gradient(sim)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        outside = ~inside
        neg_mask = outside & (sim <= self.sneg)
        if not neg_mask.any():
            neg_mask = outside
        if neg_mask.any():
            neg_score = np.where(neg_mask, grad_mag * (1.0 - sim), -1.0)
            ny, nx = np.unravel_index(int(np.argmax(neg_score)), neg_score.shape)
            negative = [(float(nx), float(ny))]
        else:
            negative = []
        return positive, negative

