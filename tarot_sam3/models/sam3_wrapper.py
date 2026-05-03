"""SAM3 image wrapper for text, box, and point prompts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tarot_sam3.utils.geometry import (
    MaskCandidate,
    box_xyxy_to_cxcywh_norm,
    clip_box_xyxy,
    point_to_tiny_box,
)


class Sam3Segmentor:
    """Single-image SAM3 prompt interface."""

    def __init__(self, cfg: dict[str, Any], paths: dict[str, Any] | None = None, device: str = "cuda"):
        paths = paths or {}
        repo_path = Path(cfg.get("repo_path") or paths.get("sam3_repo", "external/sam3"))
        if repo_path.exists():
            sys.path.insert(0, str(repo_path.resolve()))

        checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints/sam3"))
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path is None:
            candidate = checkpoint_dir / "sam3.pt"
            checkpoint_path = str(candidate) if candidate.exists() else None

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=float(cfg.get("confidence_threshold", 0.25)),
        )
        self.image: Image.Image | None = None
        self.state: dict[str, Any] | None = None
        self.width = 0
        self.height = 0

    def set_image(self, image: Image.Image) -> None:
        self.image = image.convert("RGB")
        self.width, self.height = self.image.size
        self.state = self.processor.set_image(self.image)
        predictor = getattr(self.model, "inst_interactive_predictor", None)
        if predictor is not None:
            predictor.set_image(self.image)

    def _fresh_state(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call set_image before prompting SAM3.")
        return copy.copy(self.state)

    @staticmethod
    def _to_candidates(output: dict[str, Any], prompt: str, prompt_type: str, limit: int | None = None) -> list[MaskCandidate]:
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None:
            return []

        masks_np = masks.detach().cpu().numpy()
        boxes_np = boxes.detach().cpu().numpy() if boxes is not None else [None] * len(masks_np)
        scores_np = scores.detach().cpu().numpy() if scores is not None else np.zeros((len(masks_np),), dtype=float)

        candidates: list[MaskCandidate] = []
        for mask, box, score in zip(masks_np, boxes_np, scores_np, strict=False):
            mask_2d = np.squeeze(mask).astype(bool)
            candidates.append(
                MaskCandidate(
                    mask=mask_2d,
                    score=float(score),
                    box=None if box is None else [float(v) for v in box],
                    prompt=prompt,
                    prompt_type=prompt_type,
                )
            )
        candidates.sort(key=lambda item: (item.score, item.area()), reverse=True)
        return candidates[:limit] if limit else candidates

    def predict_text(self, prompt: str, limit: int | None = None) -> list[MaskCandidate]:
        state = self._fresh_state()
        output = self.processor.set_text_prompt(prompt=prompt, state=state)
        return self._to_candidates(output, prompt=prompt, prompt_type="text", limit=limit)

    def predict_box(
        self,
        box_xyxy: list[float],
        text_hint: str = "visual",
        label: bool = True,
        limit: int | None = None,
    ) -> list[MaskCandidate]:
        state = self._fresh_state()
        state = self.processor.set_text_prompt(prompt=text_hint or "visual", state=state)
        norm_box = box_xyxy_to_cxcywh_norm(box_xyxy, self.width, self.height)
        output = self.processor.add_geometric_prompt(box=norm_box, label=label, state=state)
        candidates = self._to_candidates(output, prompt=text_hint, prompt_type="box", limit=limit)
        for candidate in candidates:
            candidate.metadata["input_box"] = clip_box_xyxy(box_xyxy, self.width, self.height)
        return candidates

    def predict_points(
        self,
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]] | None = None,
        text_hint: str = "visual",
        limit: int | None = None,
    ) -> list[MaskCandidate]:
        negative_points = negative_points or []
        predictor = getattr(self.model, "inst_interactive_predictor", None)
        if predictor is not None:
            points = np.array(positive_points + negative_points, dtype=np.float32)
            labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
                return_logits=False,
            )
            candidates = [
                MaskCandidate(
                    mask=np.squeeze(mask).astype(bool),
                    score=float(score),
                    box=None,
                    prompt=text_hint,
                    prompt_type="point",
                    metadata={"positive_points": positive_points, "negative_points": negative_points},
                )
                for mask, score in zip(masks, scores, strict=False)
            ]
            candidates.sort(key=lambda item: (item.score, item.area()), reverse=True)
            return candidates[:limit] if limit else candidates

        state = self._fresh_state()
        state = self.processor.set_text_prompt(prompt=text_hint or "visual", state=state)
        output = None
        for point in positive_points:
            box = box_xyxy_to_cxcywh_norm(point_to_tiny_box(point, self.width, self.height), self.width, self.height)
            output = self.processor.add_geometric_prompt(box=box, label=True, state=state)
        for point in negative_points:
            box = box_xyxy_to_cxcywh_norm(point_to_tiny_box(point, self.width, self.height), self.width, self.height)
            output = self.processor.add_geometric_prompt(box=box, label=False, state=state)
        if output is None:
            return []
        candidates = self._to_candidates(output, prompt=text_hint, prompt_type="point", limit=limit)
        for candidate in candidates:
            candidate.metadata["positive_points"] = positive_points
            candidate.metadata["negative_points"] = negative_points
            candidate.metadata["point_fallback"] = "tiny_box"
        return candidates

