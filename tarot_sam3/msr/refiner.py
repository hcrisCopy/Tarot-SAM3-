"""Mask Self-Refining phase for single-image Tarot-SAM3 inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tarot_sam3.eri import ERIOutput
from tarot_sam3.models import DinoFeatureExtractor, QwenVLReasoner, Sam3Segmentor
from tarot_sam3.utils.geometry import MaskCandidate, mask_center, pick_candidate
from tarot_sam3.utils.prompts import load_prompt
from tarot_sam3.utils.visualization import candidate_panel, region_panel, save_image


@dataclass
class MSROutput:
    selected_best: MaskCandidate | None
    final_mask: MaskCandidate | None
    over_segmented: bool = False
    under_segmented: bool = False
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)


class MaskSelfRefiner:
    """Implements the paper's MSR phase for a single image."""

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
        self.max_refine_rounds = int(cfg.get("msr", {}).get("max_refine_rounds", 1))

    def run(self, image: Image.Image, query: str, eri: ERIOutput) -> MSROutput:
        candidates = eri.best_candidates()
        selected_best = self._select_global_best(image, query, candidates)
        if selected_best is None:
            return MSROutput(selected_best=None, final_mask=None)

        selected_point = eri.selected_point
        if selected_point is None or self.max_refine_rounds <= 0:
            return MSROutput(
                selected_best=selected_best,
                final_mask=selected_best,
                positive_points=eri.positive_points,
                negative_points=eri.negative_points,
            )

        best_only = np.logical_and(selected_best.mask, ~selected_point.mask)
        point_only = np.logical_and(selected_point.mask, ~selected_best.mask)
        high_confidence = eri.high_confidence

        artifacts: dict[str, str] = {}
        decisions: dict[str, Any] = {}

        over_segmented = False
        under_segmented = False
        positive_points = list(eri.positive_points)
        negative_points = list(eri.negative_points)

        if best_only.any():
            path = self.output_dir / "intermediates" / "msr_best_only_region.png"
            panel = region_panel(image, high_confidence, best_only, "green=high confidence, red=best-only")
            save_image(panel, path)
            artifacts["best_only_region"] = str(path)
            belongs = self._judge_region(panel, query, "best-only discriminative region")
            decisions["best_only_belongs_to_target"] = belongs
            over_segmented = not belongs.get("belongs_to_target", True)
            if over_segmented:
                center = mask_center(best_only)
                if center is not None:
                    negative_points = [center]

        if point_only.any():
            path = self.output_dir / "intermediates" / "msr_point_only_region.png"
            panel = region_panel(image, high_confidence, point_only, "green=high confidence, red=point-only")
            save_image(panel, path)
            artifacts["point_only_region"] = str(path)
            belongs = self._judge_region(panel, query, "point-only discriminative region")
            decisions["point_only_belongs_to_target"] = belongs
            under_segmented = bool(belongs.get("belongs_to_target", False))
            if under_segmented:
                center = mask_center(point_only)
                if center is not None and center not in positive_points:
                    positive_points.append(center)

        if not over_segmented and not under_segmented:
            return MSROutput(
                selected_best=selected_best,
                final_mask=selected_best,
                over_segmented=False,
                under_segmented=False,
                positive_points=positive_points,
                negative_points=negative_points,
                artifacts=artifacts,
                decisions=decisions,
            )

        refined_candidates = self.sam3.predict_points(
            positive_points=positive_points,
            negative_points=negative_points,
            text_hint=str(eri.reasoning.get("target_name") or query),
            limit=int(self.cfg.get("mask_score_topk", 5)),
        )
        refined = self._select_global_best(image, query, refined_candidates)
        final_mask = refined or selected_best

        if refined_candidates:
            path = self.output_dir / "intermediates" / "msr_refined_candidates.png"
            save_image(candidate_panel(image, refined_candidates, "MSR refined point candidates"), path)
            artifacts["refined_candidates"] = str(path)

        return MSROutput(
            selected_best=selected_best,
            final_mask=final_mask,
            over_segmented=over_segmented,
            under_segmented=under_segmented,
            positive_points=positive_points,
            negative_points=negative_points,
            artifacts=artifacts,
            decisions=decisions,
        )

    def _select_global_best(
        self,
        image: Image.Image,
        query: str,
        candidates: list[MaskCandidate],
    ) -> MaskCandidate | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        panel = candidate_panel(image, candidates, "MSR inter-prompt candidates")
        template = load_prompt("mask_selection_prompt.txt", self.prompts_dir)
        prompt = (
            f"{template}\n\nReferring expression: {query}\n"
            f"Candidate count: {len(candidates)}\n"
            "The panel shows the best text, box, and point-prompt masks from left to right. "
            "Choose the globally best mask for the target."
        )
        try:
            response = self.qwen.generate_json(panel, prompt)
            index = int(response.get("best_index", -1))
        except Exception:
            index = None
        return pick_candidate(candidates, index)

    def _judge_region(self, panel: Image.Image, query: str, region_name: str) -> dict[str, Any]:
        template = load_prompt("region_affiliation_prompt.txt", self.prompts_dir)
        prompt = (
            f"{template}\n\nReferring expression: {query}\n"
            f"Region to judge: {region_name}\n"
            "The green area is a high-confidence part of the target. "
            "The red area is the discriminative region. Decide whether the red area belongs to the same target object."
        )
        try:
            return self.qwen.generate_json(panel, prompt)
        except Exception as exc:  # noqa: BLE001
            return {"belongs_to_target": False, "reason": f"MLLM judgment failed: {exc}"}

