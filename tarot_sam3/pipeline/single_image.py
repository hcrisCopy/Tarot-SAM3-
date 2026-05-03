"""Single-image Tarot-SAM3 pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from tarot_sam3.config import load_config
from tarot_sam3.eri import ERIOutput, ExpressionReasoningInterpreter
from tarot_sam3.models import DinoFeatureExtractor, QwenVLReasoner, Sam3Segmentor
from tarot_sam3.msr import MSROutput, MaskSelfRefiner
from tarot_sam3.utils.geometry import MaskCandidate, mask_to_box
from tarot_sam3.utils.visualization import overlay_mask, save_image


@dataclass
class SingleImageResult:
    image_path: str
    query: str
    output_path: str
    json_path: str
    eri: ERIOutput
    msr: MSROutput


def _candidate_summary(candidate: MaskCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "prompt_type": candidate.prompt_type,
        "prompt": candidate.prompt,
        "score": candidate.score,
        "area": candidate.area(),
        "box": candidate.box or mask_to_box(candidate.mask),
        "metadata": candidate.metadata,
    }


def _candidate_list_summary(candidates: list[MaskCandidate]) -> list[dict[str, Any]]:
    return [_candidate_summary(candidate) or {} for candidate in candidates]


def _write_intermediate_json(path: Path, image_path: str, query: str, eri: ERIOutput, msr: MSROutput) -> None:
    payload = {
        "image": image_path,
        "query": query,
        "eri": {
            "reasoning": eri.reasoning,
            "text_prompts": eri.text_prompts,
            "refer_objects": eri.refer_objects,
            "criteria": eri.criteria,
            "rephrased": eri.rephrased,
            "boxes": eri.boxes,
            "positive_points": eri.positive_points,
            "negative_points": eri.negative_points,
            "selected_text": _candidate_summary(eri.selected_text),
            "selected_box": _candidate_summary(eri.selected_box),
            "selected_point": _candidate_summary(eri.selected_point),
            "text_candidates": _candidate_list_summary(eri.text_candidates),
            "box_candidates": _candidate_list_summary(eri.box_candidates),
            "point_candidates": _candidate_list_summary(eri.point_candidates),
            "artifacts": eri.artifacts,
        },
        "msr": {
            "selected_best": _candidate_summary(msr.selected_best),
            "final_mask": _candidate_summary(msr.final_mask),
            "over_segmented": msr.over_segmented,
            "under_segmented": msr.under_segmented,
            "positive_points": msr.positive_points,
            "negative_points": msr.negative_points,
            "decisions": msr.decisions,
            "artifacts": msr.artifacts,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class SingleImagePipeline:
    """Load all models once and run Tarot-SAM3 on individual images."""

    def __init__(self, config_path: str | Path):
        self.config_path = str(config_path)
        self.cfg = load_config(config_path)
        runtime = self.cfg.get("runtime", {})
        models = self.cfg.get("models", {})
        self.device = runtime.get("device", "cuda")
        self.output_dir = Path(self.cfg.get("project", {}).get("output_dir", "outputs"))

        self.sam3 = Sam3Segmentor(models.get("sam3", {}), paths=self.cfg.get("paths", {}), device=self.device)
        self.dino = DinoFeatureExtractor(models.get("dino", {}), device=self.device)
        self.qwen = QwenVLReasoner(models.get("mllm", {}))

        self.eri = ExpressionReasoningInterpreter(
            self.qwen,
            self.sam3,
            self.dino,
            self.cfg.get("method", {}),
            output_dir=self.output_dir,
        )
        self.msr = MaskSelfRefiner(
            self.qwen,
            self.sam3,
            self.dino,
            self.cfg.get("method", {}),
            output_dir=self.output_dir,
        )

    def run(self, image_path: str | Path, query: str, output_path: str | Path) -> SingleImageResult:
        image_path = str(image_path)
        output_path = Path(output_path)
        image = Image.open(image_path).convert("RGB")

        self.sam3.set_image(image)
        self.dino.set_image(image)

        eri_output = self.eri.run(image, query)
        msr_output = self.msr.run(image, query, eri_output)

        final_candidate = msr_output.final_mask or eri_output.selected_point or eri_output.selected_box or eri_output.selected_text
        if final_candidate is None:
            raise RuntimeError("No mask candidate was produced by SAM3.")

        rendered = overlay_mask(image, final_candidate.mask, (230, 57, 70), alpha=0.45)
        save_image(rendered, output_path)

        json_path = output_path.with_suffix(".json")
        _write_intermediate_json(json_path, image_path, query, eri_output, msr_output)

        return SingleImageResult(
            image_path=image_path,
            query=query,
            output_path=str(output_path),
            json_path=str(json_path),
            eri=eri_output,
            msr=msr_output,
        )
