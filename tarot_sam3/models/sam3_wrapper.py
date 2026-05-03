"""SAM3 image wrapper for text, box, and point prompts."""

from __future__ import annotations

import copy
import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tarot_sam3.utils.geometry import (
    MaskCandidate,
    box_xyxy_to_cxcywh_norm,
    clip_box_xyxy,
    point_to_tiny_box,
)


def _torch_dtype(name: str | None) -> torch.dtype:
    if not name:
        return torch.float32
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), torch.float32)


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

        self._ensure_pkg_resources_shim(repo_path)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.set_default_dtype(torch.float32)
        self.device = device
        self.allow_low_precision = bool(cfg.get("allow_low_precision", False))
        self.dtype = _torch_dtype(cfg.get("dtype")) if self.allow_low_precision else torch.float32
        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_inst_interactivity=False,
        )
        self.model.to(dtype=self.dtype)
        self._patch_backbone_input_dtype()
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=float(cfg.get("confidence_threshold", 0.25)),
        )
        self._patch_processor_prompt_dtype()
        print(f"[Tarot-SAM3] SAM3 dtype={self.dtype}, allow_low_precision={self.allow_low_precision}")
        self.image: Image.Image | None = None
        self.state: dict[str, Any] | None = None
        self.width = 0
        self.height = 0

    def set_image(self, image: Image.Image) -> None:
        self.image = image.convert("RGB")
        self.width, self.height = self.image.size
        with self._sam3_precision_context():
            self.state = self.processor.set_image(self.image)

    @staticmethod
    def _ensure_pkg_resources_shim(repo_path: Path) -> None:
        try:
            import pkg_resources  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        shim = types.ModuleType("pkg_resources")

        def resource_filename(package_name: str, resource_name: str) -> str:
            package_path = repo_path / package_name.replace(".", "/")
            return str((package_path / resource_name).resolve())

        shim.resource_filename = resource_filename  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = shim

    def _fresh_state(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call set_image before prompting SAM3.")
        state = copy.copy(self.state)
        self._cast_state_dtype(state)
        return state

    def _cast_value_dtype(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=self.dtype)
        if isinstance(value, dict):
            return {key: self._cast_value_dtype(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cast_value_dtype(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cast_value_dtype(item) for item in value)
        return value

    def _cast_state_dtype(self, state: dict[str, Any]) -> None:
        if "backbone_out" in state:
            state["backbone_out"] = self._cast_value_dtype(state["backbone_out"])
        self._cast_geometric_prompt(state)

    def _patch_backbone_input_dtype(self) -> None:
        original_forward_image = self.model.backbone.forward_image
        dtype = self.dtype

        def forward_image_with_dtype(image, *args, **kwargs):
            if hasattr(image, "to"):
                image = image.to(dtype=dtype)
            return original_forward_image(image, *args, **kwargs)

        self.model.backbone.forward_image = forward_image_with_dtype

    def _patch_processor_prompt_dtype(self) -> None:
        original_forward_grounding = self.processor._forward_grounding

        def forward_grounding_with_prompt_dtype(state):
            self._cast_state_dtype(state)
            self._cast_geometric_prompt(state)
            return original_forward_grounding(state)

        self.processor._forward_grounding = forward_grounding_with_prompt_dtype

    def _cast_geometric_prompt(self, state: dict[str, Any]) -> None:
        prompt = state.get("geometric_prompt")
        if prompt is None:
            return
        for attr in ["box_embeddings", "point_embeddings", "mask_embeddings"]:
            value = getattr(prompt, attr, None)
            if value is not None and hasattr(value, "to") and value.is_floating_point():
                setattr(prompt, attr, value.to(dtype=self.dtype))

    def _sam3_precision_context(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            if self.allow_low_precision and self.dtype in {torch.float16, torch.bfloat16}:
                return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True)
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

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
        with self._sam3_precision_context():
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
        norm_box = box_xyxy_to_cxcywh_norm(box_xyxy, self.width, self.height)
        with self._sam3_precision_context():
            state = self.processor.set_text_prompt(prompt=text_hint or "visual", state=state)
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
        state = self._fresh_state()
        all_points = positive_points + negative_points
        if not all_points:
            return []

        try:
            with self._sam3_precision_context():
                state = self.processor.set_text_prompt(prompt=text_hint or "visual", state=state)
                norm_points = [
                    [float(x) / max(self.width, 1), float(y) / max(self.height, 1)]
                    for x, y in all_points
                ]
                point_tensor = torch.tensor(
                    norm_points,
                    device=self.device,
                    dtype=self.dtype,
                ).view(len(norm_points), 1, 2)
                label_tensor = torch.tensor(
                    [1] * len(positive_points) + [0] * len(negative_points),
                    device=self.device,
                    dtype=torch.long,
                ).view(len(norm_points), 1)
                state["geometric_prompt"].append_points(point_tensor, label_tensor)
                output = self.processor._forward_grounding(state)
            candidates = self._to_candidates(output, prompt=text_hint, prompt_type="point", limit=limit)
        except Exception:
            output = None
            with self._sam3_precision_context():
                state = self.processor.set_text_prompt(prompt=text_hint or "visual", state=self._fresh_state())
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
                candidate.metadata["point_fallback"] = "tiny_box"

        for candidate in candidates:
            candidate.metadata["positive_points"] = positive_points
            candidate.metadata["negative_points"] = negative_points
        return candidates
