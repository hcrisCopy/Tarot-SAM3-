"""DINOv3 feature wrapper for similarity maps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoFeatureExtractor:
    """Extract dense feature similarity maps with a Hugging Face DINOv3 model."""

    def __init__(self, cfg: dict[str, Any], device: str = "cuda"):
        model_path = cfg.get("local_path") or cfg.get("name", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self._features: torch.Tensor | None = None
        self._grid_hw: tuple[int, int] | None = None
        self._image_size: tuple[int, int] | None = None

    @torch.inference_mode()
    def set_image(self, image: Image.Image) -> None:
        image = image.convert("RGB")
        self._image_size = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            tokens = outputs[0]
        tokens = tokens.squeeze(0)

        n_tokens = tokens.shape[0]
        side = int((n_tokens - 1) ** 0.5)
        if side * side == n_tokens - 1:
            tokens = tokens[1:]
            grid_h = grid_w = side
        else:
            side = int(n_tokens**0.5)
            if side * side != n_tokens:
                raise RuntimeError(f"Cannot infer DINO feature grid from {n_tokens} tokens.")
            grid_h = grid_w = side

        features = F.normalize(tokens.float(), dim=-1)
        self._features = features.reshape(grid_h, grid_w, -1)
        self._grid_hw = (grid_h, grid_w)

    def similarity_map(self, point_xy: tuple[float, float]) -> np.ndarray:
        if self._features is None or self._grid_hw is None or self._image_size is None:
            raise RuntimeError("Call set_image before requesting DINO similarity.")

        width, height = self._image_size
        grid_h, grid_w = self._grid_hw
        x = min(max(point_xy[0], 0.0), width - 1)
        y = min(max(point_xy[1], 0.0), height - 1)
        gx = int(round(x / max(width - 1, 1) * (grid_w - 1)))
        gy = int(round(y / max(height - 1, 1) * (grid_h - 1)))

        anchor = self._features[gy, gx]
        sim = torch.einsum("hwc,c->hw", self._features, anchor)
        sim = sim[None, None]
        sim = F.interpolate(sim, size=(height, width), mode="bilinear", align_corners=False)
        sim_np = sim.squeeze().detach().cpu().numpy()
        sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)
        return sim_np

    def accumulated_similarity(self, points: list[tuple[float, float]]) -> np.ndarray:
        if not points:
            raise ValueError("At least one point is required.")
        maps = [self.similarity_map(point) for point in points]
        return np.mean(np.stack(maps, axis=0), axis=0)

