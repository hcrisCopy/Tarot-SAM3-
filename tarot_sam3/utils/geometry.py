"""Geometry and mask helpers for single-image inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaskCandidate:
    """A SAM3 mask candidate with enough metadata for ERI/MSR decisions."""

    mask: np.ndarray
    score: float = 0.0
    box: list[float] | None = None
    prompt: str = ""
    prompt_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def area(self) -> int:
        return int(self.mask.astype(bool).sum())


def clip_box_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return [x1, y1, x2, y2]


def box_xyxy_to_cxcywh_norm(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = clip_box_xyxy(box, width, height)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return [cx, cy, bw, bh]


def point_to_tiny_box(point: tuple[float, float], width: int, height: int, size: int = 8) -> list[float]:
    x, y = point
    half = max(1.0, size / 2.0)
    return clip_box_xyxy([x - half, y - half, x + half, y + half], width, height)


def box_to_mask(box: list[float], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = clip_box_xyxy(box, width, height)
    mask = np.zeros((height, width), dtype=bool)
    mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
    return mask


def mask_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def mask_box_iou(mask: np.ndarray, box: list[float]) -> float:
    return mask_iou_np(mask, box_to_mask(box, mask.shape))


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def pick_candidate(candidates: list[MaskCandidate], index: int | None = None) -> MaskCandidate | None:
    if not candidates:
        return None
    if index is not None and 0 <= index < len(candidates):
        return candidates[index]
    return max(candidates, key=lambda c: (c.score, c.area()))

