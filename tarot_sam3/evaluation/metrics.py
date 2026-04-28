"""Segmentation metrics."""

from __future__ import annotations

import numpy as np


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU for two binary masks."""
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    intersection = np.logical_and(pred_bool, target_bool).sum()
    union = np.logical_or(pred_bool, target_bool).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def cumulative_iou(preds: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Compute dataset-level cIoU."""
    total_intersection = 0
    total_union = 0
    for pred, target in zip(preds, targets, strict=True):
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        total_intersection += np.logical_and(pred_bool, target_bool).sum()
        total_union += np.logical_or(pred_bool, target_bool).sum()
    if total_union == 0:
        return 1.0
    return float(total_intersection / total_union)

