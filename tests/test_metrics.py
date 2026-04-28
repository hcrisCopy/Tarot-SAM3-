from __future__ import annotations

import numpy as np

from tarot_sam3.evaluation.metrics import cumulative_iou, mask_iou


def test_mask_iou_identical_masks() -> None:
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    assert mask_iou(mask, mask) == 1.0


def test_mask_iou_partial_overlap() -> None:
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert mask_iou(pred, target) == 1 / 3


def test_cumulative_iou() -> None:
    preds = [np.array([[1, 0]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8)]
    targets = [np.array([[1, 1]], dtype=np.uint8), np.array([[0, 1]], dtype=np.uint8)]
    assert cumulative_iou(preds, targets) == 0.5
