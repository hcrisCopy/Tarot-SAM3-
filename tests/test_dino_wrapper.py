from __future__ import annotations

import torch

from tarot_sam3.models.dino_wrapper import DinoFeatureExtractor


def test_patch_tokens_with_cls_and_register_tokens() -> None:
    tokens = torch.zeros(201, 8)
    patch_tokens, grid_h, grid_w = DinoFeatureExtractor._patch_tokens(tokens)
    assert patch_tokens.shape == (196, 8)
    assert (grid_h, grid_w) == (14, 14)


def test_patch_tokens_without_prefix() -> None:
    tokens = torch.zeros(196, 8)
    patch_tokens, grid_h, grid_w = DinoFeatureExtractor._patch_tokens(tokens)
    assert patch_tokens.shape == (196, 8)
    assert (grid_h, grid_w) == (14, 14)
