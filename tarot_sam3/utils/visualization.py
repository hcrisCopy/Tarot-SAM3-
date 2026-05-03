"""Visualization helpers used for MLLM mask selection and user outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tarot_sam3.utils.geometry import MaskCandidate, mask_to_box


PALETTE = [
    (230, 57, 70),
    (29, 128, 159),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
]


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    mask_bool = mask.astype(bool)
    arr[mask_bool, :3] = color
    arr[mask_bool, 3] = int(alpha * 255)
    overlay = Image.fromarray(arr, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width: int = 4) -> None:
    draw.rectangle([float(v) for v in box], outline=color, width=width)


def candidate_panel(image: Image.Image, candidates: list[MaskCandidate], title: str = "candidates") -> Image.Image:
    width, height = image.size
    label_h = 32
    panel = Image.new("RGB", (width * max(1, len(candidates)), height + label_h), "white")
    font = _font()
    for idx, candidate in enumerate(candidates):
        color = PALETTE[idx % len(PALETTE)]
        rendered = overlay_mask(image, candidate.mask, color)
        draw = ImageDraw.Draw(rendered)
        box = candidate.box or mask_to_box(candidate.mask)
        if box is not None:
            draw_box(draw, box, color)
        panel.paste(rendered, (idx * width, label_h))
        label = f"{idx}: {candidate.prompt_type} score={candidate.score:.3f}"
        ImageDraw.Draw(panel).text((idx * width + 8, 6), label, fill=color, font=font)
    ImageDraw.Draw(panel).text((8, height + 6), title, fill=(20, 20, 20), font=font)
    return panel


def region_panel(
    image: Image.Image,
    high_confidence: np.ndarray,
    region: np.ndarray,
    title: str,
) -> Image.Image:
    rendered = overlay_mask(image, high_confidence, (42, 157, 143), alpha=0.42)
    rendered = overlay_mask(rendered, region, (230, 57, 70), alpha=0.55)
    draw = ImageDraw.Draw(rendered)
    draw.text((8, 8), title, fill=(255, 255, 255), font=_font())
    return rendered


def save_image(image: Image.Image, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

