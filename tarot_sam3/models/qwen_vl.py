"""Qwen2.5-VL wrapper for structured multimodal reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from tarot_sam3.utils.json_utils import extract_json_object


def _torch_dtype(name: str | None) -> torch.dtype | str:
    if not name:
        return "auto"
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(name.lower(), "auto")


@dataclass
class QwenVLConfig:
    name: str
    local_path: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 512


class QwenVLReasoner:
    """Small wrapper around Qwen2.5-VL chat generation."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = QwenVLConfig(
            name=cfg.get("name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            local_path=cfg.get("local_path"),
            dtype=cfg.get("dtype", "bfloat16"),
            device_map=cfg.get("device_map", "auto"),
            max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        )
        model_path = self.cfg.local_path or self.cfg.name

        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=_torch_dtype(self.cfg.dtype),
            device_map=self.cfg.device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate a text response for one image and one prompt."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=False,
            )

        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()

    def generate_json(self, image: Image.Image, prompt: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        return extract_json_object(self.generate(image, prompt, max_new_tokens=max_new_tokens))

