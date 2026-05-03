"""Prompt template loading."""

from __future__ import annotations

from pathlib import Path


def load_prompt(name: str, prompts_dir: str | Path = "prompts") -> str:
    path = Path(prompts_dir) / name
    return path.read_text(encoding="utf-8").strip()

