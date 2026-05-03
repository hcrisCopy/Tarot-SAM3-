"""Configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving the repo-local ``inherits`` key."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherited = cfg.pop("inherits", None)
    if inherited:
        parent_path = Path(inherited)
        if not parent_path.is_absolute():
            parent_path = cfg_path.parent.parent / parent_path
            if not parent_path.exists():
                parent_path = cfg_path.parent / inherited
        parent = load_config(parent_path)
        return deep_update(parent, cfg)
    return cfg

