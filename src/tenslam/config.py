"""Load TOML configs as plain nested dicts (stdlib only).

A config may say `extends = "other.toml"`: the parent is loaded first and the child's
tables are deep-merged on top, so a Kaggle config only has to state what differs.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path = "configs/data.toml") -> dict[str, Any]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent) if Path(parent).is_absolute() else Path(path).parent.parent / parent
        if not parent_path.exists():  # also accept paths relative to the config's own folder
            parent_path = Path(path).parent / parent
        cfg = _deep_merge(load_config(parent_path), cfg)
    return cfg
