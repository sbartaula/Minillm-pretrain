"""Atomic training checkpoints (fully resumable) and RNG-state capture.

Everything stored is plain Python types or tensors, so the file can be read with
torch.load(weights_only=True) and never executes pickled code.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1


def get_rng_state() -> dict[str, Any]:
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, key.tolist(), pos, has_gauss, cached),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, key, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def atomic_save(obj: Any, path: str | Path) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace.

    If the process is interrupted mid-write, the previous checkpoint stays untouched and a
    reader never sees a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)  # persist the rename itself
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {ckpt.get('format_version')}")
    return ckpt
