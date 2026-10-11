"""Inference-only export: weights + config + tokenizer. No optimizer, no training state."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from .data import sha256_file
from .model import ModelConfig, TenslamLM


def export_inference(model: TenslamLM, tokenizer_path: str | Path, out_dir: str | Path,
                     info: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / "model.safetensors"))  # tied tensors are stored once
    (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2, sort_keys=True))
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    meta = {"format": "tenslam-inference-v1", "torch": torch.__version__,
            "n_params": model.num_params(), "tokenizer_sha256": sha256_file(out / "tokenizer.json"),
            "weights_sha256": sha256_file(out / "model.safetensors"), **(info or {})}
    (out / "export_info.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out


def load_exported(export_dir: str | Path, device: str | torch.device = "cpu") -> TenslamLM:
    d = Path(export_dir)
    cfg = ModelConfig(**json.loads((d / "config.json").read_text()))
    model = TenslamLM(cfg)
    load_model(model, str(d / "model.safetensors"), strict=True)
    return model.to(device).eval()
