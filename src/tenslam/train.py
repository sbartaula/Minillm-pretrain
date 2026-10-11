"""Pretraining engine: AdamW + warmup/cosine LR, gradient accumulation, mixed precision,
validation, structured logging, atomic resumable checkpoints and controlled stops.

One optimizer step ("step") consumes `grad_accum_steps` micro-batches of
`micro_batch_size` windows of `seq_len` tokens:
    tokens_per_step = micro_batch_size * grad_accum_steps * seq_len.
Each micro-batch loss is divided by grad_accum_steps so the summed gradient equals the
gradient of the mean loss over the whole effective batch.
"""
from __future__ import annotations

import dataclasses
import json
import math
import platform
import signal
import threading
import time
import tomllib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import tokenizer as T
from .checkpoint import (FORMAT_VERSION, atomic_save, get_rng_state, load_checkpoint,
                         set_rng_state)
from .data import sha256_file
from .dataset import eval_batches, open_tokens, sample_batch
from .export import export_inference
from .generate import generate
from .model import ModelConfig, TenslamLM
from .schedule import make_scheduler

# The scheduler legitimately advances on a step the GradScaler skipped (inf gradients).
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")


class NonFiniteError(RuntimeError):
    """Raised when the loss, gradients or weights become NaN/Inf and training must stop."""


@dataclass
class TrainConfig:
    name: str = "run"
    out_dir: str = "runs/run"
    seed: int = 1337
    train_bin: str = "data/tokens/train.bin"
    val_bin: str = "data/tokens/validation.bin"
    manifest: str = "data/tokens/prepare_manifest.json"
    tokenizer: str = "artifacts/tokenizer/tokenizer.json"
    model: dict = field(default_factory=dict)
    seq_len: int = 512
    micro_batch_size: int = 32
    grad_accum_steps: int = 4
    max_tokens: int = 100_000_000
    max_steps: int = 0  # 0 = derive from max_tokens
    max_hours: float = 0.0  # 0 = no wall-clock limit
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    device: str = "auto"  # auto | cpu | cuda
    compile: bool = False
    verify_data_hash: bool = True
    eval_interval: int = 250
    eval_batches: int = 50  # 0 = whole validation file
    eval_at_start: bool = True
    log_interval: int = 10
    ckpt_interval: int = 500
    sample_interval: int = 500
    max_consecutive_skips: int = 20
    prompts: list = field(default_factory=lambda: ["Once upon a time"])
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40
    hourly_price_eur: float | None = None
    setup_minutes: float = 15.0

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def total_steps(self) -> int:
        return self.max_steps or max(1, math.ceil(self.max_tokens / self.tokens_per_step))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        flat: dict[str, Any] = {}
        for section in raw.values():
            flat.update(section)
        model_path = flat.pop("model_config", None)
        if model_path:
            with open(model_path, "rb") as f:
                flat["model"] = tomllib.load(f)["model"]
        return cls.from_dict(flat)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {name!r}")
    return torch.device(name)


def resolve_precision(device: torch.device, name: str = "auto") -> tuple[torch.dtype | None, bool]:
    """-> (autocast dtype or None for plain fp32, whether a GradScaler is needed).

    bf16 keeps fp32's exponent range, so it needs no loss scaling. fp16 has a narrow range:
    small gradients underflow to 0, so the loss is scaled up and gradients scaled back down
    (GradScaler). 'auto' picks bf16 only on GPUs with native bf16 (compute capability >= 8);
    older GPUs (Kaggle T4 / P100) get fp16 + GradScaler.
    """
    if name == "auto":
        if device.type == "cuda":
            native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
            name = "bf16" if native_bf16 else "fp16"
        else:
            name = "fp32"
    if name == "fp32":
        return None, False
    if name == "bf16":
        return torch.bfloat16, False
    if name == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 mixed precision requires CUDA")
        return torch.float16, True
    raise ValueError(f"unknown precision {name!r}")


def check_finite_params(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise NonFiniteError(f"non-finite values in parameter {name}")


def next_skip_count(current: int, skipped: bool, limit: int) -> int:
    """Consecutive GradScaler-skipped steps; too many in a row means real divergence."""
    count = current + 1 if skipped else 0
    if count > limit:
        raise NonFiniteError(f"{count} consecutive optimizer steps skipped for inf/NaN gradients")
    return count


def safe_exp(x: float) -> float:
    return math.exp(min(x, 700.0))


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: str | Path | None = None,
                 write_files: bool = True) -> None:
        self.cfg = cfg
        self.write_files = write_files
        self.out_dir = Path(cfg.out_dir)
        self.device = resolve_device(cfg.device)
        self.autocast_dtype, self.use_scaler = resolve_precision(self.device, cfg.precision)

        self.tok = T.load_tokenizer(cfg.tokenizer)
        self.tokenizer_sha = sha256_file(cfg.tokenizer)
        self.model_cfg = ModelConfig(**cfg.model)
        if self.tok.get_vocab_size() != self.model_cfg.vocab_size:
            raise ValueError("tokenizer vocab size != model vocab size")
        if cfg.seq_len > self.model_cfg.max_seq_len:
            raise ValueError("seq_len exceeds model max_seq_len")
        self.data_info = self._verify_data()
        self.train_tokens = open_tokens(cfg.train_bin)
        self.val_tokens = open_tokens(cfg.val_bin)
        self.source_val: dict[str, np.ndarray] = {}
        if Path(cfg.manifest).exists():
            with open(cfg.manifest, encoding="utf-8") as f:
                for src in json.load(f).get("sources", []):
                    vb = Path(src.get("validation_bin", ""))
                    if src.get("validation_tokens", 0) > cfg.seq_len + 1 and vb.exists():
                        self.source_val[src["name"]] = open_tokens(vb)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        import random
        random.seed(cfg.seed)
        self.model = TenslamLM(self.model_cfg).to(self.device)
        self.opt = self.model.configure_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
        self.scheduler = make_scheduler(self.opt, cfg.lr, cfg.min_lr, cfg.warmup_steps,
                                        cfg.total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self._fwd = torch.compile(self.model) if cfg.compile else self.model

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.consecutive_skips = 0
        self.stop_requested = False
        self.stop_reason: str | None = None
        if resume:
            self.load(resume)

    # ----------------------------------------------------------------- data / identity
    def _verify_data(self) -> dict:
        cfg = self.cfg
        if not Path(cfg.manifest).exists():
            return {}
        with open(cfg.manifest, encoding="utf-8") as f:
            man = json.load(f)
        if man["tokenizer"]["sha256"] != self.tokenizer_sha:
            raise ValueError("tokenizer does not match the one used to prepare the data")
        info = {"train_sha256": man["train"]["sha256"], "validation_sha256": man["validation"]["sha256"],
                "train_tokens": man["train"]["n_tokens"], "validation_tokens": man["validation"]["n_tokens"]}
        if cfg.verify_data_hash:
            for key, path in (("train_sha256", cfg.train_bin), ("validation_sha256", cfg.val_bin)):
                if sha256_file(path) != info[key]:
                    raise ValueError(f"{path} does not match the manifest hash")
        return info

    # ----------------------------------------------------------------- core step
    def micro_batches(self, step: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        c = self.cfg
        return [sample_batch(self.train_tokens, c.micro_batch_size, c.seq_len, c.seed, step, m)
                for m in range(c.grad_accum_steps)]

    def _autocast(self):
        if self.autocast_dtype is None:
            return torch.autocast(self.device.type, enabled=False)
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def train_step(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        """One optimizer update from a list of micro-batches. Returns loss / grad_norm / skipped."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        n = len(batches)
        loss_sum = torch.zeros((), device=self.device)
        for x, y in batches:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss = self._fwd(x, y)
            loss_sum += loss.detach().float() / n
            self.scaler.scale(loss / n).backward()
        loss_val = loss_sum.item()
        if not math.isfinite(loss_val):
            raise NonFiniteError(f"non-finite training loss at step {self.step}")
        if self.use_scaler:
            self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        grad_norm_val = float(grad_norm)
        skipped = not math.isfinite(grad_norm_val)
        if skipped and not self.use_scaler:
            raise NonFiniteError(f"non-finite gradient norm at step {self.step}")
        self.consecutive_skips = next_skip_count(self.consecutive_skips, skipped,
                                                 self.cfg.max_consecutive_skips)
        self.scaler.step(self.opt)  # the scaler itself skips the update on inf/NaN gradients
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        self.tokens_seen += self.cfg.tokens_per_step
        return {"loss": loss_val, "grad_norm": grad_norm_val, "skipped": skipped}

    # ----------------------------------------------------------------- evaluation / samples
    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        """Mean next-token cross-entropy over fixed, non-overlapping validation windows."""
        mb = self.cfg.eval_batches if max_batches is None else max_batches
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        for x, y in eval_batches(self.val_tokens, self.cfg.micro_batch_size, self.cfg.seq_len, mb):
            x, y = x.to(self.device), y.to(self.device)
            with self._autocast():
                _, loss = self._fwd(x, y)
            total += loss.item() * y.numel()
            count += y.numel()
        if was_training:
            self.model.train()
        if count == 0:
            raise ValueError("validation file too short for one window")
        mean = total / count
        out = {"val_loss": mean, "val_ppl": safe_exp(mean), "val_tokens": count}
        if self.source_val:  # per-source validation loss (Stage 2 mixes)
            self.model.eval()
            for name, toks in self.source_val.items():
                tot, cnt = 0.0, 0
                for x, y in eval_batches(toks, self.cfg.micro_batch_size, self.cfg.seq_len, max(1, mb // 5)):
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.no_grad(), self._autocast():
                        _, loss = self._fwd(x, y)
                    tot += loss.item() * y.numel()
                    cnt += y.numel()
                if cnt:
                    out[f"val_loss/{name}"] = tot / cnt
            if was_training:
                self.model.train()
        return out

    def sample_texts(self) -> list[dict[str, str]]:
        c = self.cfg
        out = []
        with self._autocast():
            for i, prompt in enumerate(c.prompts):
                text = generate(self.model, self.tok, prompt, c.max_new_tokens, c.temperature,
                                c.top_k, seed=c.seed + i, device=self.device)
                out.append({"prompt": prompt, "text": text})
        return out

    # ----------------------------------------------------------------- checkpoints
    def state(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_scaler else None,
            "step": self.step, "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss, "consecutive_skips": self.consecutive_skips,
            "rng": get_rng_state(),
            "config": {"train": self.cfg.to_dict(), "model": self.model_cfg.to_dict()},
            "tokenizer_sha256": self.tokenizer_sha, "data": self.data_info,
        }

    def save(self, path: str | Path | None = None) -> Path:
        check_finite_params(self.model)  # never persist a corrupted state
        path = Path(path or self.out_dir / "last.pt")
        atomic_save(self.state(), path)
        return path

    def load(self, path: str | Path) -> None:
        ck = load_checkpoint(path, map_location=self.device)
        if ck["config"]["model"] != self.model_cfg.to_dict():
            raise ValueError("checkpoint model architecture differs from the configuration")
        if ck["tokenizer_sha256"] != self.tokenizer_sha:
            raise ValueError("checkpoint tokenizer differs from the configured tokenizer")
        if ck.get("data") and self.data_info and ck["data"] != self.data_info:
            raise ValueError("checkpoint was trained on different data files")
        self.model.load_state_dict(ck["model"])  # weight tying survives: one shared tensor
        self.opt.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        if self.use_scaler and ck["scaler"] is not None:
            self.scaler.load_state_dict(ck["scaler"])
        self.step, self.tokens_seen = ck["step"], ck["tokens_seen"]
        self.best_val_loss = ck["best_val_loss"]
        self.consecutive_skips = ck["consecutive_skips"]
        set_rng_state(ck["rng"])

    def export(self, out_dir: str | Path | None = None, extra: dict | None = None) -> Path:
        info = {"step": self.step, "tokens_seen": self.tokens_seen,
                "best_val_loss": self.best_val_loss, **(extra or {})}
        return export_inference(self.model, self.cfg.tokenizer,
                                out_dir or self.out_dir / "export", info)

    # ----------------------------------------------------------------- logging
    def _log(self, record: dict[str, Any]) -> None:
        if not self.write_files:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def request_stop(self, reason: str = "stop_requested") -> None:
        self.stop_requested = True
        self.stop_reason = self.stop_reason or reason

    def _write_run_info(self) -> None:
        if not self.write_files:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        info = {"config": self.cfg.to_dict(), "model_params": self.model.num_params(),
                "device": str(self.device),
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "precision": str(self.autocast_dtype or "fp32"), "grad_scaler": self.use_scaler,
                "tokens_per_step": self.cfg.tokens_per_step, "total_steps": self.cfg.total_steps,
                "tokenizer_sha256": self.tokenizer_sha, "data": self.data_info,
                "python": platform.python_version(), "torch": torch.__version__}
        (self.out_dir / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))

    # ----------------------------------------------------------------- main loop
    def run(self, on_step: Callable[["Trainer", dict], None] | None = None,
            install_signal_handlers: bool = True) -> dict[str, Any]:
        c = self.cfg
        total = c.total_steps
        self._write_run_info()
        handlers = {}
        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, lambda s, f: self.request_stop(signal.Signals(s).name))
        start, start_step = time.time(), self.step
        last_val: dict[str, float] = {}
        try:
            if c.eval_at_start and self.step == 0:
                last_val = self.evaluate()
                self._log({"type": "eval", "step": 0, "tokens": 0, **last_val})
                print(f"step 0 | val_loss {last_val['val_loss']:.4f} | val_ppl {last_val['val_ppl']:.1f}")
            while self.step < total and not self.stop_requested:
                t0 = time.time()
                lr = self.scheduler.get_last_lr()[0]
                res = self.train_step(self.micro_batches(self.step))
                dt = time.time() - t0
                if on_step:
                    on_step(self, res)
                if self.step % c.log_interval == 0 or self.step == 1 or self.step == total:
                    rec = {"type": "train", "step": self.step, "tokens": self.tokens_seen,
                           "loss": res["loss"], "lr": lr, "grad_norm": res["grad_norm"],
                           "step_time_s": dt, "tokens_per_s": c.tokens_per_step / dt,
                           "skipped": res["skipped"],
                           "loss_scale": self.scaler.get_scale() if self.use_scaler else None,
                           "epochs": self.tokens_seen / len(self.train_tokens)}
                    if self.device.type == "cuda":
                        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                    self._log(rec)
                    print(f"step {self.step}/{total} | loss {res['loss']:.4f} | lr {lr:.2e} | "
                          f"gnorm {res['grad_norm']:.2f} | {rec['tokens_per_s']:,.0f} tok/s")
                if self.step % c.eval_interval == 0 or self.step == total:
                    last_val = self.evaluate()
                    self.best_val_loss = min(self.best_val_loss, last_val["val_loss"])
                    self._log({"type": "eval", "step": self.step, "tokens": self.tokens_seen, **last_val})
                    print(f"step {self.step} | val_loss {last_val['val_loss']:.4f} | "
                          f"val_ppl {last_val['val_ppl']:.1f}")
                if c.sample_interval and (self.step % c.sample_interval == 0 or self.step == total):
                    samples = self.sample_texts()
                    self._log({"type": "samples", "step": self.step, "samples": samples})
                    print(f"  sample: {samples[0]['text'][:160]!r}")
                if self.write_files and c.ckpt_interval and self.step % c.ckpt_interval == 0:
                    self.save()
                if c.max_hours and (time.time() - start) >= c.max_hours * 3600:
                    self.request_stop("max_hours")
            reason = self.stop_reason or "completed"
            if self.write_files:
                path = self.save()
                self._log({"type": "stop", "step": self.step, "tokens": self.tokens_seen,
                           "reason": reason, "checkpoint": str(path)})
                if reason in ("completed", "max_hours"):
                    self.export(extra={"stop_reason": reason, **last_val})
            return {"reason": reason, "step": self.step, "tokens": self.tokens_seen,
                    "seconds": time.time() - start, "steps_run": self.step - start_step, **last_val}
        except NonFiniteError as e:  # keep the last good checkpoint; do not overwrite it
            self._log({"type": "error", "step": self.step, "error": str(e)})
            raise
        finally:
            for sig, h in handlers.items():
                signal.signal(sig, h)
