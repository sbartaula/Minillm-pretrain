"""Throughput / memory benchmark and run-time & cost estimates.

Nothing here is guessed: tokens/s, VRAM, evaluation time and checkpoint time are measured on
the device the benchmark runs on. A cost is reported only if the benchmark ran on CUDA AND an
hourly price was configured explicitly; otherwise it is reported as unknown.
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch

from .train import Trainer, TrainConfig

TOKEN_TARGETS = (10_000_000, 50_000_000, 100_000_000, 200_000_000)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(cfg: TrainConfig, warmup_steps: int = 3, steps: int = 10,
                  eval_batches: int = 5) -> dict:
    """Short real training run (no files written) after a warmup that is excluded from timing."""
    tr = Trainer(cfg, write_files=False)
    dev = tr.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    times: list[float] = []
    losses: list[float] = []
    for i in range(warmup_steps + steps):
        _sync(dev)
        t0 = time.perf_counter()
        res = tr.train_step(tr.micro_batches(tr.step))
        _sync(dev)
        if i >= warmup_steps:  # warmup absorbs CUDA context init, kernel autotune, allocator growth
            times.append(time.perf_counter() - t0)
        losses.append(res["loss"])
    tps = [cfg.tokens_per_step / t for t in times]

    _sync(dev); t0 = time.perf_counter()
    tr.evaluate(eval_batches)
    _sync(dev); eval_s = (time.perf_counter() - t0) / eval_batches  # seconds per eval batch
    _sync(dev); t0 = time.perf_counter()
    tr.sample_texts()
    _sync(dev); sample_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        ck = tr.save(Path(d) / "bench.pt")
        ckpt_s, ckpt_bytes = time.perf_counter() - t0, ck.stat().st_size

    full_eval_batches = cfg.eval_batches or max(1, (len(tr.val_tokens) - 1) // (cfg.seq_len * cfg.micro_batch_size))
    return {
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "gpu_total_gb": round(torch.cuda.get_device_properties(dev).total_memory / 2**30, 2)
        if dev.type == "cuda" else None,
        "torch": torch.__version__, "precision": str(tr.autocast_dtype or "fp32"),
        "grad_scaler": tr.use_scaler, "compile": cfg.compile,
        "model_params": tr.model.num_params(),
        "micro_batch_size": cfg.micro_batch_size, "grad_accum_steps": cfg.grad_accum_steps,
        "seq_len": cfg.seq_len, "tokens_per_step": cfg.tokens_per_step,
        "warmup_steps": warmup_steps, "timed_steps": steps,
        "step_time_s_median": statistics.median(times), "step_time_s_mean": statistics.fmean(times),
        "tokens_per_s_mean": statistics.fmean(tps), "tokens_per_s_median": statistics.median(tps),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3)
        if dev.type == "cuda" else None,
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2**30, 3)
        if dev.type == "cuda" else None,
        "eval_seconds_per_batch": eval_s,
        "eval_pass_seconds": eval_s * full_eval_batches,
        "sample_round_seconds": sample_s,
        "checkpoint_save_seconds": ckpt_s, "checkpoint_bytes": ckpt_bytes,
        "loss_first": losses[0], "loss_last": losses[-1],
        "all_losses_finite": all(math.isfinite(x) for x in losses),
        "train_tokens_available": int(len(tr.train_tokens)),
        "eval_interval": cfg.eval_interval, "ckpt_interval": cfg.ckpt_interval,
        "sample_interval": cfg.sample_interval,
    }


def estimate(bench: dict, price_eur_per_hour: float | None = None, setup_minutes: float = 15.0,
             targets: tuple[int, ...] = TOKEN_TARGETS) -> dict:
    """Wall-clock and cost estimates scaled from a measured benchmark."""
    tps = bench["tokens_per_s_mean"]
    tps_step = bench["tokens_per_step"]
    on_cuda = str(bench["device"]).startswith("cuda")
    rows = []
    for tokens in targets:
        steps = math.ceil(tokens / tps_step)
        train_s = tokens / tps
        n_eval = steps // bench["eval_interval"] + 1
        n_ckpt = steps // bench["ckpt_interval"] + 1
        n_sample = steps // bench["sample_interval"] + 1 if bench["sample_interval"] else 0
        overhead_s = (n_eval * bench["eval_pass_seconds"] + n_ckpt * bench["checkpoint_save_seconds"]
                      + n_sample * bench["sample_round_seconds"])
        total_h = (train_s + overhead_s) / 3600 + setup_minutes / 60
        if not on_cuda:
            cost, why = None, "UNKNOWN: this benchmark ran on CPU, not a GPU"
        elif price_eur_per_hour is None:
            cost, why = None, "UNKNOWN: no hourly price configured"
        else:
            cost, why = round(total_h * price_eur_per_hour, 2), "measured throughput x configured price"
        rows.append({
            "tokens": tokens, "steps": steps,
            "epochs_over_available_train_tokens": round(tokens / bench["train_tokens_available"], 2),
            "train_hours": round(train_s / 3600, 3), "eval_ckpt_sample_overhead_hours": round(overhead_s / 3600, 3),
            "setup_hours_ASSUMED": round(setup_minutes / 60, 3), "total_hours": round(total_h, 3),
            "cost_eur": cost, "cost_note": why})
    return {"based_on_device": bench["device"], "gpu_name": bench["gpu_name"],
            "measured_tokens_per_s": round(tps, 1), "price_eur_per_hour": price_eur_per_hour,
            "setup_minutes_ASSUMPTION_not_measured": setup_minutes, "rows": rows}


def format_estimate(est: dict) -> str:
    lines = [f"Estimate based on measured {est['measured_tokens_per_s']:,.0f} tok/s on "
             f"{est['gpu_name'] or est['based_on_device']} "
             f"(price/h: {est['price_eur_per_hour'] if est['price_eur_per_hour'] is not None else 'not set'})",
             f"{'tokens':>12} {'epochs':>7} {'train h':>8} {'overhead h':>10} {'total h':>8}  cost"]
    for r in est["rows"]:
        cost = f"EUR {r['cost_eur']:.2f}" if r["cost_eur"] is not None else r["cost_note"]
        lines.append(f"{r['tokens']:>12,} {r['epochs_over_available_train_tokens']:>7} {r['train_hours']:>8} "
                     f"{r['eval_ckpt_sample_overhead_hours']:>10} {r['total_hours']:>8}  {cost}")
    lines.append(f"total includes {est['setup_minutes_ASSUMPTION_not_measured']:.0f} min setup (an assumption, not measured)")
    return "\n".join(lines)


def run_sweep(cfg: TrainConfig, micro_batch_sizes: list[int], **kw) -> list[dict]:
    """Try several micro-batch sizes; report OOM instead of crashing."""
    out = []
    for mb in micro_batch_sizes:
        c = dataclasses.replace(cfg, micro_batch_size=mb)
        try:
            out.append(run_benchmark(c, **kw))
        except torch.cuda.OutOfMemoryError:
            out.append({"micro_batch_size": mb, "error": "CUDA out of memory"})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def save_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
