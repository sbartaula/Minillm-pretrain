"""Plot training/validation loss from a run's metrics.jsonl.

    python scripts/plot_metrics.py runs/tenslam_30m_runpod/metrics.jsonl --out reports/training_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", default="reports/training_curve.png")
    args = ap.parse_args()

    rows = load(Path(args.metrics))
    train = [r for r in rows if r["type"] == "train"]
    evals = [r for r in rows if r["type"] == "eval"]
    if not train or not evals:
        raise SystemExit("metrics file has no train/eval records")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot([r["tokens"] / 1e6 for r in train], [r["loss"] for r in train], label="train loss", alpha=0.7)
    ax1.plot([r["tokens"] / 1e6 for r in evals], [r["val_loss"] for r in evals], "o-", label="validation loss")
    ax1.set_xlabel("tokens processed (millions)")
    ax1.set_ylabel("cross-entropy (nats)")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([r["tokens"] / 1e6 for r in train], [r["lr"] for r in train], color="tab:green")
    ax2.set_xlabel("tokens processed (millions)")
    ax2.set_ylabel("learning rate")
    ax2.set_title("Warmup + cosine schedule")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    best = min(evals, key=lambda r: r["val_loss"])
    print(f"wrote {out} | best val_loss {best['val_loss']:.4f} (ppl {best['val_ppl']:.2f}) at step {best['step']}")


if __name__ == "__main__":
    main()
