"""Learning-rate schedule: linear warmup, then cosine decay to a floor."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def lr_at(step: int, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """LR used for the update that *starts* at optimizer step `step` (0-based).

    warmup : lr = peak * (step+1)/warmup          (never exactly 0 at step 0)
    cosine : lr = min + 0.5*(peak-min)*(1+cos(pi*p)),  p = (step-warmup)/(total-warmup)
    after `total_steps` the LR stays at min_lr.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_scheduler(opt: Optimizer, peak_lr: float, min_lr: float, warmup_steps: int,
                   total_steps: int) -> LambdaLR:
    # LambdaLR multiplies the optimizer's base lr (= peak_lr) by this factor.
    return LambdaLR(opt, lambda s: lr_at(s, peak_lr, min_lr, warmup_steps, total_steps) / peak_lr)
