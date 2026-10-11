"""Memory-mapped token files and deterministic next-token batch sampling.

Storage: one flat uint16 file per split, documents back to back, each followed by <|endoftext|>.

Training sampling ("random windows"): a window is `seq_len + 1` consecutive tokens starting
at a uniformly random offset; x = window[:-1], y = window[1:] (y[t] is the token after x[t]).
Windows are NOT aligned to document starts, so one window can hold the tail of one story,
an EOS, and the start of the next. Consequences (deliberate, standard GPT-style packing):
  * the EOS token is a normal target: the model learns to end a story;
  * the token after an EOS is predicted from the previous story's context (loss crosses
    documents) and attention is not masked at EOS, so a story can attend to its predecessor;
  * no padding is ever needed, so every token costs useful compute.

The batch for (seed, step, micro_index) is a pure function of those three integers, so
training is exactly reproducible and resuming needs no sampler state.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16  # vocab 8,192 << 65,535


def open_tokens(path: str | Path) -> np.ndarray:
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def sample_batch(tokens: np.ndarray, batch_size: int, seq_len: int, seed: int, step: int,
                 micro: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(tokens)
    if n < seq_len + 1:
        raise ValueError(f"token file too short ({n}) for seq_len {seq_len}")
    rng = np.random.default_rng([seed, step, micro])
    starts = rng.integers(0, n - seq_len, size=batch_size)  # last valid start is n-seq_len-1
    window = tokens[starts[:, None] + np.arange(seq_len + 1)[None, :]].astype(np.int64)
    data = torch.from_numpy(window)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def eval_batches(tokens: np.ndarray, batch_size: int, seq_len: int,
                 max_batches: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping consecutive windows covering the file in order (fixed, seed-free)."""
    n_windows = (len(tokens) - 1) // seq_len
    yielded = 0
    for first in range(0, n_windows, batch_size):
        if max_batches and yielded >= max_batches:
            return
        idx = range(first, min(first + batch_size, n_windows))
        window = np.stack([tokens[i * seq_len: i * seq_len + seq_len + 1] for i in idx]).astype(np.int64)
        data = torch.from_numpy(window)
        yield data[:, :-1].contiguous(), data[:, 1:].contiguous()
        yielded += 1
