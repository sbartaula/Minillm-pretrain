"""Streaming TinyStories sampling plus JSONL / hashing helpers.

We never download the full dataset: `streaming=True` reads parquet shards lazily and
`.take(n)` stops the iteration after n rows.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_stream(
    rows: Iterable[dict[str, Any]],
    n: int,
    text_field: str = "text",
) -> list[str | None]:
    """Take the first n rows of an (already shuffled) row iterator, keeping raw values.

    Missing / non-string values are kept as None so that preprocessing can count them
    honestly instead of the loader silently hiding them.
    """
    out: list[str | None] = []
    for row in rows:
        if len(out) >= n:
            break
        value = row.get(text_field) if isinstance(row, dict) else None
        out.append(value if isinstance(value, str) else None)
    return out


def stream_tinystories(
    name: str,
    revision: str,
    split: str,
    n: int,
    seed: int,
    shuffle_buffer: int,
    text_field: str = "text",
) -> list[str | None]:
    """Deterministic sample of n stories from a streamed HF split."""
    from datasets import load_dataset  # imported lazily so unit tests stay offline

    ds = load_dataset(name, split=split, revision=revision, streaming=True)
    # Buffered shuffle: shard order and in-buffer order are both seeded -> reproducible.
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return sample_stream(ds, n, text_field)


def write_jsonl(path: str | Path, texts: Iterable[str | None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[str | None]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
