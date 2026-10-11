"""Corpus -> clean, de-duplicated, tokenized uint16 files + manifest."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import tokenizer as T
from .data import (read_jsonl, sha256_file, sha256_text, stream_tinystories, write_json,
                   write_jsonl)
from .dataset import TOKEN_DTYPE
from .preprocess import preprocess_corpus
from .stats import length_summary


def loose_key(text: str) -> str:
    """Case- and whitespace-insensitive identity, used for the train/validation leak check."""
    return sha256_text(" ".join(text.casefold().split()))


def filter_special_text(docs: list[str], special_tokens: list[str]) -> tuple[list[str], int]:
    """Drop documents containing a special-token string: the tokenizer would turn that text
    into a real control token (e.g. a fake document boundary)."""
    kept = [d for d in docs if not any(s in d for s in special_tokens)]
    return kept, len(docs) - len(kept)


def tokenize_to_bin(tok: Tokenizer, docs: list[str], path: str | Path, chunk: int = 2000) -> dict:
    """Write doc_1 EOS doc_2 EOS ... as uint16. Atomic (tmp file + rename)."""
    if tok.get_vocab_size() > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("vocab too large for uint16 storage")
    eos = tok.token_to_id(T.EOS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    lengths: list[int] = []
    with open(tmp, "wb") as f:
        for i in range(0, len(docs), chunk):
            encs = tok.encode_batch(docs[i:i + chunk])
            for e in encs:
                ids = e.ids + [eos]
                lengths.append(len(ids))
                f.write(np.asarray(ids, dtype=TOKEN_DTYPE).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    n_tokens = sum(lengths)
    assert path.stat().st_size == n_tokens * np.dtype(TOKEN_DTYPE).itemsize
    return {"n_docs": len(docs), "n_tokens": n_tokens,
            "tokens_per_doc": length_summary(lengths) if lengths else {},
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_token_files(train_raw, val_raw, tok: Tokenizer, train_bin, val_bin,
                      preprocess_cfg: dict, special_tokens: list[str]) -> dict:
    kw = dict(min_chars=preprocess_cfg["min_chars"], max_chars=preprocess_cfg["max_chars"],
              unicode_form=preprocess_cfg["unicode_form"])
    train, tr_rep = preprocess_corpus(train_raw, **kw)
    train, tr_special = filter_special_text(train, special_tokens)
    train_hashes = {sha256_text(t) for t in train}
    val, va_rep = preprocess_corpus(val_raw, exclude_hashes=train_hashes, **kw)
    val, va_special = filter_special_text(val, special_tokens)
    # Stricter second pass (case/whitespace-insensitive) on top of the exact-hash check.
    train_loose = {loose_key(t) for t in train}
    n_before = len(val)
    val = [t for t in val if loose_key(t) not in train_loose]
    loose_removed = n_before - len(val)
    assert not ({sha256_text(t) for t in val} & train_hashes)
    assert not ({loose_key(t) for t in val} & train_loose), "train/validation overlap"
    train_info = tokenize_to_bin(tok, train, train_bin)
    val_info = tokenize_to_bin(tok, val, val_bin)
    eos = tok.token_to_id(T.EOS)
    for info, path in ((train_info, train_bin), (val_info, val_bin)):
        arr = np.fromfile(path, dtype=TOKEN_DTYPE)
        assert int((arr == eos).sum()) == info["n_docs"], "EOS count must equal document count"
        assert arr[-1] == eos
    return {
        "train": {**train_info, "n_raw": tr_rep.n_input, "preprocess": tr_rep.to_dict(),
                  "removed_special_token_text": tr_special},
        "validation": {**val_info, "n_raw": va_rep.n_input, "preprocess": va_rep.to_dict(),
                       "removed_special_token_text": va_special,
                       "removed_loose_overlap_with_train": loose_removed,
                       "n_overlap_with_train_after_filtering": 0},
        "eos_id": eos, "dtype": np.dtype(TOKEN_DTYPE).name,
    }


def _collect(cfg: dict, split: str, n: int, seed: int, raw_path: str, force: bool) -> list:
    path = Path(raw_path)
    if path.exists() and not force:
        texts = list(read_jsonl(path))
        if len(texts) == n:
            print(f"{split}: reusing {raw_path} ({n} raw docs)")
            return texts
    d = cfg["dataset"]
    texts = stream_tinystories(d["name"], d["revision"], split, n, seed,
                               cfg["sample"]["shuffle_buffer"], d["text_field"])
    write_jsonl(raw_path, texts)
    gc.collect()  # let HF streaming threads finish before interpreter shutdown
    print(f"{split}: streamed {len(texts)} docs -> {raw_path}")
    return texts


def prepare_corpus(cfg: dict, train_docs: int | None = None, validation_docs: int | None = None,
                   force: bool = False) -> dict:
    from importlib.metadata import version
    prep, p = cfg["prepare"], cfg["paths"]
    n_train = train_docs or prep["train_docs"]
    n_val = validation_docs or prep["validation_docs"]
    seed = cfg["seed"]
    train_raw = _collect(cfg, "train", n_train, seed, prep["raw_train"], force)
    val_raw = _collect(cfg, "validation", n_val, seed + 1, prep["raw_validation"], force)
    tok = T.load_tokenizer(p["tokenizer"])
    result = build_token_files(train_raw, val_raw, tok, prep["train_bin"], prep["validation_bin"],
                               cfg["preprocess"], cfg["tokenizer"]["special_tokens"])
    manifest = {
        "dataset": cfg["dataset"]["name"], "revision": cfg["dataset"]["revision"],
        "license": "cdla-sharing-1.0",
        "sampling": {"seed_train": seed, "seed_validation": seed + 1,
                     "shuffle_buffer": cfg["sample"]["shuffle_buffer"],
                     "requested_train_docs": n_train, "requested_validation_docs": n_val,
                     "method": "HF streaming + seeded buffered shuffle, first N rows"},
        "preprocess_config": cfg["preprocess"],
        "tokenizer": {"path": p["tokenizer"], "sha256": sha256_file(p["tokenizer"]),
                      "vocab_size": tok.get_vocab_size()},
        "raw_files": {"train": {"path": prep["raw_train"], "sha256": sha256_file(prep["raw_train"])},
                      "validation": {"path": prep["raw_validation"],
                                     "sha256": sha256_file(prep["raw_validation"])}},
        "files": {"train": prep["train_bin"], "validation": prep["validation_bin"]},
        "libraries": {"datasets": version("datasets"), "tokenizers": version("tokenizers"),
                      "numpy": version("numpy")},
        **result,
    }
    write_json(prep["manifest"], manifest)
    return manifest


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
