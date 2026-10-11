"""Stage 2: multi-source corpus -> one tokenizer -> uint16 token files.

Pipeline (everything streams to disk, memory stays bounded):
  1. collect   per source: stream, chunk long documents, clean, exact-dedup, MinHash fuzzy-dedup,
               split train/validation BY PARENT DOCUMENT (a book never straddles the split),
               drop validation docs that are near-duplicates of training docs.
  2. tokenizer train a byte-level BPE on a sample of the TRAINING documents only.
  3. tokenize  write train.bin, validation.bin (all sources) and one validation bin per source.
  4. manifest  hashes, token counts per source, dataset revisions, licences.

The manifest keeps the same `train` / `validation` / `tokenizer` keys as Stage 1, so the trainer's
integrity check works unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import tokenizer as T
from .data import read_jsonl, sha256_file, sha256_text, write_json
from .dataset import TOKEN_DTYPE
from .preprocess import normalize_text, validate_text
from .prepare import loose_key

_WORD = re.compile(r"\w+")


# ------------------------------------------------------------------ chunking
def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split long text at paragraph (then line, then space) boundaries into pieces <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind("\n\n"), -1)
        if cut < max_chars // 2:
            cut = max(window.rfind("\n"), window.rfind(". ") + 1, window.rfind(" "))
        if cut < max_chars // 4:
            cut = max_chars
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return [c for c in chunks if c]


# ------------------------------------------------------------------ MinHash dedup
class MinHashLSH:
    """Near-duplicate detector: 5-word shingles, `perms` minhashes, banded LSH."""

    def __init__(self, perms: int = 64, bands: int = 16, shingle: int = 5, seed: int = 0):
        if perms % bands:
            raise ValueError("perms must be divisible by bands")
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**31 - 1, size=perms, dtype=np.uint64)
        self.b = rng.integers(0, 2**31 - 1, size=perms, dtype=np.uint64)
        self.perms, self.bands, self.rows, self.shingle = perms, bands, perms // bands, shingle
        self.buckets: list[set[int]] = [set() for _ in range(bands)]

    def signature(self, text: str) -> np.ndarray | None:
        words = _WORD.findall(text.casefold())
        if len(words) < self.shingle:
            return None
        hs = {zlib.crc32(" ".join(words[i:i + self.shingle]).encode()) & 0x7FFFFFFF
              for i in range(len(words) - self.shingle + 1)}
        x = np.fromiter(hs, dtype=np.uint64, count=len(hs))
        # (a*x + b) mod 2^32-ish universal hash; uint64 arithmetic wraps, which is fine here
        sig = ((x[:, None] * self.a[None, :] + self.b[None, :]) & np.uint64(0xFFFFFFFF)).min(axis=0)
        return sig

    def _keys(self, sig: np.ndarray) -> list[int]:
        r = self.rows
        return [int.from_bytes(hashlib.blake2b(sig[i * r:(i + 1) * r].tobytes(), digest_size=8).digest(), "little")
                for i in range(self.bands)]

    def query(self, sig: np.ndarray) -> bool:
        return any(k in self.buckets[i] for i, k in enumerate(self._keys(sig)))

    def add(self, sig: np.ndarray) -> None:
        for i, k in enumerate(self._keys(sig)):
            self.buckets[i].add(k)

    def seen_or_add(self, sig: np.ndarray) -> bool:
        dup = self.query(sig)
        if not dup:
            self.add(sig)
        return dup


# ------------------------------------------------------------------ sources
def iter_source(src: dict, seed: int) -> Iterator[str]:
    """Yield raw parent documents. kind = 'hf' (streaming, pinned revision) or 'jsonl' (tests)."""
    field = src.get("text_field", "text")
    if src.get("kind", "hf") == "jsonl":
        with open(src["path"], encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                yield row[field] if isinstance(row, dict) else row
        return
    from datasets import load_dataset
    ds = load_dataset(src["dataset"], src.get("config"), split=src.get("split", "train"),
                      revision=src.get("revision"), streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=int(src.get("shuffle_buffer", 5000)))
    for row in ds:
        yield row[field]


def _is_val(parent_text: str, seed: int, permille: int) -> bool:
    h = int(sha256_text(f"{seed}:{parent_text[:500]}")[:8], 16)
    return h % 1000 < permille


def collect_source(src: dict, mix: dict, out_dir: Path, seed: int, limit_docs: int | None = None) -> dict:
    """Stream one source into <out_dir>/<name>.{train,validation}.jsonl. Returns its report."""
    name = src["name"]
    pp = {**mix["preprocess"], **{k: src[k] for k in ("min_chars", "max_chars") if k in src}}
    specials = mix["tokenizer"]["special_tokens"]
    cpt = float(mix.get("chars_per_token_estimate", 4.0))
    train_budget = int(src["train_tokens"] * cpt)
    val_budget = int(src.get("validation_tokens", 0) * cpt)
    permille = int(mix.get("validation_permille", 20))
    chunk = int(src.get("chunk_chars", pp["max_chars"]))
    lsh = MinHashLSH(seed=seed)
    exact: set[str] = set()
    paths = {s: out_dir / f"{name}.{s}.jsonl" for s in ("train", "validation")}
    files = {s: open(p, "w", encoding="utf-8") for s, p in paths.items()}
    chars = {"train": 0, "validation": 0}
    counts = {"train": 0, "validation": 0}
    rej = {"invalid": 0, "special_token": 0, "exact_dup": 0, "fuzzy_dup": 0, "val_overlap_train": 0}
    n_parents = 0
    val_sigs: list[tuple[np.ndarray, str]] = []
    try:
        for raw in iter_source(src, seed):
            if limit_docs and n_parents >= limit_docs:
                break
            if chars["train"] >= train_budget and chars["validation"] >= val_budget:
                break
            n_parents += 1
            if not isinstance(raw, str):
                rej["invalid"] += 1
                continue
            split = "validation" if _is_val(raw, seed, permille) else "train"
            if chars[split] >= (train_budget if split == "train" else val_budget):
                continue
            for piece in chunk_text(normalize_text(raw, pp["unicode_form"]), chunk):
                if validate_text(piece, pp["min_chars"], pp["max_chars"]):
                    rej["invalid"] += 1
                    continue
                if any(s in piece for s in specials):
                    rej["special_token"] += 1
                    continue
                key = loose_key(piece)
                if key in exact:
                    rej["exact_dup"] += 1
                    continue
                exact.add(key)
                sig = lsh.signature(piece)
                if sig is not None:
                    if split == "train":
                        if lsh.seen_or_add(sig):
                            rej["fuzzy_dup"] += 1
                            continue
                    else:
                        val_sigs.append((sig, piece))  # checked against train at the end
                        continue
                files[split].write(json.dumps({"text": piece}, ensure_ascii=False) + "\n")
                chars[split] += len(piece)
                counts[split] += 1
        # validation: keep only docs with no near-duplicate in this source's training set
        for sig, piece in val_sigs:
            if chars["validation"] >= val_budget:
                break
            if lsh.query(sig):
                rej["val_overlap_train"] += 1
                continue
            files["validation"].write(json.dumps({"text": piece}, ensure_ascii=False) + "\n")
            chars["validation"] += len(piece)
            counts["validation"] += 1
    finally:
        for f in files.values():
            f.close()
    return {"name": name, "parents_read": n_parents, "docs": counts, "chars": chars,
            "rejected": rej, "paths": {k: str(v) for k, v in paths.items()}}


# ------------------------------------------------------------------ tokenizer + token files
def tokenizer_sample(reports: list[dict], max_docs_per_source: int, specials: list[str]) -> Iterator[str]:
    for r in reports:
        n = r["docs"]["train"]
        stride = max(1, n // max_docs_per_source)
        for i, text in enumerate(read_jsonl_text(r["paths"]["train"])):
            if i % stride == 0:
                yield text


def read_jsonl_text(path: str | Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def _tokenize_stream(tok, path: Path, out_f, chunk: int = 2000) -> int:
    eos = tok.token_to_id(T.EOS)
    n, batch = 0, []

    def flush():
        nonlocal n
        for e in tok.encode_batch(batch):
            ids = np.asarray(e.ids + [eos], dtype=TOKEN_DTYPE)
            out_f.write(ids.tobytes())
            n += len(ids)
        batch.clear()

    for text in read_jsonl_text(path):
        batch.append(text)
        if len(batch) >= chunk:
            flush()
    if batch:
        flush()
    return n


def prepare_mix(cfg: dict, limit_docs: int | None = None, force: bool = False) -> dict:
    mix = cfg["mix"]
    seed = int(cfg.get("seed", 1337))
    out = Path(mix["out_dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    sources = cfg["source"]
    reports: list[dict] = []
    for src in sources:
        rp = work / f"{src['name']}.report.json"
        if rp.exists() and not force:
            reports.append(json.loads(rp.read_text()))
            print(f"[{src['name']}] reusing {rp}")
            continue
        print(f"[{src['name']}] collecting ...", flush=True)
        r = collect_source(src, mix, work, seed, limit_docs)
        r.update(dataset=src.get("dataset", src.get("path")), config=src.get("config"),
                 revision=src.get("revision"), license=src.get("license"))
        write_json(rp, r)
        reports.append(r)
        print(f"[{src['name']}] docs {r['docs']} chars {r['chars']} rejected {r['rejected']}", flush=True)

    tk = mix["tokenizer"]
    tok_path = Path(tk["path"])
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    print("training tokenizer on training documents only ...", flush=True)
    tok = T.train_tokenizer(tokenizer_sample(reports, int(tk["sample_docs_per_source"]), tk["special_tokens"]),
                            vocab_size=int(tk["vocab_size"]), special_tokens=tk["special_tokens"],
                            min_frequency=int(tk.get("min_frequency", 2)))
    if tok.get_vocab_size() != int(tk["vocab_size"]):
        print(f"WARNING: tokenizer has {tok.get_vocab_size()} tokens, requested {tk['vocab_size']}; "
              "set model vocab_size to the real value (too little data?)", flush=True)
    tok.save(str(tok_path))
    tok_sha = sha256_file(tok_path)

    train_bin, val_bin = Path(mix["train_bin"]), Path(mix["validation_bin"])
    train_bin.parent.mkdir(parents=True, exist_ok=True)
    per_source = []
    tmp_t, tmp_v = train_bin.with_suffix(".bin.tmp"), val_bin.with_suffix(".bin.tmp")
    with open(tmp_t, "wb") as ft, open(tmp_v, "wb") as fv:
        for r in reports:
            name = r["name"]
            print(f"[{name}] tokenizing ...", flush=True)
            nt = _tokenize_stream(tok, Path(r["paths"]["train"]), ft)
            sv = out / f"validation_{name}.bin"
            with open(sv, "wb") as fs:
                nv = _tokenize_stream(tok, Path(r["paths"]["validation"]), fs)
            with open(sv, "rb") as fs:
                fv.write(fs.read())
            per_source.append({"name": name, "train_tokens": nt, "validation_tokens": nv,
                               "validation_bin": str(sv), "validation_sha256": sha256_file(sv),
                               "dataset": r.get("dataset"), "config": r.get("config"),
                               "revision": r.get("revision"), "license": r.get("license"),
                               "docs": r["docs"], "rejected": r["rejected"]})
    for tmp, final in ((tmp_t, train_bin), (tmp_v, val_bin)):
        os.replace(tmp, final)
    n_train = sum(s["train_tokens"] for s in per_source)
    n_val = sum(s["validation_tokens"] for s in per_source)
    for s in per_source:
        s["train_share"] = round(s["train_tokens"] / max(n_train, 1), 4)
    manifest = {
        "stage": 2, "seed": seed,
        "train": {"n_tokens": n_train, "bytes": train_bin.stat().st_size, "sha256": sha256_file(train_bin)},
        "validation": {"n_tokens": n_val, "bytes": val_bin.stat().st_size, "sha256": sha256_file(val_bin)},
        "tokenizer": {"path": str(tok_path), "sha256": tok_sha, "vocab_size": tok.get_vocab_size(),
                      "trained_on": "training documents only (sampled per source)"},
        "sources": per_source,
    }
    write_json(mix["manifest"], manifest)
    return manifest
