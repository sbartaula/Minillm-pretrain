import json
import random

import numpy as np

from tenslam.dataset import open_tokens
from tenslam.mix import MinHashLSH, chunk_text, prepare_mix
from tenslam import tokenizer as T

WORDS = ("light energy force mass nature mind virtue reason matter motion wave field time space "
         "justice soul habit memory cell atom planet orbit story river garden window").split()


def _doc(rng, n=120):
    return " ".join(rng.choice(WORDS) + str(rng.randint(0, 400)) for _ in range(n)) + "."


def test_chunk_text_respects_limit_and_loses_nothing():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(30))
    parts = chunk_text(text, 500)
    assert all(len(p) <= 500 for p in parts)
    assert "".join(p.replace(" ", "").replace("\n", "") for p in parts) == text.replace(" ", "").replace("\n", "")


def test_minhash_flags_near_duplicates_only():
    rng = random.Random(1)
    a = _doc(rng, 200)
    b = a.replace("light", "lamp", 1)  # one-word edit
    c = _doc(rng, 200)
    lsh = MinHashLSH()
    assert not lsh.seen_or_add(lsh.signature(a))
    assert lsh.seen_or_add(lsh.signature(b))
    assert not lsh.seen_or_add(lsh.signature(c))


def _write(path, docs, field="text"):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({field: d}) + "\n")


def test_prepare_mix_end_to_end(tmp_path):
    rng = random.Random(7)
    a = [_doc(rng) for _ in range(300)]
    a += [a[0], a[1], a[2].replace("force", "power", 1)]  # exact + near duplicates
    a.append("bad <|endoftext|> doc " + _doc(rng))
    books = ["\n\n".join(_doc(rng, 60) for _ in range(40)) for _ in range(20)]
    _write(tmp_path / "a.jsonl", a)
    _write(tmp_path / "b.jsonl", books, field="TEXT")
    out = tmp_path / "mix"
    cfg = {
        "seed": 3, "chars_per_token_estimate": 4.0,
        "mix": {"out_dir": str(out), "train_bin": str(out / "train.bin"),
                "validation_bin": str(out / "validation.bin"), "manifest": str(out / "m.json"),
                "validation_permille": 150,
                "preprocess": {"min_chars": 50, "max_chars": 1500, "unicode_form": "NFC"},
                "tokenizer": {"path": str(tmp_path / "tok" / "tokenizer.json"), "vocab_size": 600,
                              "special_tokens": ["<|pad|>", "<|bos|>", "<|endoftext|>"],
                              "sample_docs_per_source": 100}},
        "source": [
            {"name": "a", "kind": "jsonl", "path": str(tmp_path / "a.jsonl"),
             "train_tokens": 10**7, "validation_tokens": 10**6},
            {"name": "b", "kind": "jsonl", "path": str(tmp_path / "b.jsonl"), "text_field": "TEXT",
             "chunk_chars": 1000, "train_tokens": 10**7, "validation_tokens": 10**6},
        ],
    }
    m = prepare_mix(cfg)
    assert m["train"]["n_tokens"] > 0 and m["validation"]["n_tokens"] > 0
    by = {s["name"]: s for s in m["sources"]}
    assert by["a"]["rejected"]["special_token"] == 1
    assert by["a"]["rejected"]["exact_dup"] + by["a"]["rejected"]["fuzzy_dup"] >= 3
    assert sum(s["train_tokens"] for s in m["sources"]) == m["train"]["n_tokens"]
    train, val = open_tokens(out / "train.bin"), open_tokens(out / "validation.bin")
    assert len(train) == m["train"]["n_tokens"] and len(val) == m["validation"]["n_tokens"]
    tok = T.load_tokenizer(tmp_path / "tok" / "tokenizer.json")
    assert tok.get_vocab_size() <= 600 and int(train.max()) < tok.get_vocab_size()
    # no validation document appears in the training text
    def docs(bin_):
        eos = tok.token_to_id(T.EOS)
        arr = np.asarray(bin_)
        idx = np.flatnonzero(arr == eos)
        start = 0
        for i in idx:
            yield tok.decode(arr[start:i].tolist())
            start = i + 1
    tr = {d for d in docs(train)}
    assert not any(d in tr for d in docs(val))
    # both splits are non-empty for the book source
    assert by["b"]["validation_tokens"] > 0 and by["b"]["train_tokens"] > 0
