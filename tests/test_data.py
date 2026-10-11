import sys
import types

from tenslam import data
from tenslam.config import load_config


class FakeStream:
    """Mimics a streaming HF dataset: .shuffle(seed, buffer_size) is deterministic."""

    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed, buffer_size):
        import random
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeStream(rows)

    def __iter__(self):
        return iter(self.rows)


def _patch_datasets(monkeypatch, n_rows=100):
    rows = [{"text": f"story {i}"} for i in range(n_rows)]
    calls = {}

    def fake_load_dataset(name, split, revision, streaming):
        calls.update(name=name, split=split, revision=revision, streaming=streaming)
        return FakeStream(rows)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    return calls


def test_sample_stream_takes_exactly_n_and_keeps_missing_as_none():
    rows = [{"text": "a"}, {"text": None}, {"other": "x"}, {"text": 5}, {"text": "e"}]
    assert data.sample_stream(iter(rows), 4) == ["a", None, None, None]


def test_sample_stream_handles_short_input():
    assert data.sample_stream(iter([{"text": "a"}]), 10) == ["a"]


def test_stream_is_streaming_pinned_and_reproducible(monkeypatch):
    calls = _patch_datasets(monkeypatch)
    a = data.stream_tinystories("ds", "rev123", "train", 10, seed=1, shuffle_buffer=50)
    b = data.stream_tinystories("ds", "rev123", "train", 10, seed=1, shuffle_buffer=50)
    c = data.stream_tinystories("ds", "rev123", "train", 10, seed=2, shuffle_buffer=50)
    assert a == b and a != c and len(a) == 10
    assert calls["streaming"] is True and calls["revision"] == "rev123"


def test_jsonl_roundtrip_preserves_unicode_newlines_and_none(tmp_path):
    texts = ["héllo\nworld", None, "😀 \"quoted\""]
    p = tmp_path / "x.jsonl"
    data.write_jsonl(p, texts)
    assert list(data.read_jsonl(p)) == texts
    assert len(p.read_text(encoding="utf-8").splitlines()) == 3  # one line per document


def test_hashes_are_stable(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    assert data.sha256_file(p) == data.sha256_text("abc")


def test_project_config_matches_session_goals():
    cfg = load_config("configs/data.toml")
    assert cfg["tokenizer"]["vocab_size"] == 8192
    assert cfg["sample"]["train_docs"] == 10000
    assert len(cfg["dataset"]["revision"]) == 40  # pinned commit sha
