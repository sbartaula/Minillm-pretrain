from pathlib import Path

import pytest

from tenslam import tokenizer as T
from tenslam.data import read_jsonl

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play.",
    "Tom and his dog went to the park. \"Look!\" said Tom. The dog barked.",
    "The sun was bright and the birds sang. Everyone was happy that day.",
] * 40
SPECIALS = [T.PAD, T.BOS, T.EOS]


@pytest.fixture(scope="module")
def tok():
    return T.train_tokenizer(CORPUS, vocab_size=400, special_tokens=SPECIALS, min_frequency=1)


def test_special_token_ids_are_first_and_ordered(tok):
    assert [tok.token_to_id(t) for t in SPECIALS] == [0, 1, 2]


def test_vocab_never_exceeds_requested_and_contains_all_256_bytes(tok):
    assert tok.get_vocab_size() <= 400
    assert tok.get_vocab_size() >= 256 + len(SPECIALS)


@pytest.mark.parametrize("text", [
    "Once upon a time.",
    "multi\nline\n\ntext with\ttabs",
    "  leading and  double  spaces ",
    "café naïve 日本語 😀",
    "",
    "<|endoftext|> literal",
])
def test_round_trip_is_lossless(tok, text):
    assert T.decode(tok, tok.encode(text).ids) == text


def test_unseen_bytes_never_fail(tok):
    text = "".join(chr(c) for c in range(32, 400))
    assert T.decode(tok, tok.encode(text).ids) == text


def test_encode_document_appends_eos_only_once(tok):
    ids = T.encode_document(tok, "The dog barked.")
    assert ids[-1] == tok.token_to_id(T.EOS) and ids.count(tok.token_to_id(T.EOS)) == 1
    assert T.encode_document(tok, "x", add_eos=False)[-1] != tok.token_to_id(T.EOS)


def test_encode_corpus_matches_single_document_encoding(tok):
    docs = CORPUS[:3]
    assert T.encode_corpus(tok, docs) == [T.encode_document(tok, d) for d in docs]


def test_training_is_deterministic(tok):
    again = T.train_tokenizer(CORPUS, vocab_size=400, special_tokens=SPECIALS, min_frequency=1)
    assert again.to_str() == tok.to_str()


def test_tokenizer_only_learns_from_its_training_docs(tok):
    held_out = "Zzyzx quokka"
    assert tok.token_to_id("Ġquokka") is None and tok.token_to_id("quokka") is None
    assert T.decode(tok, tok.encode(held_out).ids) == held_out  # still encodable via bytes


def test_save_and_load_round_trip(tok, tmp_path):
    p = tmp_path / "tokenizer.json"
    tok.save(str(p))
    assert T.load_tokenizer(p).encode("Hello there").ids == tok.encode("Hello there").ids


ARTIFACT = Path("artifacts/tokenizer/tokenizer.json")
VAL = Path("data/processed/validation_clean.jsonl")


@pytest.mark.skipif(not (ARTIFACT.exists() and VAL.exists()), reason="run `tenslam all` first")
def test_real_tokenizer_on_validation_documents():
    real = T.load_tokenizer(ARTIFACT)
    assert real.get_vocab_size() == 8192
    assert [real.token_to_id(t) for t in SPECIALS] == [0, 1, 2]
    for doc in list(read_jsonl(VAL))[:200]:
        assert T.decode(real, real.encode(doc).ids) == doc
