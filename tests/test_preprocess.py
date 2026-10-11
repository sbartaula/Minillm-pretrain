import pytest

from tenslam.data import sha256_text
from tenslam.preprocess import normalize_text, preprocess_corpus, validate_text

LONG = "Once upon a time there was a small cat who liked to nap in the sun all day."


def test_mojibake_is_repaired_and_punctuation_folded():
    assert normalize_text("Itâ€™s â€œfineâ€ â€“ really") == "It's \"fine\" - really"
    assert normalize_text("cafÃ©") == "café"


def test_whitespace_rules_keep_paragraph_breaks():
    out = normalize_text("Hi.  \r\nThere   you\t\tgo. \n\n\n\nEnd.  ")
    assert out == "Hi.\nThere you go.\n\nEnd."
    assert normalize_text("a\tb") == "a\tb"  # a single tab is kept


def test_normalization_is_idempotent_and_deterministic():
    raw = "â€œHiâ€ she said.  \n\nBye\u200b!"
    once = normalize_text(raw)
    assert normalize_text(once) == once == normalize_text(raw)


def test_control_characters_removed_but_newline_kept():
    assert normalize_text("a\x00b\x07c\nd") == "abc\nd"


@pytest.mark.parametrize("text,reason", [
    (None, "missing"), (123, "missing"), ("   ", "empty"),
    ("x" * 10, "too_short"), ("x" * 200, "too_long"),
    (LONG + " \ufffd", "replacement_char"),
])
def test_validate_rejections(text, reason):
    assert validate_text(text, min_chars=50, max_chars=100) == reason


def test_validate_accepts_good_text():
    assert validate_text(LONG, 50, 10_000) is None


def test_corpus_counts_dedup_and_order():
    a, b = LONG, LONG.replace("cat", "dog")
    docs, rep = preprocess_corpus([a, None, b, a + "  ", "short", a, b])
    assert docs == [a, b]  # first occurrences, original order
    assert rep.n_input == 7 and rep.n_kept == 2
    assert rep.n_duplicates == 3  # a+"  " normalizes to a
    assert rep.rejected == {"missing": 1, "too_short": 1}
    assert rep.n_invalid == 2
    assert rep.n_kept + rep.n_duplicates + rep.n_invalid == rep.n_input


def test_documents_are_never_merged_or_split():
    a = LONG + "\n\nSecond paragraph of the same story, still one document."
    docs, _ = preprocess_corpus([a, LONG.replace("cat", "fox")])
    assert len(docs) == 2 and docs[0] == a  # internal paragraph break preserved


def test_exclude_hashes_removes_train_validation_leakage():
    docs, rep = preprocess_corpus([LONG], exclude_hashes={sha256_text(LONG)})
    assert docs == [] and rep.n_duplicates == 1


def test_pipeline_is_reproducible():
    raw = [LONG, None, LONG.upper(), "â€œ" + LONG + "â€"]
    assert preprocess_corpus(raw)[0] == preprocess_corpus(raw)[0]
