import pytest

from tenslam import tokenizer as T
from tenslam.prepare import build_token_files

STORIES = [
    f"Once upon a time there was a little {w} named {n}. {n} loved to play in the {p}. "
    f"One day {n} found a shiny {o} and was very happy."
    for w, n, p, o in [
        ("girl", "Lily", "park", "stone"), ("boy", "Tom", "garden", "key"),
        ("robot", "Beep", "yard", "coin"), ("cat", "Mia", "forest", "ball"),
        ("dog", "Rex", "field", "bone"), ("bird", "Pip", "tree", "seed"),
    ] * 12
]
STORIES = [f"{s} Story number {i}." for i, s in enumerate(STORIES)]
SPECIALS = [T.PAD, T.BOS, T.EOS]
PREP = {"min_chars": 10, "max_chars": 10000, "unicode_form": "NFC"}


@pytest.fixture(scope="session")
def tok():
    return T.train_tokenizer(STORIES, vocab_size=400, special_tokens=SPECIALS, min_frequency=1)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, tok):
    """Tiny prepared corpus (train/validation bins + manifest-free) in a temp dir."""
    d = tmp_path_factory.mktemp("corpus")
    train, val = STORIES[:60], STORIES[60:] + [STORIES[0].upper(), STORIES[1] + "  "]
    tok_path = d / "tokenizer.json"
    tok.save(str(tok_path))
    info = build_token_files(train, val, tok, d / "train.bin", d / "val.bin", PREP, SPECIALS)
    return {"dir": d, "tok_path": tok_path, "train": d / "train.bin", "val": d / "val.bin",
            "info": info, "train_docs": train, "val_docs": val}
