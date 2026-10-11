"""Deterministic text cleaning, validation and exact de-duplication.

Design rules:
  * Pure functions, no randomness, no global state -> same input always gives same output.
  * One document in -> one document out (or a counted rejection). Documents are never
    merged or split here, so document boundaries survive to tokenization.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .data import sha256_text

# TinyStories contains UTF-8 curly quotes that were wrongly decoded as cp1252, e.g.
# "’" (E2 80 99) shows up as "â€™". The closing quote "”" (E2 80 9D) lost its last byte
# (0x9D is undefined in cp1252), leaving a bare "â€" -- so that one is a best guess.
# Order matters: longer / more specific sequences first, bare "â€" last.
MOJIBAKE_REPAIRS: tuple[tuple[str, str], ...] = (
    ("â€™", "’"),
    ("â€œ", "“"),
    ("â€˜", "‘"),
    ("â€“", "–"),
    ("â€”", "—"),
    ("â€¦", "…"),
    ("â€‹", ""),  # zero-width space bytes
    ("Ã©", "é"),
    ("â€", "”"),
)

# Typographic -> ASCII. Lossy on purpose: with only 8,192 tokens we do not want to spend
# merges on 3 byte-long punctuation variants that carry no extra story meaning.
PUNCT_FOLD = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
)

_SPACES = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text: str, unicode_form: str = "NFC") -> str:
    text = unicodedata.normalize(unicode_form, text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for bad, good in MOJIBAKE_REPAIRS:
        text = text.replace(bad, good)
    text = text.translate(PUNCT_FOLD)
    # Drop control / format characters (zero-width etc.) but keep newline and tab.
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn")
    )
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def validate_text(text: object, min_chars: int, max_chars: int) -> str | None:
    """Return a rejection reason, or None if the (normalized) document is acceptable."""
    if text is None or not isinstance(text, str):
        return "missing"
    if not text.strip():
        return "empty"
    if "\ufffd" in text:
        return "replacement_char"
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if "â€" in text or "Ã" in text:
        return "unrepaired_mojibake"
    return None


@dataclass
class PreprocessReport:
    n_input: int = 0
    n_kept: int = 0
    n_modified: int = 0  # kept docs whose text changed during normalization
    n_duplicates: int = 0
    rejected: Counter = field(default_factory=Counter)

    @property
    def n_invalid(self) -> int:
        return sum(self.rejected.values())

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_modified_by_normalization": self.n_modified,
            "n_exact_duplicates_removed": self.n_duplicates,
            "n_invalid": self.n_invalid,
            "invalid_by_reason": dict(sorted(self.rejected.items())),
        }


def preprocess_corpus(
    texts: Iterable[object],
    min_chars: int = 50,
    max_chars: int = 10_000,
    unicode_form: str = "NFC",
    exclude_hashes: set[str] | None = None,
) -> tuple[list[str], PreprocessReport]:
    """Clean, validate and de-duplicate. Keeps first occurrence and original order.

    `exclude_hashes` lets us drop validation documents that also appear in train
    (train/validation leakage); they are counted as duplicates.
    """
    report = PreprocessReport()
    seen: set[str] = set(exclude_hashes or ())
    kept: list[str] = []
    for raw in texts:
        report.n_input += 1
        if not isinstance(raw, str):
            report.rejected["missing"] += 1
            continue
        clean = normalize_text(raw, unicode_form)
        reason = validate_text(clean, min_chars, max_chars)
        if reason:
            report.rejected[reason] += 1
            continue
        digest = sha256_text(clean)
        if digest in seen:
            report.n_duplicates += 1
            continue
        seen.add(digest)
        kept.append(clean)
        report.n_modified += clean != raw
    report.n_kept = len(kept)
    return kept, report
