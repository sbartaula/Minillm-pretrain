"""Corpus statistics used in the data report."""
from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence


def _pct(sorted_vals: Sequence[int], p: float) -> int:
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def length_summary(values: Sequence[int]) -> dict:
    s = sorted(values)
    return {
        "min": s[0], "p1": _pct(s, .01), "p50": _pct(s, .5), "p99": _pct(s, .99),
        "max": s[-1], "mean": round(statistics.fmean(s), 2),
    }


def corpus_stats(docs: Sequence[str]) -> dict:
    chars = [len(d) for d in docs]
    words = [len(d.split()) for d in docs]
    charset = Counter(ch for d in docs for ch in d)
    non_ascii = {ch: n for ch, n in charset.items() if ord(ch) > 127}
    return {
        "n_docs": len(docs),
        "total_chars": sum(chars),
        "total_words": sum(words),
        "chars_per_doc": length_summary(chars),
        "words_per_doc": length_summary(words),
        "distinct_characters": len(charset),
        "non_ascii_characters": dict(sorted(non_ascii.items(), key=lambda kv: -kv[1])),
    }
