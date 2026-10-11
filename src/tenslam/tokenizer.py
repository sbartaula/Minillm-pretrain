"""Byte-level BPE tokenizer trained only on our training documents."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"
PAD = "<|pad|>"
BOS = "<|bos|>"


def train_tokenizer(
    docs: Iterable[str],
    vocab_size: int = 8192,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
) -> Tokenizer:
    special_tokens = special_tokens or [PAD, BOS, EOS]
    tok = Tokenizer(models.BPE())  # byte-level => no <unk> needed, every string is encodable
    # GPT-2 style: split into words/punctuation first (regex), then map bytes to
    # printable unicode. Merges never cross word boundaries.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,  # receive ids 0..k-1, in order
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes always present
        show_progress=False,
    )
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_document(tok: Tokenizer, text: str, add_eos: bool = True) -> list[int]:
    """Encode one document; the EOS id marks the boundary in the packed training stream."""
    ids = tok.encode(text).ids
    if add_eos:
        ids.append(tok.token_to_id(EOS))
    return ids


def encode_corpus(tok: Tokenizer, docs: Iterable[str]) -> list[list[int]]:
    docs = list(docs)
    eos = tok.token_to_id(EOS)
    encs = tok.encode_batch(docs)
    return [e.ids + [eos] for e in encs]


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)
