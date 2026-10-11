"""Autoregressive sampling for qualitative checks."""
from __future__ import annotations

import torch
from tokenizers import Tokenizer

from . import tokenizer as T
from .model import TenslamLM


@torch.no_grad()
def generate(model: TenslamLM, tok: Tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int = 40, seed: int = 0,
             device: torch.device | str | None = None) -> str:
    """Sample a continuation. temperature <= 0 means greedy.

    The context starts with <|endoftext|>, because in the training stream every story
    follows an EOS. A private torch.Generator is used so sampling never disturbs the
    global RNG state of a training run.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    eos = tok.token_to_id(T.EOS)
    ids = [eos] + tok.encode(prompt).ids
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: list[int] = []
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long, device=device)
        logits = model(ctx)[0][0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        if nxt == eos:
            break
        ids.append(nxt)
        out.append(nxt)
    if was_training:
        model.train()
    return prompt + tok.decode(out, skip_special_tokens=False)
