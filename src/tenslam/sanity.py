"""The three pre-GPU checks: forward shape, backward gradients, overfit a tiny batch."""
from __future__ import annotations

import math
import time

import torch

from . import tokenizer as T
from .data import read_jsonl
from .model import ModelConfig, TenslamLM


def make_batch(tok, path: str, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack documents (EOS-separated) into one stream; return (inputs, next-token targets)."""
    need = batch_size * (seq_len + 1)
    stream: list[int] = []
    for doc in read_jsonl(path):
        stream += T.encode_document(tok, doc)
        if len(stream) >= need:
            break
    data = torch.tensor(stream[:need]).view(batch_size, seq_len + 1)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()  # targets = inputs shifted by 1


def check_forward(model: TenslamLM, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
    b, t = x.shape
    ok = tuple(logits.shape) == (b, t, model.cfg.vocab_size) and torch.isfinite(logits).all().item()
    return {"ok": bool(ok), "logits_shape": tuple(logits.shape), "initial_loss": loss.item(),
            "expected_loss_ln_V": math.log(model.cfg.vocab_size)}


def check_backward(model: TenslamLM, x: torch.Tensor, y: torch.Tensor, lr: float) -> dict:
    model.train()
    opt = model.configure_optimizer(lr)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    gnorm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    opt.step()
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p, before[n])]
    return {"ok": not no_grad and not unchanged and bool(torch.isfinite(gnorm)),
            "params_without_gradient": no_grad, "params_not_updated": unchanged,
            "grad_norm": gnorm.item()}


def check_overfit(model: TenslamLM, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float,
                  log_every: int = 25) -> dict:
    model.train()
    opt = model.configure_optimizer(lr, weight_decay=0.0)
    history, t0 = [], time.time()
    for step in range(steps + 1):
        _, loss = model(x, y)
        if step % log_every == 0 or step == steps:
            history.append((step, loss.item()))
            print(f"  step {step:4d}  loss {loss.item():.4f}", flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    first, last = history[0][1], history[-1][1]
    return {"ok": last < 0.1 * first, "first_loss": first, "final_loss": last,
            "seconds": round(time.time() - t0, 1), "history": history}


def run_sanity(tokenizer_path: str, data_path: str, model_cfg: str, sanity_cfg: dict) -> bool:
    torch.manual_seed(sanity_cfg["seed"])
    cfg = ModelConfig.from_toml(model_cfg)
    tok = T.load_tokenizer(tokenizer_path)
    assert tok.get_vocab_size() == cfg.vocab_size, "tokenizer / model vocab mismatch"
    model = TenslamLM(cfg)
    x, y = make_batch(tok, data_path, sanity_cfg["batch_size"], sanity_cfg["seq_len"])
    print(f"model: {model.num_params():,} params ({model.num_params(True):,} non-embedding); "
          f"batch {tuple(x.shape)} = {x.numel()} tokens")
    results = {}
    print("[1/3] forward"); results["forward"] = check_forward(model, x, y); print("  ", results["forward"])
    print("[2/3] backward"); results["backward"] = check_backward(model, x, y, sanity_cfg["lr"])
    print("  ", results["backward"])
    # Fresh model for the overfit test so check 2's single optimizer step does not leak in.
    torch.manual_seed(sanity_cfg["seed"]); model = TenslamLM(cfg)
    print(f"[3/3] overfit one batch ({sanity_cfg['steps']} steps)")
    results["overfit"] = check_overfit(model, x, y, sanity_cfg["steps"], sanity_cfg["lr"])
    print("  ", {k: v for k, v in results["overfit"].items() if k != "history"})
    ok = all(r["ok"] for r in results.values())
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok
