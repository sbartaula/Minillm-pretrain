import json
import math

import numpy as np
import pytest
import torch

from tenslam import tokenizer as T
from tenslam.checkpoint import atomic_save, get_rng_state, load_checkpoint, set_rng_state
from tenslam.data import sha256_file
from tenslam.dataset import eval_batches, open_tokens, sample_batch
from tenslam.export import export_inference, load_exported
from tenslam.prepare import filter_special_text, loose_key
from tenslam.schedule import lr_at, make_scheduler
from tenslam.train import (NonFiniteError, TrainConfig, Trainer, check_finite_params,
                           next_skip_count, resolve_device, resolve_precision)

MODEL = dict(vocab_size=400, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=32)


def make_cfg(corpus, tmp_path, **kw):
    base = dict(out_dir=str(tmp_path / "run"), train_bin=str(corpus["train"]),
                val_bin=str(corpus["val"]), manifest=str(tmp_path / "none.json"),
                tokenizer=str(corpus["tok_path"]), model=dict(MODEL), seq_len=16,
                micro_batch_size=4, grad_accum_steps=2, max_steps=8, lr=3e-3, min_lr=3e-4,
                warmup_steps=2, device="cpu", precision="fp32", eval_interval=4, eval_batches=2,
                log_interval=2, ckpt_interval=4, sample_interval=0, prompts=["Once upon"],
                max_new_tokens=5)
    base.update(kw)
    return TrainConfig.from_dict(base)


def weights(tr):
    return {k: v.clone() for k, v in tr.model.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ data
def test_next_token_alignment():
    toks = np.arange(1000, dtype=np.uint16)
    x, y = sample_batch(toks, 4, 16, seed=1, step=0)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted left by one
    assert torch.equal(y, x + 1)  # on arange data target == next value
    for xe, ye in eval_batches(toks, 4, 16):
        assert torch.equal(ye, xe + 1)


def test_sampling_is_deterministic_and_varies_by_step_and_micro():
    toks = np.random.default_rng(0).integers(0, 400, 5000).astype(np.uint16)
    a = sample_batch(toks, 4, 16, 7, 3, 1)
    assert torch.equal(a[0], sample_batch(toks, 4, 16, 7, 3, 1)[0])
    assert not torch.equal(a[0], sample_batch(toks, 4, 16, 7, 4, 1)[0])
    assert not torch.equal(a[0], sample_batch(toks, 4, 16, 7, 3, 2)[0])
    assert not torch.equal(a[0], sample_batch(toks, 4, 16, 8, 3, 1)[0])


def test_sampler_never_reads_out_of_bounds():
    toks = np.arange(20, dtype=np.uint16)
    for s in range(200):
        x, y = sample_batch(toks, 8, 19, 0, s)
        assert int(y.max()) <= 19


def test_eval_windows_cover_file_without_overlap():
    toks = np.arange(101, dtype=np.uint16)
    seen = torch.cat([x.flatten() for x, _ in eval_batches(toks, 3, 10)])
    assert len(seen) == 100 and len(set(seen.tolist())) == 100


def test_eos_after_every_document_and_boundaries(corpus, tok):
    eos = tok.token_to_id(T.EOS)
    arr = np.fromfile(corpus["train"], dtype=np.uint16)
    info = corpus["info"]["train"]
    assert arr[-1] == eos and int((arr == eos).sum()) == info["n_docs"] == len(corpus["train_docs"])
    docs, cur = [], []
    for t in arr.tolist():  # splitting on EOS recovers each document exactly
        if t == eos:
            docs.append(T.decode(tok, cur))
            cur = []
        else:
            cur.append(t)
    assert docs == corpus["train_docs"]
    assert corpus["info"]["eos_id"] == eos


def test_documents_containing_special_token_text_are_dropped():
    kept, n = filter_special_text(["fine", f"bad {T.EOS} text", "ok"], [T.EOS])
    assert kept == ["fine", "ok"] and n == 1


def test_train_validation_separation(corpus):
    v = corpus["info"]["validation"]
    # exact copies, case variants and whitespace variants of train docs were all removed
    assert v["n_docs"] == len(corpus["val_docs"]) - 2
    assert v["n_overlap_with_train_after_filtering"] == 0
    assert v["preprocess"]["n_exact_duplicates_removed"] + v["removed_loose_overlap_with_train"] >= 2
    assert loose_key("A  b\n") == loose_key("a b")


# ------------------------------------------------------------------ schedule
def test_scheduler_warmup_cosine_and_floor():
    kw = dict(peak_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    lrs = [lr_at(s, **kw) for s in range(130)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))  # strictly rising in warmup
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.55)  # cosine midpoint = (peak+min)/2
    assert all(a >= b for a, b in zip(lrs[10:110], lrs[11:111]))  # monotone decay
    assert lrs[110] == pytest.approx(0.1) and lrs[129] == pytest.approx(0.1)


def test_scheduler_drives_optimizer_and_state_roundtrips():
    def build():
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        return opt, make_scheduler(opt, 1.0, 0.1, 5, 50)
    opt, sch = build()
    seen = []
    for _ in range(20):
        seen.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    assert seen == pytest.approx([lr_at(s, 1.0, 0.1, 5, 50) for s in range(20)])
    opt2, sch2 = build()
    sch2.load_state_dict(sch.state_dict())
    assert sch2.get_last_lr() == pytest.approx(sch.get_last_lr())


# ------------------------------------------------------------------ training engine
def test_gradient_accumulation_matches_one_big_batch(corpus, tmp_path):
    a = Trainer(make_cfg(corpus, tmp_path, grad_clip=1e9, warmup_steps=0), write_files=False)
    b = Trainer(make_cfg(corpus, tmp_path, grad_clip=1e9, warmup_steps=0), write_files=False)
    assert same(weights(a), weights(b))
    micro = a.micro_batches(0)
    big = [(torch.cat([x for x, _ in micro]), torch.cat([y for _, y in micro]))]
    ra, rb = a.train_step(micro), b.train_step(big)
    assert ra["loss"] == pytest.approx(rb["loss"], rel=1e-5)
    assert ra["grad_norm"] == pytest.approx(rb["grad_norm"], rel=1e-4)
    # Adam rescales every update to ~lr, so float summation-order differences between CPUs can flip
    # near-zero-gradient entries. Require the same update on average, not bit-for-bit.
    wb = weights(b)
    for k, v in weights(a).items():
        diff = (v - wb[k]).abs()
        assert diff.mean() < 1e-5, k
        assert diff.max() < 1e-2, k


def test_grad_clipping_bounds_the_applied_norm(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, grad_clip=0.01), write_files=False)
    tr.train_step(tr.micro_batches(0))
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in tr.model.parameters() if p.grad is not None))
    assert float(norm) <= 0.0101


def test_training_reduces_loss(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, max_steps=30), write_files=False)
    losses = [tr.train_step(tr.micro_batches(s))["loss"] for s in range(30)]
    assert losses[0] == pytest.approx(math.log(400), abs=0.5)
    assert losses[-1] < losses[0] - 1.0
    assert tr.tokens_seen == 30 * tr.cfg.tokens_per_step


def test_evaluate_is_gradient_free_and_does_not_change_weights(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path), write_files=False)
    tr.train_step(tr.micro_batches(0))
    before, grads = weights(tr), [p.grad.clone() for p in tr.model.parameters()]
    flags = []
    orig = tr.model.forward
    tr.model.forward = lambda *a, **k: (flags.append((tr.model.training, torch.is_grad_enabled())), orig(*a, **k))[1]
    r = tr.evaluate()
    tr.model.forward = orig
    assert flags and all(not t and not g for t, g in flags)
    assert tr.model.training  # restored
    assert same(before, weights(tr))
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, tr.model.parameters()))
    assert math.isfinite(r["val_loss"]) and r["val_ppl"] == pytest.approx(math.exp(r["val_loss"]))
    assert tr.evaluate()["val_loss"] == r["val_loss"]  # deterministic


def test_mixed_validation_aggregates_source_losses_by_source_token_count(corpus, tmp_path):
    source_values = {"short_source": (1, 100), "long_source": (3, 300)}
    sources = []
    for name, (token_id, token_count) in source_values.items():
        path = tmp_path / f"{name}.bin"
        np.full(token_count, token_id, dtype=np.uint16).tofile(path)
        sources.append({"name": name, "validation_bin": str(path), "validation_tokens": token_count})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "tokenizer": {"sha256": sha256_file(corpus["tok_path"])},
        "train": {"sha256": "", "n_tokens": 0},
        "validation": {"sha256": "", "n_tokens": sum(v[1] for v in source_values.values())},
        "sources": sources,
    }))
    tr = Trainer(make_cfg(corpus, tmp_path, manifest=str(manifest), eval_batches=2,
                          verify_data_hash=False), write_files=False)
    tr._fwd = lambda x, y: (None, y.float().mean())

    result = tr.evaluate()

    assert result["val_loss/short_source"] == pytest.approx(1.0)
    assert result["val_loss/long_source"] == pytest.approx(3.0)
    assert result["val_loss"] == pytest.approx((1.0 * 100 + 3.0 * 300) / 400)
    assert result["val_tokens"] == 2 * 4 * 16
    assert result["val_ppl"] == pytest.approx(math.exp(result["val_loss"]))


def test_run_logs_metrics_and_writes_checkpoint_and_export(corpus, tmp_path):
    cfg = make_cfg(corpus, tmp_path, sample_interval=4)
    res = Trainer(cfg).run(install_signal_handlers=False)
    assert res["reason"] == "completed" and res["step"] == 8
    recs = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    kinds = {r["type"] for r in recs}
    assert {"train", "eval", "samples", "stop"} <= kinds
    tr = next(r for r in recs if r["type"] == "train")
    assert {"lr", "tokens", "step_time_s", "grad_norm", "tokens_per_s"} <= set(tr)
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "export" / "model.safetensors").exists()


# ------------------------------------------------------------------ checkpoint / resume
def test_resume_is_identical_to_uninterrupted_training(corpus, tmp_path):
    full = Trainer(make_cfg(corpus, tmp_path / "a"), write_files=False)
    full.run(install_signal_handlers=False)

    first = Trainer(make_cfg(corpus, tmp_path / "b", max_steps=8))
    first.cfg.max_steps = 8
    while first.step < 5:
        first.train_step(first.micro_batches(first.step))
    path = first.save()
    resumed = Trainer(make_cfg(corpus, tmp_path / "b"), resume=path, write_files=False)
    assert resumed.step == 5 and resumed.tokens_seen == 5 * resumed.cfg.tokens_per_step
    assert resumed.scheduler.get_last_lr() == pytest.approx(first.scheduler.get_last_lr())
    resumed.run(install_signal_handlers=False)
    assert resumed.step == 8
    assert same(weights(full), weights(resumed))


def test_checkpoint_contains_everything_and_loads_weights_only(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path))
    tr.train_step(tr.micro_batches(0))
    ck = load_checkpoint(tr.save())  # uses weights_only=True
    assert {"model", "optimizer", "scheduler", "scaler", "step", "tokens_seen", "rng", "config",
            "tokenizer_sha256"} <= set(ck)
    assert {"python", "numpy", "torch_cpu"} <= set(ck["rng"])
    assert ck["step"] == 1 and ck["scaler"] is None
    assert ck["config"]["train"]["seq_len"] == 16


def test_rng_state_roundtrip():
    import random
    set_rng_state(get_rng_state())
    st = get_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_rng_state(st)
    assert a == (random.random(), np.random.rand(), torch.rand(1).item())


def test_weight_tying_survives_checkpoint_and_export(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path))
    tr.train_step(tr.micro_batches(0))
    path = tr.save()
    r = Trainer(make_cfg(corpus, tmp_path), resume=path, write_files=False)
    assert r.model.lm_head.weight is r.model.tok_emb.weight
    r.train_step(r.micro_batches(1))
    assert r.model.lm_head.weight is r.model.tok_emb.weight
    assert torch.equal(r.model.lm_head.weight, r.model.tok_emb.weight)
    out = export_inference(tr.model, corpus["tok_path"], tmp_path / "exp")
    m = load_exported(out)
    assert m.lm_head.weight is m.tok_emb.weight
    x = torch.randint(0, 400, (1, 8))
    assert torch.allclose(m(x)[0], tr.model.eval()(x)[0], atol=1e-5)


def test_atomic_save_keeps_old_checkpoint_when_write_fails(tmp_path, monkeypatch):
    p = tmp_path / "c.pt"
    atomic_save({"v": 1}, p)
    monkeypatch.setattr(torch, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        atomic_save({"v": 2}, p)
    monkeypatch.undo()
    assert torch.load(p, weights_only=True)["v"] == 1
    assert [f.name for f in tmp_path.iterdir()] == ["c.pt"]  # no stray temp files


def test_resume_rejects_different_architecture_or_tokenizer(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path))
    path = tr.save()
    other = dict(MODEL, n_layers=1)
    with pytest.raises(ValueError, match="architecture"):
        Trainer(make_cfg(corpus, tmp_path, model=other), resume=path, write_files=False)


def test_export_does_not_contain_training_state(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path))
    out = tr.export()
    assert not (out / "optimizer.pt").exists()
    assert {p.name for p in out.iterdir()} == {"model.safetensors", "config.json", "tokenizer.json",
                                               "export_info.json"}


# ------------------------------------------------------------------ devices / precision
def test_device_resolution(monkeypatch):
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_precision_selection(monkeypatch):
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert resolve_precision(cpu, "auto") == (None, False)
    assert resolve_precision(cpu, "bf16") == (torch.bfloat16, False)
    with pytest.raises(ValueError):
        resolve_precision(cpu, "fp16")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5))  # T4
    assert resolve_precision(cuda, "auto") == (torch.float16, True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 0))  # A100
    assert resolve_precision(cuda, "auto") == (torch.bfloat16, False)


def test_scaler_is_disabled_on_cpu_and_state_is_none(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path), write_files=False)
    assert not tr.use_scaler and tr.state()["scaler"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_training_step(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, device="cuda", precision="auto"), write_files=False)
    assert math.isfinite(tr.train_step(tr.micro_batches(0))["loss"])


# ------------------------------------------------------------------ safety
def test_nan_loss_stops_safely_and_keeps_last_good_checkpoint(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, ckpt_interval=2, max_steps=20))
    calls = {"n": 0}

    def poison(t, res):
        calls["n"] += 1
        if t.step == 4:  # corrupt a weight after the step-4 update; next loss is NaN
            with torch.no_grad():
                next(t.model.parameters()).fill_(float("nan"))
    with pytest.raises(NonFiniteError):
        tr.run(on_step=poison, install_signal_handlers=False)
    ck = load_checkpoint(tmp_path / "run" / "last.pt")
    assert ck["step"] == 2  # the step-4 save was refused (NaN weights); step 2 stays intact
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
    recs = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert recs[-1]["type"] == "error"


def test_save_refuses_non_finite_weights(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path))
    with torch.no_grad():
        next(tr.model.parameters())[0, 0] = float("inf")
    with pytest.raises(NonFiniteError):
        check_finite_params(tr.model)
    with pytest.raises(NonFiniteError):
        tr.save()
    assert not (tmp_path / "run" / "last.pt").exists()


def test_skip_counter_resets_and_limits():
    assert next_skip_count(0, True, 3) == 1 and next_skip_count(2, False, 3) == 0
    with pytest.raises(NonFiniteError):
        next_skip_count(3, True, 3)


def test_controlled_stop_saves_resumable_checkpoint(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, ckpt_interval=0, max_steps=20))
    res = tr.run(on_step=lambda t, r: t.request_stop("test") if t.step == 3 else None,
                 install_signal_handlers=False)
    assert res["reason"] == "test" and res["step"] == 3
    assert not (tmp_path / "run" / "export").exists()  # only completed runs export
    r = Trainer(make_cfg(corpus, tmp_path, max_steps=20), resume=tmp_path / "run" / "last.pt",
                write_files=False)
    assert r.step == 3


def test_max_hours_stops_and_saves(corpus, tmp_path):
    tr = Trainer(make_cfg(corpus, tmp_path, max_steps=50, max_hours=1e-9, ckpt_interval=0))
    res = tr.run(install_signal_handlers=False)
    assert res["reason"] == "max_hours" and res["step"] == 1
    assert (tmp_path / "run" / "last.pt").exists()


def test_token_budget_sets_total_steps(corpus, tmp_path):
    cfg = make_cfg(corpus, tmp_path, max_steps=0, max_tokens=1000)
    assert cfg.tokens_per_step == 128 and cfg.total_steps == 8


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({"lr_typo": 1})


def test_tokenizer_mismatch_with_manifest_is_rejected(corpus, tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"tokenizer": {"sha256": "0" * 64},
                               "train": {"sha256": "x", "n_tokens": 1},
                               "validation": {"sha256": "x", "n_tokens": 1}}))
    with pytest.raises(ValueError, match="tokenizer"):
        Trainer(make_cfg(corpus, tmp_path, manifest=str(man)), write_files=False)


def test_benchmark_cost_is_unknown_on_cpu_and_without_price():
    from tenslam.benchmark import estimate
    bench = {"device": "cpu", "gpu_name": None, "tokens_per_s_mean": 1000.0,
             "tokens_per_s_median": 1000.0, "eval_pass_seconds": 1.0, "sample_round_seconds": 1.0,
             "checkpoint_save_seconds": 1.0, "eval_interval": 10, "ckpt_interval": 10,
             "sample_interval": 10, "tokens_per_step": 1000, "train_tokens_available": 10**7}
    rows = estimate(bench, 1.0, 15)["rows"]
    assert all(r["cost_eur"] is None for r in rows)
    assert [r["tokens"] for r in rows] == [10_000_000, 50_000_000, 100_000_000, 200_000_000]
    gpu = dict(bench, device="cuda:0", gpu_name="T4")
    assert all(r["cost_eur"] is None for r in estimate(gpu, None, 15)["rows"])  # no price
    priced = estimate(gpu, 2.0, 15)["rows"]
    assert priced[0]["cost_eur"] == pytest.approx(priced[0]["total_hours"] * 2.0, abs=0.01)
