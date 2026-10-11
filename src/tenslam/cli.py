"""Command line entry point: `tenslam <command> [--config configs/data.toml]`."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from . import tokenizer as T
from .config import load_config
from .data import (read_jsonl, sha256_file, sha256_text, stream_tinystories, write_json,
                   write_jsonl)
from .preprocess import preprocess_corpus
from .stats import corpus_stats, length_summary

DEMO_TEXTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tim said, \"Let's go to the park!\" Mom smiled.\n\nThey played all day.",
    "The café was cozy; naïve rabbits ate 3 carrots.",
    "Emoji and rare bytes: 😀 日本語",
    "   leading spaces and  double  spaces\tand a tab",
    "Supercalifragilisticexpialidocious",
]


def _libs() -> dict:
    return {"python": platform.python_version(),
            **{p: version(p) for p in ("datasets", "tokenizers", "torch", "numpy")}}


def cmd_collect(cfg: dict) -> None:
    d, s, p = cfg["dataset"], cfg["sample"], cfg["paths"]
    for split, n, key in (("train", s["train_docs"], "raw_train"),
                          ("validation", s["validation_docs"], "raw_validation")):
        # Different seed offset per split: the samples are independent draws.
        seed = cfg["seed"] + (0 if split == "train" else 1)
        texts = stream_tinystories(d["name"], d["revision"], split, n, seed,
                                   s["shuffle_buffer"], d["text_field"])
        write_jsonl(p[key], texts)
        print(f"{split}: wrote {len(texts)} raw docs -> {p[key]} sha256={sha256_file(p[key])[:16]}")
        del texts
        gc.collect()  # let streaming threads finish before interpreter shutdown


def cmd_preprocess(cfg: dict) -> None:
    p, pp, d = cfg["paths"], cfg["preprocess"], cfg["dataset"]
    kw = dict(min_chars=pp["min_chars"], max_chars=pp["max_chars"], unicode_form=pp["unicode_form"])
    train, tr_rep = preprocess_corpus(read_jsonl(p["raw_train"]), **kw)
    train_hashes = {sha256_text(t) for t in train}
    val, va_rep = preprocess_corpus(read_jsonl(p["raw_validation"]),
                                    exclude_hashes=train_hashes, **kw)
    reports = {}
    for name, docs, rep, raw_key, clean_key, man_key, split, seed in (
        ("train", train, tr_rep, "raw_train", "clean_train", "train_manifest", "train", cfg["seed"]),
        ("validation", val, va_rep, "raw_validation", "clean_validation", "validation_manifest",
         "validation", cfg["seed"] + 1),
    ):
        write_jsonl(p[clean_key], docs)
        st = corpus_stats(docs)
        reports[name] = {"preprocess": rep.to_dict(), "stats": st}
        write_json(p[man_key], {
            "dataset": d["name"], "revision": d["revision"], "split": split, "seed": seed,
            "sampling": f"streaming + buffered shuffle(buffer={cfg['sample']['shuffle_buffer']})",
            "license": "cdla-sharing-1.0",
            "raw_file": p[raw_key], "raw_sha256": sha256_file(p[raw_key]),
            "clean_file": p[clean_key], "clean_sha256": sha256_file(p[clean_key]),
            "preprocess_config": pp, "libraries": _libs(),
            "n_raw": rep.n_input, "n_clean": rep.n_kept,
        })
        print(f"{name}: {rep.n_input} raw -> {rep.n_kept} kept | invalid={rep.n_invalid} "
              f"{dict(rep.rejected)} dup={rep.n_duplicates} normalized={rep.n_modified} "
              f"chars={st['total_chars']:,}")
    write_json(p["report"], reports)


def cmd_train_tokenizer(cfg: dict) -> None:
    p, tk = cfg["paths"], cfg["tokenizer"]
    docs = list(read_jsonl(p["clean_train"]))  # training split ONLY
    tok = T.train_tokenizer(docs, tk["vocab_size"], tk["special_tokens"], tk["min_frequency"])
    Path(p["tokenizer"]).parent.mkdir(parents=True, exist_ok=True)
    tok.save(p["tokenizer"])
    val = list(read_jsonl(p["clean_validation"]))
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {t: tok.token_to_id(t) for t in tk["special_tokens"]},
        "tokenizer_sha256": sha256_file(p["tokenizer"]),
        "trained_on": p["clean_train"],
        "trained_on_sha256": sha256_file(p["clean_train"]),
        "config": tk, "libraries": _libs(),
    }
    for name, d in (("train", docs), ("validation", val)):
        lens = [len(x) for x in T.encode_corpus(tok, d)]  # includes the EOS token
        chars = sum(len(x) for x in d)
        meta[f"{name}_tokens"] = {
            "n_docs": len(d), "total_tokens_incl_eos": sum(lens),
            "tokens_per_doc": length_summary(lens),
            "chars_per_token": round(chars / (sum(lens) - len(d)), 3),
            "docs_over_512_tokens": sum(n > 512 for n in lens),
        }
    write_json(p["tokenizer_meta"], meta)
    print(f"vocab={meta['vocab_size']} specials={meta['special_tokens']}")
    for n in ("train", "validation"):
        print(n, meta[f"{n}_tokens"])


def cmd_demo(cfg: dict) -> None:
    tok = T.load_tokenizer(cfg["paths"]["tokenizer"])
    for text in DEMO_TEXTS:
        enc = tok.encode(text)
        back = T.decode(tok, enc.ids)
        print(f"text   : {text!r}\ntokens : {enc.tokens}\nids    : {enc.ids}\n"
              f"decoded: {back!r}  round-trip={'OK' if back == text else 'DIFFERS'}\n")
    ids = T.encode_document(tok, "The end.")
    print("with EOS:", ids, "->", T.decode(tok, ids))


def cmd_check_model(cfg: dict) -> None:
    import tomllib

    from .sanity import run_sanity
    with open("configs/model.toml", "rb") as f:
        sanity_cfg = tomllib.load(f)["sanity"]
    ok = run_sanity(cfg["paths"]["tokenizer"], cfg["paths"]["clean_train"],
                    "configs/model.toml", sanity_cfg)
    if not ok:
        raise SystemExit(1)


def cmd_prepare(cfg: dict, args: argparse.Namespace) -> None:
    from .prepare import prepare_corpus
    m = prepare_corpus(cfg, args.train_docs, args.validation_docs, args.force)
    for split in ("train", "validation"):
        r = m[split]
        print(f"{split}: raw {r['n_raw']} -> kept {r['n_docs']} docs | {r['n_tokens']:,} tokens | "
              f"invalid {r['preprocess']['n_invalid']} dup {r['preprocess']['n_exact_duplicates_removed']} "
              f"| {r['bytes']:,} bytes | sha256 {r['sha256'][:16]}")
    print(f"validation docs removed for train overlap: exact "
          f"{m['validation']['preprocess']['n_exact_duplicates_removed']}, loose "
          f"{m['validation']['removed_loose_overlap_with_train']}")
    print(f"manifest -> {cfg['prepare']['manifest']}")


def cmd_train(args: argparse.Namespace) -> None:
    from .train import Trainer, TrainConfig
    cfg = TrainConfig.from_toml(args.config)
    for key in ("max_steps", "max_hours", "max_tokens", "out_dir", "device", "precision"):
        if getattr(args, key, None) is not None:
            setattr(cfg, key, getattr(args, key))
    resume = args.resume
    if resume == "auto":
        last = Path(cfg.out_dir) / "last.pt"
        resume = str(last) if last.exists() else None
    tr = Trainer(cfg, resume=resume)
    print(f"device {tr.device} | precision {tr.autocast_dtype or 'fp32'} | scaler {tr.use_scaler} | "
          f"params {tr.model.num_params():,} | tokens/step {cfg.tokens_per_step:,} | "
          f"total steps {cfg.total_steps:,} | resumed at step {tr.step}")
    result = tr.run()
    print("finished:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import (estimate, format_estimate, run_benchmark, run_sweep, save_report)
    from .train import TrainConfig
    cfg = TrainConfig.from_toml(args.config)
    if args.device:
        cfg.device = args.device
    if args.precision:
        cfg.precision = args.precision
    if args.sweep:
        results = run_sweep(cfg, [int(x) for x in args.sweep.split(",")],
                            warmup_steps=args.warmup, steps=args.steps)
        for r in results:
            print({k: r.get(k) for k in ("micro_batch_size", "tokens_per_s_mean",
                                          "peak_vram_allocated_gb", "error")})
        ok = [r for r in results if "error" not in r]
        if not ok:
            raise SystemExit("every micro-batch size failed")
        bench = max(ok, key=lambda r: r["tokens_per_s_mean"])
        sweep = results
    else:
        bench, sweep = run_benchmark(cfg, args.warmup, args.steps), None
    price = args.price if args.price is not None else cfg.hourly_price_eur
    est = estimate(bench, price, cfg.setup_minutes)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in bench.items()}, indent=2))
    print(format_estimate(est))
    if bench["device"].startswith("cpu"):
        print("NOTE: CPU benchmark only. GPU speed and GPU cost are UNKNOWN until this is run on CUDA.")
    out = args.out or f"reports/benchmark_{bench['device'].split(':')[0]}.json"
    save_report(out, {"benchmark": bench, "estimate": est, "sweep": sweep})
    print(f"report -> {out}")


def cmd_estimate(args: argparse.Namespace) -> None:
    from .benchmark import estimate, format_estimate
    bench = json.loads(Path(args.benchmark).read_text())["benchmark"]
    print(format_estimate(estimate(bench, args.price, args.setup_minutes)))


def cmd_export(args: argparse.Namespace) -> None:
    from .checkpoint import load_checkpoint
    from .export import export_inference
    from .model import ModelConfig, TenslamLM
    ck = load_checkpoint(args.checkpoint)
    model = TenslamLM(ModelConfig(**ck["config"]["model"]))
    model.load_state_dict(ck["model"])
    out = export_inference(model, ck["config"]["train"]["tokenizer"], args.out,
                           {"step": ck["step"], "tokens_seen": ck["tokens_seen"]})
    print(f"exported inference-only model ({model.num_params():,} params) -> {out}")


def cmd_generate(args: argparse.Namespace) -> None:
    from .export import load_exported
    from .generate import generate
    model = load_exported(args.export_dir)
    tok = T.load_tokenizer(Path(args.export_dir) / "tokenizer.json")
    for i, prompt in enumerate(args.prompt):
        print(generate(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_k, seed=i), "\n")


PIPELINE = ["collect", "preprocess", "train-tokenizer", "demo"]
COMMANDS = {"collect": cmd_collect, "preprocess": cmd_preprocess,
            "train-tokenizer": cmd_train_tokenizer, "demo": cmd_demo,
            "check-model": cmd_check_model}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tenslam")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in (*COMMANDS, "all"):
        sub.add_parser(name).add_argument("--config", default="configs/data.toml")
    p = sub.add_parser("prepare", help="stream, clean, dedupe, tokenize -> uint16 token files")
    p.add_argument("--config", default="configs/data.toml")
    p.add_argument("--train-docs", type=int)
    p.add_argument("--validation-docs", type=int)
    p.add_argument("--force", action="store_true", help="re-stream even if raw files exist")
    p = sub.add_parser("train", help="pretrain (CPU or CUDA)")
    p.add_argument("--config", default="configs/train_cpu_smoke.toml")
    p.add_argument("--resume", help="checkpoint path, or 'auto' to continue <out_dir>/last.pt")
    p.add_argument("--max-steps", type=int, dest="max_steps")
    p.add_argument("--max-hours", type=float, dest="max_hours")
    p.add_argument("--max-tokens", type=int, dest="max_tokens")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"])
    p = sub.add_parser("benchmark", help="measure tokens/s, VRAM, overheads; estimate time/cost")
    p.add_argument("--config", default="configs/train_kaggle.toml")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sweep", help="comma-separated micro-batch sizes, e.g. 16,32,64")
    p.add_argument("--price", type=float, help="hourly price in EUR (cost stays unknown without it)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--out")
    p = sub.add_parser("estimate", help="re-estimate from a saved benchmark report")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--price", type=float)
    p.add_argument("--setup-minutes", type=float, default=15.0)
    p = sub.add_parser("export", help="training checkpoint -> inference-only export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("generate", help="sample text from an exported model")
    p.add_argument("--export-dir", required=True)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args(argv)

    if args.command in ("train", "benchmark", "estimate", "export", "generate"):
        {"train": cmd_train, "benchmark": cmd_benchmark, "estimate": cmd_estimate,
         "export": cmd_export, "generate": cmd_generate}[args.command](args)
        return 0
    cfg = load_config(args.config)
    if args.command == "prepare":
        cmd_prepare(cfg, args)
        # HF streaming threads can crash the interpreter during teardown; our files are
        # already closed and fsynced, so leave immediately with a clean status.
        sys.stdout.flush()
        os._exit(0)
    for name in (PIPELINE if args.command == "all" else [args.command]):
        print(f"== {name} ==")
        COMMANDS[name](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
