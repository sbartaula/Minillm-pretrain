"""Run a repeatable CPU quality check against a Tenslam inference export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tenslam.data import sha256_file
from tenslam.export import load_exported
from tenslam.generate import generate
from tenslam.tokenizer import load_tokenizer


PROMPTS = [
    ("story", "Once upon a time, a young girl found a map"),
    ("english_completion", "The boy was tired, so he decided to"),
    ("factual_continuation", "In 1905, Albert Einstein published"),
    ("expository", "The water cycle describes how water moves"),
    ("arithmetic", "Three apples are placed on a table. Then"),
    ("question_answering", "Question: What is the capital of France? Answer:"),
    ("out_of_domain", "A microscope reveals a miniature city beneath the soil."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0 or args.top_k < 0 or args.threads < 1:
        parser.error("max-new-tokens and threads must be positive; temperature must be positive; top-k cannot be negative")

    export_dir = args.export_dir.resolve()
    output = args.output.resolve()
    info = json.loads((export_dir / "export_info.json").read_text(encoding="utf-8"))
    weights_path = export_dir / "model.safetensors"
    tokenizer_path = export_dir / "tokenizer.json"
    weights_sha256 = sha256_file(weights_path)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    if weights_sha256 != info["weights_sha256"]:
        raise ValueError("model.safetensors SHA-256 does not match export_info.json")
    if tokenizer_sha256 != info["tokenizer_sha256"]:
        raise ValueError("tokenizer.json SHA-256 does not match export_info.json")

    torch.set_num_threads(args.threads)
    model = load_exported(export_dir, device="cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match model config")
    if model.num_params() != info["n_params"]:
        raise ValueError("loaded model parameter count does not match export_info.json")

    results = []
    for index, (category, prompt) in enumerate(PROMPTS):
        results.append({
            "category": category,
            "prompt": prompt,
            "greedy": generate(model, tokenizer, prompt, args.max_new_tokens,
                               temperature=0, top_k=0, seed=args.seed + index),
            "sampled": generate(model, tokenizer, prompt, args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k,
                                seed=args.seed + index),
        })

    report = {
        "device": "cpu",
        "model_params": model.num_params(),
        "weights_sha256": info["weights_sha256"],
        "tokenizer_sha256": tokenizer_sha256,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {"temperature": args.temperature, "top_k": args.top_k, "seed": args.seed},
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CPU evaluation complete: {len(results)} prompts, {model.num_params():,} parameters")
    print(f"results -> {output}")


if __name__ == "__main__":
    main()
