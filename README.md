# Tenslam MiniLM

A small language model built **entirely from scratch**: data pipeline, tokenizer, Transformer, training engine,
checkpointing and cost estimator. About 30M parameters, trained on TinyStories. The goal is to understand every
part of how an LLM is built, in code small enough to read in an afternoon, with the engineering habits of a
real project (pinned data, hashes, tests, reproducible runs).

> **Status:** the data pipeline, model and training engine are done and tested (91 tests pass, 1 skipped).
> **Stage 1 is trained:** a 29.9M-parameter model on 100M TinyStories tokens (1,004 s on one GPU) reached
> validation loss **1.631 (perplexity 5.11)**; curve and logs in [reports/stage1_30m_tinystories](reports/stage1_30m_tinystories).
> Stage 2 (60M parameters, multi-source data) is prepared but not trained yet; see
> [docs/RUNBOOK_STAGE2.md](docs/RUNBOOK_STAGE2.md). No pretrained weights are used anywhere.

## What is in here

| Part | What it does | Code |
|---|---|---|
| Data | stream TinyStories (pinned revision), clean, exact-dedup, remove train/validation overlap | [data.py](src/tenslam/data.py), [preprocess.py](src/tenslam/preprocess.py), [prepare.py](src/tenslam/prepare.py) |
| Tokenizer | byte-level BPE, 8,192 tokens, trained on training split only | [tokenizer.py](src/tenslam/tokenizer.py) |
| Model | decoder-only Transformer: RMSNorm, RoPE, causal attention, SwiGLU, tied embeddings | [model.py](src/tenslam/model.py) |
| Training | AdamW, warmup + cosine, grad accumulation, clipping, bf16/fp16, NaN guard, eval, sampling | [train.py](src/tenslam/train.py), [schedule.py](src/tenslam/schedule.py) |
| Recovery | atomic checkpoints with optimizer, scheduler, scaler and RNG state; exact resume | [checkpoint.py](src/tenslam/checkpoint.py) |
| Export | inference-only `safetensors` weights, separate from training state | [export.py](src/tenslam/export.py) |
| Benchmark | measured tokens/s, peak VRAM, time and cost estimates (never guessed) | [benchmark.py](src/tenslam/benchmark.py) |

Model: 8 layers, d_model 512, 8 heads, SwiGLU d_ff 1408, 512-token context, **29,893,120 parameters**.

## Measured so far

**Data (RunPod profile):** 573,932 training stories, **125,952,650 tokens**; 4,924 official validation stories,
992,483 tokens. Dropped: 109 invalid and 25,959 exact duplicates from 600,000 raw stories; 76 validation stories
removed for overlapping training. Hashes in [data/tokens_runpod/prepare_manifest.json](data/tokens_runpod/prepare_manifest.json).

**GPU benchmark** ([reports/benchmark_cuda_rtx_a5000.json](reports/benchmark_cuda_rtx_a5000.json)), NVIDIA RTX A5000 24 GB,
bf16, sequence 512, 10 timed steps:

| Micro-batch | Tokens/s | Peak VRAM |
|---|---|---|
| 16 | 140,600 | 3.3 GB |
| 32 | 148,200 | 6.3 GB |
| 64 | 153,100 | 12.1 GB |

Extrapolated from this (an estimate, not a full run): 100M tokens takes about 11 minutes of pure training.

## Quick start (CPU, no GPU needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install datasets tokenizers numpy safetensors matplotlib tqdm pytest
pip install -e . --no-build-isolation

pytest -q                                              # full test suite
tenslam prepare                                        # stream + clean + tokenize a 50k-story subset
tenslam train --config configs/train_cpu_smoke.toml    # ~25 s tiny-model smoke run
tenslam generate --export-dir runs/cpu_smoke/export --prompt "Once upon a time"
```

> If your shell sets `PYTHONPATH` (for example ROS), pytest plugins can leak in. Use `env -u PYTHONPATH pytest -q`.

## Train on a rented GPU

The same code runs on CPU or CUDA with no source changes. Step-by-step instructions, including setup,
monitoring, getting the weights out safely and where to publish them, are in
**[docs/RUNBOOK_100M.md](docs/RUNBOOK_100M.md)**. A GPU is never started automatically; the benchmark command
estimates cost only from measured speed and a price you pass explicitly:

```bash
tenslam benchmark --config configs/train_runpod.toml --sweep 16,32,64 --price <EUR_PER_HOUR>
tenslam train     --config configs/train_runpod.toml --resume auto
```

## Design choices in one page

- **TinyStories**, because very small models can learn coherent English from simple, narrow data, which keeps
  the project cheap and the results visible.
- **8,192-token byte-level BPE**, not 50k: a 50k vocabulary would put about 26M of the 30M parameters into
  embeddings. Byte-level means no unknown token.
- **Pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings, no biases:** the LLaMA-style recipe, adding no position
  parameters and saving 14% of the parameters through tying.
- **Packed 512-token windows** sampled from one EOS-separated token stream. Loss does cross document
  boundaries and attention is not masked at EOS (the usual GPT-style choice).
- **Stateless deterministic sampling** (`rng(seed, step, micro)`): reproducible, and resume is exact.
- **Integrity first:** pinned dataset revision, SHA-256 manifests, a hard assertion of zero train/validation
  overlap, and the trainer re-verifies data hashes before training.
- **Honest cost:** the estimator prints `UNKNOWN` without a real CUDA benchmark and an explicit price.

## Documentation

- [docs/BUILD_JOURNAL.md](docs/BUILD_JOURNAL.md): the whole build explained, decision by decision, with alternatives.
- [docs/RUNBOOK_100M.md](docs/RUNBOOK_100M.md): exact steps for the 100M-token run and what to do afterwards.
- [docs/SCALING.md](docs/SCALING.md): what changes to build 1B or larger models (compute, memory, data, code).
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md): model card template, to be filled with real results.
- [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md): the earlier detailed per-session notes.

## Scaling this up to a big LLM, in short

Compute is about `6 x parameters x tokens`. A 1B-parameter model on about 20B tokens needs roughly 6,700 times the
compute of this project (about 50 days on the single GPU used here, so in practice multi-GPU), and the tokens must
come from a large, diverse, deduplicated corpus rather than TinyStories. What changes in the code is mainly
parallelism (DDP/FSDP, tensor and pipeline parallel), memory tricks (activation checkpointing, mixed precision),
sharded data and fuzzy deduplication. The architecture and training recipe stay the same family.
See [docs/SCALING.md](docs/SCALING.md) for the numbers and a step-by-step ladder.

## Limitations

- Trained on synthetic children's stories: not a general-purpose assistant, no safety tuning.
- No pretrained weights were used; no benchmark comparisons yet.
- The TinyStories dataset has its own licence (CDLA-Sharing-1.0). Re-check it before publishing derived data or weights.
- A code licence has not been chosen yet.

## Layout

```
configs/    model, data and training configs (TOML; `extends` supported)
src/tenslam model, data, tokenizer, training, checkpoint, export, benchmark, CLI
tests/      offline unit tests
docs/       journal, runbook, scaling guide, model card
scripts/    plot_metrics.py (training curves from metrics.jsonl)
artifacts/  trained tokenizer
reports/    data statistics and benchmark results
```
