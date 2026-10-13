# Tenslam MiniLM

A small language model built **entirely from scratch**: data pipeline, tokenizer, Transformer, training engine,
checkpointing and cost estimator. The project includes a 29.9M-parameter TinyStories model and a 60.1M-parameter
multi-source model. The goal is to understand every part of how an LLM is built, in code small enough to read in an
afternoon, with the engineering habits of a real project (pinned data, hashes, tests, reproducible runs).

**Source repository:** [sbartaula/Minillm-pretrain](https://github.com/sbartaula/Minillm-pretrain)

> **Status:** the data pipeline, model and training engine are implemented and tested.
> **Stage 1:** a 29.9M-parameter model trained on 100M TinyStories tokens (1,004 s on one GPU), with validation
> loss **1.631 (perplexity 5.11)**.
> **Stage 2:** a 60.1M-parameter model trained on 791M tokens from six sources before its 5.5-hour time limit.
> Its export loads and generates on CPU. The logged aggregate validation value was TinyStories-only because of
> the original validation windowing. See [the recovery report](reports/stage2_recovery.md); do not cite that
> aggregate as mixed-corpus validation.
> No pretrained weights are used anywhere. This is a learning project, not an instruction-tuned assistant.
> Stage 2 weight redistribution terms are under review; there is no public model upload link yet.

## What is in here

| Part | What it does | Code |
|---|---|---|
| Data | pinned dataset revisions, document cleaning/deduplication, train-only tokenizer sample, manifests and hashes | [mix.py](src/tenslam/mix.py), [prepare.py](src/tenslam/prepare.py) |
| Tokenizer | byte-level BPE trained on training text only; 8,192 tokens (Stage 1), 16,384 (Stage 2) | [tokenizer.py](src/tenslam/tokenizer.py) |
| Model | decoder-only Transformer: RMSNorm, RoPE, causal attention, SwiGLU, tied embeddings | [model.py](src/tenslam/model.py) |
| Training | AdamW, warmup + cosine, grad accumulation, clipping, bf16/fp16, NaN guard, eval, sampling | [train.py](src/tenslam/train.py), [schedule.py](src/tenslam/schedule.py) |
| Recovery | atomic checkpoints with optimizer, scheduler, scaler and RNG state; exact resume | [checkpoint.py](src/tenslam/checkpoint.py) |
| Export | inference-only `safetensors` weights, separate from training state | [export.py](src/tenslam/export.py) |
| Benchmark | measured tokens/s, peak VRAM, time and cost estimates (never guessed) | [benchmark.py](src/tenslam/benchmark.py) |

Stage 2: 10 layers, width 640, 10 heads, SwiGLU width 1,728, 512-token context,
**60,060,800 parameters**. The implementation is custom PyTorch, not directly loadable through
Hugging Face `AutoModelForCausalLM`.

## Measured so far

**Stage 1 data (RunPod profile):** 573,932 training stories, **125,952,650 tokens**; 4,924 official validation stories,
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

## Evaluate a recovered Stage 2 export

The export uses Tenslam's custom model implementation. The verified local recovery is outside this source
repository in `../recovered_stage2/`. Run the deterministic CPU prompt suite from the repository root:

```bash
python scripts/evaluate_export.py \
  --export-dir ../recovered_stage2/runs/tenslam_60m_mix/export \
  --output ../recovered_stage2/evaluation.json
```

The script checks the tokenizer hash, vocabulary size and model parameter count, then saves greedy and sampled
continuations for story, factual, expository, arithmetic, question-answering and out-of-domain prompts.

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
- [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md): beginner-friendly architecture, pipeline and results.
- [docs/HUGGINGFACE_MODEL_CARD.md](docs/HUGGINGFACE_MODEL_CARD.md): model-card draft and source/licensing disclosures.
- [reports/stage2_recovery.md](reports/stage2_recovery.md): verified recovery, CPU outputs and validation caveat.
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

- Stage 1 was trained on synthetic children's stories; Stage 2 uses a small, mixed-source corpus. Neither is a
  general-purpose assistant or safety-tuned.
- Stage 2 generations can repeat and hallucinate facts; it is a base text-continuation model, not a chatbot.
- Stage 2 uses sources with different and sometimes item-specific terms, including Wikipedia and Project
  Gutenberg. Review rights before redistributing weights; this model is not cleared for public upload.
- No pretrained weights were used; no benchmark comparisons yet.
- The TinyStories dataset has its own licence (CDLA-Sharing-1.0). Re-check it before publishing derived data or weights.
- Source code is MIT-licensed; that licence does not cover training data or model weights.

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
