# Tenslam MiniLM: from blank weights to a working model

Tenslam MiniLM is an educational project that builds a small language model from random initialization. It
covers the practical lifecycle: data preparation, tokenization, model implementation, pretraining, checkpointing,
export and local inference. It is not a foundation-model-scale training run and it is not a chat assistant.

## What “from scratch” means here

We did not download pretrained model weights. We implemented the Transformer and training loop in PyTorch,
trained a byte-level BPE tokenizer on our training text, initialized the model randomly, and optimized it to
predict the next token. In contrast, fine-tuning starts with an already pretrained model and changes some or all
of its weights.

“From scratch” describes the model initialization and training approach; it does not mean every text in the
training corpus was authored by us. The datasets have their own licences and provenance, documented in the
model card.

## Model in plain language

The model reads token IDs, builds contextual representations, and predicts a probability distribution for the
next token. It repeats this to continue a prompt. The Stage 2 model has:

- 10 Transformer blocks and 60,060,800 trainable parameters.
- Width 640; 10 attention heads, each with width 64; feed-forward width 1,728.
- Causal self-attention, so each position can use earlier tokens but not future ones.
- RoPE for position information, RMSNorm for normalization, and SwiGLU feed-forward layers.
- Tied token embedding and output weights, reducing duplicated parameters.
- A 16,384-entry byte-level BPE vocabulary and a 512-token context.

These are common decoder-only Transformer techniques. The project's value is its transparent, small end-to-end
implementation and measured run, not a new architecture.

## Training pipeline

1. **Pin and prepare sources.** Stage 2 used pinned revisions of TinyStories, FineWeb-Edu, English Wikipedia,
   Project Gutenberg English books, and two Cosmopedia subsets. The pipeline cleaned documents, separated
   validation by parent document, filtered exact and near-duplicate/overlapping documents, and recorded counts,
   source revisions and SHA-256 hashes.
2. **Train the tokenizer on train text only.** Byte-level BPE avoids unknown characters. Validation text is not
   used to learn vocabulary merges.
3. **Pack text into token windows.** Documents end with an EOS marker and windows contain 512 input tokens plus
   next-token targets. Sampling is deterministic from the run seed, step and micro-batch.
4. **Optimize next-token prediction.** AdamW, BF16 autocast, gradient accumulation, gradient clipping, warmup
   followed by cosine learning-rate decay, periodic validation, logs and atomic resumable checkpoints.
5. **Export inference artifacts.** `model.safetensors` stores model weights; `config.json` and `tokenizer.json`
   tell Tenslam how to interpret them. The inference export is separate from the larger training checkpoint.

## Results and interpretation

Stage 1 trained a 29.9M-parameter model on 100M TinyStories tokens and achieved validation loss 1.631
(perplexity 5.11) on its TinyStories validation set.

Stage 2 used a 60.1M-parameter model, saw 791,412,736 tokens, and stopped at its 5.5-hour time limit at step
12,076 of 15,259 planned. The export loads and generates on CPU. Simple story continuations are often readable,
but the model repeats phrases and makes factual errors. It is a base text continuer, not an instruction-following
system.

**Validation caveat:** the archived Stage 2 aggregate validation loss (1.567 at step 12,000) covers only the
first TinyStories section of a concatenated validation file. Per-source losses are also partial slices. There is
no reliable full-mix validation result in the recovered files. Do not present that aggregate as representative
of all six sources. See [the recovery report](../reports/stage2_recovery.md).

## Why two weight formats?

The training checkpoint `last.pt` contains weights plus optimizer, scheduler, step, RNG and configuration state.
With the matching data and tokenizer, it supports an exact training resume. The inference export
`model.safetensors` contains weights only, accompanied by model configuration and tokenizer; it is intended for
loading and generation. The Stage 2 `.pt` checkpoint was not downloaded, so exact continuation is unavailable.
The `.safetensors` export was recovered and verified.

The `.pt` suffix itself is not a universal specification: inspect what a particular project saved inside it.
Likewise, large open-source models may use `.safetensors`, often split into multiple shards. The extension
alone does not indicate model size or capability.

For this project, use the formats like this:

- **Generate text:** load `model.safetensors` together with `config.json`, `tokenizer.json`, and Tenslam's code.
  Example: `tenslam generate --export-dir path/to/export --prompt "Once upon a time"`.
- **Resume training:** load `last.pt` with the matching config, tokenizer and data. The `.pt` checkpoint restores
  optimizer/scheduler progress and training state as well as weights. Stage 2's `last.pt` was not downloaded, so
  this exact resume is not possible for that run.

## Reproduce and inspect locally

Install the project dependencies, then run:

```bash
pip install -e . --no-build-isolation
pytest -q
python scripts/evaluate_export.py \
  --export-dir ../recovered_stage2/runs/tenslam_60m_mix/export \
  --output ../recovered_stage2/evaluation.json
```

To prepare and train from scratch, inspect the TOML configs and
[`RUNBOOK_STAGE2.md`](RUNBOOK_STAGE2.md) first. Training requires rebuilding the data and using a suitable GPU;
do not launch a paid run just to perform inference.

The basic Stage 2 commands are:

```bash
tenslam prepare-mix --config configs/data_mix.toml
tenslam train --config configs/train_mix.toml
```

Preparation streams datasets and training needs a CUDA GPU for a practical run. Review the configs, runbook,
cost controls and dataset terms before starting; this is not a quick CPU command.

## Lessons and next steps

- A loss number is meaningful only when validation coverage is representative and the calculation is correct.
- Reproducibility needs more than a seed: pin dataset revisions, hash prepared artifacts, preserve the config,
  and retain checkpoints and logs.
- Small models are valuable for learning engineering, not substitutes for large pretrained assistants.
- Resolve dataset and source-text rights before redistributing Stage 2 weights. For a separate instruction-tuning
  project, start from a suitably licensed larger pretrained model and a dataset with compatible terms.
