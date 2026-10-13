# Historical Model Card Template: Tenslam MiniLM 30M

> This is the original Stage 1 template and is now historical. It still contains `TBD` values and is not ready
> to publish. For the trained Stage 2 model, use [HUGGINGFACE_MODEL_CARD.md](HUGGINGFACE_MODEL_CARD.md),
> which is also marked HOLD pending dataset/source-text rights review.

## Model details

- **Architecture:** decoder-only Transformer, 8 layers, d_model 512, 8 heads, SwiGLU (d_ff 1408), RMSNorm, RoPE,
  tied embeddings, no biases, 512-token context. 29,893,120 parameters.
- **Tokenizer:** byte-level BPE, 8,192 tokens, trained on the training split only.
- **Trained from scratch:** yes, random initialization. No pretrained weights were used.
- **Code:** <https://github.com/TBD/tenslam-minilm>
- **Weights file:** `model.safetensors` (fp32 values, tied tensors stored once).

## Training data

- `roneneldan/TinyStories`, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Licence CDLA-Sharing-1.0 (check the
  dataset card before redistribution).
- 573,932 training stories (125,952,650 tokens) after cleaning and exact deduplication; 4,924 official validation
  stories (992,483 tokens); no train/validation overlap after normalization.

## Training procedure

- 100,000,000 tokens, 1,526 steps, 65,536 tokens per step, AdamW (betas 0.9/0.95, weight decay 0.1 on matrices),
  warmup 150 steps then cosine decay from 1e-3 to 1e-4, gradient clipping 1.0, bf16 autocast.
- Hardware: TBD (planned: one RTX A5000). Wall-clock time: TBD. Cost: TBD.

## Evaluation

| Metric | Value |
|---|---|
| Validation loss (nats/token) | TBD |
| Validation perplexity | TBD |
| Best step | TBD |

Perplexity here is over **our own 8,192-token vocabulary**, so it is not comparable with other models' numbers.

## Intended use and limitations

- Educational. Produces short, simple, children's-story style English.
- Knows nothing outside TinyStories-style text: no facts, no instructions, no safety tuning, no multilingual ability.
- May produce repetitive, incoherent or factually nonsensical text. Not for any production use.

## Reproduce

See `docs/RUNBOOK_100M.md` in the code repository.
