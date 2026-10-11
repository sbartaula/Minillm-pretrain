# Scaling Up: From This 30M Model to Billion-Parameter LLMs

You asked about "1 billion". That can mean 1 billion **tokens** of training data or 1 billion **parameters**.
They have very different costs, so both are covered.

All numbers use one rule of thumb and one measurement:

- **Rule of thumb:** training compute is about `6 x parameters x training tokens` floating-point operations
  (a forward pass is roughly 2 operations per parameter per token, the backward pass about twice that).
- **Measurement:** our 29.9M-parameter model ran at 153,000 tokens/s on one RTX A5000 in bf16. That is
  `6 x 29.9M x 153k = 27.5 TFLOP/s` of useful compute.
- **Everything below is an extrapolation.** Real efficiency changes with model size, kernels, memory and
  communication, so treat these as order-of-magnitude figures, not quotes.

## 1. Same 30M model, 1 billion training tokens

- Time on the same GPU: `1e9 / 153,000 = 6,540 s`, about **1.8 hours**.
- Cost at about 0.25 EUR/h: roughly **0.45 EUR** of GPU time, plus setup (about 0.6 EUR in total).
- That is only about 9x the 100M plan. Cheap.
- The catch is **data, not money**. TinyStories is only roughly 450 to 500M tokens in total (my estimate from
  the measured ~219 tokens per story; not counted exactly), so 1B tokens means about 2 passes over all of it.
  Repeating data brings diminishing returns, and a 30M model has little room to use more. It would also start
  memorizing.

Conclusion: more tokens help up to a point for a fixed model. For the same effort, a **bigger model** on the full
dataset or a **better, more diverse dataset** helps more than endlessly repeating TinyStories.

## 2. A 1 billion parameter model

The Chinchilla result suggests about **20 training tokens per parameter** for compute-optimal training:
1B parameters, 20B tokens.

| Quantity | Our 30M model (100M tokens) | 1B-parameter model (20B tokens) |
|---|---|---|
| Compute (6 N D) | 1.8e16 FLOPs | 1.2e20 FLOPs (about 6,700x more) |
| On one RTX A5000 (27.5 TFLOP/s achieved) | 11 minutes | about **50 days** |
| Cost at 0.25 EUR/h | about 0.05 EUR | about **300 EUR** |
| Memory for weights + AdamW + gradients (fp32) | about 0.5 GB | about **16 GB** before activations |

Why this does not simply fit on the same hardware:

- **Memory:** AdamW with fp32 weights, gradients and two moment buffers is about 16 bytes per parameter.
  1B parameters is about 16 GB before activations. It is tight on 24 GB and not practical.
- **Time:** 50 days of one GPU is not usable. In practice you use 8 or more modern GPUs (H100 class, several
  times faster each) so it finishes in days, not months. Cloud cost is then in the low hundreds to
  thousands of euros depending on GPU price and efficiency.
- **Data:** you need about 20B tokens of text. TinyStories has under 0.5B. You would need something like
  FineWeb-Edu, a deduplicated and filtered web corpus (public, tens of billions of tokens).

So 1B parameters is more than 6,000 times the compute of this project. It is not "a little more".

## 3. Trillion parameters

Same formula: `6 x 1e12 x 20e12 = 1.2e26` FLOPs. That is a million times the 1B case. On one A5000 that is
about **140,000 years**. Real trillion-parameter systems use thousands to tens of thousands of GPUs for months,
and budgets of tens to hundreds of millions of dollars. Many of those models are also **mixture-of-experts**,
where only a fraction of the parameters is active per token, which cuts the compute per token.

## 4. What changes in the code when you scale

Our code is deliberately small. This is what you would add, in roughly the order you would hit each wall:

| Problem you hit | Technique | Where in this repo it would go |
|---|---|---|
| Model does not fit in one GPU's memory | **Sharded training** (PyTorch FSDP or DeepSpeed ZeRO) splits weights, gradients and optimizer states across GPUs | wrap the model in `Trainer.__init__` ([train.py](../src/tenslam/train.py)) |
| One GPU is too slow | **Data parallel** training (DDP): every GPU has a copy and sees different batches; gradients are averaged | add `torch.distributed`, give each rank a different `micro` index in `sample_batch` |
| Layers do not fit even sharded | **Tensor parallel** (split each matrix) and **pipeline parallel** (split by layers) | change `CausalSelfAttention` and `SwiGLU` linear layers |
| Activations do not fit | **Activation checkpointing** (recompute in backward) | wrap `Block` calls in `torch.utils.checkpoint` |
| Compute is the bottleneck | **`torch.compile`**, FlashAttention (already used via SDPA), **FP8** on H100 | `compile = true` in the config |
| Very long context | **GQA** (fewer key/value heads) and RoPE scaling | `CausalSelfAttention` |
| Compute per token too high | **Mixture of experts** | replace `SwiGLU` with routed experts |
| Token files too big for one machine | **Sharded `.bin` files** and streaming from object storage | extend `dataset.py` |
| Runs last weeks | **Frequent checkpoints, automatic restart, loss-spike handling** | already partly in `checkpoint.py` |

What does **not** change: the model equations (attention, RoPE, RMSNorm, SwiGLU), the loss, AdamW with warmup and
cosine decay, deterministic sampling, and checkpoint design. Our architecture is the same family used by LLaMA-class
models, only smaller.

## 5. What changes in the data when you scale

1. **Source:** instead of one synthetic children's dataset, use large, diverse text: filtered web (FineWeb /
   FineWeb-Edu), books, code, Wikipedia, math. Diversity is what produces general ability.
2. **Quality filtering:** classifier- or heuristic-based filtering of spam, boilerplate and low-quality text.
3. **Deduplication:** our exact dedup is not enough at web scale. Use **fuzzy** dedup (MinHash/LSH) and
   substring-level dedup. Duplicates cause memorization and inflate validation scores.
4. **Decontamination:** remove any text that appears in evaluation benchmarks.
5. **Tokenizer:** a larger vocabulary (32k to 128k tokens) pays off for multilingual and code data. The
   embedding cost shrinks relative to a big model, so the 8k trade-off we made for a 30M model no longer applies.
   Train it on a representative sample.
6. **Data mix and curriculum:** weights per source, and often a final phase on the cleanest data.
7. **Storage:** many shards (for example 100 MB to 1 GB each), `uint16` or `uint32` tokens, listed in a
   manifest with hashes (our manifest idea scales directly).

## 6. A realistic ladder for this project

Each step is a real, cheaper learning experience before the next:

| Step | Model | Tokens | Rough GPU time on one A5000 | Notes |
|---|---|---|---|---|
| Now | 30M | 100M | about 11 min | this repo's planned run |
| 1 | 30M | about 450M (all TinyStories) | about 50 min | needs more data prepared |
| 2 | about 125M (12 layers, d_model 768) | 2.5B | about a day or more | needs more memory-efficient settings; new web-text data |
| 3 | about 350M | 7B | weeks on one GPU, so multi-GPU | introduces DDP/FSDP |
| 4 | 1B and beyond | 20B+ | multi-node cluster | all of section 4 |

The GPU times above for steps 1 to 3 are rough extrapolations, not measurements. The first real validation of the
scaling law for this codebase would be to run steps 0 and 1 and see whether the loss follows the predicted curve.

## 7. Scaling laws in one paragraph

Loss falls smoothly as a power law in model size, data and compute. For a fixed compute budget there is a best
split between parameters and tokens (about 20 tokens per parameter in the Chinchilla fits). Training a model on
far more tokens than that (as LLaMA-class models do) costs extra training compute but gives a smaller model that is
cheaper to run later. That is the reason a small model trained on a lot of data can be a good practical choice.
