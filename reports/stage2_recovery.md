# Stage 2 recovery and evaluation

**Run:** `tenslam_60m_mix`  
**Status:** inference export recovered, SHA-256 verified, CPU generation tested  
**Scope:** facts below come from the downloaded archive, the local Tenslam source, and a local CPU evaluation. No RunPod access was used.

## Recovered files

The downloaded archive is named `../stage2_results.tgz.gz` on disk. Its SHA-256 is
`e86e6253f10baf922989989db18bac5c9071eb7fe3dc86769dda442d0121ba1f`, which matches the checksum entry
named `stage2_results.tgz` in `../stage2_results.sha256`. The extension differs; the bytes match the supplied
checksum. Both gzip and tar integrity checks passed.

The archive was extracted without overwriting project files into `../recovered_stage2/`. It contains:

- `runs/tenslam_60m_mix/export/model.safetensors`, `config.json`, `tokenizer.json`, `export_info.json`
- `runs/tenslam_60m_mix/metrics.jsonl` and `run_info.json`
- `data/mix/prepare_manifest.json`
- `train_mix.log`, `prepare_mix.log`, `smoke_mix.log` and `stage2.log`

Verified export hashes:

- Weights: `890c4bb83409aab9e7b45bd5decf0aa3e72524520c6987ccfc53f75a9e677440`
- Tokenizer: `00244176f00b4be2fa22c6eaf6703e2ce08b0654fcd24cbeb276992c33281d70`

The package does **not** contain `runs/tenslam_60m_mix/last.pt`, the packed train/validation token files, or the
raw datasets. A local `runs/cpu_smoke/last.pt` exists, but it belongs to the separate CPU smoke run and is not the
Stage 2 checkpoint. Exact Stage 2 resume is therefore unavailable.

## Confirmed architecture and data

- **Parameters:** 60,060,800, confirmed by the export metadata and the custom model's parameter counter.
- **Architecture:** 10-layer pre-norm decoder-only Transformer; width 640; 10 heads (head dimension 64);
  feed-forward width 1,728; 512-token context; vocabulary 16,384.
- **Components:** causal self-attention, RoPE (`theta=10,000`), RMSNorm (`epsilon=1e-5`), SwiGLU, tied token
  embeddings/output weights, no linear biases.
- **Tokenizer:** train-only byte-level BPE, 16,384-token vocabulary.
- **Dataset:** 946,087,312 training tokens and 5,747,658 validation tokens prepared across TinyStories,
  FineWeb-Edu, Wikipedia, Gutenberg, Cosmopedia Stanford and Cosmopedia OpenStax. The prepared training shares
  were about 12.8%, 40.0%, 16.9%, 18.1%, 8.6% and 3.7%, respectively.
- **Export precision:** the saved safetensors are F32. Training used CUDA BF16 autocast; those are different
  facts, not a contradiction.
- **Format:** this is a Tenslam inference export, not a Hugging Face Transformers `AutoModelForCausalLM`
  package. The project loader strictly loaded the weights against its custom model and generated text.

## Training outcome

The planned run was 1,000,000,000 tokens / 15,259 optimizer steps, with 65,536 tokens per step
(micro-batch 32 × accumulation 4 × sequence length 512), AdamW, peak learning rate `6e-4`, minimum
`6e-5`, 500 warmup steps, cosine decay, weight decay 0.1, and a 5.5-hour time limit. It stopped normally for
`max_hours`, not from a reported numerical or memory failure:

- Final saved step: **12,076 / 15,259**
- Tokens seen: **791,412,736** (79.1% of the 1B target; about 0.84 token-equivalents of the prepared training
  stream, not proof that 84% of distinct documents were visited)
- Runtime: **19,801 seconds** (about 5.5 hours)
- Typical throughput: **about 40.7k tokens/s** on an NVIDIA L4
- Logged peak CUDA allocation: **about 10.27 GiB**
- Last scheduled validation record: step 12,000; the current export was saved at step 12,076

### Important validation limitation

At step 12,000 the logs report `val_loss=1.5670`, `val_ppl=4.7923`, `val_tokens=819,200`. **That is not a
representative mixed-corpus validation score.** The old evaluator took the first 50 batches from a concatenated
validation file. TinyStories comes first and contributes 1,009,411 validation tokens, so all 819,200 evaluated
targets were from TinyStories.

The per-source values logged at that step were:

| Source | Logged loss |
|---|---:|
| TinyStories | 1.6000 |
| FineWeb-Edu | 3.3959 |
| Wikipedia | 3.1045 |
| Gutenberg | 3.3116 |
| Cosmopedia Stanford | 2.5654 |
| Cosmopedia OpenStax | 2.6276 |

Each source number uses only the first 10 non-overlapping batches (163,840 targets) of that source's validation
file, not its entire split. Weighting these partial source measurements by the full source validation-token
counts gives a **post-hoc estimate** of about 2.917 nats/token (perplexity about 18.5); this is not a measured
full-validation result. The raw validation bins were not included in the download, so the full mixed loss cannot
be recomputed from the recovered files.

The trainer now evaluates mixed validation data per source and weights the aggregate by each source's full
validation-token count. This corrects the first-source-only aggregate for future runs; it does not rewrite the
historical metrics in this archive.

## Local CPU generation

The export loaded on CPU and completed the seven-prompt greedy/sampled evaluation using
`scripts/evaluate_export.py`. Results, prompts, parameters and artifact hashes are preserved in
`../recovered_stage2/evaluation.json`.

Observed behavior:

- Simple story and sentence completion were often fluent and locally coherent.
- Continuations repeated phrases, especially on expository and arithmetic prompts.
- The model hallucinated factual details (for example, an incorrect account of Einstein's 1905 publication and
  an incorrect response to the capital-of-France prompt).
- It did not reliably solve arithmetic or follow a question-answer format.
- Sampling increased variety but did not make the output factually reliable.

This is a small **base text-continuation model**, not an instruction-tuned chatbot. These qualitative examples
are a small diagnostic set, not a benchmark or a capability claim.

Representative greedy outputs from the fixed prompt suite:

> **Prompt:** “Once upon a time, a young girl found a map”
>
> **Continuation:** “She was very excited and wanted to show it to her mom. She ran to her mom and said,
> ‘Mom, look what I found!’”

> **Prompt:** “In 1905, Albert Einstein published”
>
> **Continuation:** “a paper on the theory of relativity, which was published in the Journal of Physics.
> Einstein's paper was published in the Journal of Physics in 1908.”

The story continuation is locally coherent; the historical continuation invents incorrect details. More
unfiltered prompt/output pairs (including repetitive arithmetic and expository examples) are in the local
evaluation JSON, which is kept outside the GitHub repository along with the model export.

## Recovery and next step

The F32 model weights and tokenizer survive, so CPU inference and a fresh weights-only fine-tuning experiment
are possible. Exact training resume is not: the model export has no optimizer, scheduler, random-state or data
sampler checkpoint, and the training data files are absent. Do not pay for more pretraining until there is a
specific experiment and a plan to rebuild or obtain the data.

Before publishing weights, review the original dataset terms, especially Wikipedia's CC-BY-SA/GFDL terms and
any attribution/share-alike obligations. A separate post-training exercise should use a suitably licensed,
larger pretrained base model rather than trying to turn this 60M model into a capable assistant.
