---
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- from-scratch
- educational
- tenslam
---

# Tenslam MiniLM Stage 2

This page describes the project; model files are not hosted here. See the
[GitHub source repository](https://github.com/sbartaula/Minillm-pretrain) for code and instructions. The
source code is MIT-licensed; that licence does not apply to the datasets or trained weights.

## Model summary

Tenslam MiniLM is an educational project that implements and trains small decoder-only language models from
random initialization. Stage 2 is a 60.1M-parameter base language model trained to predict the next token from a
prompt. It is useful for studying the end-to-end mechanics of pretraining; it is **not** an instruction-tuned
chatbot or a reliable source of facts.

**Model files are not hosted on this page.** This page provides the project description, training information,
sample outputs and limitations. The [GitHub repository](https://github.com/sbartaula/Minillm-pretrain) provides
source code, tests and instructions. A separate lawful copy of the inference export is required to run Stage 2.

## What the project includes

- **Stage 1:** 29.9M parameters, trained from scratch on 100M TinyStories tokens.
- **Stage 2:** 60,060,800 parameters, trained from scratch on a six-source text mix.
- A custom PyTorch Transformer, byte-level BPE tokenizer, data preparation pipeline, training/checkpoint code,
  CPU generation and reproducibility tools.

The project is intended to help learners understand and reproduce the engineering workflow—not to compete with
large pretrained foundation models.

## Architecture

- Decoder-only, pre-norm Transformer with 10 layers
- Width 640; 10 causal attention heads (64 dimensions per head)
- RoPE positional encoding, RMSNorm, SwiGLU feed-forward layers, tied token embedding/output weights
- 512-token context; 16,384-token byte-level BPE vocabulary
- 60,060,800 parameters
- Implemented in PyTorch; this is a Tenslam export, not a Transformers `AutoModelForCausalLM` checkpoint

## Training

Stage 2 was trained on an NVIDIA L4 with BF16 autocast, AdamW, gradient accumulation, gradient clipping, warmup
and cosine learning-rate decay. It stopped normally at its configured 5.5-hour limit:

- Steps completed: 12,076 of 15,259 planned
- Tokens processed: 791,412,736 of the 1B-token target
- Throughput: approximately 40.7k tokens/second
- Inference export: F32 values in `model.safetensors`

The original resumable `last.pt` training checkpoint was not included in the recovered download. The recovered
inference export supports generation, but not exact resumption of that training run.

## Evaluation

Local CPU inference was verified using greedy and sampled decoding on seven prompts. Simple story continuations
can be readable, but the model also repeats itself and generates false details.

Examples from greedy generation:

> **Story prompt:** “Once upon a time, a young girl found a map”
>
> “She was very excited and wanted to show it to her mom. She ran to her mom and said, ‘Mom, look what I found!’”

> **Factual prompt:** “In 1905, Albert Einstein published”
>
> “a paper on the theory of relativity, which was published in the Journal of Physics. Einstein's paper was
> published in the Journal of Physics in 1908.”

The story continuation is locally coherent; the historical continuation includes invented details. These are
illustrative examples, not a benchmark.

### Validation metric limitation

The archived `val_loss=1.567` / perplexity `4.79` at step 12,000 is **not** a mixed-corpus score: the original
evaluator used the first 819,200 tokens in the concatenated validation file, which were from TinyStories.
Per-source losses are also partial slices. The recovered package does not include the validation token files,
so full mixed-source validation cannot be recomputed. Do not compare or present that aggregate as a
representative six-source result. Details are in the
[recovery report](https://github.com/sbartaula/Minillm-pretrain/blob/main/reports/stage2_recovery.md).

## Training data and rights status

Stage 2 used pinned dataset revisions. Preparation split by source document, filtered invalid and duplicate
documents, kept tokenizer training restricted to training text, and recorded source revisions and hashes. No
raw training data is included in this project page.

| Source | Pinned revision | Dataset card's listed terms | Important qualification |
|---|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64) | `f54c09f` | CDLA-Sharing-1.0 | Synthetic stories; review sharing terms for model derivatives. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | `87f0914` | ODC-BY | Dataset/database card terms do not resolve every source web page's rights. |
| [Wikipedia (English, 20231101)](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa) | `b04c8d1` | CC-BY-SA-3.0 and GFDL | Attribution/share-alike terms are relevant; model-weight implications need review. |
| [Project Gutenberg English](https://huggingface.co/datasets/sedthh/gutenberg_english/tree/28973b04f28fd7be4a6186a042bc26159d4366ca) | `28973b0` | MIT on dataset card | The card warns that some texts may remain copyrighted and need item/country-specific review. |
| [Cosmopedia Stanford](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/0ae6ec63f91742bd2d1eaef4f02232c55d719385) | `0ae6ec6` | Apache-2.0 on dataset card | Generated text; upstream seed-data provenance and terms still matter. |
| [Cosmopedia OpenStax](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/0ae6ec63f91742bd2d1eaef4f02232c55d719385) | `0ae6ec6` | Apache-2.0 on dataset card | Generated text; upstream seed-data provenance and terms still matter. |

These are the terms and cautions reported by the upstream dataset cards; they are not a determination that
redistribution of every source text or the combined model weights is permitted. In particular, the Gutenberg
card warns that some books may still be copyrighted, and FineWeb's database terms do not settle rights for every
source web page. This is a provenance summary, not legal advice. This page does not declare a licence for the
trained weights.

Primary terms to review:

- [CDLA Sharing 1.0](https://cdla.dev/sharing-1-0/), [ODC Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/)
- [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and [GNU FDL 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
- [Project Gutenberg permissions and copyright guidance](https://www.gutenberg.org/policy/permission.html)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Intended use and limitations

For education, experimentation and local inference. The model may repeat, produce incoherent passages or state
false facts. It has no instruction tuning, chat template, safety training or reliable question-answering ability.
Do not use it for high-stakes decisions or as a factual source.

## Files and local use

When available, the Tenslam inference export consists of:

- `model.safetensors` — model weights only
- `config.json` — architecture configuration
- `tokenizer.json` — tokenizer model and vocabulary
- `export_info.json` — parameter count and artifact hashes

The export is separate from the `.pt` training checkpoint. If you have a lawful copy of all four export files,
clone the source code and install the CPU dependencies:

```bash
git clone https://github.com/sbartaula/Minillm-pretrain.git
cd Minillm-pretrain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers numpy safetensors
pip install -e . --no-build-isolation
tenslam generate --export-dir /path/to/downloaded/model-files \
  --prompt "Once upon a time, a young girl found a map"
```

`--export-dir` must point to a local folder containing those four files. This is a completion interface, not a
chat interface. This project page contains documentation only; it does not offer a model download.

## Reproducibility

See the [source repository](https://github.com/sbartaula/Minillm-pretrain) for implementation, configuration,
tests and a beginner-friendly [technical overview](https://github.com/sbartaula/Minillm-pretrain/blob/main/docs/TECHNICAL_OVERVIEW.md).
The recovered run records source revisions, tokenizer and weight hashes, metrics and logs. The full training
corpus and resumable Stage 2 checkpoint are not included.
