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

> **Publication status: HOLD.** Dataset and source-text redistribution rights have not been resolved, so do not
> upload these weights publicly yet. This Hugging Face repository is a project page with this card only; it
> contains no model weights, tokenizer or config files. This card deliberately does not declare a model-weight
> licence.
>
> The GitHub source code is MIT-licensed. That code licence does not apply to datasets or these trained weights.

## Model summary

Tenslam MiniLM Stage 2 is a 60.1M-parameter decoder-only language model pretrained from random initialization
to predict the next token. It is an educational artifact for studying tokenization, Transformer training,
evaluation and inference. It is a base text-completion model, not an instruction-tuned chatbot.

**Files are not hosted here.** This page is for project information and points to the
[MIT-licensed source repository](https://github.com/sbartaula/Minillm-pretrain). To run the trained model, a user
needs a separate copy of its four-file inference export. The weights are not publicly downloadable while source
and dataset redistribution terms are being reviewed.

## Architecture

- 10 decoder blocks; width 640; 10 attention heads; feed-forward width 1,728
- Causal attention, RoPE, RMSNorm, SwiGLU, tied embeddings
- 512-token context; byte-level BPE tokenizer with 16,384 tokens
- 60,060,800 parameters
- Custom PyTorch implementation; this export is not a Transformers `AutoModelForCausalLM` checkpoint

## Training

Trained on an NVIDIA L4 using BF16 autocast, AdamW, gradient accumulation, gradient clipping, warmup and cosine
learning-rate decay. The run stopped at its configured 5.5-hour wall-clock limit:

- Step 12,076; 791,412,736 tokens seen (planned target: 1B)
- Throughput around 40.7k tokens/s
- Inference weights exported as F32 safetensors

### Evaluation and caveat

The archived `val_loss=1.567` / perplexity `4.79` at step 12,000 is a TinyStories-only partial validation
slice, not a representative mixed-corpus metric. Per-source metrics are also partial slices. The recovered
package does not include validation token files, so a full mixed-source validation score cannot be calculated.
Seven-prompt greedy and sampled CPU continuations are recorded in the project recovery report; they show
repetition and factual hallucinations. These are qualitative probes, not a benchmark.

Examples from deterministic greedy generation:

> **Story prompt:** “Once upon a time, a young girl found a map”
>
> “She was very excited and wanted to show it to her mom. She ran to her mom and said, ‘Mom, look what I found!’”

> **Factual prompt:** “In 1905, Albert Einstein published”
>
> “a paper on the theory of relativity, which was published in the Journal of Physics. Einstein's paper was
> published in the Journal of Physics in 1908.”

The story continuation is coherent, while the historical details are false. This illustrates why the model
should not be treated as a factual reference or assistant.

## Training data and rights status

Data preparation pinned revisions, separated validation at document level, filtered duplicates and recorded
source/content hashes. No raw training data is redistributed with the model.

| Source | Pinned revision | Dataset card's listed terms | Important qualification |
|---|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64) | `f54c09f` | CDLA-Sharing-1.0 | Synthetic stories; review sharing terms for model derivatives. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | `87f0914` | ODC-BY | Dataset/database card terms do not resolve every source web page's rights. |
| [Wikipedia (English, 20231101)](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa) | `b04c8d1` | CC-BY-SA-3.0 and GFDL | Attribution/share-alike terms are relevant; model-weight implications need review. |
| [Project Gutenberg English](https://huggingface.co/datasets/sedthh/gutenberg_english/tree/28973b04f28fd7be4a6186a042bc26159d4366ca) | `28973b0` | MIT on dataset card | The card warns that some texts may remain copyrighted and need item/country-specific review. |
| [Cosmopedia Stanford](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/0ae6ec63f91742bd2d1eaef4f02232c55d719385) | `0ae6ec6` | Apache-2.0 on dataset card | Generated text; upstream seed-data provenance and terms still matter. |
| [Cosmopedia OpenStax](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/0ae6ec63f91742bd2d1eaef4f02232c55d719385) | `0ae6ec6` | Apache-2.0 on dataset card | Generated text; upstream seed-data provenance and terms still matter. |

Dataset-card labels alone do not establish that every underlying text or model weight can be redistributed
under one common licence. This is a provenance disclosure, not legal advice. Public weight release is on hold
pending a source-specific rights review.

Primary terms to review:

- [CDLA Sharing 1.0](https://cdla.dev/sharing-1-0/)
- [ODC Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/)
- [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
  [GNU FDL 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
- [Project Gutenberg permissions and copyright guidance](https://www.gutenberg.org/policy/permission.html)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

These source terms do not by themselves settle whether this trained model's weights can be redistributed.

## Intended use and limitations

Educational experimentation, local inference and studying small-scale pretraining. The model may repeat,
produce incoherent passages, or state false facts. It has no instruction tuning, safety training, or reliable
question-answering ability. Do not use it for high-stakes applications or as a factual source.

## Files and use

`model.safetensors`, `config.json`, `tokenizer.json`, and `export_info.json` form the Tenslam inference export.
Use the Tenslam repository's custom loader and tokenizer. `model.safetensors` contains weights only; it is not
a training-resume checkpoint.

If you have a lawfully obtained local copy of all four files, clone the code and run:

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

The export directory must contain the four files listed above. `--export-dir` is a local folder, not a
Transformers model ID; Tenslam loads its own custom architecture. It generates text continuations and does not
provide a chat interface or instruction-following behavior.

## Reproducibility

Source code: [sbartaula/Minillm-pretrain](https://github.com/sbartaula/Minillm-pretrain). See its technical
overview for implementation, configuration and tests. The recovered archive records source revisions, tokenizer
and weight hashes, metrics and logs. The full training corpus and resumable Stage 2 checkpoint are not included.
