# Runbook: Pretraining Tenslam MiniLM on 100M Tokens (RunPod)

> **Superseded for the pod procedure:** use [RUNBOOK_SINGLE_SESSION.md](RUNBOOK_SINGLE_SESSION.md). Phases 4-7 (after training) here still apply.

This is the exact, ordered procedure for the real training run (Session 4), including what to do afterwards.
Nothing here has been run yet. **Every time and cost below is an estimate extrapolated from the measured
benchmark (153k tokens/s on an RTX A5000, bf16), not a measured result of a full run.**

Run configuration: [configs/train_runpod.toml](../configs/train_runpod.toml)

| Setting | Value |
|---|---|
| Training tokens | 100,000,000 (about 0.8 epoch of the 125.95M prepared tokens) |
| Tokens per optimizer step | 65,536 (micro-batch 32 x accumulation 4 x 512) |
| Optimizer steps | 1,526 |
| Learning rate | warmup 150 steps to 1e-3, cosine decay to 1e-4 |
| Precision | bf16 (auto-selected on the A5000, no GradScaler) |
| Safety cap | `max_hours = 0.5`, token budget 100M |
| Expected pure training time | about 11 minutes (100M / 153k tok/s = 653 s) |
| Expected GPU cost | roughly 0.10 to 0.30 EUR including setup (assumed 15 to 30 min at about 0.25 EUR/h) |

## Phase 0: before you rent anything (laptop)

1. Confirm the data files exist and are intact:
   ```bash
   cd /home/saroj/Aihackathon/ai2026/tenslam-minilm
   ls -lh data/tokens_runpod          # train.bin ~241M, validation.bin ~1.9M, prepare_manifest.json
   ```
2. The upload pieces from last time are still in `/home/saroj/Aihackathon/pods_parts/` (13 files,
   `bundle.part_00` to `bundle.part_12`). Reuse them. The joined bundle must hash to
   `eeada3a2978395d9b81ffa7251305517f9dc318cdbd70f5beef6afe898c6fff2`.
3. Also upload `tenslam-minilm-runpod.zip` (149 KB source). If you changed any code since, rebuild it first.
   The docs and scripts are not needed on the pod.

## Phase 1: create the Pod

1. RunPod, Pods, Deploy. Pick **RTX A5000 (24 GB)** or anything with 24 GB or more. Template: *Runpod PyTorch
   2.8.0* (CUDA 12.8), the same one the benchmark used.
2. Storage: the benchmark used no persistent volume. For the real run, **everything you need to keep must be
   downloaded before you terminate the Pod**. A small network volume survives termination but adds a cost and
   can limit which GPUs you can pick. Check the price shown in the UI. I am not certain of RunPod's exact
   rules for what survives a stop versus a terminate, so do not rely on it: download first.
3. Note the price per hour shown in the UI (in USD). Set a reminder for 45 minutes after you start.

## Phase 2: set up the Pod (web terminal)

```bash
cd /workspace
# 1. upload bundle.part_* and tenslam-minilm-runpod.zip in JupyterLab first, wait for all uploads to finish
ls -l bundle.part_* | wc -l                                  # must print 13
cat bundle.part_* > tenslam-runpod-bundle.zip
sha256sum tenslam-runpod-bundle.zip                          # must match eeada3a2...c6fff2
unzip -t tenslam-runpod-bundle.zip | tail -2                 # "No errors detected"
unzip -o -q tenslam-runpod-bundle.zip -d /workspace/tenslam-minilm
unzip -o -q tenslam-minilm-runpod.zip -d /workspace/tenslam-minilm
cd /workspace/tenslam-minilm
ls -lh data/tokens_runpod                                    # train.bin ~241M

pip install -r requirements-runpod.txt
pip install -e . --no-build-isolation
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Stop here if CUDA prints `False`. Do not train on CPU.

## Phase 3: start training (keep it alive if the browser disconnects)

A web terminal can disconnect. Run training in the background with a log file:

```bash
cd /workspace/tenslam-minilm
nohup tenslam train --config configs/train_runpod.toml --resume auto > train.log 2>&1 &
sleep 60 && tail -n 20 train.log
```

`--resume auto` is safe on a fresh start (it simply starts at step 0) and continues from `last.pt` if you ever
have to restart. The trainer verifies the SHA-256 of `train.bin`, `validation.bin` and the tokenizer against the
manifest before it begins. A truncated upload fails here, loudly, before spending GPU time.

Watch progress any time with `tail -n 30 train.log`.

### What you should see

- `step 0 | val_loss ~9.0 | val_ppl ~8,000`. This is `ln(8192) = 9.01`, i.e. random guessing.
  If it is far from 9, something is wrong: stop.
- `step N/1526 | loss ... | lr ... | gnorm ... | ~150,000 tok/s`. Throughput near 150k tok/s means the GPU is
  healthy. If it is under 50k tok/s, stop and investigate before paying for more time.
- Validation every 250 steps (6 times in total), checkpoint every 250 steps, a text sample every 500 steps.
- Loss falling quickly in the first ~150 steps, then more slowly. Samples go from gibberish, to real words, to
  grammatical sentences, to small stories. I will not promise a final loss: that is exactly what this run measures.

### When to stop early (and how)

- Loss becomes `nan` or `inf`: the trainer stops itself and keeps the last good checkpoint.
- Validation loss rises for several evaluations in a row, or `gnorm` explodes: stop and tell me.
- To stop cleanly (saves a checkpoint, no export): `ps aux | grep tenslam`, then send SIGINT to that PID
  with `kill -INT <PID>`.

### Finishing

When the log prints the final validation line and `stop`, the trainer has written:

```
runs/tenslam_30m_runpod/
  last.pt          full resumable checkpoint (~359 MB): weights + optimizer + scheduler + RNG states
  export/          inference-only: model.safetensors (~120 MB), config.json, tokenizer.json, export_info.json
  metrics.jsonl    every logged step: loss, lr, grad norm, tokens/s, validation, samples
  run_info.json    config and environment of this run
```

(The export is written when the run completes or hits `max_hours`.)

## Phase 4: get your files out BEFORE terminating the Pod

Large JupyterLab transfers have silently truncated before, in both directions, so use small pieces and checksums.

```bash
cd /workspace/tenslam-minilm/runs/tenslam_30m_runpod
tar -cf /workspace/tenslam_run_artifacts.tar export metrics.jsonl run_info.json ../../train.log
cd /workspace
sha256sum tenslam_run_artifacts.tar | tee tenslam_run_artifacts.tar.sha256
split -b 10M -d -a 2 tenslam_run_artifacts.tar artifacts.part_
ls -l artifacts.part_*
```

In JupyterLab, download `tenslam_run_artifacts.tar.sha256` and every `artifacts.part_*` file.
You do **not** need `last.pt` unless you plan to continue training later (then download it the same way).

On the laptop, put the pieces in one folder and run:

```bash
cat artifacts.part_* > tenslam_run_artifacts.tar
sha256sum tenslam_run_artifacts.tar          # must equal the value in the .sha256 file
```

**Alternative that avoids browser downloads:** publish straight from the Pod to the Hugging Face Hub (see Phase 6).
Create a **write** token at huggingface.co/settings/tokens, use it once, and delete the token afterwards.
Never paste a token into a chat, a file in the repo, or a screenshot.

Only after the checksum matches locally:

1. Terminate the Pod in the RunPod UI (Stop alone may keep billing for disk, so confirm in the Billing page).
2. Check that no Pods or volumes are still listed.

## Phase 5: after training, on the laptop

```bash
cd /home/saroj/Aihackathon/ai2026/tenslam-minilm
mkdir -p models/tenslam-30m-v1 reports/run_100m
tar -xf /path/to/tenslam_run_artifacts.tar -C models/tenslam-30m-v1 export
mv models/tenslam-30m-v1/export/* models/tenslam-30m-v1/ && rmdir models/tenslam-30m-v1/export
tar -xf /path/to/tenslam_run_artifacts.tar -C reports/run_100m metrics.jsonl run_info.json

source .venv/bin/activate && unset PYTHONPATH
tenslam generate --export-dir models/tenslam-30m-v1 \
  --prompt "Once upon a time, there was a little girl named" --temperature 0.8
python scripts/plot_metrics.py reports/run_100m/metrics.jsonl --out reports/training_curve.png
```

Evaluate honestly and record:

- final validation loss and perplexity (from the last `eval` line in `metrics.jsonl`),
- the curve image `reports/training_curve.png`,
- 5 to 10 sample stories at temperatures 0.5, 0.8 and 1.0, including **bad** ones,
- the real time and real cost from the RunPod billing page, so the estimate can be corrected.

`models/` and `*.pt` are not committed to git (weights are too large for normal git; see Phase 6).

## Phase 6: where to put the weights

| What | Where | Why |
|---|---|---|
| Code, configs, tests, docs, tokenizer, small reports, curve image | **GitHub** | normal git files, all small |
| `model.safetensors` (~120 MB), `config.json`, `tokenizer.json`, model card | **Hugging Face Hub** (model repo) | built for weights, versioned, easy `snapshot_download`, free for public models |
| `last.pt` (359 MB, full training state) | your laptop / cloud drive, optional | only needed to resume training |
| Processed token files (`train.bin`) | not published; regenerate with `tenslam prepare` | manifests carry hashes; the dataset has its own license (CDLA-Sharing-1.0) |

GitHub rejects files over 100 MB in normal git, which is why the 120 MB weights go elsewhere. A GitHub Release
attachment (up to 2 GB) is a second option.

Publishing to Hugging Face (after you create an account and a token; confirm the licence terms first):

```bash
pip install -U huggingface_hub
huggingface-cli login                       # paste the token in the prompt, not in a command line
huggingface-cli upload <your-username>/tenslam-minilm-30m models/tenslam-30m-v1 . 
```

Add the model card ([docs/MODEL_CARD.md](MODEL_CARD.md)) as `README.md` in that model repo after filling in the
real results. Do not fill in numbers you have not measured.

Loading later from anywhere:

```python
from huggingface_hub import snapshot_download
from tenslam.export import load_exported
model = load_exported(snapshot_download("<your-username>/tenslam-minilm-30m"))
```

## Phase 7: commit and publish the code (GitHub)

1. Create an empty repo on github.com (no README, no licence, so the first push is clean).
2. Add the remote and push (the first commit is already prepared locally):
   ```bash
   git remote add origin git@github.com:<your-username>/tenslam-minilm.git
   git push -u origin master
   ```
3. After training, add a second commit with the real results: fill in the README results table, add
   `reports/training_curve.png` and the real benchmark and cost numbers.
4. Choose and add a licence file for the code (for example MIT or Apache-2.0). That is your decision. The
   TinyStories data carries its own CDLA-Sharing-1.0 terms; re-read them before publishing weights.

## Cost and time guard rails

- Total expected GPU time: setup 15 to 30 min (assumed), training about 11 min, export and download 5 to 10 min.
- Worst realistic case if you are slow or something fails once: 1.5 hours, about 0.40 EUR at 0.25 EUR/h.
- The trainer's `max_hours = 0.5` stops a runaway training loop. It does **not** stop the Pod itself, so billing
  continues until **you** terminate it.
