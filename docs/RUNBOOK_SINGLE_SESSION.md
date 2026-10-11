# One pod session: Stage 1 baseline, then Stage 2 mix

Numbers marked (est.) are estimates from the measured A5000 benchmark, not measurements.
You create and terminate the pod. Terminating (not just stopping) ends the billing.

## What you upload: ONE file
`/home/saroj/Aihackathon/tenslam-minilm-runpod.zip` (code + the 8k tokenizer, ~190 KB).
Large uploads were truncated by JupyterLab (56 MB, 20 MB, and 11 of 13 pieces of 10 MB). Small files survived,
so nothing large is uploaded: the pod downloads and tokenizes the datasets itself.

## Pod
RTX A5000 (24 GB) or similar, Runpod PyTorch 2.8 template, **container disk 30 GB**. Note the EUR/hour price.

## Order of work (chosen so the GPU never waits for data)
1. Setup + checks (~3 min)
2. Stage 1 data rebuild (~5-15 min, est.)
3. Stage 2 dry run: proves all 6 sources work, makes a small tokenizer + token files
4. Benchmark the 60M model on the dry-run files (~1 min)
5. Start Stage 1 training (~11 min, est.) AND, in a second terminal, the full Stage 2 data build at the same time
6. When both finish: 100-step smoke test, then the long Stage 2 run
7. Download results, terminate

### 1. Setup
```bash
cd /workspace && ls -l tenslam-minilm-runpod.zip     # must be exactly 192223 bytes; else re-upload
unzip -tq tenslam-minilm-runpod.zip | tail -1         # "No errors detected"
unzip -q tenslam-minilm-runpod.zip -d /tmp/x && bash /tmp/x/scripts/pod_setup.sh
```
Expect: `tokenizer OK`, a GPU line, `pytest` all passed (one test skipped), `SETUP DONE`.
Any failure: stop and send me the output.

### 2. Stage 1 data
```bash
export HF_TOKEN=...        # type it yourself
cd /workspace/tenslam-minilm && bash scripts/pod_stage1_data.sh
```
Success ends with `STAGE 1 DATA IDENTICAL TO LAPTOP: OK` (125,952,650 train tokens).
Exit code 2 means the hashes differ: data is valid TinyStories but not identical. Tell me before training.

### 3. Stage 2 dry run
```bash
tenslam prepare-mix --config configs/data_mix.toml --limit-docs 200 --force
```
Expect a table with 6 sources and no traceback. Likely failure points: a wrong field name (Gutenberg) or a
download error. Paste the error. If it prints `WARNING: tokenizer has N tokens`, the dry-run sample was too
small for 16,384 merges; for the benchmark only, run
`sed -i "s/vocab_size = 16384/vocab_size = N/" configs/model_60m.toml` (put 16384 back before the real run).

### 4. Benchmark the 60M model
```bash
tenslam benchmark --config configs/train_mix.toml --sweep 16,32,64 --price <EUR_PER_HOUR>
```
Send me the output. I use it to set micro-batch and `max_hours`. (My unverified guess: ~75k tokens/s.)

### 5. Stage 1 training + Stage 2 data build in parallel
```bash
cd /workspace/tenslam-minilm
nohup tenslam train --config configs/train_runpod.toml --resume auto > train_stage1.log 2>&1 &
nohup tenslam prepare-mix --config configs/data_mix.toml --force > prepare_mix.log 2>&1 &
tail -f train_stage1.log        # Ctrl+C stops only the tail
```
Stage 1 should start near loss 9.0, run ~150k tok/s, and end with `finished: reason completed`;
export in `runs/tenslam_30m_runpod/export`. Watch the other job with `tail -f prepare_mix.log`.
Data-build duration is unknown (network/CPU bound). If it is still running after Stage 1 finishes and
2 more hours pass, message me. Check `df -h /workspace` now and then (30 GB disk).

### 6. Long Stage 2 run
When `prepare_mix.log` ends with the share table and `manifest -> data/mix/prepare_manifest.json`:
```bash
tenslam train --config configs/train_mix.toml --max-steps 100 --out-dir runs/smoke_mix   # sanity only
nohup tenslam train --config configs/train_mix.toml --resume auto > train_mix.log 2>&1 &
```
Expect loss to start near 9.7 and fall; `val_loss/<source>` appears every 500 steps (TinyStories lowest).
If it dies, rerun the same command (`--resume auto`). `max_hours` stops the trainer, not the billing.

### 7. Finish
```bash
cd /workspace/tenslam-minilm
tar czf results.tgz runs/tenslam_30m_runpod runs/tenslam_60m_mix/export runs/tenslam_60m_mix/metrics.jsonl \
  runs/tenslam_60m_mix/run_info.json data/mix/prepare_manifest.json artifacts/tokenizer_mix *.log
sha256sum results.tgz
split -b 5M -d results.tgz results.part_      # if a single download is truncated, download the pieces
sha256sum results.part_*
```
Verify hashes on the laptop, then **terminate** the pod. Next: Phases 4-7 of `RUNBOOK_100M.md`.

## Estimated budget (assumes EUR 0.25/h; check the real price)
Setup + Stage 1 ~30 min (~EUR 0.12); Stage 2 data build mostly overlaps Stage 1, then maybe 0.5-2 h more;
Stage 2 training ~4 h (~EUR 1). Probably under EUR 2 total; the cap is EUR 8.
