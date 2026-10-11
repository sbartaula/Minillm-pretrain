# One pod session: Stage 1 baseline, then Stage 2 mix

Numbers marked (est.) are estimates from the measured A5000 benchmark, not measurements.
I never start the pod; you do. Terminate it at the end (stop still bills disk).

## Pod setup
- RTX A5000 (or similar 24 GB), PyTorch template, **container disk 30 GB**, volume at /workspace.
- Upload only: `tenslam-minilm-runpod.zip` (code, ~0.2 MB) and, for Stage 1, `bundle.part_00..12`.
- A Hugging Face read token helps download speed: `export HF_TOKEN=...` (type it in the pod terminal yourself).

```bash
cd /workspace && rm -rf tenslam-minilm && unzip -q tenslam-minilm-runpod.zip -d tenslam-minilm
cd tenslam-minilm && pip install -r requirements-runpod.txt && pip install -e . --no-deps
python -m pytest -q          # expect all pass, CUDA test runs
```

## Stage 1: 100M-token TinyStories baseline (~11 min training, est.)
```bash
cd /workspace && cat bundle.part_* > tenslam-runpod-bundle.zip
sha256sum tenslam-runpod-bundle.zip   # must be eeada3a2978395d9b81ffa7251305517f9dc318cdbd70f5beef6afe898c6fff2
unzip -q tenslam-runpod-bundle.zip -d /tmp/b          # bundle holds OLD code too; copy only data + tokenizer
cp -r /tmp/b/data /tmp/b/artifacts /workspace/tenslam-minilm/
cd /workspace/tenslam-minilm
nohup tenslam train --config configs/train_runpod.toml --resume auto > train_stage1.log 2>&1 &
tail -f train_stage1.log
```
Stage 1 needs `data/tokens_runpod/` and `artifacts/tokenizer/`. Download `runs/tenslam_30m_runpod/` before moving on.

## Stage 2: build the mix ON the pod (data never touches your laptop)
```bash
# 1. dry run: ~200 docs per source, proves every dataset/field name works (a few minutes)
tenslam prepare-mix --config configs/data_mix.toml --limit-docs 200 --force
# 2. real run (time depends on pod network/CPU; unknown until measured). Resumable per source.
nohup tenslam prepare-mix --config configs/data_mix.toml --force > prepare_mix.log 2>&1 &
tail -f prepare_mix.log
```
Read the printed token shares and `data/mix/prepare_manifest.json`. The manifest records per-source counts,
rejections, dataset revisions and licences. If the tokenizer vocab is below 16384, set `vocab_size` in
`configs/model_60m.toml` to the real value.

## Benchmark + smoke the 60M model, then the long run
```bash
tenslam benchmark --config configs/train_mix.toml --sweep 16,32,64 --price <EUR_PER_HOUR_FROM_RUNPOD>
tenslam train --config configs/train_mix.toml --max-steps 100 --out-dir runs/smoke_mix   # sanity only
```
Set `max_hours` in `configs/train_mix.toml` from the benchmark (est. ~75k tok/s -> 1B tokens ~3.7 h, ~EUR 1, at an
assumed EUR 0.25/h). Then:
```bash
nohup tenslam train --config configs/train_mix.toml --resume auto > train_mix.log 2>&1 &
```
`max_hours` stops the trainer, not the billing. Per-source validation losses (`val_loss/<source>`) are in
`runs/tenslam_60m_mix/metrics.jsonl`.

## Finish
```bash
cd /workspace/tenslam-minilm
tar czf results.tgz runs/tenslam_30m_runpod runs/tenslam_60m_mix/export runs/tenslam_60m_mix/metrics.jsonl \
  runs/tenslam_60m_mix/run_info.json data/mix/prepare_manifest.json artifacts/tokenizer_mix *.log
sha256sum results.tgz
```
Download via JupyterLab (use the split trick if the file is large), verify the hash, then **terminate** the pod.
Then follow Phases 4-7 of `RUNBOOK_100M.md` (evaluate, publish weights, push code).
