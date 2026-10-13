#!/usr/bin/env bash
# Stage 2 driver. Start ONCE, in the background (see docs/RUNBOOK_STAGE2.md):
#   nohup bash scripts/pod_stage2.sh > stage2.log 2>&1 &
# Steps: build mix data -> 100-step smoke test -> long 60M training -> package results.
# Needs HF_TOKEN in the environment. Safe to re-run after an interruption (resumes).
set -uo pipefail
cd /workspace/tenslam-minilm
exec 9>/tmp/stage2.lock
flock -n 9 || { echo "Stage 2 is already running. Exiting."; exit 1; }
ts() { date '+%H:%M:%S'; }

if [ ! -f data/mix/prepare_manifest.json ]; then
  echo "[$(ts)] STEP 1/4: building the data mix"
  tenslam prepare-mix --config configs/data_mix.toml --force > prepare_mix.log 2>&1
  [ -f data/mix/prepare_manifest.json ] || { echo "[$(ts)] DATA BUILD FAILED, see prepare_mix.log"; exit 1; }
fi
echo "[$(ts)] data ready:"; tail -8 prepare_mix.log
V=$(python -c "from tokenizers import Tokenizer;print(Tokenizer.from_file('artifacts/tokenizer_mix/tokenizer.json').get_vocab_size())")
echo "tokenizer vocab: $V"
[ "$V" = "16384" ] || { echo "vocab $V != 16384: model config must match. Stopping."; exit 1; }

echo "[$(ts)] STEP 2/4: smoke test (100 steps)"
tenslam train --config configs/train_mix.toml --max-steps 100 --out-dir runs/smoke_mix > smoke_mix.log 2>&1
tail -3 smoke_mix.log
grep -q "finished" smoke_mix.log || { echo "SMOKE TEST FAILED, see smoke_mix.log"; exit 1; }

echo "[$(ts)] STEP 3/4: long training (resumes automatically)"
tenslam train --config configs/train_mix.toml --resume auto > train_mix.log 2>&1
tail -3 train_mix.log

echo "[$(ts)] STEP 4/4: packaging"
bash scripts/pod_package.sh
echo "[$(ts)] ALL DONE. You can download the files listed above, then TERMINATE the pod."
