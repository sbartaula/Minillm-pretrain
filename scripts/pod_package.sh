#!/usr/bin/env bash
# Pack the Stage 2 results into small pieces (large single downloads were truncated before).
set -uo pipefail
cd /workspace/tenslam-minilm
rm -f stage2_results.tgz stage2.part_*
tar czf stage2_results.tgz runs/tenslam_60m_mix/export runs/tenslam_60m_mix/metrics.jsonl \
  runs/tenslam_60m_mix/run_info.json data/mix/prepare_manifest.json artifacts/tokenizer_mix \
  train_mix.log prepare_mix.log smoke_mix.log stage2.log 2>/dev/null
ls -l stage2_results.tgz
sha256sum stage2_results.tgz | tee stage2_results.sha256
split -b 20M -d stage2_results.tgz stage2.part_
ls -l stage2.part_*
sha256sum stage2.part_* | tee -a stage2_results.sha256
