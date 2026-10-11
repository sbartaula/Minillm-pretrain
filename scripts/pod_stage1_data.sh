#!/usr/bin/env bash
# Rebuild the Stage 1 TinyStories token files on the pod and compare with the laptop's hashes.
set -uo pipefail
cd /workspace/tenslam-minilm
tenslam prepare --config configs/data_runpod.toml
T=$(sha256sum data/tokens_runpod/train.bin | cut -d' ' -f1)
V=$(sha256sum data/tokens_runpod/validation.bin | cut -d' ' -f1)
WT=be4b33c015720068c89ae3778c1e9ab8f4048b89968cf7965a0fb7f31b8e8d1c
WV=c86d7dec007073f599613aad560e35447aec3a2ee1eabded610b639419e564af
echo "train      $T"; echo "validation $V"
if [ "$T" = "$WT" ] && [ "$V" = "$WV" ]; then echo "STAGE 1 DATA IDENTICAL TO LAPTOP: OK"
else echo "HASHES DIFFER from laptop (see docs/RUNBOOK_SINGLE_SESSION.md). Do not train until you have checked."; exit 2; fi
