#!/usr/bin/env bash
# Run once on a fresh RunPod pod, from /workspace, after uploading tenslam-minilm-runpod.zip.
set -euo pipefail
cd /workspace
EXPECTED_TOKENIZER_SHA=ffe5be10f6b3765f4d10c97724c885ec2c53113d2fba8d3f02f34e793597d47f
rm -rf tenslam-minilm && unzip -q tenslam-minilm-runpod.zip -d tenslam-minilm
cd tenslam-minilm
pip install -q -r requirements-runpod.txt && pip install -q -e . --no-deps
got=$(sha256sum artifacts/tokenizer/tokenizer.json | cut -d' ' -f1)
[ "$got" = "$EXPECTED_TOKENIZER_SHA" ] || { echo "TOKENIZER HASH MISMATCH: $got"; exit 1; }
echo "tokenizer OK"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
python -m pytest -q
echo "SETUP DONE. Next: export HF_TOKEN=..., then bash scripts/pod_stage1_data.sh"
