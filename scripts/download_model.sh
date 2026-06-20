#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models/instruct-pix2pix"
MODEL_REPO="timbrooks/instruct-pix2pix"
MODEL_REVISION="31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"

cd "$ROOT"

if [[ -f "$MODEL_DIR/vae/config.json" \
   && -f "$MODEL_DIR/vae/diffusion_pytorch_model.safetensors" \
   && -f "$MODEL_DIR/scheduler/scheduler_config.json" ]]; then
  echo "The pinned offline model is already present."
  python scripts/validate_bundle.py
  exit 0
fi

echo "Downloading the pinned SafeShot model files..."
echo "Source: https://huggingface.co/$MODEL_REPO"
echo "Revision: $MODEL_REVISION"

MODEL_DIR="$MODEL_DIR" \
MODEL_REPO="$MODEL_REPO" \
MODEL_REVISION="$MODEL_REVISION" \
python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_dir = Path(os.environ["MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
    allow_patterns=[
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ],
    local_dir=model_dir,
)
PY

python scripts/validate_bundle.py
echo "Model download complete: $MODEL_DIR"
