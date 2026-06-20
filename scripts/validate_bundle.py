"""Validate files required before building an offline installer."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "instruct-pix2pix"
REQUIRED_FILES = (
    MODEL_ROOT / "vae" / "config.json",
    MODEL_ROOT / "vae" / "diffusion_pytorch_model.safetensors",
    MODEL_ROOT / "scheduler" / "scheduler_config.json",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Offline model bundle is incomplete:\n{formatted}")

    for config_path in (
        MODEL_ROOT / "vae" / "config.json",
        MODEL_ROOT / "scheduler" / "scheduler_config.json",
    ):
        with config_path.open(encoding="utf-8") as file:
            json.load(file)

    weight_path = MODEL_ROOT / "vae" / "diffusion_pytorch_model.safetensors"
    weight_size_mb = weight_path.stat().st_size / (1024**2)
    if weight_size_mb < 100:
        raise SystemExit(
            f"VAE checkpoint looks incomplete: only {weight_size_mb:.1f} MB"
        )

    print(f"Offline model bundle valid ({weight_size_mb:.1f} MB checkpoint).")


if __name__ == "__main__":
    main()
