"""SafeShot desktop entry point."""

import os
import sys

from PIL import Image

from imageshield.ui import launch

# PyInstaller console=False sets sys.stdout/stderr to None.
# uvicorn's DefaultFormatter calls sys.stdout.isatty() during logging setup, crashing on None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


if __name__ == "__main__":
    if os.environ.get("SAFESHOT_SMOKE_TEST") == "1" or os.environ.get("IMAGESHIELD_SMOKE_TEST") == "1":
        from imageshield.protection import ProtectionService, ProtectionSettings

        smoke_service = ProtectionService(
            settings=ProtectionSettings(resolution=64, steps=1)
        )
        smoke_service.protect(Image.new("RGB", (64, 64), color=(128, 128, 128)))
    else:
        launch()
