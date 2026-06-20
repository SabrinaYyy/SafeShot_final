"""Paths that work from source and from a PyInstaller bundle."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bundle_root() -> Path:
    """Return the root containing bundled application resources."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Return a writable per-user directory for generated files."""
    override = os.environ.get("SAFESHOT_DATA_DIR") or os.environ.get("IMAGESHIELD_DATA_DIR")
    if override:
        base = Path(override).expanduser()
        path = base
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        path = base / "SafeShot"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        path = base / "SafeShot"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        path = base / "SafeShot"

    path.mkdir(parents=True, exist_ok=True)
    return path
