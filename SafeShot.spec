# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

PROJECT_ROOT = Path(SPECPATH)

datas = [
    (
        str(PROJECT_ROOT / "models" / "instruct-pix2pix"),
        "models/instruct-pix2pix",
    ),
    (str(PROJECT_ROOT / "licenses"), "licenses"),
    (str(PROJECT_ROOT / "imageshield" / "demo_image"), "imageshield/demo_image"),
]
binaries = []
hiddenimports = [
    "diffusers.models.autoencoders.autoencoder_kl",
    "diffusers.schedulers.scheduling_ddpm",
    "gradio.routes",
]

datas += collect_data_files("gradio", include_py_files=True)
datas += collect_data_files("gradio_client")
datas += collect_data_files("groovy")
datas += collect_data_files("safehttpx")
for distribution in (
    "accelerate",
    "diffusers",
    "filelock",
    "gradio",
    "gradio_client",
    "groovy",
    "huggingface_hub",
    "numpy",
    "packaging",
    "PyYAML",
    "regex",
    "requests",
    "safetensors",
    "safehttpx",
    "scipy",
    "scikit_image",
    "tokenizers",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
):
    datas += copy_metadata(distribution)

analysis = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "notebook",
        "pytest",
        "tensorboard",
        "transformers",
        "bitsandbytes",
        "datasets",
        "tensorflow",
        "torchaudio",
        "PyQt5",
        "PySide6",
        "PyQt6",
        "PySide2",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SafeShot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SafeShot",
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name="SafeShot.app",
        icon=None,
        bundle_identifier="org.ai4good.safeshot",
        info_plist={
            "CFBundleDisplayName": "SafeShot",
            "CFBundleName": "SafeShot",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
