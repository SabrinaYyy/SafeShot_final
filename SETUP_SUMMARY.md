# SafeShot Setup Summary

SafeShot final is a self-contained release folder for building and testing the
SafeShot desktop app. The release-facing app name is `SafeShot`, and the app
contains both protection paths:

- **IP2P / EditShield**: faster InstructPix2Pix-oriented VAE latent protection.
- **SD / BlurGuard**: stronger Stable Diffusion-oriented protection with
  adaptive blur warmup plus VAE latent optimization.

## Ready Windows Release Build

Windows users should download the installer directly from the GitHub Releases
page:

```text
https://github.com/SabrinaYyy/SafeShot_final/releases
```

The release installer asset is:

```text
SafeShot-Setup-0.1.0.exe
```

If a portable zip is included in the release, it is:

```text
dist/windows/SafeShot-Windows-Portable-0.1.0.zip
```

Extract the zip, open the extracted `SafeShot` folder, and run `SafeShot.exe`.
Keep the `_internal` folder beside the executable.

The current portable build was rebuilt with CUDA-enabled PyTorch. On compatible
NVIDIA systems, SafeShot should detect CUDA automatically; otherwise it falls
back to CPU.

## Validate the Source Bundle

Download the pinned offline model files first. From Git Bash on Windows, or
from Terminal on macOS:

```bash
./scripts/download_model.sh
```

```powershell
python scripts\validate_bundle.py
python -m pytest
```

The model bundle is expected at:

```text
models/instruct-pix2pix/
```

It contains the VAE and scheduler files from `timbrooks/instruct-pix2pix`,
pinned to revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`.

## Build Windows Portable and Installer

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
```

For a CUDA build, install CUDA-enabled PyTorch before running PyInstaller. For
example:

```powershell
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Download the pinned offline model files before packaging if you have not already
done so. From Git Bash:

```bash
./scripts/download_model.sh
```

Then validate and test from PowerShell:

```powershell
python scripts\validate_bundle.py
python -m pytest
```

Build the PyInstaller app:

```powershell
python -m PyInstaller --noconfirm --clean SafeShot.spec
```

PyInstaller creates:

```text
dist/SafeShot/SafeShot.exe
```

The release portable copy belongs at:

```text
dist/windows/portable/SafeShot/
```

The uploaded GitHub Release installer should be available as:

```text
SafeShot-Setup-0.1.0.exe
```

To rebuild that installer locally, compile the Inno Setup script:

```powershell
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" "packaging\windows\SafeShot.iss"
```

If `ISCC.exe` is on your `PATH`, this also works:

```powershell
ISCC.exe "packaging\windows\SafeShot.iss"
```

Expected installer name:

```text
SafeShot-Setup-0.1.0.exe
```

Expected installer folder:

```text
dist/installer/
```

## Build macOS DMG

macOS DMG creation requires macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
chmod +x scripts/download_model.sh scripts/build_macos.sh
./scripts/download_model.sh
./scripts/build_macos.sh
```

Expected DMG:

```text
dist/dmg/SafeShot.dmg
```

Upload `dist/dmg/SafeShot.dmg` to the GitHub Release. Public macOS releases
should be signed and notarized.

## Documentation

See `README.md` for user instructions, protection-mode descriptions,
limitations, citations, and third-party acknowledgements.
