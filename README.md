# SafeShot

SafeShot is an offline desktop tool for adding adversarial protection to images
before they are shared. It is designed to raise the cost of unwanted AI image
editing by applying small pixel-level perturbations that target latent diffusion
editing systems.

SafeShot runs locally on the user's computer. It does not require an account,
does not enable Gradio public sharing, and does not upload images to a remote
server.

SafeShot includes two protection methods:

- **IP2P / EditShield**: a faster VAE latent-divergence protection mode aimed at
  InstructPix2Pix-style instruction-based editing. This protection path is
  related to EditShield / Chen et al., ECCV 2024. It is the recommended first
  mode to try because it is usually less visible and runs faster.
- **SD / BlurGuard**: a stronger Stable Diffusion protection mode inspired by
  PhotoGuard / Salman et al., 2023 and BlurGuard / Kim et al., 2025. It
  combines a low-frequency, per-region blur warmup with latent-space
  optimization. This mode can be slower and may produce more visible changes,
  but it is intended for stronger protection tests.

SafeShot does not provide a universal guarantee against every AI editor,
face-swap model, image restoration system, or manual attack. Treat protected
images as a defensive layer, not as proof that an image can never be edited.

## Ready-to-Use Windows App

The ready portable Windows build is:

```text
dist/windows/SafeShot-Windows-Portable-0.1.0.zip
```

To test it:

1. Extract the zip file.
2. Open the extracted `SafeShot` folder.
3. Run `SafeShot.exe`.
4. Keep the full folder together. Do not move `SafeShot.exe` away from the
   `_internal` folder.
5. When the browser interface opens, upload an image, choose a protection mode,
   adjust the settings, and click the protect button.
6. Download the protected PNG after processing finishes.

The current Windows portable build includes CUDA-enabled PyTorch. On a machine
with a compatible NVIDIA GPU and driver, SafeShot should use CUDA automatically.
If CUDA is unavailable, it falls back to CPU.

## Protection Modes

### IP2P / EditShield

Use this mode when you want the default SafeShot protection path.

- Targets InstructPix2Pix-style image editing.
- Uses VAE latent divergence to push the protected image away from the original
  image in the editor's latent representation.
- Default perturbation budget: `8/255`.
- Usually faster and less visible than SD / BlurGuard.
- Recommended for quick tests, CPU fallback, and lower-visibility protection.

Suggested first test:

```text
Mode: IP2P / EditShield
Resolution: 128 or 256
Steps: 20
Epsilon: 8/255
```

### SD / BlurGuard

Use this mode when you want the stronger Stable Diffusion-oriented path.

- Targets Stable Diffusion latent/VAE behavior.
- Starts with adaptive per-region blur guidance, then runs VAE latent attack
  optimization.
- Default perturbation budget: `16/255`.
- More computationally expensive than IP2P / EditShield.
- Best tested on a CUDA GPU, especially for 256 or 512 pixel resolution.

Suggested GPU test:

```text
Mode: SD / BlurGuard
Resolution: 256
Steps: 40 to 80
Epsilon: 16/255
```

## Recommended Settings

For the first functionality test, use a small image or select a low resolution.

```text
Fast smoke test:
  Mode: IP2P / EditShield
  Resolution: 128
  Steps: 20

Good GPU test:
  Mode: IP2P / EditShield
  Resolution: 256 or 512
  Steps: 50 to 100

Stronger SD test:
  Mode: SD / BlurGuard
  Resolution: 256
  Steps: 40 to 80
```

CPU mode works, but protection can take 10 to 60+ minutes per image depending
on image size, selected mode, and step count. SD / BlurGuard is not recommended
for CPU-only testing unless you use very small settings.

## GPU Check

To confirm that the source Python environment can see your NVIDIA GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result on a CUDA-ready machine:

```text
True
NVIDIA ...
```

You can also run:

```powershell
nvidia-smi
```

The packaged Windows portable app must be built from an environment that already
has CUDA-enabled PyTorch installed. If it is built with CPU-only PyTorch, the
final `.exe` will also run as CPU-only even when the user's computer has an
NVIDIA GPU.

## Build From Source

Use Python 3.10 on Windows or macOS.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
```

For a CUDA Windows build, install a CUDA-enabled PyTorch wheel before running
PyInstaller. For example, for CUDA 12.6:

```powershell
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then validate and test:

```powershell
python scripts\validate_bundle.py
python -m pytest
```

Build the Windows portable app:

```powershell
python -m PyInstaller --noconfirm --clean SafeShot.spec
```

The PyInstaller output is:

```text
dist/SafeShot/SafeShot.exe
```

The release portable folder is:

```text
dist/windows/portable/SafeShot/
```

The release zip is:

```text
dist/windows/SafeShot-Windows-Portable-0.1.0.zip
```

If Inno Setup is installed, compile:

```text
packaging/windows/SafeShot.iss
```

The installer output name is:

```text
SafeShot-Setup-0.1.0.exe
```

## macOS Build

macOS DMG creation requires macOS tools such as `hdiutil`, so it must be built
on macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
chmod +x scripts/download_model.sh scripts/build_macos.sh
./scripts/download_model.sh
./scripts/build_macos.sh
```

The finished DMG is expected at:

```text
dist/dmg/SafeShot.dmg
```

Unsigned development builds may require right-clicking the app and choosing
**Open** on first launch. Public macOS releases should be signed, notarized, and
stapled with an Apple Developer account.

## Model Bundle

SafeShot uses a local offline copy of the VAE and scheduler files from:

```text
Repository: timbrooks/instruct-pix2pix
Revision:   31519b5cb02a7fd89b906d88731cd4d6a7bbf88d
```

Expected files:

```text
models/instruct-pix2pix/
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
  scheduler/scheduler_config.json
```

The app uses local model files at runtime. Missing model files are not
downloaded automatically by `python app.py` or by the packaged app.

## Privacy and Storage

- Processing happens on the user's computer.
- The local server binds to `127.0.0.1`.
- Gradio public sharing is disabled.
- Generated files are stored in the user's application-data directory.
- Old generated files are removed during later startups.

Closing the browser tab does not always quit SafeShot. Quit the desktop app
from the operating system when finished.

## Limitations

- Protection is model-specific and does not cover every editing or generation
  system.
- Stronger settings may create more visible artifacts.
- Resizing, recompression, screenshots, denoising, or manual retouching may
  weaken or remove perturbations.
- Output is saved as a PNG at the selected processing resolution.
- One protection job runs at a time to reduce accelerator memory pressure.
- The app is large because it bundles Python, PyTorch, Gradio, dependencies,
  and local model files.

## Release Checklist

1. Run `python scripts/validate_bundle.py`.
2. Run `python -m pytest`.
3. Build on a clean machine for each target operating system.
4. Test startup, upload, protection, cancellation, and download.
5. Test once with networking disabled.
6. Confirm whether CUDA is detected on the Windows portable build.
7. Publish SHA-256 checksums for every downloadable artifact.
8. Include third-party notices, model terms, and research acknowledgements.
9. Sign Windows and macOS public releases when distributing broadly.

## Research Acknowledgements and Citations

SafeShot is a defensive research prototype built on ideas from adversarial
image protection and latent diffusion editing research. The names
`EditShield` and `BlurGuard` are SafeShot release labels for the two protection
paths in this app.

Core references:

- Ruoxi Chen, Haibo Jin, Yixin Liu, Jinyin Chen, Haohan Wang, and Lichao Sun.
  "EditShield: Protecting Unauthorized Image Editing by Instruction-guided
  Diffusion Models." ECCV 2024.
  <https://arxiv.org/abs/2311.12066>
- Jinsu Kim, Yunhun Nam, Minseon Kim, Sangpil Kim, and Jongheon Jeong.
  "BlurGuard: A Simple Approach for Robustifying Image Protection Against
  AI-Powered Editing." 2025.
  <https://arxiv.org/abs/2511.00143>
- Tim Brooks, Aleksander Holynski, and Alexei A. Efros. "InstructPix2Pix:
  Learning to Follow Image Editing Instructions." 2022.
  <https://arxiv.org/abs/2211.09800>
- Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander
  Madry. "Raising the Cost of Malicious AI-Powered Image Editing"
  (PhotoGuard). 2023.
  <https://arxiv.org/abs/2302.06588>
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn
  Ommer. "High-Resolution Image Synthesis with Latent Diffusion Models." 2021.
  <https://arxiv.org/abs/2112.10752>

BibTeX:

```bibtex
@inproceedings{chen2024editshield,
  title={EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models},
  author={Chen, Ruoxi and Jin, Haibo and Liu, Yixin and Chen, Jinyin and Wang, Haohan and Sun, Lichao},
  booktitle={European Conference on Computer Vision},
  year={2024}
}

@article{kim2025blurguard,
  title={BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing},
  author={Kim, Jinsu and Nam, Yunhun and Kim, Minseon and Kim, Sangpil and Jeong, Jongheon},
  journal={arXiv preprint arXiv:2511.00143},
  year={2025}
}

@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A.},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}

@article{salman2023raising,
  title={Raising the Cost of Malicious AI-Powered Image Editing},
  author={Salman, Hadi and Khaddaj, Alaa and Leclerc, Guillaume and Ilyas, Andrew and Madry, Aleksander},
  journal={arXiv preprint arXiv:2302.06588},
  year={2023}
}

@article{rombach2021highresolution,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bjorn},
  journal={arXiv preprint arXiv:2112.10752},
  year={2021}
}
```

For release and redistribution notes, also see `licenses/THIRD_PARTY_NOTICES.md`.
SafeShot source code is released under the Apache License 2.0 unless otherwise
noted. Bundled dependencies, model files, and research-derived methods remain
under their own licenses, terms, and citation requirements.

SafeShot also depends on open-source tooling and libraries including PyTorch,
Diffusers, Hugging Face Hub, Gradio, scikit-image, SciPy, Pillow, NumPy, and
PyInstaller. See `licenses/THIRD_PARTY_NOTICES.md` for packaging and license
notes.
