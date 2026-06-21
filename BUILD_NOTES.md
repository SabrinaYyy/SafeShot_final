# SafeShot Build Notes

- Offline model bundle validation passed: `319.1 MB` checkpoint.
- Unit tests passed: `8 passed`.
- Documentation was refreshed in `README.md`, `SETUP_SUMMARY.md`, and
  `licenses/THIRD_PARTY_NOTICES.md` to describe both protection modes:
  IP2P / EditShield and SD / BlurGuard.
- Windows PyInstaller portable build completed successfully.
- Built executable smoke launch returned successfully, but because it is a
  windowed app, it briefly left a `SafeShot.exe` process that exited shortly
  afterward.
- Windows installer has been uploaded to GitHub Releases for direct download:
  `SafeShot-Setup-0.1.0.exe`.
- Local Windows installer rebuilds still require Inno Setup and
  `packaging/windows/SafeShot.iss`.
- macOS DMG creation requires macOS `hdiutil`; run `./scripts/build_macos.sh`
  on macOS to create `dist/dmg/SafeShot.dmg`.
- Build dependencies were installed into the local Windows Store Python
  environment. Pip reported conflicts with unrelated existing packages:
  `matplotlib`, `contourpy`, and `torchaudio`.
- Rebuilt Windows portable app after installing CUDA-enabled PyTorch:
  `torch 2.12.1+cu126`, CUDA runtime `12.6`.
- The rebuilt portable bundle includes CUDA libraries such as `torch_cuda.dll`
  and `cudart64_12.dll`.
- Source control should include source, packaging scripts, docs, license files,
  tests, and demo images. It should not include build outputs, downloaded model
  checkpoints, virtual environments, PyInstaller cache, Python bytecode, or
  release artifacts such as `.exe`, `.dmg`, and `.zip` files.
