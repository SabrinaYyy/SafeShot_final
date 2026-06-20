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
- Inno Setup was not installed, so no Windows installer was produced.
- No local DMG was found; macOS DMG creation requires macOS `hdiutil`.
- Build dependencies were installed into the local Windows Store Python
  environment. Pip reported conflicts with unrelated existing packages:
  `matplotlib`, `contourpy`, and `torchaudio`.
- Rebuilt Windows portable app after installing CUDA-enabled PyTorch:
  `torch 2.12.1+cu126`, CUDA runtime `12.6`.
- The rebuilt portable bundle includes CUDA libraries such as `torch_cuda.dll`
  and `cudart64_12.dll`.
