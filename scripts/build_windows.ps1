$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYINSTALLER_CONFIG_DIR = Join-Path $Root ".pyinstaller"

python scripts/validate_bundle.py
if ($LASTEXITCODE -ne 0) {
    throw "Model files are missing. Download them before building."
}
python -m PyInstaller --noconfirm --clean SafeShot.spec

$Executable = Join-Path $Root "dist\SafeShot\SafeShot.exe"
if (-not (Test-Path $Executable)) {
    throw "Build failed: $Executable was not created."
}

Write-Host "Created $Executable"
Write-Host "Run Inno Setup against packaging\windows\SafeShot.iss to create the installer."
