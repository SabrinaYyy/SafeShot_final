#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYINSTALLER_CONFIG_DIR="$ROOT/.pyinstaller"

if ! python scripts/validate_bundle.py; then
  echo
  echo "Run ./scripts/download_model.sh, then run this build again." >&2
  exit 1
fi
python -m PyInstaller --noconfirm --clean SafeShot.spec

APP="$ROOT/dist/SafeShot.app"
if [[ ! -d "$APP" ]]; then
  echo "Build failed: $APP was not created." >&2
  exit 1
fi

SAFESHOT_SMOKE_TEST=1 \
IMAGESHIELD_DATA_DIR="$ROOT/.safeshot-smoke" \
"$APP/Contents/MacOS/SafeShot"

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime \
    --entitlements "$ROOT/packaging/macos/entitlements.plist" \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
else
  echo "APPLE_SIGNING_IDENTITY is not set; creating an unsigned development DMG."
fi

mkdir -p "$ROOT/dist/dmg"
rm -f "$ROOT/dist/dmg/SafeShot.dmg"
hdiutil create \
  -volname "SafeShot" \
  -srcfolder "$APP" \
  -ov \
  -format UDZO \
  "$ROOT/dist/dmg/SafeShot.dmg"

echo "Created $ROOT/dist/dmg/SafeShot.dmg"
