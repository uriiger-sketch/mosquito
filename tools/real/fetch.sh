#!/usr/bin/env bash
# Fetch the real recordings used by tools/real (see SOURCES.md) at pinned versions.
#   tools/real/fetch.sh [dest]        (default: ~/mosquito-real-data)
set -euo pipefail
DEST="${1:-$HOME/mosquito-real-data}"
mkdir -p "$DEST"
clone() {   # url dir commit
  if [ ! -d "$DEST/$2/.git" ]; then GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$1" "$DEST/$2"; fi
  git -C "$DEST/$2" checkout --quiet "$3"
}
clone https://github.com/HumBug-Mosquito/MozzBNN mozzbnn 8a808ea6be15c3de5ff7c8ca5a36bc6a9c1f7a74
clone https://github.com/karolpiczak/ESC-50       esc-50  33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6
clone https://github.com/HumBug-Mosquito/MFCC_features MFCC_features 152a1e0bf3402235802cb5265447de29cc622570
clone https://github.com/HumBug-Mosquito/classification-platform classification-platform 4e043c159e116d40cd06ea69903953063b27127d
echo "export MOZZBNN_DIR=$DEST/mozzbnn/Code/lib/VAD-Humbug/input ESC50_DIR=$DEST/esc-50 HUMBUG_CULEX_DIR='$DEST/MFCC_features/mfcc_weak - Culex/inputs'"
