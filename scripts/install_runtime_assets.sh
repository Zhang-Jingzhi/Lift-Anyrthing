#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$repo_root/runtime_assets/data"
target_root="$repo_root/data"

if [[ ! -d "$target_root/data_urdf/robot/allegro" ]]; then
  echo "Official data is missing: extract data.zip into $target_root first." >&2
  exit 1
fi

mkdir -p "$target_root"
cp -a "$source_root/." "$target_root/"

echo "Installed the tracked left/right Allegro and xlarge-object overlay."
