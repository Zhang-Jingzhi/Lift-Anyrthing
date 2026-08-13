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
# The repository may live on NFS with root-squash or mapped ownership.  In
# that case `cp -a` copies the bytes but fails while preserving metadata,
# causing a false installation failure.  These runtime files do not require
# source ownership, modes, or timestamps; copy only the directory contents.
cp -R --no-preserve=mode,ownership,timestamps \
  "$source_root/." "$target_root/"

echo "Installed the tracked left/right Allegro and xlarge-object overlay."
