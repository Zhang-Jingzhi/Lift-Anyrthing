#!/usr/bin/env bash
set -euo pipefail

repo_root="${TRO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRO_REPO="$repo_root"
export TRO_PYTHON="${TRO_PYTHON:-$(command -v python3)}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"
export XHAND_BANK_ROOT="${XHAND_BANK_ROOT:-$repo_root/migration_4090/xhand_six_object_smoke_v1_v3/banks}"
export XHAND_SMOKE_ROOT="${XHAND_SMOKE_ROOT:-$repo_root/migration_4090/xhand_six_object_smoke_v4}"

for required in "$TRO_PYTHON" "$ISAAC_PYTHON" "$XHAND_BANK_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required local dependency: $required" >&2
    exit 2
  fi
done

exec "$repo_root/migration_4090/search_six_object_smoke_v4.sh"
