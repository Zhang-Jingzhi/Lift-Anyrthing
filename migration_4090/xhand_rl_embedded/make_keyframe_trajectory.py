#!/usr/bin/env python3
"""Create a compact, non-destructive visualization trajectory.

For an exact RL prefix trajectory this keeps one recorded state at the end of
each renderable phase.  It is useful on heavily loaded machines where a full
mesh render is expensive; no training or source data is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    src = json.loads(args.input.read_text())
    states = src.get("states", [])
    wanted = ("pregrasp", "closure", "lift", "hold")
    selected = []
    for stage in wanted:
        rows = [row for row in states if row.get("stage") == stage]
        if not rows:
            raise RuntimeError(f"missing stage {stage}")
        selected.append(copy.deepcopy(rows[-1]))
    # Keep recorded step numbers so the renderer still orders the phases.
    out = dict(src)
    out["states"] = selected
    out["schema"] = "xhand_rl_exact_prefix_keyframe_visualization_v1"
    out["keyframes_only"] = True
    out["note"] = "One exact recorded state per pregrasp/closure/lift/hold phase; visualization-only."
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
