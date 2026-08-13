#!/usr/bin/env python3
"""Print every stored metric from the three Isaac validation modes."""

from pathlib import Path
import sys

import torch


base = Path(sys.argv[1])
for mode in ("both", "left", "right"):
    path = base / f"isaac_{mode}_chunks/00000_00002/isaac_result.pt"
    result = torch.load(path, map_location="cpu")
    print(f"\nMODE {mode} TYPE {type(result).__name__} PATH {path}")
    if not isinstance(result, dict):
        print(repr(result))
        continue
    print("KEYS", list(result))
    for key, value in result.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape), value.tolist())
        else:
            print(key, repr(value))
