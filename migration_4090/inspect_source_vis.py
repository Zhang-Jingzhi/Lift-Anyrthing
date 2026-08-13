"""Inspect the downloaded source_vis payload before data generation."""

from collections import Counter

import torch


path = "data/bimanual/source_vis.pt"
payload = torch.load(path, map_location="cpu", weights_only=False)
print(f"SOURCE_VIS_TYPE {type(payload).__name__}")
print(f"SOURCE_VIS_LENGTH {len(payload)}")
if not isinstance(payload, list) or not payload:
    raise RuntimeError("source_vis.pt must contain a non-empty list")

required = {"object_name"}
objects = Counter()
tensor_count = 0
floating_count = 0
for index, row in enumerate(payload):
    if not isinstance(row, dict):
        raise TypeError(f"entry {index} is not a dictionary")
    missing = required - set(row)
    if missing:
        raise KeyError(f"entry {index} is missing {sorted(missing)}")
    objects[row["object_name"]] += 1
    for value in row.values():
        if torch.is_tensor(value):
            tensor_count += 1
            if value.is_floating_point():
                floating_count += 1
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"entry {index} contains a non-finite tensor")

print("SOURCE_VIS_ENTRY_KEYS " + " ".join(sorted(payload[0])))
print(f"SOURCE_VIS_TENSOR_COUNT {tensor_count}")
print(f"SOURCE_VIS_FLOATING_TENSOR_COUNT {floating_count}")
print(f"SOURCE_VIS_OBJECT_COUNT {len(objects)}")
for name, count in sorted(objects.items()):
    print(f"SOURCE_VIS_OBJECT {name} {count}")
print("SOURCE_VIS_VALIDATION_OK")
