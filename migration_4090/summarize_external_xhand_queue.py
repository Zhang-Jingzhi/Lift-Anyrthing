#!/usr/bin/env python3
"""Summarize the resumable external-approx XHand Isaac screening queue."""
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("migration_4090/results/xhand_external_strict_seed_search_v1/candidates")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main():
    rows = defaultdict(lambda: {
        "both_done": 0,
        "both_pass": 0,
        "strict_done": 0,
        "strict_pass": 0,
        "repeat2_pass": 0,
        "repeat3_pass": 0,
        "verified_3_of_3": 0,
    })
    for candidate in sorted(ROOT.glob("*/*/candidate_*")):
        if not candidate.is_dir():
            continue
        key = f"{candidate.parts[-3]}/{candidate.parts[-2]}"
        both = load(candidate / "both.json")
        strict = load(candidate / "strict.json")
        repeat2 = load(candidate / "repeat2.json")
        repeat3 = load(candidate / "repeat3.json")
        if both is not None:
            rows[key]["both_done"] += 1
            rows[key]["both_pass"] += int(bool(both.get("physical_pass")))
        if strict is not None:
            rows[key]["strict_done"] += 1
            rows[key]["strict_pass"] += int(bool(strict.get("physical_pass")))
        if repeat2 is not None:
            rows[key]["repeat2_pass"] += int(bool(repeat2.get("physical_pass")))
        if repeat3 is not None:
            rows[key]["repeat3_pass"] += int(bool(repeat3.get("physical_pass")))
        if all(x is not None and x.get("physical_pass") for x in (strict, repeat2, repeat3)):
            rows[key]["verified_3_of_3"] += 1
    totals = {field: sum(row[field] for row in rows.values()) for field in next(iter(rows.values()), {})}
    print(json.dumps({"groups": dict(rows), "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
