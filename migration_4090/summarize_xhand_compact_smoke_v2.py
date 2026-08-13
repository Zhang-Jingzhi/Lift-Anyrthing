#!/usr/bin/env python3
"""Summarize the twelve compact XHand smoke combinations."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHYS = ROOT / "migration_4090/results/xhand_compact_6x100_v2_smoke/physics"
OBJECTS = ("sphere", "cube", "cracker", "bleach", "pitcher", "drill")
METHODS = ("baseline", "bidex_v3")


def main():
    rows = []
    for method in METHODS:
        for obj in OBJECTS:
            summaries = sorted((PHYS / obj / method).glob("seed*/smoke_success.json"))
            both_reports = sorted((PHYS / obj / method).glob("seed*/candidate_*/both.json"))
            passed_both = 0
            for path in both_reports:
                try:
                    if json.loads(path.read_text())["physical_pass"]:
                        passed_both += 1
                except Exception:
                    pass
            row = {
                "method": method,
                "object": obj,
                "strict_success": bool(summaries),
                "both_reports": len(both_reports),
                "both_passes": passed_both,
                "summary": str(summaries[0]) if summaries else None,
            }
            if summaries:
                row["metrics"] = json.loads(summaries[0].read_text())
            rows.append(row)
    payload = {
        "schema": "xhand_compact_12way_smoke_progress_v2",
        "strict_success_count": sum(row["strict_success"] for row in rows),
        "target_count": len(rows),
        "rows": rows,
    }
    out = PHYS / "progress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
