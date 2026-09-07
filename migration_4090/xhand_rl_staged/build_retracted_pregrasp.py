#!/usr/bin/env python3
"""Solve a retracted pregrasp so the hands have somewhere to approach from.

Every BODex candidate places the hands within 0.89-7.39 mm of the ball at the
pregrasp pose, and the environment resets the robot exactly there, so the
"approach" phase has nothing to approach: `_base_target` clamps its interpolation
factor to zero and simply holds the pregrasp pose.  Measured 2026-09-06/07, the
object is knocked to 0.9 m/s on the fourth physics step and rises 21-28 mm before
the policy's residual is even active, and that survives a zero-overlap bank, a
2.5x longer approach phase and zero reset noise alike.

UltraDexGrasp approaches from 0.1 m; this solves the equivalent pose for our
candidates by pushing each palm back along its own outward normal and re-solving
the arms only, exactly as build_micro_lift_targets.py does for the lift.  The
BODex-optimised finger joints are preserved untouched, so the retracted pose is
the same open hand, just further away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jaxlie
import numpy as np
import torch

MIGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(MIGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MIGRATION_ROOT))

from generate_xhand_compact_candidate_bank_v2 import (  # noqa: E402
    TARGET_LINKS,
    achieved_metrics,
    calibrate_ik_to_actual_rotations,
    make_reusable_pose_solver,
    solve_targets,
)
from migration_4090.xhand_bodex_bimanual.contracts import (  # noqa: E402
    load_bodex_bank,
    sha256_file,
)
from migration_4090.xhand_rl_staged.build_standoff_pregrasp_bank import (  # noqa: E402
    load_fixed_kinematic_robot,
)

# Both XHand palms face along local +X (CRITERIA_v2), so retracting means
# stepping along -X of each palm frame.
PALM_LOCAL_NORMAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument(
        "--retract-m",
        type=float,
        nargs="+",
        default=(0.10,),
        help="distance to pull each palm back along its own outward normal",
    )
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--wrist-rest-weight", type=float, default=0.03)
    parser.add_argument("--max-position-error-m", type=float, default=0.006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distances = tuple(sorted(set(float(value) for value in args.retract_m)))
    if not distances or any(value <= 0.0 for value in distances):
        raise ValueError("retract distances must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    bank = load_bodex_bank(
        args.bodex_bank,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    robot = load_fixed_kinematic_robot()
    ik_names = list(robot.joints.actuated_names)
    link_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    os.environ["XHAND_FORMAL_WRIST_REST_WEIGHT"] = str(args.wrist_rest_weight)

    poses_by_distance: dict[str, list[list[float]]] = {
        f"{value:.6f}": [] for value in distances
    }
    diagnostics = []
    for candidate_index, sample in enumerate(bank["samples"]):
        full_names = list(sample["joint_names"])
        values = dict(zip(full_names, sample["pregrasp_full_body_q"]))
        missing = [name for name in ik_names if name not in values]
        if missing:
            raise RuntimeError(f"candidate {candidate_index} lacks IK joints: {missing}")
        pregrasp_q = np.asarray([values[name] for name in ik_names], dtype=np.float32)
        _, _, frame_rotations = calibrate_ik_to_actual_rotations(
            robot, pregrasp_q, link_indices
        )
        solver = make_reusable_pose_solver(
            robot,
            link_indices,
            ik_to_actual_rotations=frame_rotations,
            palm_direction_weight=args.orientation_weight,
        )
        pregrasp_fk = robot.forward_kinematics(jnp.asarray(pregrasp_q))
        record = {
            "candidate_index": candidate_index,
            "candidate_id": sample["candidate_id"],
            "distances": {},
        }
        for distance in distances:
            palm_targets = []
            for link in link_indices:
                pose = jaxlie.SE3(pregrasp_fk[link])
                rotation = np.asarray(pose.rotation().as_matrix(), dtype=np.float64)
                outward = rotation @ PALM_LOCAL_NORMAL
                translation = np.asarray(pose.translation(), dtype=np.float64).copy()
                translation -= distance * outward
                palm_targets.append(
                    jaxlie.SE3.from_rotation_and_translation(
                        pose.rotation(), jnp.asarray(translation)
                    )
                )
            solved = np.asarray(solve_targets(solver, pregrasp_q, palm_targets)).copy()
            # Arms only: the open BODex finger shape must survive verbatim, or
            # the retracted pose is a different hand rather than the same hand
            # further back.
            for index, name in enumerate(ik_names):
                if "_hand_" in name:
                    solved[index] = pregrasp_q[index]
            achieved, position_error, orientation_error = achieved_metrics(
                robot, solved, palm_targets, link_indices
            )
            valid = (
                np.isfinite(solved).all()
                and np.all(solved >= lower - 1.0e-4)
                and np.all(solved <= upper + 1.0e-4)
                and max(position_error) <= args.max_position_error_m
            )
            record["distances"][f"{distance:.6f}"] = {
                "success": bool(valid),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "target_palm_positions_world_m": [
                    np.asarray(target.translation()).tolist() for target in palm_targets
                ],
                "achieved_palm_positions_world_m": achieved,
            }
            full_row = list(sample["pregrasp_full_body_q"])
            for index, name in enumerate(ik_names):
                full_row[full_names.index(name)] = float(solved[index])
            poses_by_distance[f"{distance:.6f}"].append(full_row)
            print(
                f"candidate {candidate_index} {sample['candidate_id'][:38]:40s} "
                f"retract {distance * 1000:5.1f}mm  success={valid}  "
                f"pos_err {max(position_error) * 1000:6.3f}mm"
            )
        diagnostics.append(record)

    payload = {
        "schema": "xhand_bodex_retracted_pregrasp_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "source_bodex_bank": str(args.bodex_bank.resolve()),
        "source_bodex_bank_sha256": sha256_file(args.bodex_bank),
        "candidate_ids": [sample["candidate_id"] for sample in bank["samples"]],
        "joint_names": list(bank["samples"][0]["joint_names"]),
        "retract_distances_m": list(distances),
        "retracted_pregrasp_full_body_q": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in poses_by_distance.items()
        },
        "diagnostics": diagnostics,
        "non_promotional": True,
    }
    torch.save(payload, args.output)
    summary = {key: value for key, value in payload.items() if key not in ("retracted_pregrasp_full_body_q",)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(diagnostics)}, indent=2))


if __name__ == "__main__":
    main()
