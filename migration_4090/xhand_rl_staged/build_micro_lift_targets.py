#!/usr/bin/env python3
"""Build paired-palm micro-lift IK targets for a genuine BODex contact bank."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument(
        "--lift-height-m", type=float, nargs="+", default=(0.005, 0.010)
    )
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--wrist-rest-weight", type=float, default=0.03)
    parser.add_argument("--max-position-error-m", type=float, default=0.006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    heights = tuple(sorted(set(float(value) for value in args.lift_height_m)))
    if not heights or any(value <= 0.0 for value in heights):
        raise ValueError("lift heights must be positive")
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
    targets_by_height: dict[str, list[list[float]]] = {
        f"{height:.6f}": [] for height in heights
    }
    diagnostics = []
    for candidate_index, sample in enumerate(bank["samples"]):
        full_names = list(sample["joint_names"])
        controller_grasp = sample.get(
            "controller_grasp_full_body_q", sample["full_body_q"]
        )
        values = dict(zip(full_names, controller_grasp))
        missing = [name for name in ik_names if name not in values]
        if missing:
            raise RuntimeError(
                f"candidate {candidate_index} lacks IK joints: {missing}"
            )
        grasp_q = np.asarray([values[name] for name in ik_names], dtype=np.float32)
        _, _, frame_rotations = calibrate_ik_to_actual_rotations(
            robot, grasp_q, link_indices
        )
        solver = make_reusable_pose_solver(
            robot,
            link_indices,
            ik_to_actual_rotations=frame_rotations,
            palm_direction_weight=args.orientation_weight,
        )
        grasp_fk = robot.forward_kinematics(jnp.asarray(grasp_q))
        candidate_diagnostics = {
            "candidate_index": candidate_index,
            "candidate_id": sample["candidate_id"],
            "controller_grasp_used": "controller_grasp_full_body_q" in sample,
            "heights": {},
        }
        for height in heights:
            palm_targets = []
            for link in link_indices:
                pose = jaxlie.SE3(grasp_fk[link])
                translation = np.asarray(pose.translation(), dtype=np.float64).copy()
                translation[2] += height
                palm_targets.append(
                    jaxlie.SE3.from_rotation_and_translation(
                        pose.rotation(), jnp.asarray(translation)
                    )
                )
            # JAX may expose the solved buffer as a read-only NumPy view.  The
            # hand joints below must be restored to the exact BODex values, so
            # materialize an explicitly writable copy first.
            solved = np.asarray(
                solve_targets(solver, grasp_q, palm_targets)
            ).copy()
            # The paired-palm target is an arm trajectory.  Preserve every
            # BODex-optimized hand joint exactly.
            for index, name in enumerate(ik_names):
                if "_hand_" in name:
                    solved[index] = grasp_q[index]
            achieved, position_error, orientation_error = achieved_metrics(
                robot, solved, palm_targets, link_indices
            )
            valid = (
                np.isfinite(solved).all()
                and np.all(solved >= lower - 1.0e-4)
                and np.all(solved <= upper + 1.0e-4)
                and max(position_error) <= args.max_position_error_m
            )
            candidate_diagnostics["heights"][f"{height:.6f}"] = {
                "success": bool(valid),
                "target_palm_positions_world_m": [
                    np.asarray(target.translation()).tolist()
                    for target in palm_targets
                ],
                "achieved_palm_positions_world_m": achieved,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
            }
            if not valid:
                raise RuntimeError(
                    f"candidate {candidate_index} lift IK failed at {height} m: "
                    f"position_error={position_error}"
                )
            lift_values = values.copy()
            lift_values.update(
                {name: float(value) for name, value in zip(ik_names, solved)}
            )
            targets_by_height[f"{height:.6f}"].append(
                [float(lift_values[name]) for name in full_names]
            )
        diagnostics.append(candidate_diagnostics)
    payload = {
        "schema": "xhand_bodex_nonpromotional_micro_lift_targets_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "source_bodex_bank": str(args.bodex_bank.resolve()),
        "source_bodex_bank_sha256": sha256_file(args.bodex_bank),
        "candidate_ids": [sample["candidate_id"] for sample in bank["samples"]],
        "joint_names": list(bank["samples"][0]["joint_names"]),
        "lift_heights_m": list(heights),
        "targets_by_height": targets_by_height,
        "diagnostics": diagnostics,
        "non_promotional": True,
        "grasp_source": (
            "side_clipped_controller_target_from_genuine_joint_bimanual_bodex"
            if any(
                "controller_grasp_full_body_q" in sample
                for sample in bank["samples"]
            )
            else "genuine_joint_bimanual_bodex"
        ),
        "lift_target_method": "paired_palm_orientation_preserving_ik",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "targets_by_height"}, indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_count": len(bank["samples"]),
                "lift_heights_m": list(heights),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
