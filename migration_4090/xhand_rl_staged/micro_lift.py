"""Pure reporting helpers for non-promotional Stage-2 micro-lift probes."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any


# Defaults for the stability contract.  These were the only values the scorer
# could ever use: classify_micro_lift_report read them directly and its callers
# had no way to override, so a profile declaring stable_hold_steps=125 was still
# scored at 32.  They are now defaults for keyword arguments; bridge_evaluate
# passes the profile's own contract and records it with the results.
MICRO_LIFT_LINEAR_SPEED_MAX_M_S = 0.05
MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S = 0.50
MICRO_LIFT_STABLE_HOLD_STEPS = 32

# The paired-palm nominal target moves only 5--10 mm.  A further 20 mm of
# upward travel is a deliberately generous overshoot allowance; larger motion
# is a throw/contact explosion rather than a controlled micro-lift.  Likewise,
# falling more than 20 mm below the reset height is not a retained grasp.
MICRO_LIFT_MAX_OVERSHOOT_M = 0.020
MICRO_LIFT_MIN_HEIGHT_M = -0.020


def controlled_height_upper_bound_m(target_height_m: float) -> float:
    if target_height_m <= 0.0:
        raise ValueError("micro-lift target must be positive")
    return target_height_m + MICRO_LIFT_MAX_OVERSHOOT_M


def classify_micro_lift_report(
    report: dict[str, Any],
    *,
    target_height_m: float,
    stable_hold_steps: int = MICRO_LIFT_STABLE_HOLD_STEPS,
    linear_speed_max_m_s: float = MICRO_LIFT_LINEAR_SPEED_MAX_M_S,
    angular_speed_max_rad_s: float = MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S,
) -> dict[str, Any]:
    if target_height_m <= 0.0:
        raise ValueError("micro-lift target must be positive")
    if stable_hold_steps <= 0:
        raise ValueError("stable hold steps must be positive")
    gates = report.get("stage_gates", {})
    final_height = float(report["final_lift_height_m"])
    maximum_height = float(report["maximum_lift_height_m"])
    finite_height = math.isfinite(maximum_height) and math.isfinite(final_height)
    reached = finite_height and maximum_height >= target_height_m
    retained = finite_height and final_height >= 0.8 * target_height_m
    physically_bounded = bool(report.get("micro_lift_physically_bounded", False))
    stable_terminal = (
        float(report.get("object_linear_speed_m_s", math.inf))
        <= linear_speed_max_m_s
        and float(report.get("object_angular_speed_rad_s", math.inf))
        <= angular_speed_max_rad_s
    )
    stable_hold = (
        int(report.get("maximum_micro_lift_stable_steps", 0)) >= stable_hold_steps
    )
    held_window_available = "maximum_micro_lift_held_steps" in report
    held_hold = (
        int(report["maximum_micro_lift_held_steps"]) >= stable_hold_steps
        if held_window_available
        else stable_hold
    )
    stage2_contact_pass = all(
        gates.get(name) is True
        for name in (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
        )
    )
    controlled = (
        reached
        and retained
        and physically_bounded
        and not bool(report.get("terminated_early", False))
    )
    stage4_ready = controlled and stable_terminal and stable_hold
    sustained = controlled and stable_terminal and held_hold and stage2_contact_pass
    return {
        "target_reached": reached,
        "target_retained_at_terminal": retained,
        "physically_bounded_micro_lift": physically_bounded,
        "stable_terminal_state": stable_terminal,
        "stable_micro_lift_hold": stable_hold,
        "held_micro_lift_hold": held_hold,
        "held_window_available": held_window_available,
        "controlled_micro_lift": controlled,
        "stage4_ready_micro_lift": stage4_ready,
        "stage2_contact_pass": stage2_contact_pass,
        "sustained_micro_lift": sustained,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _failure_funnel(
    rows: list[dict[str, Any]],
    stages: tuple[tuple[str, Any], ...],
) -> list[dict[str, Any]]:
    """Build a cumulative pass-rate funnel from already classified reports."""

    remaining = rows
    previous_count = len(rows)
    funnel = []
    for name, predicate in stages:
        remaining = [row for row in remaining if predicate(row)]
        count = len(remaining)
        funnel.append(
            {
                "stage": name,
                "episodes": count,
                "rate_of_all": _rate(count, len(rows)),
                "retention_from_previous": _rate(count, previous_count),
            }
        )
        previous_count = count
    return funnel


def summarize_micro_lift_reports(
    reports: list[dict[str, Any]],
    *,
    target_height_m: float,
    stable_hold_steps: int = MICRO_LIFT_STABLE_HOLD_STEPS,
    linear_speed_max_m_s: float = MICRO_LIFT_LINEAR_SPEED_MAX_M_S,
    angular_speed_max_rad_s: float = MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("micro-lift summary requires at least one report")
    criteria = {
        "target_height_m": target_height_m,
        "stable_hold_steps": stable_hold_steps,
        "linear_speed_max_m_s": linear_speed_max_m_s,
        "angular_speed_max_rad_s": angular_speed_max_rad_s,
    }
    classified = [
        {
            **report,
            **classify_micro_lift_report(
                report,
                target_height_m=target_height_m,
                stable_hold_steps=stable_hold_steps,
                linear_speed_max_m_s=linear_speed_max_m_s,
                angular_speed_max_rad_s=angular_speed_max_rad_s,
            ),
        }
        for report in reports
    ]
    metrics = (
        "target_reached",
        "target_retained_at_terminal",
        "physically_bounded_micro_lift",
        "stable_terminal_state",
        "stable_micro_lift_hold",
        "held_micro_lift_hold",
        "held_window_available",
        "controlled_micro_lift",
        "stage4_ready_micro_lift",
        "stage2_contact_pass",
        "sustained_micro_lift",
    )
    overall = {
        name: sum(bool(row[name]) for row in classified) / len(classified)
        for name in metrics
    }
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in classified:
        by_candidate[int(row["candidate_index"])].append(row)
    candidate_summary = {}
    for candidate_index, rows in sorted(by_candidate.items()):
        candidate_summary[str(candidate_index)] = {
            "candidate_id": rows[0]["candidate_id"],
            "episodes": len(rows),
            **{
                name: sum(bool(row[name]) for row in rows) / len(rows)
                for name in metrics
            },
        }
    conditional = {}
    gates = (
        "finite_state",
        "bilateral_contact",
        "contact_continuity",
        "distributed_contacts",
        "all_stage2_contacts",
    )
    for gate in gates:
        if gate == "all_stage2_contacts":
            true_rows = [row for row in classified if row["stage2_contact_pass"]]
        else:
            true_rows = [
                row for row in classified if row["stage_gates"].get(gate) is True
            ]
        true_ids = {id(row) for row in true_rows}
        false_rows = [row for row in classified if id(row) not in true_ids]
        conditional[gate] = {
            "gate_true_episodes": len(true_rows),
            "gate_false_episodes": len(false_rows),
            "sustained_lift_given_gate_true": _rate(
                sum(bool(row["sustained_micro_lift"]) for row in true_rows),
                len(true_rows),
            ),
            "target_reached_given_gate_true": _rate(
                sum(bool(row["target_reached"]) for row in true_rows),
                len(true_rows),
            ),
            "target_reached_given_gate_false": _rate(
                sum(bool(row["target_reached"]) for row in false_rows),
                len(false_rows),
            ),
            "target_retained_given_gate_true": _rate(
                sum(bool(row["target_retained_at_terminal"]) for row in true_rows),
                len(true_rows),
            ),
            "target_retained_given_gate_false": _rate(
                sum(bool(row["target_retained_at_terminal"]) for row in false_rows),
                len(false_rows),
            ),
            "controlled_lift_given_gate_true": _rate(
                sum(bool(row["controlled_micro_lift"]) for row in true_rows),
                len(true_rows),
            ),
            "controlled_lift_given_gate_false": _rate(
                sum(bool(row["controlled_micro_lift"]) for row in false_rows),
                len(false_rows),
            ),
            "stage4_ready_lift_given_gate_true": _rate(
                sum(bool(row["stage4_ready_micro_lift"]) for row in true_rows),
                len(true_rows),
            ),
            "stage4_ready_lift_given_gate_false": _rate(
                sum(bool(row["stage4_ready_micro_lift"]) for row in false_rows),
                len(false_rows),
            ),
        }
    contact_quality_funnel = _failure_funnel(
        classified,
        (
            ("finite_state", lambda row: row["stage_gates"].get("finite_state") is True),
            (
                "bilateral_contact",
                lambda row: row["stage_gates"].get("bilateral_contact") is True,
            ),
            (
                "contact_continuity",
                lambda row: row["stage_gates"].get("contact_continuity") is True,
            ),
            (
                "distributed_contacts",
                lambda row: row["stage_gates"].get("distributed_contacts") is True,
            ),
        ),
    )
    lift_outcome_funnel = _failure_funnel(
        classified,
        (
            ("target_reached", lambda row: bool(row["target_reached"])),
            (
                "target_retained_at_terminal",
                lambda row: bool(row["target_retained_at_terminal"]),
            ),
            (
                "physically_bounded_micro_lift",
                lambda row: bool(row["physically_bounded_micro_lift"]),
            ),
            ("controlled_micro_lift", lambda row: bool(row["controlled_micro_lift"])),
            ("stable_terminal_state", lambda row: bool(row["stable_terminal_state"])),
            ("stable_micro_lift_hold", lambda row: bool(row["stable_micro_lift_hold"])),
            ("stage4_ready_micro_lift", lambda row: bool(row["stage4_ready_micro_lift"])),
        ),
    )
    strict_success_funnel = _failure_funnel(
        classified,
        (
            ("stage2_contact_pass", lambda row: bool(row["stage2_contact_pass"])),
            ("target_reached", lambda row: bool(row["target_reached"])),
            (
                "target_retained_at_terminal",
                lambda row: bool(row["target_retained_at_terminal"]),
            ),
            ("controlled_micro_lift", lambda row: bool(row["controlled_micro_lift"])),
            ("stable_terminal_state", lambda row: bool(row["stable_terminal_state"])),
            ("stable_micro_lift_hold", lambda row: bool(row["stable_micro_lift_hold"])),
            ("sustained_micro_lift", lambda row: bool(row["sustained_micro_lift"])),
        ),
    )
    phase_names = ("approach", "close", "lift", "hold")
    phase_diagnostics = {}
    for phase_name in phase_names:
        phase_rows = [
            row["bridge_phase_metrics"][phase_name]
            for row in classified
            if phase_name in row.get("bridge_phase_metrics", {})
        ]
        if not phase_rows:
            continue
        phase_diagnostics[phase_name] = {
            "episodes": len(phase_rows),
            "safe_rate": sum(bool(row["safe"]) for row in phase_rows)
            / len(phase_rows),
            "height_bounded_rate": sum(
                bool(row["height_bounded"]) for row in phase_rows
            )
            / len(phase_rows),
            "speed_safe_rate": sum(bool(row["speed_safe"]) for row in phase_rows)
            / len(phase_rows),
            "mean_maximum_height_m": sum(
                float(row["maximum_height_m"]) for row in phase_rows
            )
            / len(phase_rows),
            "mean_maximum_linear_speed_m_s": sum(
                float(row["maximum_linear_speed_m_s"]) for row in phase_rows
            )
            / len(phase_rows),
            "mean_maximum_angular_speed_rad_s": sum(
                float(row["maximum_angular_speed_rad_s"]) for row in phase_rows
            )
            / len(phase_rows),
        }
    first_unsafe_counts = (
        Counter(row.get("first_unsafe_bridge_phase") or "none" for row in classified)
        if phase_diagnostics
        else Counter()
    )
    return {
        "episodes": len(classified),
        "target_height_m": target_height_m,
        "overall_rates": overall,
        "candidate_rates": candidate_summary,
        "gate_conditionals": conditional,
        "failure_funnels": {
            "contact_quality": contact_quality_funnel,
            "lift_outcome": lift_outcome_funnel,
            "strict_success": strict_success_funnel,
        },
        "phase_diagnostics": {
            "by_phase": phase_diagnostics,
            "first_unsafe_phase_counts": dict(sorted(first_unsafe_counts.items())),
        },
        "reports": classified,
    }
