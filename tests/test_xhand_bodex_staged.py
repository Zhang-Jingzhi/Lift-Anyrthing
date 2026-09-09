from __future__ import annotations

import copy

import numpy as np
import pytest

from migration_4090.xhand_bodex_bimanual.bodex_adapter import (
    bimanual_pressure_constraints,
    configure_joint_bimanual_seed,
    validate_bodex_result_shapes,
)
from migration_4090.xhand_bodex_bimanual.build_robot_config import (
    BODEX_TO_XHAND_LINK_ROTATION,
    CONTACT_MESH_LINKS,
    CONTACT_POINT_LINKS,
    select_contact_point_sphere_index,
)
from migration_4090.xhand_bodex_bimanual.contracts import (
    BODEX_BACKEND,
    BODEX_BANK_SCHEMA,
    BODEX_CURRICULUM_BANK_SCHEMA,
    BODEX_COMMIT,
    BODEX_REPOSITORY,
    REQUIRED_BODEX_SOURCE_FILES,
    SMOKE_BACKEND,
    SMOKE_BANK_SCHEMA,
    BODexContractError,
    validate_bodex_bank,
)
from migration_4090.xhand_bodex_bimanual.diversity import (
    descriptor_distance,
    diverse_subset,
    grasp_descriptor,
)
from migration_4090.xhand_bodex_bimanual.curate_bank import diagnostic_candidate_rates
from migration_4090.xhand_bodex_bimanual.paired_surface import (
    LEFT_TANGENT_REFERENCE,
    PairedSurfaceSeed,
    RIGHT_TANGENT_REFERENCE,
    pair_surface_samples,
)
from migration_4090.xhand_bodex_bimanual.run_solver import (
    _contiguous_constraint_q,
    bisect_signed_contact_boundary,
    CONTACT_MESH_LINKS_BY_SIDE,
    CONTACT_POINT_LINKS_BY_SIDE,
    base_manip_config,
    collision_pair_diagnostics,
    contact_point_names,
    finite_trajectory_mask,
    install_grouped_line_search_scales,
    install_mesh_ge_query_refresh,
    install_reluqp_device_alignment,
    json_safe_tensor_values,
    joint_group_line_search_multipliers,
    load_coordinate_seed_replay,
    load_ik_initial_pose,
    load_self_collision_sphere_overrides,
    pair_window_indices,
    transform_pair_to_world,
)
from migration_4090.xhand_rl_staged.contact_groups import grouped_filter_indices
from migration_4090.xhand_rl_staged.merge_success_shards import merge_shards
from migration_4090.xhand_rl_staged.checkpoint_migration import (
    append_candidate_context_to_checkpoint_payload,
    insert_candidate_gated_memory_to_checkpoint_payload,
    insert_gate_memory_to_checkpoint_payload,
)
from migration_4090.xhand_rl_staged.bridge_profiles import (
    BRIDGE_PROFILES,
    get_bridge_profile,
)
from migration_4090.xhand_rl_staged.assemble_candidate_bank import (
    minimum_pairwise_descriptor_distance,
)
from migration_4090.xhand_rl_staged.build_standoff_pregrasp_bank import (
    interpolation_rows,
    outward_unit_vectors,
    preserve_open_hand_posture,
)
from migration_4090.xhand_rl_staged.build_side_clipped_controller_bank import (
    arm_then_hand_rows,
    deepest_prefix_safe_contact_waypoints,
    side_clipped_joint_target,
)
from migration_4090.xhand_rl_staged.contracts import STAGED_EVALUATION_SCHEMA
from migration_4090.xhand_rl_staged.promotion import promotion_decision
from migration_4090.xhand_rl_staged.policy_noise import (
    freeze_policy_noise,
    policy_noise_std,
    reset_policy_noise_std,
    restrict_actor_updates_to_candidate_context,
    restrict_actor_updates_to_candidate_gated_action_rows,
    restrict_actor_updates_to_candidate_gated_memory,
    restrict_actor_updates_to_action_rows,
    zero_initialize_residual_actor,
    zero_initialize_residual_actor_rows,
)
from migration_4090.xhand_rl_staged.reward_profiles import (
    stage_init_noise_std,
    stage_entropy_coef,
    stage_freeze_policy_noise,
    stage_reward_profile,
    stage_learning_rate,
)
from migration_4090.xhand_rl_staged.sampling import (
    parse_candidate_indices,
    parse_candidate_repeat_factors,
)
from migration_4090.xhand_rl_staged.micro_lift import (
    classify_micro_lift_report,
    summarize_micro_lift_reports,
)
from migration_4090.xhand_rl_staged.multi_seed_screen import aggregate_evaluations
from migration_4090.xhand_rl_staged.selection import (
    balanced_checkpoint_score,
    candidate_success_rates,
    checkpoint_selection_decision,
)
from migration_4090.xhand_rl_staged.stages import (
    STAGES,
    action_mask_for_group,
    action_mask_for_stage,
    minimum_complete_episode_iterations,
    stage_with_overrides,
    stage_lift_scale,
)
from migration_4090.xhand_rl_staged.throughput import evaluation_throughput


def joint_names() -> list[str]:
    arms = [f"{side}_j{index}" for side in ("left", "right") for index in range(1, 8)]
    hands = [f"{side}_hand_joint_{index}" for side in ("left", "right") for index in range(12)]
    return arms + hands


def test_candidate_index_subset_is_sorted_unique_and_bounded() -> None:
    assert parse_candidate_indices("3,1,3", candidate_count=4) == (1, 3)
    assert parse_candidate_indices(None, candidate_count=4) is None
    with pytest.raises(ValueError, match="outside"):
        parse_candidate_indices("4", candidate_count=4)


def test_explicit_action_group_masks_preserve_the_38d_policy_abi() -> None:
    names = joint_names()
    assert len(names) == 38
    assert int(action_mask_for_group(names, "hands", device="cpu").sum()) == 24
    assert int(action_mask_for_group(names, "distal_wrist", device="cpu").sum()) == 6
    assert int(
        action_mask_for_group(names, "hands_and_distal_arms", device="cpu").sum()
    ) == 30
    assert int(action_mask_for_group(names, "all", device="cpu").sum()) == 38
    with pytest.raises(ValueError, match="unknown active action group"):
        action_mask_for_group(names, "typo", device="cpu")


def test_stage3a_overrides_are_opt_in_and_preserve_formal_stage3() -> None:
    formal = STAGES[2]
    adaptive = stage_with_overrides(
        3,
        contact_groups_per_side_min=2,
        contact_groups_total_min=5,
        penetration_max_m=0.010,
        stable_hold_steps=16,
        stability_height_tolerance_m=0.002,
        penetration_gate_mode="current",
        stability_gate_mode="historical_max",
    )
    assert formal.contact_groups_per_side_min == 3
    assert formal.contact_groups_total_min == 0
    assert formal.penetration_max_m == pytest.approx(0.005)
    assert formal.stable_hold_steps == 32
    assert formal.stability_height_tolerance_m == 0.0
    assert formal.penetration_gate_mode == "historical_max"
    assert formal.stability_gate_mode == "current"
    assert adaptive.contact_groups_per_side_min == 2
    assert adaptive.contact_groups_total_min == 5
    assert adaptive.penetration_max_m == pytest.approx(0.010)
    assert adaptive.stable_hold_steps == 16
    assert adaptive.stability_height_tolerance_m == pytest.approx(0.002)
    assert adaptive.penetration_gate_mode == "current"
    assert adaptive.stability_gate_mode == "historical_max"


def test_stage_override_validation_rejects_incoherent_contact_totals() -> None:
    with pytest.raises(ValueError, match="twice the per-side"):
        stage_with_overrides(
            3,
            contact_groups_per_side_min=3,
            contact_groups_total_min=5,
        )
    with pytest.raises(ValueError, match="penetration gate mode"):
        stage_with_overrides(3, penetration_gate_mode="terminalish")
    with pytest.raises(ValueError, match="stability gate mode"):
        stage_with_overrides(3, stability_gate_mode="sometimes")


def sample(offset: float = 0.0) -> dict:
    names = joint_names()
    q = [offset + 0.01 * index for index in range(len(names))]
    return {
        "object_name": "cube",
        "joint_names": names,
        "pregrasp_full_body_q": q,
        "full_body_q": q,
        "lift_full_body_q": q,
        "stage_seed_profile": 4,
        "lift_mode": "curobo_collision_aware_two_palm_ik_from_bodex_grasp",
        "lift_result": {"success": True},
        "candidate_id": f"candidate_{offset}",
        "bodex_result": {
            "success": True,
            "strict_bodex_success": True,
            "curriculum_seed_accepted": True,
            "optimizer": "BODex.GraspSolver",
            "joint_bimanual_optimization": True,
            "combined_grasp_matrix": True,
            "independent_single_hand_pairing": False,
            "contact_counts_by_side": {"left": 6, "right": 6},
            "grasp_matrix_shape": [1, 12, 6, 3],
        },
    }


def test_candidate_composition_detects_duplicate_grasp_descriptors() -> None:
    first = sample(0.0)
    duplicate = copy.deepcopy(first)
    duplicate["candidate_id"] = "duplicate_identifier"
    distinct = sample(0.5)
    assert minimum_pairwise_descriptor_distance([first, duplicate, distinct]) == 0.0


def test_evaluation_throughput_reports_strict_samples_per_hour() -> None:
    result = evaluation_throughput(
        episodes=100,
        overall_rates={
            "controlled_micro_lift": 0.50,
            "stage4_ready_micro_lift": 0.20,
            "sustained_micro_lift": 0.10,
        },
        end_to_end_wall_time_s=1800.0,
        rollout_wall_time_s=900.0,
    )
    assert result["episodes_per_hour_end_to_end"] == pytest.approx(200.0)
    assert result["accepted_counts"]["sustained_micro_lift"] == 10
    assert result["accepted_per_hour_end_to_end"]["sustained_micro_lift"] == pytest.approx(20.0)
    assert result["accepted_per_hour_rollout_only"]["sustained_micro_lift"] == pytest.approx(40.0)


def test_multi_seed_screen_requires_every_candidate_to_be_robust() -> None:
    def payload(seed: int, weak_ready: float) -> dict:
        candidate_rates = {}
        for index in range(4):
            ready = weak_ready if index == 2 else 0.10
            candidate_rates[str(index)] = {
                "candidate_id": f"candidate_{index}",
                "episodes": 25,
                "stage2_contact_pass": 0.80,
                "physically_bounded_micro_lift": 0.70,
                "controlled_micro_lift": 0.50,
                "stage4_ready_micro_lift": ready,
                "sustained_micro_lift": ready,
            }
        return {
            "object": "sphere",
            "profile": {"name": "controlled_5mm_decoupled_settle_v1"},
            "target_height_m": 0.005,
            "bodex_bank": "/tmp/bank.pt",
            "bodex_bank_sha256": "bank",
            "checkpoint": "/tmp/model.pt",
            "checkpoint_sha256": "checkpoint",
            "policy_mode": "checkpoint",
            "zero_action_candidates": [2],
            "seed": seed,
            "episodes": 100,
            "candidate_rates": candidate_rates,
            "overall_rates": {
                "stage2_contact_pass": 0.80,
                "physically_bounded_micro_lift": 0.70,
                "controlled_micro_lift": 0.50,
                "stage4_ready_micro_lift": 0.08,
                "sustained_micro_lift": 0.08,
            },
            "throughput": {"end_to_end_wall_time_s": 900.0},
        }

    result = aggregate_evaluations(
        [payload(1, 0.04), payload(2, 0.00), payload(3, 0.04)],
        minimum_seed_count=3,
        minimum_descriptor_distance=0.035,
        bank_descriptor_distance=0.08,
        minimum_per_candidate_controlled=0.25,
        minimum_per_candidate_ready=0.03,
        minimum_overall_sustained=0.05,
        minimum_sustained_per_hour=100.0,
        required_production_height_m=0.05,
    )
    assert result["decisions"]["diversity_pass"] is True
    assert result["decisions"]["per_candidate_controlled_pass"] is True
    assert result["decisions"]["per_candidate_ready_pass"] is False
    assert result["decisions"]["bridge_screen_pass"] is False
    assert result["decisions"]["production_pass"] is False


def production_bank() -> dict:
    return {
        "schema": BODEX_BANK_SCHEMA,
        "generation_backend": BODEX_BACKEND,
        "object": "cube",
        "stage_seed_profile": 4,
        "supported_curriculum_stages": list(range(1, 7)),
        "requires_lift_ik": True,
        "bodex_provenance": {
            "repository": BODEX_REPOSITORY,
            "commit": BODEX_COMMIT,
            "joint_bimanual_optimization": True,
            "combined_grasp_matrix": True,
            "independent_single_hand_pairing": False,
            "source_files_sha256": {
                path: "a" * 64 for path in REQUIRED_BODEX_SOURCE_FILES
            },
        },
        "samples": [sample()],
    }


def test_production_bank_requires_joint_bimanual_provenance() -> None:
    validate_bodex_bank(production_bank(), expected_object="cube")
    bad = production_bank()
    bad["bodex_provenance"]["independent_single_hand_pairing"] = True
    with pytest.raises(BODexContractError):
        validate_bodex_bank(bad)


def curriculum_bank() -> dict:
    bank = production_bank()
    bank["schema"] = BODEX_CURRICULUM_BANK_SCHEMA
    bank["stage_seed_profile"] = 1
    bank["supported_curriculum_stages"] = [1, 2, 3]
    bank["requires_lift_ik"] = False
    row = bank["samples"][0]
    row["stage_seed_profile"] = 1
    row["lift_full_body_q"] = list(row["full_body_q"])
    row["lift_mode"] = "not_required_for_curriculum_stages_1_to_3"
    row["lift_result"] = {
        "success": False,
        "requested": False,
    }
    row["bodex_result"].update(
        {
            "success": False,
            "strict_bodex_success": False,
            "curriculum_seed_accepted": True,
            "stage_seed_profile": 1,
            "stage_seed_acceptance": {
                "accepted": True,
                "maximum_gap_m": 0.006,
                "world_gap_m": 0.0045,
                "positive_world_penetration": False,
                "joint_bounds_exact_zero": True,
                "self_collision_exact_zero": True,
                "grasp_error_max": 0.08,
                "distance_error_max": 0.015,
                "grasp_threshold": 0.1,
                "distance_threshold": 0.02,
            },
        }
    )
    return bank


def test_stage1_curriculum_bank_is_genuine_bodex_but_not_lift_ready() -> None:
    bank = curriculum_bank()
    validate_bodex_bank(bank, expected_object="cube", intended_stage=1)
    validate_bodex_bank(bank, expected_object="cube", intended_stage=3)
    with pytest.raises(BODexContractError, match="does not support curriculum stage 4"):
        validate_bodex_bank(bank, expected_object="cube", intended_stage=4)


def test_stage1_curriculum_bank_rejects_penetration_or_nonzero_self_collision() -> None:
    penetrated = curriculum_bank()
    penetrated["samples"][0]["bodex_result"]["stage_seed_acceptance"][
        "positive_world_penetration"
    ] = True
    with pytest.raises(BODexContractError, match="positive world penetration"):
        validate_bodex_bank(penetrated)

    colliding = curriculum_bank()
    colliding["samples"][0]["bodex_result"]["stage_seed_acceptance"][
        "self_collision_exact_zero"
    ] = False
    with pytest.raises(BODexContractError, match="joint/self-collision"):
        validate_bodex_bank(colliding)


def test_standoff_pregrasp_uses_object_to_palm_directions() -> None:
    palms = np.asarray([[0.5, 0.2, 0.9], [0.5, -0.2, 0.8]])
    center = np.asarray([0.5, 0.0, 0.85])
    directions = outward_unit_vectors(
        palms, center, horizontal_only=False
    )
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)
    assert directions[0, 1] > 0.0
    assert directions[1, 1] < 0.0
    horizontal = outward_unit_vectors(
        palms, center, horizontal_only=True
    )
    assert np.allclose(horizontal[:, 2], 0.0)


def test_standoff_pregrasp_preserves_only_open_hand_values() -> None:
    names = ["left_j1", "right_j1", "left_hand_index_joint1"]
    solved = np.asarray([1.0, 2.0, 3.0])
    source = np.asarray([4.0, 5.0, 6.0])
    preserved = preserve_open_hand_posture(solved, source, names)
    assert preserved.tolist() == [1.0, 2.0, 6.0]


def test_standoff_pregrasp_audits_linear_close_endpoints() -> None:
    pregrasp = np.asarray([0.0, 1.0])
    grasp = np.asarray([2.0, 3.0])
    rows = interpolation_rows(pregrasp, grasp, steps=3)
    assert np.allclose(rows[0], pregrasp)
    assert np.allclose(rows[1], [1.0, 2.0])
    assert np.allclose(rows[-1], grasp)


def test_side_clipped_controller_uses_independent_bodex_prefixes() -> None:
    names = ["left_j1", "left_hand_joint_1", "right_j1", "right_hand_joint_1"]
    pregrasp = np.asarray([0.0, 1.0, 2.0, 3.0])
    grasp = np.asarray([10.0, 11.0, 12.0, 13.0])
    target = side_clipped_joint_target(
        pregrasp,
        grasp,
        names,
        left_alpha=0.25,
        right_alpha=0.75,
    )
    assert target == pytest.approx([2.5, 3.5, 9.5, 10.5])


def test_side_clipped_controller_requires_prefix_safe_contact() -> None:
    rows = [
        {
            "mesh_query_valid": True,
            "left_penetration_mm": 0.0,
            "left_contact_points": 0,
        },
        {
            "mesh_query_valid": True,
            "left_penetration_mm": 0.4,
            "left_contact_points": 2,
        },
        {
            "mesh_query_valid": True,
            "left_penetration_mm": 0.7,
            "left_contact_points": 5,
        },
        {
            "mesh_query_valid": True,
            "left_penetration_mm": 0.1,
            "left_contact_points": 6,
        },
    ]
    safe = deepest_prefix_safe_contact_waypoints(
        rows,
        side="left",
        maximum_penetration_mm=0.5,
        minimum_contact_points=1,
    )
    assert safe == [{**rows[1], "prefix_maximum_penetration_mm": 0.4}]


def test_side_clipped_controller_audits_arm_then_hand_order() -> None:
    names = ["left_j1", "left_hand_joint_1", "right_j1", "right_hand_joint_1"]
    pregrasp = np.asarray([0.0, 1.0, 2.0, 3.0])
    target = np.asarray([4.0, 5.0, 6.0, 7.0])
    rows = arm_then_hand_rows(
        pregrasp,
        target,
        names,
        arm_steps=3,
        hand_steps=3,
    )
    assert rows[0] == pytest.approx(pregrasp)
    assert rows[2] == pytest.approx([4.0, 1.0, 6.0, 3.0])
    assert rows[-1] == pytest.approx(target)


def test_decoupled_bridge_profile_preserves_legacy_profiles() -> None:
    assert get_bridge_profile("controlled_5mm_v2").arm_approach_fraction_of_close == 0.0
    assert (
        get_bridge_profile("controlled_5mm_decoupled_v1")
        .arm_approach_fraction_of_close
        == pytest.approx(0.60)
    )
    settled = get_bridge_profile("controlled_5mm_decoupled_settle_v1")
    assert settled.arm_approach_fraction_of_close == pytest.approx(0.45)
    assert settled.finger_close_end_fraction_of_close == pytest.approx(0.80)
    stable_settled = get_bridge_profile("stable_5mm_decoupled_settle_v1")
    assert stable_settled.stable_hold_steps > settled.stable_hold_steps
    assert stable_settled.residual_integration < settled.residual_integration
    slow_diverse = get_bridge_profile("stable_5mm_slow_settle_diverse_v1")
    assert slow_diverse.stable_hold_steps == stable_settled.stable_hold_steps
    assert slow_diverse.episode_length_s > stable_settled.episode_length_s
    assert slow_diverse.arm_approach_fraction_of_close < slow_diverse.finger_close_end_fraction_of_close
    assert slow_diverse.reset_xy_noise_m == pytest.approx(stable_settled.reset_xy_noise_m)
    assert slow_diverse.reset_yaw_noise_rad == pytest.approx(stable_settled.reset_yaw_noise_rad)
    assert slow_diverse.reset_joint_noise_rad == pytest.approx(stable_settled.reset_joint_noise_rad)
    no_reset_noise = get_bridge_profile("stable_5mm_decoupled_settle_no_reset_noise_v1")
    assert no_reset_noise.nominal_pose_lock is False
    assert no_reset_noise.fixed_candidate_index is None
    assert no_reset_noise.reset_xy_noise_m == pytest.approx(0.0)
    assert no_reset_noise.reset_yaw_noise_rad == pytest.approx(0.0)
    assert no_reset_noise.reset_joint_noise_rad == pytest.approx(0.0)
    assert no_reset_noise.contact_gated_residual is True
    freeze25 = get_bridge_profile("controlled_5mm_decoupled_settle_freeze25_v1")
    freeze50 = get_bridge_profile("controlled_5mm_decoupled_settle_freeze50_v1")
    assert freeze25.residual_integration_end_fraction_of_hold == pytest.approx(0.25)
    assert freeze50.residual_integration_end_fraction_of_hold == pytest.approx(0.50)
    slow = get_bridge_profile("controlled_5mm_slow_settle_v1")
    assert slow.episode_length_s > settled.episode_length_s
    assert sum(slow.phase_fractions) == pytest.approx(1.0)
    assert slow.arm_approach_fraction_of_close < slow.finger_close_end_fraction_of_close
    formation = get_bridge_profile("formation_5mm_close_wrist_v1")
    assert formation.action_group == "distal_wrist"
    assert formation.residual_activation == "close"
    assert formation.residual_activation_close_fraction == pytest.approx(0.50)
    assert formation.residual_limit_rad == pytest.approx(0.035)
    retry = get_bridge_profile("formation_5mm_close_wrist_retry_v2")
    assert retry.action_group == "distal_wrist"
    assert retry.residual_activation == "close"
    assert retry.residual_activation_close_fraction == pytest.approx(0.50)
    assert retry.residual_limit_rad == pytest.approx(0.020)
    assert retry.residual_integration < formation.residual_integration
    wrist_guard = get_bridge_profile("stable_5mm_wrist_guard_v1")
    assert wrist_guard.action_group == "distal_wrist"
    assert wrist_guard.fixed_candidate_index is None
    assert wrist_guard.penetration_limit_m == pytest.approx(0.025)
    assert wrist_guard.palm_center_alignment_reward_weight == pytest.approx(1.5)
    hold_wrist_guard = get_bridge_profile("stable_5mm_hold_wrist_guard_v1")
    assert hold_wrist_guard.action_group == "distal_wrist"
    assert hold_wrist_guard.residual_activation == "hold"
    assert hold_wrist_guard.residual_integration_end_fraction_of_hold == pytest.approx(
        0.50
    )
    assert hold_wrist_guard.residual_limit_rad < wrist_guard.residual_limit_rad
    assert hold_wrist_guard.palm_center_alignment_reward_weight == pytest.approx(0.0)
    assert hold_wrist_guard.penetration_limit_m == pytest.approx(0.025)


def test_fixed_pose_palm_alignment_profile_is_isolated() -> None:
    profile = get_bridge_profile("formation_5mm_fixed_pose_palm_align_v1")
    assert profile.nominal_pose_lock is True
    assert profile.fixed_candidate_index == 2
    assert profile.palm_center_alignment_reward_weight == pytest.approx(3.0)
    assert profile.reset_xy_noise_m == pytest.approx(0.0)
    assert profile.reset_yaw_noise_rad == pytest.approx(0.0)
    assert profile.reset_joint_noise_rad == pytest.approx(0.0)
    assert profile.action_group == "distal_wrist"
    assert profile.residual_activation == "close"
    assert profile.penetration_limit_m is None


def micro_lift_report(
    *,
    candidate_index: int,
    gates: dict[str, bool],
    maximum: float,
    final: float,
    physically_bounded: bool = True,
    stable_steps: int = 32,
    linear_speed: float = 0.0,
    angular_speed: float = 0.0,
) -> dict:
    return {
        "candidate_index": candidate_index,
        "candidate_id": f"candidate_{candidate_index}",
        "stage_gates": gates,
        "maximum_lift_height_m": maximum,
        "final_lift_height_m": final,
        "micro_lift_physically_bounded": physically_bounded,
        "maximum_micro_lift_stable_steps": stable_steps,
        "object_linear_speed_m_s": linear_speed,
        "object_angular_speed_rad_s": angular_speed,
        "terminated_early": False,
    }


def test_micro_lift_requires_reach_retention_and_all_stage2_contact_gates() -> None:
    gates = {
        "finite_state": True,
        "bilateral_contact": True,
        "contact_continuity": True,
        "distributed_contacts": True,
    }
    passed = classify_micro_lift_report(
        micro_lift_report(
            candidate_index=0, gates=gates, maximum=0.011, final=0.009
        ),
        target_height_m=0.010,
    )
    assert passed["sustained_micro_lift"] is True
    failed = classify_micro_lift_report(
        micro_lift_report(
            candidate_index=0,
            gates={**gates, "distributed_contacts": False},
            maximum=0.011,
            final=0.009,
        ),
        target_height_m=0.010,
    )
    assert failed["target_reached"] is True
    assert failed["sustained_micro_lift"] is False


def test_micro_lift_rejects_unbounded_or_unstable_height_events() -> None:
    gates = {
        "finite_state": True,
        "bilateral_contact": True,
        "contact_continuity": True,
        "distributed_contacts": True,
    }
    unbounded = classify_micro_lift_report(
        micro_lift_report(
            candidate_index=0,
            gates=gates,
            maximum=1.0,
            final=0.010,
            physically_bounded=False,
        ),
        target_height_m=0.010,
    )
    assert unbounded["target_reached"] is True
    assert unbounded["controlled_micro_lift"] is False
    unstable = classify_micro_lift_report(
        micro_lift_report(
            candidate_index=0,
            gates=gates,
            maximum=0.011,
            final=0.010,
            linear_speed=0.2,
        ),
        target_height_m=0.010,
    )
    assert unstable["stable_terminal_state"] is False
    assert unstable["controlled_micro_lift"] is True
    assert unstable["stage4_ready_micro_lift"] is False
    assert unstable["sustained_micro_lift"] is False


def test_micro_lift_summary_reports_candidate_and_gate_conditionals() -> None:
    gates = {
        "finite_state": True,
        "bilateral_contact": True,
        "contact_continuity": True,
        "distributed_contacts": True,
    }
    reports = [
        micro_lift_report(
            candidate_index=0, gates=gates, maximum=0.006, final=0.005
        ),
        micro_lift_report(
            candidate_index=1,
            gates={**gates, "distributed_contacts": False},
            maximum=0.006,
            final=0.005,
        ),
    ]
    summary = summarize_micro_lift_reports(reports, target_height_m=0.005)
    assert summary["overall_rates"]["target_reached"] == 1.0
    assert summary["overall_rates"]["controlled_micro_lift"] == 1.0
    assert summary["overall_rates"]["sustained_micro_lift"] == 0.5
    assert summary["candidate_rates"]["0"]["sustained_micro_lift"] == 1.0
    assert (
        summary["gate_conditionals"]["distributed_contacts"]
        ["target_reached_given_gate_false"]
        == 1.0
    )
    assert (
        summary["gate_conditionals"]["distributed_contacts"]
        ["controlled_lift_given_gate_false"]
        == 1.0
    )
    assert summary["failure_funnels"]["contact_quality"][-1] == {
        "stage": "distributed_contacts",
        "episodes": 1,
        "rate_of_all": 0.5,
        "retention_from_previous": 0.5,
    }
    assert summary["failure_funnels"]["lift_outcome"][-1]["episodes"] == 2
    assert summary["failure_funnels"]["strict_success"][-1]["episodes"] == 1


def test_lift_bridge_profiles_tighten_height_before_stability() -> None:
    assert set(BRIDGE_PROFILES) == {
        "controlled_5mm_v1",
        "controlled_5mm_v2",
        "controlled_5mm_decoupled_v1",
        "controlled_5mm_decoupled_settle_v1",
        "controlled_5mm_slow_settle_v1",
        "controlled_5mm_decoupled_settle_freeze25_v1",
        "controlled_5mm_decoupled_settle_freeze50_v1",
        "stable_5mm_decoupled_settle_v1",
        "stable_5mm_slow_settle_diverse_v1",
        "stable_5mm_decoupled_settle_no_reset_noise_v1",
        "stable_5mm_decoupled_contact_gated_v1",
        "stable_5mm_decoupled_contact_gated_low_joint_noise_v1",
        "controlled_5mm_v3",
        "formation_5mm_close_wrist_v1",
        "formation_5mm_close_wrist_retry_v2",
        "formation_5mm_fixed_pose_palm_align_v1",
        "stable_5mm_wrist_guard_v1",
        "stable_5mm_hold_wrist_guard_v1",
        "stable_5mm_v1",
        "controlled_10mm_v1",
        "stable_10mm_v1",
        "stable_5mm_contact_gated_hold_v1",
        "stable_5mm_contact_gated_hold_freeze_v1",
        "stable_35mm_v1",
        "stable_35mm_slow_v1",
        "stable_35mm_lownoise_v1",
        "stable_35mm_nonoise_v1",
        "stable_35mm_free_v1",
        "stable_35mm_slowapproach_v1",
        "stable_10mm_free_v1",
        "standoff_10mm_v1",
        "standoff_80mm_v1",
        "lift50_hold1s_v1",
        "lift50_freeze_v1",
    }
    controlled_5 = get_bridge_profile("controlled_5mm_v1")
    controlled_5_v2 = get_bridge_profile("controlled_5mm_v2")
    controlled_5_v3 = get_bridge_profile("controlled_5mm_v3")
    stable_5 = get_bridge_profile("stable_5mm_v1")
    controlled_10 = get_bridge_profile("controlled_10mm_v1")
    assert controlled_5.target_height_m == 0.005
    assert controlled_10.target_height_m == 0.010
    assert controlled_5.stable_hold_steps < stable_5.stable_hold_steps
    assert stable_5.stability_reward_weight > controlled_5.stability_reward_weight
    assert controlled_5.action_group == "hands"
    assert controlled_5.residual_activation == "hold"
    assert controlled_5_v2.residual_integration < controlled_5.residual_integration
    assert (
        controlled_5_v2.height_tracking_reward_weight
        > controlled_5.height_tracking_reward_weight
    )
    assert (
        controlled_5_v2.terminal_controlled_reward_weight
        < controlled_5.terminal_controlled_reward_weight
    )
    assert controlled_5_v3.reset_xy_noise_m == pytest.approx(
        0.25 * controlled_5_v2.reset_xy_noise_m
    )
    assert controlled_5_v3.reset_yaw_noise_rad == pytest.approx(
        0.25 * controlled_5_v2.reset_yaw_noise_rad
    )
    assert controlled_5_v3.reset_joint_noise_rad == pytest.approx(
        0.25 * controlled_5_v2.reset_joint_noise_rad
    )
    assert stable_5.residual_activation == "lift"
    assert stable_5.penetration_limit_m == pytest.approx(0.025)
    assert controlled_5.maximum_height_m == pytest.approx(0.025)


def test_lift_bridge_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown lift bridge profile"):
        get_bridge_profile("missing")


def test_smoke_fixture_is_rejected_by_default() -> None:
    row = sample()
    row.pop("bodex_result")
    row["test_fixture"] = True
    bank = {
        "schema": SMOKE_BANK_SCHEMA,
        "generation_backend": SMOKE_BACKEND,
        "object": "cube",
        "test_fixture": True,
        "samples": [row],
    }
    with pytest.raises(BODexContractError):
        validate_bodex_bank(bank)
    validate_bodex_bank(bank, allow_test_fixture=True)


def test_paired_surface_sampler_uses_both_sides_and_opposed_normals() -> None:
    positions = np.asarray(
        [[0, 0.1, 0], [0.01, 0.1, 0.01], [0, -0.1, 0], [0.01, -0.1, 0.01]],
        dtype=np.float64,
    )
    normals = np.asarray(
        [[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float64
    )
    pairs = pair_surface_samples(
        positions,
        normals,
        count=2,
        seed=7,
        minimum_opposition_cosine=0.9,
        minimum_span_m=0.1,
    )
    assert len(pairs) == 2
    assert all(pair.left_position_m[1] > 0 and pair.right_position_m[1] < 0 for pair in pairs)
    assert all(pair.opposition_cosine >= 0.9 for pair in pairs)
    assert all(pair.lateral_fraction >= 0.75 for pair in pairs)
    assert all(pair.side_normal_min_component >= 0.55 for pair in pairs)
    assert all(abs(pair.left_position_m[1]) == pytest.approx(0.115) for pair in pairs)
    assert all(abs(pair.right_position_m[1]) == pytest.approx(0.115) for pair in pairs)


def test_paired_surface_sampler_rejects_top_bottom_opposition() -> None:
    positions = np.asarray(
        [
            [0.0, 0.01, 0.10],
            [0.0, -0.01, -0.10],
            [0.0, 0.10, 0.00],
            [0.0, -0.10, 0.00],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [0.0, 0.10, 0.995],
            [0.0, -0.10, -0.995],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pair = pair_surface_samples(
        positions,
        normals,
        count=1,
        seed=7,
        minimum_opposition_cosine=0.9,
        minimum_span_m=0.1,
    )[0]
    assert pair.source_surface_indices == (2, 3)
    assert pair.lateral_fraction == pytest.approx(1.0)
    assert pair.z_offset_m == pytest.approx(0.0)


def test_xhand_transfer_keeps_bodex_and_xhand_palm_axes_identical() -> None:
    transfer = np.asarray(BODEX_TO_XHAND_LINK_ROTATION)
    assert np.linalg.det(transfer) == pytest.approx(1.0)
    assert transfer == pytest.approx(np.eye(3))


def test_bodex_contact_points_are_decoupled_from_physical_contact_meshes() -> None:
    point_links = {
        link for links in CONTACT_POINT_LINKS_BY_SIDE.values() for link in links
    }
    mesh_links = {
        link for links in CONTACT_MESH_LINKS_BY_SIDE.values() for link in links
    }
    assert point_links == set(CONTACT_POINT_LINKS)
    assert mesh_links == set(CONTACT_MESH_LINKS)
    assert point_links.isdisjoint(mesh_links)
    assert all(link.endswith("_tip") for link in point_links)
    assert all(link.endswith("_link2") for link in mesh_links)


def test_contact_point_selector_rejects_large_off_origin_builder_artifacts() -> None:
    spheres = [
        {"center": [-0.022, -0.023, -0.022], "radius": 0.018},
        {"center": [0.0002, -0.0005, 0.0003], "radius": 0.004},
        {"center": [0.004, 0.003, 0.002], "radius": 0.0004},
    ]
    assert select_contact_point_sphere_index(spheres) == 1


def test_side_specific_tangent_references_match_mirrored_mounts() -> None:
    assert LEFT_TANGENT_REFERENCE == pytest.approx([-1.0, 0.0, 0.0])
    assert np.linalg.norm(RIGHT_TANGENT_REFERENCE) == pytest.approx(1.0, abs=1.0e-3)
    assert RIGHT_TANGENT_REFERENCE[0] > 0.0
    assert RIGHT_TANGENT_REFERENCE[2] > 0.0


def test_bodex_adapter_sets_two_transferred_links_and_side_constraints() -> None:
    pair = PairedSurfaceSeed(
        (0.0, 0.1, 0.0),
        (0.0, -0.1, 0.0),
        (0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        1.0,
        0.2,
        (0, 1),
    )
    configured = configure_joint_bimanual_seed(
        {},
        pair,
        initial_q=[0.0] * 38,
        left_contact_names=["left_tip/0", "left_palm/0"],
        right_contact_names=["right_tip/0", "right_palm/0"],
    )
    assert len(configured["seeder_cfg"]["t"]) == 2
    assert configured["grasp_cfg"]["ge_param"]["pressure_constraints"] == [
        [[0, 1, 2, 3], 1.0],
        [[0, 1], 0.35],
        [[2, 3], 0.35],
    ]
    result = validate_bodex_result_shapes(
        solution_shape=(8, 3, 38),
        contact_point_shape=(8, 4, 3),
        contact_frame_shape=(8, 4, 3, 3),
        left_contact_count=2,
        right_contact_count=2,
    )
    assert result["combined_grasp_matrix"] is True


def test_solver_runner_transforms_both_seeds_and_keeps_bodex_defaults() -> None:
    pair = PairedSurfaceSeed(
        (0.0, 0.1, 0.0),
        (0.0, -0.1, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        1.0,
        0.2,
        (0, 1),
    )
    transformed = transform_pair_to_world(
        pair,
        translation_xyz=(0.5, 0.0, 0.8),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    assert transformed.left_position_m == pytest.approx((0.5, 0.1, 0.8))
    assert transformed.right_position_m == pytest.approx((0.5, -0.1, 0.8))
    config = base_manip_config(
        robot_config_path=__file__,
        initial_q=[0.0] * 38,
        seed_num=8,
        translation_jitter_m=0.008,
        rotation_jitter_deg=8.0,
    )
    assert config["seed_num"] == 8
    assert config["seeder_cfg"]["skip_transfer"] is False
    assert config["grasp_cfg"]["ge_param"]["solver_type"] == "batch_reluqp"
    assert config["grasp_contact_strategy"]["max_ge_stage"] == 0


def test_solver_runner_can_keep_grasp_energy_through_exact_contact_stage() -> None:
    config = base_manip_config(
        robot_config_path=__file__,
        initial_q=[0.0] * 38,
        seed_num=1,
        translation_jitter_m=0.0,
        rotation_jitter_deg=0.0,
        max_ge_stage=2,
    )
    assert config["grasp_contact_strategy"]["opt_progress"] == [0.0, 0.6, 0.8]
    assert config["grasp_contact_strategy"]["max_ge_stage"] == 2

    with pytest.raises(ValueError, match="max-ge-stage"):
        base_manip_config(
            robot_config_path=__file__,
            initial_q=[0.0] * 38,
            seed_num=1,
            translation_jitter_m=0.0,
            rotation_jitter_deg=0.0,
            max_ge_stage=3,
        )


def test_mesh_ge_query_refresh_avoids_upstream_cached_raw_distance_branch() -> None:
    import types

    calls: list[int] = []

    class FakeCost:
        count = 4

        def forward(self, value: int) -> int:
            calls.append(self.count)
            self.count += 1
            return value + 1

    cost = FakeCost()
    rollout = types.SimpleNamespace(grasp_cost=cost)
    config = types.SimpleNamespace(
        rollout_fn=rollout,
        solver=types.SimpleNamespace(rollout_fn=rollout),
    )
    provenance = install_mesh_ge_query_refresh(config, max_ge_stage=2)
    assert cost.forward(7) == 8
    assert cost.forward(8) == 9
    assert calls == [0, 0]
    assert provenance["enabled"] is True
    assert provenance["patched_cost_instance_count"] == 1
    assert provenance["source_checkout_modified"] is False

    disabled = install_mesh_ge_query_refresh(config, max_ge_stage=0)
    assert disabled["enabled"] is False


def test_reluqp_device_alignment_moves_lazy_private_solver_buffers() -> None:
    import types
    import torch

    class FakePrivateSolver:
        def __init__(self) -> None:
            self.tensor_args = types.SimpleNamespace(device=torch.device("cpu"))
            self.rhos_matrix = torch.ones(2, 1, 3, 3)
            self.help_arange = torch.arange(1)

    class FakeBatchSolver:
        def init_problem(self, G_matrix, l_matrix, h_matrix) -> None:
            self.G_matrix = G_matrix
            self.l_matrix = l_matrix
            self.h_matrix = h_matrix
            self.solver = FakePrivateSolver()

    qpsolver = FakeBatchSolver()
    energy = types.SimpleNamespace(qpsolver=qpsolver)
    cost = types.SimpleNamespace(GraspEnergy=energy)
    rollout = types.SimpleNamespace(grasp_cost=cost)
    config = types.SimpleNamespace(
        rollout_fn=rollout,
        solver=types.SimpleNamespace(rollout_fn=rollout),
    )
    tensor_args = types.SimpleNamespace(device=torch.device("meta"))
    provenance = install_reluqp_device_alignment(
        config,
        tensor_args=tensor_args,
    )
    qpsolver.init_problem(
        torch.empty(1, device="meta"),
        torch.empty(1, device="meta"),
        torch.empty(1, device="meta"),
    )
    assert qpsolver.solver.rhos_matrix.device.type == "meta"
    assert qpsolver.solver.help_arange.device.type == "meta"
    assert qpsolver.solver.tensor_args is tensor_args
    assert provenance["target_device"] == "meta"
    assert provenance["patched_qpsolver_count"] == 1
    assert provenance["source_checkout_modified"] is False


def test_nonfinite_bodex_trajectory_is_rejected_per_seed_and_json_safe() -> None:
    import torch

    trajectories = torch.zeros((3, 2, 38), dtype=torch.float32)
    trajectories[1, 1, 4] = float("nan")
    trajectories[2, 0, 7] = float("inf")
    assert finite_trajectory_mask(trajectories).tolist() == [True, False, False]
    serialized = json_safe_tensor_values(trajectories)
    assert serialized[1][1][4] is None
    assert serialized[2][0][7] is None


def test_contact_boundary_bisection_preserves_strict_signed_bracket() -> None:
    import torch

    outside = torch.as_tensor([[-1.0] + [0.0] * 37], dtype=torch.float32)
    inside = torch.as_tensor([[1.0] + [0.0] * 37], dtype=torch.float32)

    def signed(q: torch.Tensor) -> torch.Tensor:
        return q[:, 0] - 0.25

    best, low, high, provenance = bisect_signed_contact_boundary(
        outside,
        inside,
        signed_distance_fn=signed,
        iterations=32,
    )
    assert provenance["bracketed"] == [True]
    assert abs(float(signed(best)[0])) <= 1.0e-7
    assert float(signed(low)[0]) <= 0.0
    assert float(signed(high)[0]) >= 0.0


def test_ik_initial_pose_overrides_only_named_joints(tmp_path) -> None:
    path = tmp_path / "seed.yml"
    path.write_text(
        "schema: xhand_bodex_bimanual_ik_seed_v1\n"
        "joint_positions:\n"
        "  left_j1: 0.25\n"
    )
    names = joint_names()
    fallback = [0.0] * len(names)
    seeded, provenance = load_ik_initial_pose(
        path, joint_names=names, fallback_q=fallback
    )
    assert seeded[names.index("left_j1")] == pytest.approx(0.25)
    assert sum(abs(value) > 0.0 for value in seeded) == 1
    assert provenance["seeded_joint_count"] == 1


def test_coordinate_seed_replay_is_hash_locked_joint_bodex_diagnostics(tmp_path) -> None:
    import hashlib
    import json

    robot = tmp_path / "robot.yml"
    robot.write_text("robot_cfg: {}\n")
    names = joint_names()
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "schema": "xhand_bodex_rejected_solver_diagnostics_v1",
                "object": "cube",
                "robot_config_sha256": hashlib.sha256(robot.read_bytes()).hexdigest(),
                "rows": [
                    {
                        "pair_index": 0,
                        "joint_names": names,
                        "coordinate_seed_q": [[0.0] * len(names)],
                    }
                ],
            }
        )
    )
    seeds, provenance = load_coordinate_seed_replay(
        diagnostic,
        object_name="cube",
        robot_config_path=robot,
        joint_names=names,
    )
    assert seeds == {0: [[0.0] * 38]}
    assert provenance["joint_bimanual_seed_source"] is True
    assert provenance["manual_pose_source"] is False


def test_pair_window_indices_preserve_global_candidate_provenance() -> None:
    assert pair_window_indices(256, start=16, limit=48) == list(range(16, 64))
    assert pair_window_indices(20, start=18, limit=8) == [18, 19]
    with pytest.raises(ValueError, match="outside paired-seed collection"):
        pair_window_indices(16, start=16, limit=1)


def test_grouped_line_search_uses_canonical_14_arm_24_hand_split() -> None:
    import types
    import torch

    names = joint_names()
    multipliers = joint_group_line_search_multipliers(
        names,
        arm_multiplier=0.01,
        hand_multiplier=0.1,
    )
    assert multipliers[:14] == pytest.approx([0.01] * 14)
    assert multipliers[14:] == pytest.approx([0.1] * 24)

    optimizer = types.SimpleNamespace(
        cu_opt_init=False,
        line_scale=torch.full((1, 1, 38), 0.1),
        alpha_list=torch.full((1, 1, 38), 0.1),
        zero_alpha_list=torch.full((1, 1, 1), 0.1),
        n_problems=1,
    )
    grasp_config = types.SimpleNamespace(
        solver=types.SimpleNamespace(newton_optimizer=optimizer)
    )
    provenance = install_grouped_line_search_scales(
        grasp_config,
        joint_names=names,
        arm_multiplier=0.01,
        hand_multiplier=0.1,
    )
    assert optimizer.line_scale[0, 0, :14].tolist() == pytest.approx([0.001] * 14)
    assert optimizer.line_scale[0, 0, 14:].tolist() == pytest.approx([0.01] * 24)
    assert optimizer.alpha_list == pytest.approx(optimizer.line_scale)
    assert provenance["arm_joint_indices"] == list(range(14))
    assert provenance["hand_joint_indices"] == list(range(14, 38))


def test_constraint_diagnostics_make_ranked_bodex_results_contiguous() -> None:
    import torch

    ranked = torch.arange(8 * 38, dtype=torch.float32).reshape(38, 8).transpose(0, 1)
    assert ranked.shape == (8, 38)
    assert not ranked.is_contiguous()
    normalized = _contiguous_constraint_q(ranked)
    assert normalized.is_contiguous()
    assert torch.equal(normalized, ranked)


def test_contact_sphere_selection_uses_small_near_origin_marker() -> None:
    spheres = {
        "left_hand_mid_tip": [
            {"center": [0.03, 0.0, 0.0], "radius": 0.018},
            {"center": [0.001, 0.0, 0.0], "radius": 0.004},
            {"center": [0.0001, 0.0, 0.0], "radius": 0.0001},
        ],
    }
    assert contact_point_names(spheres, ["left_hand_mid_tip"]) == [
        "left_hand_mid_tip/1"
    ]


def test_collision_diagnostic_reports_only_checked_penetrating_pair() -> None:
    spheres = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.02],
                [0.03, 0.0, 0.0, 0.02],
                [0.20, 0.0, 0.0, 0.02],
            ]
        ],
        dtype=np.float32,
    )
    import torch

    rows = collision_pair_diagnostics(
        torch.as_tensor(spheres),
        sphere_link_indices=torch.as_tensor([10, 11, 12]),
        link_index_to_name={10: "left_tip", 11: "left_palm", 12: "right_tip"},
        checked_pairs=torch.as_tensor([[0, 1], [1, 2]]),
        sphere_offsets=torch.zeros(3),
    )
    assert len(rows) == 1
    assert rows[0]["sphere_indices"] == [0, 1]
    assert rows[0]["link_sphere_indices"] == [0, 0]
    assert rows[0]["link_names"] == ["left_tip", "left_palm"]
    assert rows[0]["penetration_m"] == pytest.approx(0.01)


def test_collision_sphere_overrides_are_hash_locked_and_mesh_validated() -> None:
    override_path = (
        "migration_4090/xhand_bodex_bimanual/config/"
        "xhand_bodex_self_collision_sphere_overrides.yml"
    )
    robot_config = "/tmp/xhand_bodex_robot_config_20260829_v5_distal_contacts/xhand_fullbody_bodex.yml"
    if not __import__("pathlib").Path(robot_config).is_file():
        pytest.skip("generated BODex robot config is not available")
    payload, rows, link_rows = load_self_collision_sphere_overrides(
        override_path,
        robot_config_path=robot_config,
    )
    assert payload["schema"] == "xhand_bodex_self_collision_sphere_overrides_v1"
    assert len(rows) == 2
    assert [(row["first_link"], row["second_link"]) for row in link_rows] == [
        ("left_j5", "left_j7")
    ]
    assert all(row["mesh_validation"]["surface_intersection"] is False for row in rows)


def test_stage_abi_and_action_masks_stay_checkpoint_compatible() -> None:
    names = joint_names()
    assert all(abs(sum(stage.phase_fractions) - 1.0) < 1.0e-9 for stage in STAGES)
    assert [int(action_mask_for_stage(names, stage, device="cpu").sum()) for stage in range(1, 7)] == [
        24,
        24,
        30,
        38,
        38,
        38,
    ]
    assert [stage_lift_scale(stage) for stage in range(1, 7)] == pytest.approx(
        [0.0, 0.0, 0.0, 0.2, 0.6, 1.0]
    )
    assert STAGES[0].required_gates == ("finite_state", "bilateral_contact")
    assert STAGES[0].required_bilateral_steps == 4
    assert [stage.reset_xy_noise_m for stage in STAGES] == sorted(
        stage.reset_xy_noise_m for stage in STAGES
    )
    assert [stage.reset_yaw_noise_rad for stage in STAGES] == sorted(
        stage.reset_yaw_noise_rad for stage in STAGES
    )
    assert [stage.reset_joint_noise_rad for stage in STAGES] == sorted(
        stage.reset_joint_noise_rad for stage in STAGES
    )
    assert [stage.residual_integration for stage in STAGES] == sorted(
        stage.residual_integration for stage in STAGES
    )
    assert [stage.residual_limit_rad for stage in STAGES] == sorted(
        stage.residual_limit_rad for stage in STAGES
    )
    assert STAGES[1].residual_integration == pytest.approx(0.015)
    assert STAGES[1].residual_limit_rad == pytest.approx(0.15)
    assert [minimum_complete_episode_iterations(stage) for stage in range(1, 7)] == [
        4,
        7,
        7,
        8,
        10,
        12,
    ]


def test_contact_group_layout_covers_every_filter_once() -> None:
    groups = grouped_filter_indices()
    flattened = [index for indices in groups.values() for index in indices]
    assert len(groups) == 12
    assert sorted(flattened) == list(range(60))


def test_diversity_rejects_near_duplicates() -> None:
    first = sample(0.0)
    near = copy.deepcopy(first)
    near["full_body_q"] = [value + 1.0e-4 for value in near["full_body_q"]]
    far = sample(0.5)
    assert descriptor_distance(grasp_descriptor(first), grasp_descriptor(near)) < 0.035
    assert len(diverse_subset([first, near, far], minimum_distance=0.035)) == 2


def test_curation_computes_per_candidate_physics_rates() -> None:
    rates = diagnostic_candidate_rates(
        {
            "reports": [
                {"candidate_id": "a", "stage_pass": True},
                {"candidate_id": "a", "stage_pass": False},
                {"candidate_id": "b", "stage_pass": True},
            ]
        }
    )
    assert rates == {"a": 0.5, "b": 1.0}


def test_promotion_requires_three_consecutive_evaluation_windows() -> None:
    rows = [
        {
            "schema": STAGED_EVALUATION_SCHEMA,
            "object": "cube",
            "stage": 1,
            "num_envs": 64,
            "episodes": 256,
            "candidate_selection": "round_robin",
            "deterministic_policy": True,
            "success_rate": rate,
            "earlier_gates_no_regression": True,
        }
        for rate in (0.79, 0.81, 0.82, 0.83)
    ]
    assert promotion_decision(rows[:3], stage=1, object_name="cube")["promote"] is False
    assert promotion_decision(rows, stage=1, object_name="cube")["promote"] is True


def test_promotion_rejects_mixed_or_incomplete_evaluation_protocols() -> None:
    rows = [
        {
            "schema": STAGED_EVALUATION_SCHEMA,
            "object": "cube",
            "stage": 1,
            "num_envs": 64,
            "episodes": 256,
            "candidate_selection": "round_robin",
            "deterministic_policy": True,
            "success_rate": 0.9,
            "earlier_gates_no_regression": True,
        }
        for _ in range(3)
    ]
    rows[1]["num_envs"] = 256
    decision = promotion_decision(rows, stage=1, object_name="cube")
    assert decision["window_protocol_valid"] == [True, False, True]
    assert decision["promote"] is False

    del rows[1]["num_envs"]
    assert promotion_decision(rows, stage=1, object_name="cube")["promote"] is False


def test_stage1_profile_targets_contact_clearance_intersection() -> None:
    profile = stage_reward_profile(1)
    assert profile["bilateral_contact_reward_weight"] == pytest.approx(10.0)
    assert profile["terminal_success_weight"] == pytest.approx(600.0)
    assert profile["penetration_reward_weight"] == pytest.approx(-2.0)
    assert profile["penetration_clear_reward_weight"] == pytest.approx(1.0)
    assert profile["proximity_reward_weight"] == pytest.approx(2.0)
    assert profile["stage_gate_progress_reward_weight"] == pytest.approx(24.0)
    assert profile["stage_action_anchor_penalty_weight"] == pytest.approx(4.0)
    assert profile["stage_residual_anchor_penalty_weight"] == pytest.approx(12.0)
    assert stage_init_noise_std(1) == pytest.approx(0.15)
    assert stage_entropy_coef(1) == pytest.approx(0.001)
    assert stage_freeze_policy_noise(1) is True
    assert stage_learning_rate(1) == pytest.approx(1.0e-4)
    assert stage_learning_rate(2) == pytest.approx(1.0e-4)
    assert stage_learning_rate(3) == pytest.approx(1.0e-4)
    assert stage_learning_rate(4) == pytest.approx(3.0e-4)


def test_candidate_repeat_factors_retain_all_diverse_candidates() -> None:
    assert parse_candidate_repeat_factors(None, candidate_count=4) == (1, 1, 1, 1)
    assert parse_candidate_repeat_factors("3,2,1,3", candidate_count=4) == (3, 2, 1, 3)
    with pytest.raises(ValueError, match="must match"):
        parse_candidate_repeat_factors("3,2,1", candidate_count=4)
    with pytest.raises(ValueError, match="positive"):
        parse_candidate_repeat_factors("3,2,0,3", candidate_count=4)


def test_reset_curriculum_tightens_smoothly_and_monotonically() -> None:
    assert STAGES[1].reset_xy_noise_m == pytest.approx(0.003)
    assert STAGES[1].reset_yaw_noise_rad == pytest.approx(0.045)
    assert STAGES[1].reset_joint_noise_rad == pytest.approx(0.0045)
    for field in (
        "reset_xy_noise_m",
        "reset_yaw_noise_rad",
        "reset_joint_noise_rad",
    ):
        values = [getattr(stage, field) for stage in STAGES]
        assert values == sorted(values)


def test_reward_profile_overrides_are_validated() -> None:
    profile = stage_reward_profile(1, {"terminal_success_weight": 120.0})
    assert profile["terminal_success_weight"] == pytest.approx(120.0)
    with pytest.raises(ValueError, match="unknown staged reward overrides"):
        stage_reward_profile(1, {"typo_weight": 1.0})
    with pytest.raises(ValueError, match="must be negative"):
        stage_reward_profile(1, {"penetration_reward_weight": 1.0})


def test_stage2_profile_prioritizes_continuity_distribution_and_clearance() -> None:
    profile = stage_reward_profile(2)
    assert profile["terminal_success_weight"] == pytest.approx(800.0)
    assert profile["penetration_reward_weight"] == pytest.approx(-2.0)
    assert profile["contact_continuity_reward_weight"] == pytest.approx(14.0)
    assert profile["contact_diversity_reward_weight"] == pytest.approx(10.0)
    assert profile["distributed_contact_reward_weight"] == pytest.approx(12.0)
    assert profile["terminal_contact_reward_weight"] == pytest.approx(40.0)
    assert profile["stage_gate_progress_reward_weight"] == pytest.approx(32.0)
    assert profile["stage_residual_anchor_penalty_weight"] == pytest.approx(16.0)
    assert "catastrophic_penetration" not in STAGES[1].required_gates


def test_checkpoint_selection_balances_overall_and_weakest_bodex_candidate() -> None:
    def evaluation(checkpoint: str, successes: tuple[int, int, int, int]) -> dict:
        reports = []
        for candidate, count in enumerate(successes):
            reports.extend(
                {
                    "candidate_index": candidate,
                    "stage_pass": episode < count,
                    "terminated_early": False,
                }
                for episode in range(10)
            )
        return {
            "checkpoint": checkpoint,
            "success_rate": sum(successes) / 40,
            "reports": reports,
        }

    imbalanced = evaluation("imbalanced.pt", (5, 8, 9, 6))
    balanced = evaluation("balanced.pt", (6, 7, 7, 7))
    regressed = evaluation("regressed.pt", (4, 9, 9, 8))
    assert candidate_success_rates(imbalanced) == {
        0: 0.5,
        1: 0.8,
        2: 0.9,
        3: 0.6,
    }
    assert balanced_checkpoint_score(balanced) > balanced_checkpoint_score(imbalanced)
    decision = checkpoint_selection_decision((imbalanced, balanced, regressed))
    assert decision["selected_checkpoint"] == "balanced.pt"
    assert decision["latest_selected"] is False


def test_checkpoint_policy_noise_is_explicitly_reset() -> None:
    import types
    import torch

    policy = types.SimpleNamespace(
        state_dependent_std=False,
        noise_std_type="scalar",
        std=torch.nn.Parameter(torch.full((38,), 0.38)),
    )
    assert policy_noise_std(policy) == pytest.approx(0.38)
    assert reset_policy_noise_std(policy, 0.25) == pytest.approx(0.25)
    assert policy.std.tolist() == pytest.approx([0.25] * 38)
    freeze_policy_noise(policy)
    assert policy.std.requires_grad is False


def test_fresh_residual_actor_is_zero_initialized() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(), torch.nn.Linear(4, 2))
    policy = types.SimpleNamespace(state_dependent_std=False, actor=actor)
    zero_initialize_residual_actor(policy)
    assert torch.count_nonzero(actor[-1].weight).item() == 0
    assert torch.count_nonzero(actor[-1].bias).item() == 0
    assert torch.equal(actor(torch.randn(5, 3)), torch.zeros(5, 2))


def test_newly_unmasked_actor_rows_are_zero_initialized_selectively() -> None:
    import types
    import torch

    output = torch.nn.Linear(4, 5)
    original_weight = output.weight.detach().clone()
    original_bias = output.bias.detach().clone()
    policy = types.SimpleNamespace(
        state_dependent_std=False,
        actor=torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(), output),
    )
    assert zero_initialize_residual_actor_rows(policy, [4, 1, 4]) == (1, 4)
    assert torch.count_nonzero(output.weight[[1, 4]]).item() == 0
    assert torch.count_nonzero(output.bias[[1, 4]]).item() == 0
    assert torch.equal(output.weight[[0, 2, 3]], original_weight[[0, 2, 3]])
    assert torch.equal(output.bias[[0, 2, 3]], original_bias[[0, 2, 3]])


def test_candidate_context_only_actor_update_preserves_shared_policy() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(6, 5), torch.nn.ELU(), torch.nn.Linear(5, 2)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = [parameter.detach().clone() for parameter in actor.parameters()]

    metadata = restrict_actor_updates_to_candidate_context(policy, context_width=2)
    optimizer.zero_grad()
    actor(torch.ones(4, 6)).sum().backward()
    assert torch.count_nonzero(actor[0].weight.grad[:, :-2]).item() == 0
    assert torch.count_nonzero(actor[0].weight.grad[:, -2:]).item() > 0
    optimizer.step()

    assert metadata["trainable_actor_elements"] == 10
    assert metadata["actor_observation_normalizer_frozen"] is True
    assert policy.actor_obs_normalization is False
    assert torch.equal(actor[0].weight[:, :-2], original[0][:, :-2])
    assert not torch.equal(actor[0].weight[:, -2:], original[0][:, -2:])
    assert torch.equal(actor[0].bias, original[1])
    assert torch.equal(actor[2].weight, original[2])
    assert torch.equal(actor[2].bias, original[3])


def test_candidate_context_only_actor_can_select_candidate_columns() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(8, 5), torch.nn.ELU(), torch.nn.Linear(5, 2)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = actor[0].weight.detach().clone()

    metadata = restrict_actor_updates_to_candidate_context(
        policy, context_width=4, trainable_context_indices=(0, 2)
    )
    optimizer.zero_grad()
    actor(torch.ones(4, 8)).sum().backward()
    optimizer.step()

    assert metadata["trainable_context_indices"] == [0, 2]
    assert metadata["trainable_actor_elements"] == 10
    assert torch.equal(actor[0].weight[:, :4], original[:, :4])
    assert not torch.equal(actor[0].weight[:, 4], original[:, 4])
    assert torch.equal(actor[0].weight[:, 5], original[:, 5])
    assert not torch.equal(actor[0].weight[:, 6], original[:, 6])
    assert torch.equal(actor[0].weight[:, 7], original[:, 7])


def test_candidate_context_actor_can_train_preceding_gate_memory_columns() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(28, 5), torch.nn.ELU(), torch.nn.Linear(5, 2)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = actor[0].weight.detach().clone()

    metadata = restrict_actor_updates_to_candidate_context(
        policy,
        context_width=4,
        trainable_context_indices=(1,),
        gate_memory_width=4,
        intervening_width=16,
    )
    optimizer.zero_grad()
    actor(torch.ones(4, 28)).sum().backward()
    optimizer.step()

    assert metadata["gate_memory_width"] == 4
    assert metadata["intervening_width"] == 16
    assert metadata["trainable_actor_elements"] == 25
    assert torch.equal(actor[0].weight[:, :4], original[:, :4])
    assert not torch.equal(actor[0].weight[:, 4:8], original[:, 4:8])
    assert torch.equal(actor[0].weight[:, 8:24], original[:, 8:24])
    assert torch.equal(actor[0].weight[:, 24], original[:, 24])
    assert not torch.equal(actor[0].weight[:, 25], original[:, 25])
    assert torch.equal(actor[0].weight[:, 26:], original[:, 26:])


def test_candidate_gated_memory_actor_can_select_candidate_blocks() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(28, 5), torch.nn.ELU(), torch.nn.Linear(5, 2)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = actor[0].weight.detach().clone()

    metadata = restrict_actor_updates_to_candidate_gated_memory(
        policy,
        context_width=4,
        gate_memory_width=4,
        trainable_candidate_indices=(0, 3),
    )
    optimizer.zero_grad()
    actor(torch.ones(4, 28)).sum().backward()
    optimizer.step()

    assert metadata["trainable_candidate_indices"] == [0, 3]
    assert metadata["candidate_gated_memory_width"] == 16
    assert metadata["trainable_actor_elements"] == 40
    assert torch.equal(actor[0].weight[:, :8], original[:, :8])
    assert not torch.equal(actor[0].weight[:, 8:12], original[:, 8:12])
    assert torch.equal(actor[0].weight[:, 12:20], original[:, 12:20])
    assert not torch.equal(actor[0].weight[:, 20:24], original[:, 20:24])
    assert torch.equal(actor[0].weight[:, 24:], original[:, 24:])


def test_candidate_gated_action_rows_update_only_selected_adapters_and_rows() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(28, 5), torch.nn.ELU(), torch.nn.Linear(5, 4)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = [parameter.detach().clone() for parameter in actor.parameters()]

    metadata = restrict_actor_updates_to_candidate_gated_action_rows(
        policy,
        context_width=4,
        gate_memory_width=4,
        action_indices=(1, 3),
        trainable_candidate_indices=(0, 2),
    )
    optimizer.zero_grad()
    actor(torch.ones(4, 28)).sum().backward()
    optimizer.step()

    assert metadata["trainable_candidate_indices"] == [0, 2]
    assert metadata["trainable_action_indices"] == [1, 3]
    assert metadata["candidate_gated_memory_width"] == 16
    assert metadata["trainable_actor_elements"] == 52
    assert policy.actor_obs_normalization is False
    # Candidate blocks are columns 8:12 and 16:20 for a 28D input.
    assert not torch.equal(actor[0].weight[:, 8:12], original[0][:, 8:12])
    assert torch.equal(actor[0].weight[:, 12:16], original[0][:, 12:16])
    assert not torch.equal(actor[0].weight[:, 16:20], original[0][:, 16:20])
    assert torch.equal(actor[0].weight[:, :8], original[0][:, :8])
    assert torch.equal(actor[0].weight[:, 20:], original[0][:, 20:])
    assert torch.equal(actor[0].bias, original[1])
    assert torch.equal(actor[2].weight[[0, 2]], original[2][[0, 2]])
    assert not torch.equal(actor[2].weight[[1, 3]], original[2][[1, 3]])
    assert torch.equal(actor[2].bias[[0, 2]], original[3][[0, 2]])
    assert not torch.equal(actor[2].bias[[1, 3]], original[3][[1, 3]])


def test_output_head_only_actor_update_preserves_backbone_and_inactive_rows() -> None:
    import types
    import torch

    actor = torch.nn.Sequential(
        torch.nn.Linear(6, 5), torch.nn.ELU(), torch.nn.Linear(5, 4)
    )
    policy = types.SimpleNamespace(actor=actor, actor_obs_normalization=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    original = [parameter.detach().clone() for parameter in actor.parameters()]

    metadata = restrict_actor_updates_to_action_rows(policy, (1, 3))
    optimizer.zero_grad()
    actor(torch.ones(4, 6)).sum().backward()
    assert torch.count_nonzero(actor[-1].weight.grad[[0, 2]]).item() == 0
    assert torch.count_nonzero(actor[-1].weight.grad[[1, 3]]).item() > 0
    optimizer.step()

    assert metadata["trainable_action_indices"] == [1, 3]
    assert metadata["trainable_actor_elements"] == 12
    assert metadata["actor_backbone_frozen"] is True
    assert policy.actor_obs_normalization is False
    assert torch.equal(actor[0].weight, original[0])
    assert torch.equal(actor[0].bias, original[1])
    assert torch.equal(actor[-1].weight[[0, 2]], original[2][[0, 2]])
    assert not torch.equal(actor[-1].weight[[1, 3]], original[2][[1, 3]])
    assert torch.equal(actor[-1].bias[[0, 2]], original[3][[0, 2]])
    assert not torch.equal(actor[-1].bias[[1, 3]], original[3][[1, 3]])


@pytest.mark.parametrize("indices", [(), (-1,), (4,)])
def test_candidate_context_only_actor_rejects_invalid_candidate_columns(indices) -> None:
    import types
    import torch

    policy = types.SimpleNamespace(
        actor=torch.nn.Sequential(torch.nn.Linear(8, 5)),
        actor_obs_normalization=True,
    )
    with pytest.raises(ValueError):
        restrict_actor_updates_to_candidate_context(
            policy, context_width=4, trainable_context_indices=indices
        )


def test_candidate_context_checkpoint_migration_preserves_network_outputs() -> None:
    import torch

    actor_weight = torch.randn(5, 269)
    critic_weight = torch.randn(3, 269)
    payload = {
        "model_state_dict": {
            "actor.0.weight": actor_weight.clone(),
            "critic.0.weight": critic_weight.clone(),
            "actor_obs_normalizer._mean": torch.zeros(1, 269),
            "actor_obs_normalizer._var": torch.ones(1, 269),
            "actor_obs_normalizer._std": torch.ones(1, 269),
            "critic_obs_normalizer._mean": torch.zeros(1, 269),
            "critic_obs_normalizer._var": torch.ones(1, 269),
            "critic_obs_normalizer._std": torch.ones(1, 269),
        }
    }
    old_observation = torch.randn(7, 269)
    candidate_context = torch.nn.functional.one_hot(
        torch.arange(7) % 4, num_classes=4
    ).float()
    append_candidate_context_to_checkpoint_payload(payload)
    state = payload["model_state_dict"]
    new_observation = torch.cat((old_observation, candidate_context), dim=-1)
    assert state["actor.0.weight"].shape == (5, 273)
    assert state["critic.0.weight"].shape == (3, 273)
    torch.testing.assert_close(
        old_observation @ actor_weight.T,
        new_observation @ state["actor.0.weight"].T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_gate_memory_checkpoint_migration_preserves_candidate_context_and_outputs() -> None:
    import torch

    actor_weight = torch.randn(5, 273)
    critic_weight = torch.randn(3, 273)
    payload = {
        "model_state_dict": {
            "actor.0.weight": actor_weight.clone(),
            "critic.0.weight": critic_weight.clone(),
            "actor_obs_normalizer._mean": torch.zeros(1, 273),
            "actor_obs_normalizer._var": torch.ones(1, 273),
            "actor_obs_normalizer._std": torch.ones(1, 273),
            "critic_obs_normalizer._mean": torch.zeros(1, 273),
            "critic_obs_normalizer._var": torch.ones(1, 273),
            "critic_obs_normalizer._std": torch.ones(1, 273),
        }
    }
    old_observation = torch.randn(7, 273)
    gate_memory = torch.zeros(7, 4)
    new_observation = torch.cat(
        (old_observation[:, :-4], gate_memory, old_observation[:, -4:]), dim=-1
    )

    insert_gate_memory_to_checkpoint_payload(payload)
    state = payload["model_state_dict"]
    assert state["actor.0.weight"].shape == (5, 277)
    assert state["critic.0.weight"].shape == (3, 277)
    assert torch.equal(state["actor.0.weight"][:, -4:], actor_weight[:, -4:])
    assert torch.count_nonzero(state["actor.0.weight"][:, -8:-4]).item() == 0
    torch.testing.assert_close(
        old_observation @ actor_weight.T,
        new_observation @ state["actor.0.weight"].T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert state["actor_obs_normalizer._mean"].shape[-1] == 277
    assert torch.equal(
        state["actor_obs_normalizer._mean"][:, -4:],
        torch.zeros(1, 4),
    )
    torch.testing.assert_close(
        old_observation @ critic_weight.T,
        new_observation @ state["critic.0.weight"].T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_merge_success_shards_balances_candidates_and_deduplicates(tmp_path) -> None:
    import torch

    def sample(candidate_index: int, value: float) -> dict:
        return {
            "candidate_index": candidate_index,
            "candidate_id": f"candidate-{candidate_index}",
            "final_full_body_q": torch.full((38,), value),
        }

    common = {
        "schema": "xhand_bodex_staged_success_state_shard_v1",
        "object": "sphere",
        "stage": 3,
        "stage_spec": {"stage_id": 3},
        "checkpoint_sha256": "checkpoint",
        "bodex_bank_sha256": "bank",
        "episodes": 4,
        "successes": 2,
    }
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(
        {
            **common,
            "seed": 1,
            "samples": [sample(0, 0.0), sample(1, 0.1)],
        },
        first,
    )
    torch.save(
        {
            **common,
            "seed": 2,
            "samples": [sample(0, 0.001), sample(1, 0.2)],
        },
        second,
    )

    merged = merge_shards(
        [first, second],
        target_count=2,
        quantum_rad=0.005,
    )

    assert merged["selected_count"] == 2
    assert merged["quantized_duplicate_count"] == 1
    assert merged["candidate_selected_counts"] == {"0": 1, "1": 1}


def test_candidate_gated_memory_checkpoint_migration_preserves_outputs() -> None:
    import torch

    actor_weight = torch.randn(5, 277)
    critic_weight = torch.randn(3, 277)
    payload = {
        "model_state_dict": {
            "actor.0.weight": actor_weight.clone(),
            "critic.0.weight": critic_weight.clone(),
            "actor_obs_normalizer._mean": torch.zeros(1, 277),
            "actor_obs_normalizer._var": torch.ones(1, 277),
            "actor_obs_normalizer._std": torch.ones(1, 277),
            "critic_obs_normalizer._mean": torch.zeros(1, 277),
            "critic_obs_normalizer._var": torch.ones(1, 277),
            "critic_obs_normalizer._std": torch.ones(1, 277),
        }
    }
    old_observation = torch.randn(7, 277)
    candidate_gated_memory = torch.zeros(7, 16)
    new_observation = torch.cat(
        (
            old_observation[:, :-4],
            candidate_gated_memory,
            old_observation[:, -4:],
        ),
        dim=-1,
    )

    insert_candidate_gated_memory_to_checkpoint_payload(payload)
    state = payload["model_state_dict"]
    assert state["actor.0.weight"].shape == (5, 293)
    assert state["critic.0.weight"].shape == (3, 293)
    assert torch.equal(state["actor.0.weight"][:, -4:], actor_weight[:, -4:])
    assert torch.count_nonzero(state["actor.0.weight"][:, -20:-4]).item() == 0
    torch.testing.assert_close(
        old_observation @ actor_weight.T,
        new_observation @ state["actor.0.weight"].T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert state["actor_obs_normalizer._mean"].shape[-1] == 293
    assert torch.equal(
        state["actor_obs_normalizer._mean"][:, -20:-4],
        torch.zeros(1, 16),
    )
    torch.testing.assert_close(
        old_observation @ critic_weight.T,
        new_observation @ state["critic.0.weight"].T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
