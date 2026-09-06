#!/usr/bin/env python3
"""Train/evaluate stages in order and refuse promotion before convergence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from migration_4090.xhand_bodex_bimanual.contracts import load_bodex_bank

from .promotion import promotion_decision
from .selection import checkpoint_selection_decision
from .stages import minimum_complete_episode_iterations


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument(
        "--lift-bodex-bank",
        type=Path,
        help="optional lift-ready BODex bank used for curriculum stages 4--6",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--eval-num-envs", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--train-seed-base", type=int, default=20260829)
    parser.add_argument("--eval-seed-base", type=int, default=20260830)
    parser.add_argument("--iteration-chunk", type=int, default=100)
    parser.add_argument("--max-chunks-per-stage", type=int, default=40)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--candidate-context-only-actor", action="store_true")
    parser.add_argument("--gate-memory-context-only-actor", action="store_true")
    parser.add_argument("--candidate-gated-memory-only-actor", action="store_true")
    parser.add_argument(
        "--candidate-context-train-indices",
        help=(
            "comma-separated zero-based BODex candidate columns to update in "
            "candidate-context-only actor mode"
        ),
    )
    parser.add_argument("--init-noise-std", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument(
        "--freeze-policy-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--bilateral-contact-reward-weight", type=float)
    parser.add_argument("--terminal-success-weight", type=float)
    parser.add_argument("--penetration-reward-weight", type=float)
    parser.add_argument("--penetration-clear-reward-weight", type=float)
    parser.add_argument("--proximity-reward-weight", type=float)
    parser.add_argument("--contact-diversity-reward-weight", type=float)
    parser.add_argument("--contact-continuity-reward-weight", type=float)
    parser.add_argument("--stage-gate-progress-reward-weight", type=float)
    parser.add_argument("--stage-residual-anchor-penalty-weight", type=float)
    parser.add_argument("--distributed-contact-reward-weight", type=float)
    parser.add_argument("--terminal-contact-reward-weight", type=float)
    parser.add_argument(
        "--terminal-contact-progress-mode",
        choices=("bilateral", "joint_gate"),
        default="bilateral",
    )
    parser.add_argument(
        "--distributed-reward-requires-current-bilateral-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--zero-output-action-names-on-external-load",
        help=(
            "comma-separated actor rows to reset when the selected checkpoint "
            "comes from outside this stage training directory"
        ),
    )
    parser.add_argument(
        "--candidate-repeat-factors",
        help="comma-separated positive repeats used only for training bank sampling",
    )
    parser.add_argument("--start-stage", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--stop-stage", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    adapter_modes = (
        args.candidate_context_only_actor,
        args.gate_memory_context_only_actor,
        args.candidate_gated_memory_only_actor,
    )
    if sum(bool(enabled) for enabled in adapter_modes) > 1:
        raise ValueError("actor observation-adapter modes are mutually exclusive")
    adapter_only_actor = any(adapter_modes)
    if args.start_stage > args.stop_stage:
        raise ValueError("start-stage must not exceed stop-stage")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise FileNotFoundError(args.initial_checkpoint)
    def bank_for_stage(stage: int) -> Path:
        if stage >= 4 and args.lift_bodex_bank is not None:
            return args.lift_bodex_bank
        return args.bodex_bank

    # Reject synthetic fixtures and incompatible stage profiles before any GPU
    # process starts.
    for stage in range(args.start_stage, args.stop_stage + 1):
        minimum_iterations = minimum_complete_episode_iterations(stage)
        if args.iteration_chunk < minimum_iterations:
            raise ValueError(
                f"stage {stage} needs at least {minimum_iterations} iterations "
                "per chunk to observe a complete episode"
            )
        load_bodex_bank(
            bank_for_stage(stage),
            expected_object=args.object,
            verify_source=False,
            intended_stage=stage,
        )
    repository = Path(__file__).resolve().parents[2]
    previous_checkpoint: Path | None = (
        args.initial_checkpoint.resolve() if args.initial_checkpoint is not None else None
    )
    for stage in range(args.start_stage, args.stop_stage + 1):
        stage_bank = bank_for_stage(stage)
        stage_root = args.root.resolve() / args.object / f"stage_{stage}"
        train_root = stage_root / "training"
        evaluation_root = stage_root / "evaluations"
        train_root.mkdir(parents=True, exist_ok=True)
        evaluation_root.mkdir(parents=True, exist_ok=True)
        latest_result = train_root / "latest_training_result.json"

        def load_evaluations() -> list[dict[str, object]]:
            return [
                json.loads(path.read_text())
                for path in sorted(evaluation_root.glob("evaluation_*.json"))
            ]

        def record_selection(
            evaluations: list[dict[str, object]],
        ) -> dict[str, object]:
            decision = checkpoint_selection_decision(evaluations)
            (stage_root / "checkpoint_selection_latest.json").write_text(
                json.dumps(decision, indent=2) + "\n"
            )
            return decision

        def evaluate_checkpoint(checkpoint: Path) -> dict[str, object]:
            evaluation_index = len(list(evaluation_root.glob("evaluation_*.json")))
            evaluation_path = evaluation_root / f"evaluation_{evaluation_index:06d}.json"
            evaluate_command = [
                str(args.isaac_python),
                "-m",
                "migration_4090.xhand_rl_staged.evaluate",
                "--manifest",
                str(args.manifest.resolve()),
                "--object",
                args.object,
                "--bodex-bank",
                str(stage_bank.resolve()),
                "--stage",
                str(stage),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(evaluation_path),
                "--num-envs",
                str(args.eval_num_envs),
                "--episodes",
                str(args.eval_episodes),
                "--seed",
                str(args.eval_seed_base + evaluation_index),
                "--device",
                args.device,
                "--headless",
            ]
            _run(evaluate_command, cwd=repository)
            return json.loads(evaluation_path.read_text())

        evaluations = load_evaluations()
        if evaluations:
            selection = record_selection(evaluations)
            previous_checkpoint = Path(str(selection["selected_checkpoint"]))
        elif latest_result.is_file():
            previous_checkpoint = Path(json.loads(latest_result.read_text())["checkpoint"])
        if previous_checkpoint is not None and not evaluations:
            evaluate_checkpoint(previous_checkpoint)
            evaluations = load_evaluations()
            selection = record_selection(evaluations)
            previous_checkpoint = Path(str(selection["selected_checkpoint"]))
            decision = promotion_decision(
                evaluations,
                stage=stage,
                object_name=args.object,
                evaluation_num_envs=args.eval_num_envs,
                evaluation_episodes=args.eval_episodes,
            )
            (stage_root / "promotion_latest.json").write_text(
                json.dumps(decision, indent=2) + "\n"
            )
            if decision["promote"]:
                continue
        for _ in range(args.max_chunks_per_stage):
            training_attempt_index = len(
                list(train_root.glob("training_attempt_*.json"))
            )
            train_command = [
                str(args.isaac_python),
                "-m",
                "migration_4090.xhand_rl_staged.train",
                "--manifest",
                str(args.manifest.resolve()),
                "--object",
                args.object,
                "--bodex-bank",
                str(stage_bank.resolve()),
                "--stage",
                str(stage),
                "--output",
                str(train_root),
                "--num-envs",
                str(args.num_envs),
                "--iterations",
                str(args.iteration_chunk),
                "--seed",
                str(args.train_seed_base + training_attempt_index),
                "--device",
                args.device,
                "--headless",
            ]
            optional_train_arguments = (
                ("--learning-rate", args.learning_rate),
                ("--init-noise-std", args.init_noise_std),
                ("--entropy-coef", args.entropy_coef),
                (
                    "--bilateral-contact-reward-weight",
                    args.bilateral_contact_reward_weight,
                ),
                ("--terminal-success-weight", args.terminal_success_weight),
                ("--penetration-reward-weight", args.penetration_reward_weight),
                (
                    "--penetration-clear-reward-weight",
                    args.penetration_clear_reward_weight,
                ),
                ("--proximity-reward-weight", args.proximity_reward_weight),
                (
                    "--contact-diversity-reward-weight",
                    args.contact_diversity_reward_weight,
                ),
                (
                    "--contact-continuity-reward-weight",
                    args.contact_continuity_reward_weight,
                ),
                (
                    "--stage-gate-progress-reward-weight",
                    args.stage_gate_progress_reward_weight,
                ),
                (
                    "--stage-residual-anchor-penalty-weight",
                    args.stage_residual_anchor_penalty_weight,
                ),
                (
                    "--distributed-contact-reward-weight",
                    args.distributed_contact_reward_weight,
                ),
                (
                    "--terminal-contact-reward-weight",
                    args.terminal_contact_reward_weight,
                ),
            )
            for option, value in optional_train_arguments:
                if value is not None:
                    train_command.extend((option, str(value)))
            if args.candidate_repeat_factors is not None:
                train_command.extend(
                    ("--candidate-repeat-factors", args.candidate_repeat_factors)
                )
            train_command.extend(
                ("--terminal-contact-progress-mode", args.terminal_contact_progress_mode)
            )
            train_command.append(
                "--distributed-reward-requires-current-bilateral-contact"
                if args.distributed_reward_requires_current_bilateral_contact
                else "--no-distributed-reward-requires-current-bilateral-contact"
            )
            if args.freeze_policy_noise is not None:
                train_command.append(
                    "--freeze-policy-noise"
                    if args.freeze_policy_noise
                    else "--no-freeze-policy-noise"
                )
            if args.candidate_context_only_actor:
                train_command.append("--candidate-context-only-actor")
            elif args.gate_memory_context_only_actor:
                train_command.append("--gate-memory-context-only-actor")
            elif args.candidate_gated_memory_only_actor:
                train_command.append("--candidate-gated-memory-only-actor")
            if adapter_only_actor:
                if args.candidate_context_train_indices is not None:
                    train_command.extend(
                        (
                            "--candidate-context-train-indices",
                            args.candidate_context_train_indices,
                        )
                    )
            if previous_checkpoint is not None:
                train_command.extend(("--resume-checkpoint", str(previous_checkpoint)))
                if (
                    previous_checkpoint.parent == train_root
                    and not adapter_only_actor
                ):
                    train_command.append("--load-optimizer")
                else:
                    train_command.append("--reset-policy-noise-after-load")
                    if args.zero_output_action_names_on_external_load:
                        train_command.extend(
                            (
                                "--zero-output-action-names-after-load",
                                args.zero_output_action_names_on_external_load,
                            )
                        )
            _run(train_command, cwd=repository)
            train_result = json.loads(latest_result.read_text())
            raw_checkpoint = Path(train_result["checkpoint"])
            challenger_checkpoint = (
                train_root
                / f"challenger_{training_attempt_index:06d}_{raw_checkpoint.name}"
            )
            if challenger_checkpoint.exists():
                raise FileExistsError(challenger_checkpoint)
            # Branches from the same champion can produce the same RSL-RL
            # iteration filename (for example model_40.pt).  Preserve every
            # evaluated challenger under an immutable attempt-qualified path
            # before a later branch can overwrite the raw runner output.
            shutil.copy2(raw_checkpoint, challenger_checkpoint)
            evaluate_checkpoint(challenger_checkpoint)
            evaluations = load_evaluations()
            selection = record_selection(evaluations)
            previous_checkpoint = Path(str(selection["selected_checkpoint"]))
            decision = promotion_decision(
                evaluations,
                stage=stage,
                object_name=args.object,
                evaluation_num_envs=args.eval_num_envs,
                evaluation_episodes=args.eval_episodes,
            )
            (stage_root / "promotion_latest.json").write_text(
                json.dumps(decision, indent=2) + "\n"
            )
            if decision["promote"]:
                break
        else:
            print(
                json.dumps(
                    {
                        "status": "stage_not_converged",
                        "object": args.object,
                        "stage": stage,
                        "chunks": args.max_chunks_per_stage,
                        "next_stage_started": False,
                    },
                    indent=2,
                )
            )
            return 2
    print(
        json.dumps(
            {
                "status": "requested_stages_converged",
                "object": args.object,
                "start_stage": args.start_stage,
                "stop_stage": args.stop_stage,
                "checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
