#!/usr/bin/env python3
"""Resume-safe multi-GPU collection of audited staged-RL grasp successes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from .merge_success_shards import merge_shards


PROFILE_ARGS = {
    "stage3a_diverse": (
        "--stage-contact-groups-per-side-min", "2",
        "--stage-contact-groups-total-min", "5",
        "--stage-penetration-max-m", "0.025",
        "--stage-stable-hold-steps", "8",
        "--stage-stability-height-tolerance-m", "0.002",
        "--stage-penetration-gate-mode", "current",
        "--stage-stability-gate-mode", "historical_max",
    ),
    "stage3b_quality": (
        "--stage-contact-groups-per-side-min", "3",
        "--stage-penetration-max-m", "0.010",
        "--stage-stable-hold-steps", "8",
        "--stage-stability-height-tolerance-m", "0.0",
        "--stage-penetration-gate-mode", "historical_max",
        "--stage-stability-gate-mode", "historical_max",
    ),
    "stage3_strict": (),
}

# A shard that reaches its rollout writes the Isaac banner; a shard that aborts
# during interpreter startup writes only the allocator error (32-90 bytes) and
# one that hangs before Isaac's logging comes up writes nothing at all.  The
# three groups are cleanly separated, with healthy logs just under 5 kB.
#
# This only holds with unbuffered output.  Redirected to a file, the banner sits
# in the interpreter's stdout buffer for the whole run and flushes at exit:
# measured 2026-09-06, a healthy shard's log was still 0 bytes after 285 s, so
# the watchdog below would have killed it.  With PYTHONUNBUFFERED the same shard
# crosses the threshold 10 s in, while it is still running.
HEALTHY_STARTUP_LOG_BYTES = 4096
SHARD_ENVIRONMENT = {"PYTHONUNBUFFERED": "1"}


def _shard_successes(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "xhand_bodex_staged_success_state_shard_v1":
        raise RuntimeError(f"unexpected shard schema: {path}")
    return int(payload["successes"])


def _existing_successes(shards: list[Path], cache: dict[Path, int]) -> int:
    """Total successes over shards, reading each shard file at most once.

    Collection used to re-read every finished shard before every launch, so the
    bytes read grew quadratically with the number of shards.
    """
    total = 0
    for path in shards:
        if path not in cache:
            cache[path] = _shard_successes(path)
        total += cache[path]
    return total


def _stop_shard(process: subprocess.Popen, grace_seconds: float) -> None:
    """Stop a shard process, escalating to SIGKILL if it ignores SIGTERM.

    Measured 2026-09-05: a shard hung during Isaac startup stayed alive for
    eleven minutes after SIGTERM, so the escalation is required rather than
    defensive.
    """
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _shard_command(args: argparse.Namespace, index: int, gpu: int, report: Path, state: Path) -> list[str]:
    command = [
        str(args.evaluator_python), "-m", "migration_4090.xhand_rl_staged.evaluate",
        "--manifest", str(args.manifest), "--object", args.object,
        "--bodex-bank", str(args.bodex_bank), "--stage", "3",
        "--checkpoint", str(args.checkpoint), "--output", str(report),
        "--successful-state-output", str(state), "--num-envs", str(args.num_envs),
        "--episodes", str(args.episodes_per_shard), "--seed", str(args.seed_base + index),
        "--active-action-group-override", args.active_action_group,
        "--residual-activation-phase", args.residual_activation_phase,
        "--residual-integration", str(args.residual_integration),
        "--residual-limit-rad", str(args.residual_limit_rad),
        *PROFILE_ARGS[args.profile], "--headless", "--device", f"cuda:{gpu}",
    ]
    if args.residual_activation_close_fraction is not None:
        command.extend(
            ["--residual-activation-close-fraction", str(args.residual_activation_close_fraction)]
        )
    if args.candidate_repeat_factors:
        command.extend(
            ["--candidate-repeat-factors", args.candidate_repeat_factors]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--raw-oversample", type=float, default=1.25)
    parser.add_argument("--episodes-per-shard", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--gpus", default="1,2,3,4")
    parser.add_argument(
        "--candidate-repeat-factors",
        help="comma-separated positive repeats used to balance candidate yields",
    )
    parser.add_argument("--seed-base", type=int, default=20261000)
    parser.add_argument("--profile", choices=tuple(PROFILE_ARGS), default="stage3a_diverse")
    parser.add_argument("--active-action-group", default="hands")
    parser.add_argument("--residual-activation-phase", default="hold")
    parser.add_argument("--residual-activation-close-fraction", type=float)
    parser.add_argument("--residual-integration", type=float, default=0.00075)
    parser.add_argument("--residual-limit-rad", type=float, default=0.02)
    parser.add_argument("--joint-quantization-rad", type=float, default=0.005)
    parser.add_argument("--maximum-shards", type=int, default=10_000)
    parser.add_argument("--maximum-consecutive-failed-waves", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--shard-startup-timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "kill a shard that has not written a usable startup log by this "
            "point; a healthy shard crosses the threshold about 10 s in, so "
            "this catches the Isaac startup hang without risking a slow start"
        ),
    )
    parser.add_argument(
        "--shard-timeout-seconds",
        type=float,
        default=14_400.0,
        help="absolute per-shard wall-clock cap",
    )
    parser.add_argument(
        "--shard-stall-multiple",
        type=float,
        default=4.0,
        help=(
            "once enough shards have finished, also cap a shard at this "
            "multiple of the median completed duration; catches a shard that "
            "starts normally and then stalls mid-rollout, which the startup "
            "watchdog cannot see.  Zero disables it."
        ),
    )
    parser.add_argument(
        "--kill-grace-seconds",
        type=float,
        default=30.0,
        help="seconds to wait after SIGTERM before sending SIGKILL",
    )
    args = parser.parse_args()
    if args.target_count <= 0 or args.episodes_per_shard <= 0 or args.num_envs <= 0:
        raise ValueError("counts must be positive")
    if args.raw_oversample < 1.0:
        raise ValueError("raw-oversample must be at least one")
    if args.maximum_consecutive_failed_waves <= 0:
        raise ValueError("maximum-consecutive-failed-waves must be positive")
    if min(args.shard_startup_timeout_seconds, args.shard_timeout_seconds, args.kill_grace_seconds) <= 0:
        raise ValueError("timeouts must be positive")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")

    shard_root = args.output_root / "state_shards"
    report_root = args.output_root / "reports"
    log_root = args.output_root / "logs"
    for root in (shard_root, report_root, log_root):
        root.mkdir(parents=True, exist_ok=True)
    raw_target = int(math.ceil(args.target_count * args.raw_oversample))

    # One shard per GPU is started as soon as that GPU falls idle, rather than
    # one wave at a time.  Under the wave barrier a GPU whose shard aborted
    # during startup sat idle for the rest of the wave: on the 2026-09-02
    # sphere run that was 21,798 idle GPU-seconds, 28.7% of the time the
    # launched processes were alive.  A failed shard is retried under a fresh
    # index, so an aborted startup now costs its own runtime and nothing else.
    success_cache: dict[Path, int] = {}
    shards = sorted(shard_root.glob("shard_*.pt"))
    raw_successes = _existing_successes(shards, success_cache)
    next_index = max((int(path.stem.removeprefix("shard_")) for path in shards), default=-1) + 1
    running: dict[int, dict] = {}
    completed_durations: list[float] = []
    consecutive_failures = 0
    failure_budget = args.maximum_consecutive_failed_waves * len(gpus)
    completed_shards = 0
    failed_shards = 0
    print(
        json.dumps({"shards": len(shards), "raw_successes": raw_successes, "raw_target": raw_target}),
        flush=True,
    )

    while raw_successes < raw_target or running:
        for gpu in gpus:
            if gpu in running or raw_successes >= raw_target:
                continue
            if next_index >= args.maximum_shards:
                if not running:
                    raise RuntimeError("maximum shard count reached before raw target")
                break
            index = next_index
            next_index += 1
            stem = f"shard_{index:06d}"
            state = shard_root / f"{stem}.pt"
            report = report_root / f"{stem}.json"
            log_path = log_root / f"{stem}.log"
            log_stream = log_path.open("wb")
            process = subprocess.Popen(
                _shard_command(args, index, gpu, report, state),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, **SHARD_ENVIRONMENT},
            )
            running[gpu] = {
                "process": process,
                "stream": log_stream,
                "state": state,
                "log_path": log_path,
                "index": index,
                "started": time.monotonic(),
            }

        if not running:
            break
        time.sleep(args.poll_seconds)

        # A shard that starts normally and then stalls mid-rollout is invisible
        # to the startup watchdog, and an absolute cap wide enough to be safe is
        # far wider than any real shard.  Measured shards run within a narrow
        # band, so a multiple of the observed median bounds a stall tightly
        # without any prior knowledge of episode count or hardware.
        stall_timeout = args.shard_timeout_seconds
        if args.shard_stall_multiple > 0 and len(completed_durations) >= 3:
            median = sorted(completed_durations)[len(completed_durations) // 2]
            # the floor only matters when shards are very short, where a small
            # multiple would be within ordinary run-to-run variation
            stall_timeout = min(stall_timeout, max(60.0, args.shard_stall_multiple * median))

        now = time.monotonic()
        for gpu, job in list(running.items()):
            elapsed = now - job["started"]
            reason = None
            if job["process"].poll() is None:
                log_size = job["log_path"].stat().st_size if job["log_path"].exists() else 0
                if elapsed >= args.shard_timeout_seconds:
                    reason = "shard exceeded the absolute timeout"
                elif elapsed >= stall_timeout:
                    reason = (
                        "shard stalled: "
                        f"{elapsed:.0f}s exceeds {args.shard_stall_multiple}x the median "
                        "completed shard"
                    )
                elif (
                    elapsed >= args.shard_startup_timeout_seconds
                    and log_size < HEALTHY_STARTUP_LOG_BYTES
                ):
                    reason = "shard produced no usable startup log"
                else:
                    continue
                _stop_shard(job["process"], args.kill_grace_seconds)

            job["stream"].close()
            returncode = job["process"].returncode
            del running[gpu]
            if reason is None and returncode == 0 and job["state"].is_file():
                success_cache[job["state"]] = _shard_successes(job["state"])
                raw_successes += success_cache[job["state"]]
                completed_durations.append(elapsed)
                completed_shards += 1
                consecutive_failures = 0
                print(
                    json.dumps(
                        {
                            "shard": job["index"],
                            "gpu": gpu,
                            "raw_successes": raw_successes,
                            "raw_target": raw_target,
                            "completed_shards": completed_shards,
                            "failed_shards": failed_shards,
                        }
                    ),
                    flush=True,
                )
                continue

            failed_shards += 1
            consecutive_failures += 1
            print(
                json.dumps(
                    {
                        "warning": "collection shard failed and will be retried under a new index",
                        "shard": job["index"],
                        "gpu": gpu,
                        "returncode": returncode,
                        "reason": reason or "nonzero exit or missing state shard",
                        "elapsed_seconds": round(elapsed, 1),
                        "log": str(job["log_path"]),
                        "failed_shards": failed_shards,
                    }
                ),
                flush=True,
            )
            if consecutive_failures >= failure_budget:
                for other in list(running.values()):
                    _stop_shard(other["process"], args.kill_grace_seconds)
                    other["stream"].close()
                raise RuntimeError(
                    f"{consecutive_failures} consecutive collection shards failed; "
                    f"last log {job['log_path']}"
                )

    shards = sorted(shard_root.glob("shard_*.pt"))
    payload = merge_shards(
        shards,
        target_count=args.target_count,
        quantum_rad=args.joint_quantization_rad,
    )
    output = args.output_root / f"{args.object}_{args.profile}_{args.target_count}.pt"
    if output.exists():
        raise FileExistsError(output)
    torch.save(payload, output)
    summary = {key: value for key, value in payload.items() if key != "samples"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), **summary}, indent=2))


if __name__ == "__main__":
    main()
