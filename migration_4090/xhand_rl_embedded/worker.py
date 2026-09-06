#!/usr/bin/env python3
"""Persistent per-object v2 train then direct-collect worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .contracts import latest_policy_checkpoint, sha256_file, validate_manifest, validate_root
from .storage import accepted_count


def run(command: list[str], *, repo: Path, log: Path) -> int:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        return subprocess.run(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT).returncode


def event(path: Path, **payload: object) -> None:
    row = {"schema": "xhand_rl_embedded_timing_event_v2", "time": time.time(), **payload}
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--train-num-envs", type=int, default=64)
    parser.add_argument("--train-iterations", type=int, default=4000)
    parser.add_argument("--collect-num-envs", type=int, default=64)
    parser.add_argument("--penetration-reward-weight", type=float)
    parser.add_argument("--init-noise-std", type=float)
    parser.add_argument("--entropy-coef", type=float)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    validate_root(args.root, args.manifest, manifest)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    run_root = args.root / "runs" / args.object
    training_root = run_root / "training"
    logs_root = args.root / "logs" / args.object
    timing = run_root / "timing.jsonl"
    root_token = hashlib.sha256(str(args.root.resolve()).encode()).hexdigest()[:16]
    kit_portable_base = (
        Path(tempfile.gettempdir())
        / "xhand_rl_embedded_kit"
        / root_token
        / args.object
        / f"worker_{os.getpid()}"
    )
    kit_portable_root = kit_portable_base / "attempt_1"
    for path in (run_root, training_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)
    complete = training_root / "TRAINING_COMPLETE.json"
    if not complete.is_file():
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            kit_portable_root = kit_portable_base / f"attempt_{attempt}"
            prewarm_command = [
                str(args.isaac_python),
                "-m",
                "migration_4090.xhand_rl_embedded.smoke_env",
                "--manifest",
                str(args.manifest.resolve()),
                "--object",
                args.object,
                "--nominal-dataset",
                str(args.nominal.resolve()),
                "--steps",
                "64",
                "--num-envs",
                "1",
                "--device",
                args.device,
                "--headless",
                "--kit-portable-root",
                str(kit_portable_root),
            ]
            started = time.monotonic()
            code = run(prewarm_command, repo=repo, log=logs_root / "cache_prewarm.log")
            event(
                timing,
                stage="kit_cache_prewarm",
                attempt=attempt,
                duration_s=time.monotonic() - started,
                return_code=code,
            )
            if code != 0:
                if attempt == max_attempts:
                    raise RuntimeError(f"v2 Kit cache prewarm failed for {args.object}")
                time.sleep(10)
                continue
            # Isaac/PhysX teardown releases some driver resources asynchronously.
            time.sleep(10)
            command = [
                str(args.isaac_python),
                "-m",
                "migration_4090.xhand_rl_embedded.train",
                "--manifest",
                str(args.manifest.resolve()),
                "--object",
                args.object,
                "--nominal-dataset",
                str(args.nominal.resolve()),
                "--output",
                str(training_root.resolve()),
                "--num-envs",
                str(args.train_num_envs),
                "--max-iterations",
                str(args.train_iterations),
                "--device",
                args.device,
                "--headless",
                "--kit-portable-root",
                str(kit_portable_root),
            ]
            if args.penetration_reward_weight is not None:
                command.extend((
                    "--penetration-reward-weight",
                    str(args.penetration_reward_weight),
                ))
            if args.init_noise_std is not None:
                command.extend((
                    "--init-noise-std",
                    str(args.init_noise_std),
                ))
            if args.entropy_coef is not None:
                command.extend((
                    "--entropy-coef",
                    str(args.entropy_coef),
                ))
            checkpoint = latest_policy_checkpoint(training_root)
            if checkpoint is not None:
                command.extend(("--resume-checkpoint", str(checkpoint.resolve())))
            started = time.monotonic()
            code = run(command, repo=repo, log=logs_root / "training.log")
            event(
                timing,
                stage="train",
                attempt=attempt,
                duration_s=time.monotonic() - started,
                return_code=code,
            )
            if code == 0 and complete.is_file():
                break
            if attempt == max_attempts:
                raise RuntimeError(f"v2 training failed for {args.object}")
            time.sleep(10)
    training = json.loads(complete.read_text())
    checkpoint = Path(training["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != training["checkpoint_sha256"]:
        raise RuntimeError("v2 completed checkpoint is missing or changed")
    if accepted_count(args.root, args.object) < args.target:
        command = [
            str(args.isaac_python),
            "-m",
            "migration_4090.xhand_rl_embedded.collect",
            "--manifest",
            str(args.manifest.resolve()),
            "--object",
            args.object,
            "--nominal-dataset",
            str(args.nominal.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--root",
            str(args.root.resolve()),
            "--target",
            str(args.target),
            "--num-envs",
            str(args.collect_num_envs),
            "--device",
            args.device,
            "--headless",
            "--kit-portable-root",
            str(kit_portable_root),
        ]
        if args.penetration_reward_weight is not None:
            command.extend((
                "--penetration-reward-weight",
                str(args.penetration_reward_weight),
            ))
        started = time.monotonic()
        code = run(command, repo=repo, log=logs_root / "collect.log")
        event(timing, stage="direct_collect", duration_s=time.monotonic() - started, return_code=code)
        if code != 0:
            raise RuntimeError(f"v2 direct collector failed for {args.object}")


if __name__ == "__main__":
    main()
