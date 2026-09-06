#!/usr/bin/env python3
"""Report v2 training and direct-collection progress."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .contracts import checkpoint_iteration, latest_policy_checkpoint


def _object_process_live(object_name: str) -> bool:
    """Return whether a current v2 bridge/trainer for this object exists.

    A detached screen and an old ``training_provenance.json`` are not proof
    that PPO is still running.  The dashboard uses this read-only process
    probe to avoid showing a stopped retry as live while the append-only
    supervisor is in its cooldown/relaunch window.
    """

    try:
        result = subprocess.run(
            ["ps", "-ww", "-eo", "comm=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # Match the complete value.  A substring test makes the ``sphere`` card
    # look live whenever the unrelated ``sphere_small`` worker is running
    # (``--object sphere`` is a prefix of ``--object sphere_small``).  The
    # dashboard must not attribute another object's PPO event to this card.
    object_pattern = re.compile(rf"(?:^|\s)--object(?:=|\s+){re.escape(object_name)}(?:\s|$)")
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if not fields or not fields[0].startswith("python"):
            # A detached SCREEN/bash wrapper contains the module text in its
            # command line but is not a live Isaac/PPO worker.
            continue
        line = fields[1] if len(fields) > 1 else ""
        if "migration_4090.xhand_rl_" not in line:
            continue
        if not object_pattern.search(line):
            continue
        if any(
            module in line
            for module in (
                "xhand_rl_embedded.bridge_worker",
                "xhand_rl_embedded.train",
                "xhand_rl_curriculum.train",
                "xhand_rl_embedded.collect",
            )
        ):
            return True
    return False


def _gpu_snapshot() -> dict:
    """Return a short, read-only GPU health snapshot for the live pages.

    A running PPO event file is not evidence that the CUDA device is still
    usable: after a driver reset the last event can remain fresh for a short
    time while all Isaac workers are already gone.  Keep this check separate
    from acceptance and never infer success from it.
    """

    checked_at = time.time()
    nodes = ["/dev/nvidiactl", "/dev/nvidia-uvm"]
    node_present = all(os.path.exists(path) for path in nodes)
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "checked_at": checked_at,
            "device_nodes_present": node_present,
            "gpu_count": 0,
            "devices": [],
            "reason": f"nvidia-smi probe failed: {exc}",
        }
    devices = [
        line.strip()
        for line in probe.stdout.splitlines()
        if line.strip() and "NVIDIA-SMI has failed" not in line
    ]
    available = probe.returncode == 0 and bool(devices) and node_present
    reason = "ok" if available else (
        probe.stderr.strip()
        or probe.stdout.strip()
        or "GPU device/driver unavailable"
    )
    return {
        "available": available,
        "checked_at": checked_at,
        "device_nodes_present": node_present,
        "gpu_count": len(devices) if probe.returncode == 0 else 0,
        "devices": devices,
        "reason": reason,
    }


def _handoff_started_name(marker_name: str) -> str:
    """Map an append-only handoff request to its generation lock."""
    for version in ("V24", "V23", "V22", "V21", "V20", "V19", "V18", "V17", "V16", "V15", "V14", "V13", "V12", "V11", "V10", "V9", "V8", "V7", "V6", "V5", "V4", "V3", "V2"):
        if marker_name.endswith(f"_{version}.json"):
            return f"SUPERVISOR_HANDOFF_STARTED_{version}"
    return "SUPERVISOR_HANDOFF_STARTED"


def _maybe_handoff_reconcile_supervisor(root: Path) -> None:
    """Apply an explicit one-shot supervisor reload on the Isaac host.

    ``status_monitor_loop.sh`` starts this module afresh every minute, so it
    can reload a modified shell supervisor even when the old detached screen
    has not exited.  The request/lock files are append-only and the operation
    is gated on a healthy six-GPU host; local read-only status checks without
    NVIDIA devices therefore do not start anything.
    """
    marker = next(
        (
            candidate
            for candidate in (
                root / "SUPERVISOR_HANDOFF_REQUEST_V24.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V23.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V22.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V21.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V20.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V19.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V18.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V17.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V16.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V15.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V14.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V13.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V12.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V11.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V10.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V9.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V8.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V7.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V6.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V5.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V4.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V3.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V2.json",
                root / "SUPERVISOR_HANDOFF_REQUEST.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if marker is None:
        return
    try:
        payload = json.loads(marker.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if payload.get("root") and Path(str(payload["root"])).resolve() != root.resolve():
        return
    if payload.get("do_not_delete") is not True:
        return
    gpu = _gpu_snapshot()
    if not gpu.get("available") or int(gpu.get("gpu_count", 0)) < 6:
        return
    lock = root / _handoff_started_name(marker.name)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"time": time.time(), "source": "status_monitor"}, handle)
            handle.write("\n")
        log = root / "logs" / "reconcile_target100.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} status_monitor_supervisor_handoff "
                f"marker={marker.name}\n"
            )
        for session in (
            "xhand_formal_embedded_reconcile_target100",
            "xhand_formal_v5r_watchdog",
        ):
            subprocess.run(
                ["screen", "-S", session, "-X", "quit"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        repo = Path(__file__).resolve().parents[2]
        supervisor = repo / "migration_4090" / "xhand_rl_embedded" / "reconcile_target100.sh"
        command = f"exec '{supervisor}' >> '{log}' 2>&1"
        subprocess.run(
            [
                "screen",
                "-dmS",
                "xhand_formal_embedded_reconcile_target100",
                "bash",
                "-lc",
                command,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return


def _training_scalar_snapshot(event_dir: Path) -> dict:
    """Read the active PPO scalars for the live dashboard.

    Collection heartbeats are intentionally kept separate from PPO training.
    During a retry there may be no fresh heartbeat at all, so exposing only
    heartbeat diagnostics makes a healthy trainer look idle and can display a
    stale collector's contact numbers as if they belonged to the current
    checkpoint.  This compact snapshot is read-only and is written into the
    status JSON for the web page; TensorBoard remains the detailed source.
    """

    event_files = list(event_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    current = max(event_files, key=lambda path: path.stat().st_mtime)
    try:
        accumulator = EventAccumulator(str(current), size_guidance={"scalars": 0})
        accumulator.Reload()
    except Exception:
        return {}

    def latest(tag: str):
        values = accumulator.Scalars(tag) if tag in accumulator.Tags().get("scalars", []) else []
        if not values:
            return None
        # RSL-RL can append a resumed/replayed scalar with a lower step after
        # a higher-step scalar.  ``values[-1]`` is therefore not necessarily
        # the newest observation.  The live dashboard must follow write time
        # while TensorBoard remains the detailed step-based view.
        event = max(values, key=lambda item: item.wall_time)
        return {
            "step": int(event.step),
            "value": float(event.value),
            "wall_time": float(event.wall_time),
        }

    def maximum(tag: str):
        values = accumulator.Scalars(tag) if tag in accumulator.Tags().get("scalars", []) else []
        if not values:
            return None
        best = max(values, key=lambda item: item.value)
        return {"step": int(best.step), "value": float(best.value)}

    scalar_tags = (
        "Train/mean_reward",
        "reward/total_scaled",
        "reward/approach_proximity",
        "reward/bilateral_contact",
        "reward/contact_continuity",
        "reward/lift_height",
        "reward/gravity_margin",
        "reward/force_closure",
        "reward/hard_gate_frontier",
        "reward/disturbance_direction",
        "embedded/gate_bilateral_contact_continuity_fraction",
        "embedded/gate_lift_height_fraction",
        "embedded/gate_gravity_hold_fraction",
        "embedded/gate_physx_penetration_fraction",
        "embedded/gate_formal_force_closure_fraction",
        "embedded/force_closure_evaluated_fraction",
        "embedded/force_closure_reward_ready_fraction",
        "embedded/terminal_success_fraction",
    )
    metrics = {tag: latest(tag) for tag in scalar_tags}
    metrics = {tag: value for tag, value in metrics.items() if value is not None}
    prefixes = (
        "contact",
        "lifted",
        "clear",
        "stable_force_closure",
        "disturbed",
        "ablated",
    )
    prefix_current = {}
    prefix_max = {}
    for level, name in enumerate(prefixes, start=1):
        tag = f"embedded/hard_prefix_{name}_fraction"
        current_value = latest(tag)
        max_value = maximum(tag)
        if current_value is not None:
            prefix_current[name] = current_value
        if max_value is not None:
            prefix_max[name] = max_value
    observed_levels = [
        level
        for level, name in enumerate(prefixes, start=1)
        if prefix_max.get(name, {}).get("value", 0.0) > 0.0
    ]
    # The ordered prefix is the operator-facing bottleneck.  Per-gate
    # fractions can be non-zero on different environments, while this value
    # only advances when the same trajectory has passed every earlier gate.
    current_prefix_level = 0
    for name in prefixes:
        if prefix_current.get(name, {}).get("value", 0.0) > 0.0:
            current_prefix_level += 1
        else:
            break
    # A boolean prefix level detects a rare exact trajectory, but it is not a
    # stability/confidence measure. Keep the coverage separately so one
    # trajectory in a large rollout cannot be presented as a stable basin.
    current_prefix_coverage = (
        float(prefix_current[prefixes[current_prefix_level - 1]]["value"])
        if current_prefix_level > 0
        else 0.0
    )
    current_prefix_min_fraction = (
        min(
            float(prefix_current[name]["value"])
            for name in prefixes[:current_prefix_level]
            if name in prefix_current
        )
        if current_prefix_level > 0
        else 0.0
    )
    prefix_stability = (
        "stable"
        if current_prefix_min_fraction >= 0.05
        else ("rare" if current_prefix_min_fraction > 0.0 else "none")
    )
    bottleneck = (
        prefixes[current_prefix_level]
        if current_prefix_level < len(prefixes)
        else "terminal_success"
    )
    mean_reward = metrics.get("Train/mean_reward", {}).get("value")
    terminal_success = metrics.get("embedded/terminal_success_fraction", {}).get("value", 0.0)
    force_closure_reward = metrics.get("reward/force_closure", {}).get("value", 0.0)
    formal_force_closure = metrics.get(
        "embedded/gate_formal_force_closure_fraction", {}
    ).get("value", 0.0)
    force_closure_evaluated = metrics.get(
        "embedded/force_closure_evaluated_fraction", {}
    ).get("value", 0.0)
    force_closure_ready = metrics.get(
        "embedded/force_closure_reward_ready_fraction", {}
    ).get("value", 0.0)
    observed_iteration = max(
        (entry["step"] for entry in metrics.values()),
        default=0,
    )
    # Expose a small diagnostic so the UI can warn that step-based TensorBoard
    # lines may contain resumed/out-of-order points.  This does not alter the
    # training data or acceptance logic.
    step_order_warning = False
    for tag in scalar_tags:
        values = accumulator.Scalars(tag) if tag in accumulator.Tags().get("scalars", []) else []
        ordered = sorted(values, key=lambda item: item.wall_time)
        if any(curr.step < prev.step for prev, curr in zip(ordered, ordered[1:])):
            step_order_warning = True
            break
    return {
        "event_file": str(current.resolve()),
        "event_age_s": max(time.time() - current.stat().st_mtime, 0.0),
        "step_order_warning": step_order_warning,
        "iteration": max(
            (entry["step"] for entry in metrics.values()),
            default=None,
        ),
        "metrics": metrics,
        "prefix_current": prefix_current,
        "prefix_max": prefix_max,
        "max_prefix_level": max(observed_levels, default=0),
        "current_prefix_level": current_prefix_level,
        "current_prefix_coverage": current_prefix_coverage,
        "current_prefix_min_fraction": current_prefix_min_fraction,
        "prefix_stability": prefix_stability,
        "current_prefix_bottleneck": bottleneck,
        "force_closure_reward_gate_gap": bool(
            force_closure_reward > 0.05
            and formal_force_closure < 1.0e-6
            and (
                force_closure_ready > 0.0
                or force_closure_evaluated > 0.25
            )
        ),
        "prefix_retention_gap": bool(
            max(
                (entry.get("value", 0.0) for entry in prefix_max.values()),
                default=0.0,
            ) >= 0.05
            and current_prefix_min_fraction < 0.05
        ),
        "reward_gate_divergence": bool(
            mean_reward is not None
            and mean_reward > 0.0
            # Positive dense shaping is expected during the warm-up window.
            # Call it divergence only after enough updates have elapsed and
            # the same-trajectory prefix is still effectively absent/rare.
            and observed_iteration >= 100
            and terminal_success <= 0.0
            and current_prefix_min_fraction < 0.05
        ),
    }


def worker_status(root: Path, object_name: str, accepted: int) -> dict:
    run = root / "runs" / object_name
    training = run / "training"
    curriculum = run / "curriculum"
    # Resolve the active nominal lineage before selecting an attempt.  If all
    # attempts in the current lineage have just been marked ABANDONED, falling
    # back to an older non-abandoned v4b/v3 attempt makes the dashboard look as
    # if stale training is still current.  Keep the newest matching abandoned
    # attempt as read-only evidence instead; the stage will be reported as
    # stopped_waiting_reconcile and no acceptance/launch decision is changed.
    try:
        root_marker = json.loads((root / "EMBEDDED_RL_PIPELINE_ROOT.json").read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        root_marker = {}
    nominal_root = str(root_marker.get("nominal_root", ""))
    target_nominal = (
        str((Path(nominal_root) / f"{object_name}.pt").resolve())
        if nominal_root
        else ""
    )

    def attempt_nominal(path: Path) -> str:
        candidates = [path / "training_provenance.json"]
        candidates.extend(sorted(path.glob("curriculum_training_attempt_*.json"), reverse=True))
        for metadata in candidates:
            if not metadata.is_file():
                continue
            try:
                payload = json.loads(metadata.read_text())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            value = payload.get("nominal_dataset")
            if value:
                return str(Path(str(value)).resolve())
        return ""

    def select_attempts(pattern: str, primary: Path) -> list[Path]:
        def order(path: Path) -> tuple[int, str]:
            try:
                return (int(path.name.rsplit("_", 1)[1]), path.name)
            except (IndexError, ValueError):
                return (-1, path.name)

        all_paths = [primary] + sorted(run.glob(pattern), key=order)
        active_paths = [path for path in all_paths if not (path / "ABANDONED.json").is_file()]
        matching_active = [path for path in active_paths if target_nominal and attempt_nominal(path) == target_nominal]
        if matching_active:
            return matching_active
        matching_any = [path for path in all_paths if target_nominal and attempt_nominal(path) == target_nominal]
        if matching_any:
            return matching_any
        return active_paths or all_paths
    # The reconcile supervisor keeps every PPO retry in its own immutable
    # directory.  Report the newest attempt instead of continuing to expose
    # the completed primary formal run while a retry is actively training.
    attempts = select_attempts("training_retry_*", training)
    active_training = attempts[-1]
    formal_provenance_path = active_training / "training_provenance.json"
    curriculum_roots = select_attempts("curriculum_retry_*", curriculum)
    active_curriculum = curriculum_roots[-1]
    curriculum_attempts = sorted(active_curriculum.glob("curriculum_training_attempt_*.json"))
    curriculum_provenance_path = curriculum_attempts[-1] if curriculum_attempts else None
    # A corrected nominal may be training in a fresh curriculum directory
    # while the original ``training/`` provenance still exists.  Prefer the
    # newer live curriculum in that case; once its formal retry starts, the
    # new retry provenance becomes newer and takes precedence again.
    curriculum_is_newer = bool(
        curriculum_provenance_path is not None
        and curriculum_provenance_path.is_file()
        and (
            not formal_provenance_path.is_file()
            or curriculum_provenance_path.stat().st_mtime > formal_provenance_path.stat().st_mtime
        )
    )
    # Provenance timestamps can be misleading during a live bridge: a
    # resumable curriculum attempt may have a newer metadata file even after
    # formal PPO has started.  Prefer the event stream that is actually being
    # updated.  Otherwise the dashboard/TensorBoard selector can show an old
    # curriculum curve while the card says ``training_retry``.
    formal_event_mtime = max(
        (path.stat().st_mtime for path in run.glob("training_retry_*/events.out.tfevents.*")
         if not (path.parent / "ABANDONED.json").is_file()),
        default=0.0,
    )
    formal_event_mtime = max(
        formal_event_mtime,
        max((path.stat().st_mtime for path in training.glob("events.out.tfevents.*")), default=0.0),
    )
    curriculum_event_mtime = max(
        (path.stat().st_mtime for path in run.glob("curriculum*/events.out.tfevents.*")
         if not (path.parent / "ABANDONED.json").is_file()),
        default=0.0,
    )
    if formal_event_mtime > curriculum_event_mtime:
        curriculum_is_newer = False
    use_curriculum = curriculum_is_newer
    # The bridge writes curriculum provenance per resumable chunk, while formal
    # training has one stable training_provenance.json.  Prefer formal unless
    # the fresh curriculum is demonstrably newer.
    provenance_path = (
        curriculum_provenance_path
        if use_curriculum
        else (formal_provenance_path if formal_provenance_path.is_file() else curriculum_provenance_path)
    )
    provenance = json.loads(provenance_path.read_text()) if provenance_path is not None and provenance_path.is_file() else {}
    # The active strategy is append-only and can change nominal lineage while
    # old retries remain on disk.  Surface a mismatch explicitly so the web
    # page cannot present an obsolete worker as progress toward the current
    # mesh-aware target.
    try:
        root_marker = json.loads((root / "EMBEDDED_RL_PIPELINE_ROOT.json").read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        root_marker = {}
    target_nominal = ""
    nominal_root = str(root_marker.get("nominal_root", ""))
    if nominal_root:
        target_nominal = str((Path(nominal_root) / f"{object_name}.pt").resolve())
    active_nominal = str(Path(str(provenance.get("nominal_dataset", ""))).resolve()) if provenance.get("nominal_dataset") else ""
    nominal_mismatch = bool(target_nominal and active_nominal and target_nominal != active_nominal)
    target_iterations = int(provenance.get("max_iterations", 0))
    checkpoint_dir = active_curriculum if use_curriculum else active_training
    checkpoint = latest_policy_checkpoint(checkpoint_dir)
    latest_iteration = checkpoint_iteration(checkpoint) if checkpoint is not None else None
    rate = None
    success_fraction = None
    success_fraction_max = None
    success_fraction_max_iteration = None
    event_dir = active_curriculum if use_curriculum else active_training
    training_metrics = _training_scalar_snapshot(event_dir)
    event_files = list(event_dir.glob("events.out.tfevents.*"))
    if event_files:
        current = max(event_files, key=lambda path: path.stat().st_mtime)
        try:
            accumulator = EventAccumulator(str(current), size_guidance={"scalars": 0})
            accumulator.Reload()
            fps = accumulator.Scalars("Perf/total_fps")
            if fps:
                fps_ordered = sorted(fps, key=lambda event: event.wall_time)
                latest_iteration = fps_ordered[-1].step
                if len(fps_ordered) >= 2:
                    elapsed = fps_ordered[-1].wall_time - fps_ordered[0].wall_time
                    advanced = fps_ordered[-1].step - fps_ordered[0].step
                    if elapsed > 0.0 and advanced > 0:
                        rate = 3600.0 * advanced / elapsed
            values = accumulator.Scalars("embedded/terminal_success_fraction")
            if values:
                current_success = max(values, key=lambda event: event.wall_time)
                success_fraction = max(0.0, min(1.0, current_success.value))
                # A restarted run can leave older event files behind.  Use
                # the current file for the live value/rate, but merge this
                # scalar across all event files for historical evidence so a
                # brief success is not hidden after resume.
                merged_values = []
                for event_path in event_files:
                    try:
                        old = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
                        old.Reload()
                        merged_values.extend(old.Scalars("embedded/terminal_success_fraction"))
                    except Exception:
                        continue
                if merged_values:
                    best = max(merged_values, key=lambda event: event.value)
                    success_fraction_max = max(0.0, min(1.0, best.value))
                    success_fraction_max_iteration = int(best.step)
        except Exception:
            pass
    eta = None
    if rate and latest_iteration is not None and target_iterations:
        eta = max(target_iterations - latest_iteration - 1, 0) / rate
    if use_curriculum and (active_curriculum / "CURRICULUM_TRAINING_COMPLETE.json").is_file():
        stage = "formal_pending"
    elif use_curriculum and curriculum_provenance_path is not None:
        stage = "curriculum_training"
    elif (run / "COMPLETE.json").is_file():
        stage = "complete"
    elif active_training != training and (active_training / "TRAINING_COMPLETE.json").is_file():
        stage = "direct_collect_retry"
    elif (training / "TRAINING_COMPLETE.json").is_file() and active_training == training:
        stage = "direct_collect"
    # Formal provenance is written before the formal PPO process starts and
    # remains present while it is running.  Check it before the curriculum
    # completion marker; otherwise an active formal worker is incorrectly
    # reported as merely waiting for formal training.
    elif formal_provenance_path.is_file() and not use_curriculum:
        stage = "training_retry" if active_training != training else "training"
    elif (active_curriculum / "CURRICULUM_TRAINING_COMPLETE.json").is_file():
        stage = "formal_pending"
    elif curriculum_provenance_path is not None:
        stage = "curriculum_training"
    else:
        stage = "not_started"
    process_live = _object_process_live(object_name)
    # If no current bridge/trainer exists and the selected event file has not
    # changed recently, expose the stopped state explicitly.  This prevents a
    # historical retry with a nonzero reward/prefix maximum from being read as
    # an active run during worker rotation.
    newest_event_mtime = max(
        (path.stat().st_mtime for path in event_dir.glob("events.out.tfevents.*")),
        default=0.0,
    )
    if not process_live and time.time() - newest_event_mtime > 120.0:
        stage = "stopped_waiting_reconcile"
    heartbeat_path = run / "collection_heartbeat.jsonl"
    last_heartbeat = None
    if heartbeat_path.is_file():
        try:
            rows = [json.loads(line) for line in heartbeat_path.read_text().splitlines() if line.strip()]
            last_heartbeat = rows[-1] if rows else None
        except (OSError, TypeError, ValueError):
            last_heartbeat = None
    collection = None
    if last_heartbeat is not None:
        heartbeat_time = float(last_heartbeat.get("time", 0.0))
        collection = {
            "attempt_id": last_heartbeat.get("attempt_id"),
            "seed": last_heartbeat.get("seed"),
            "policy_steps": last_heartbeat.get("policy_steps", 0),
            "max_steps": last_heartbeat.get("max_steps"),
            "accepted": last_heartbeat.get("accepted", accepted),
            "final": bool(last_heartbeat.get("final", False)),
            "heartbeat_time": heartbeat_time or None,
            "heartbeat_age_s": None if not heartbeat_time else max(time.time() - heartbeat_time, 0.0),
            "live": bool(heartbeat_time and time.time() - heartbeat_time < 180.0),
            "diagnostics": last_heartbeat.get("diagnostics", {}),
            "diagnostics_max": last_heartbeat.get("diagnostics_max", {}),
            "diagnostics_min": last_heartbeat.get("diagnostics_min", {}),
        }
    return {
        "stage": stage,
        "process_live": process_live,
        "accepted": accepted,
        "latest_iteration": latest_iteration,
        "target_iterations": target_iterations or None,
        "iteration_rate_per_hour": rate,
        "training_eta_hours": eta,
        "embedded_terminal_success_fraction": success_fraction,
        "embedded_terminal_success_fraction_max_current_event": success_fraction_max,
        "embedded_terminal_success_max_iteration_current_event": success_fraction_max_iteration,
        "latest_checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "training_metrics": training_metrics,
        "step_order_warning": bool(training_metrics.get("step_order_warning", False)),
        "reward_revision": provenance.get("reward_shaping_revision")
        or provenance.get("reward_revision"),
        "ppo_revision": provenance.get("ppo_revision"),
        "device": provenance.get("device"),
        "residual_activation_phase": provenance.get("residual_activation_phase"),
        "formal_residual_activation_phase_override": provenance.get(
            "formal_residual_activation_phase_override"
        ),
        "nominal_dataset": provenance.get("nominal_dataset"),
        "target_nominal_dataset": target_nominal or None,
        "nominal_mismatch": nominal_mismatch,
        "nominal_lineage": (
            "meshaware_v4c_closed_hand"
            if "objectflow_xhand_rl_meshaware_nominal_v4c_closed_20260826" in str(provenance.get("nominal_dataset", ""))
            else (
                "meshaware_v4b"
                if "objectflow_xhand_rl_meshaware_nominal_v4b_20260826" in str(provenance.get("nominal_dataset", ""))
                else (
                    "meshaware_v4"
                    if "objectflow_xhand_rl_meshaware_nominal_v4_20260826" in str(provenance.get("nominal_dataset", ""))
                    else (
                        "meshaware_v3"
                        if "objectflow_xhand_rl_meshaware_nominal_v3_20260825" in str(provenance.get("nominal_dataset", ""))
                        else (
                            "meshaware_v2"
                            if "objectflow_xhand_rl_meshaware_nominal_v2_20260825" in str(provenance.get("nominal_dataset", ""))
                            else "legacy_or_previous"
                        )
                    )
                )
            )
        ),
        "collection": collection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--window-minutes", type=float, default=60.0)
    args = parser.parse_args()
    _maybe_handoff_reconcile_supervisor(args.root)
    marker_path = args.root / "EMBEDDED_RL_PIPELINE_ROOT.json"
    if not marker_path.is_file():
        raise RuntimeError(f"not a v2 embedded root: {args.root}")
    marker = json.loads(marker_path.read_text())
    cutoff = time.time() - args.window_minutes * 60.0
    counts = {}
    strict_counts = {}
    strict_rejected = {}
    strict_pending = {}
    recent = {}
    workers = {}
    for name in marker["objects"]:
        receipts = list((args.root / "accepted" / name).glob("sample_*/receipt.json"))
        counts[name] = len(receipts)
        strict_count = 0
        rejected_count = 0
        pending_count = 0
        for receipt in receipts:
            audit_receipt = receipt.parent / "strict_post_audit" / "receipt.json"
            if not audit_receipt.is_file():
                pending_count += 1
                continue
            try:
                audit = json.loads(audit_receipt.read_text())
            except (OSError, TypeError, ValueError):
                pending_count += 1
                continue
            if audit.get("accepted") is True:
                strict_count += 1
            elif audit.get("accepted") is False:
                rejected_count += 1
            else:
                pending_count += 1
        strict_counts[name] = strict_count
        strict_rejected[name] = rejected_count
        strict_pending[name] = pending_count
        recent[name] = sum(path.stat().st_mtime >= cutoff for path in receipts)
        workers[name] = worker_status(args.root, name, counts[name])
    hours = max(args.window_minutes / 60.0, 1.0e-9)
    handoff_requests = [
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V24.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V23.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V22.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V21.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V20.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V19.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V18.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V17.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V16.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V15.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V14.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V13.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V12.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V11.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V10.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V9.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V8.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V7.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V6.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V5.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V4.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V3.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST_V2.json",
        args.root / "SUPERVISOR_HANDOFF_REQUEST.json",
    ]
    strategy_warnings = {
        f"{name}:active_nominal_mismatch"
        for name, worker in workers.items()
        if worker.get("nominal_mismatch")
    }
    active_handoff = next((path for path in handoff_requests if path.is_file()), None)
    if active_handoff is not None:
        started = args.root / _handoff_started_name(active_handoff.name)
        if started.is_file():
            strategy_warnings.add("supervisor_handoff_started_reload_reconcile_target100")
        else:
            strategy_warnings.add("supervisor_handoff_requested_reload_reconcile_target100")
    result = {
        "schema": "xhand_rl_embedded_generation_status_v2",
        "root": str(args.root.resolve()),
        "acceptance_profile": "embedded_physics_pass",
        "reward_revision": marker["reward_revision"],
        "ppo_revision": marker["ppo_revision"],
        "training_reward_scale": marker["training_reward_scale"],
        "strict_visual_mesh_accepted": False,
        "omitted_expensive_gates": marker["omitted_expensive_gates"],
        "target_per_object": marker["target_per_object"],
        "accepted": counts,
        "accepted_total": sum(counts.values()),
        "strict_accepted": strict_counts,
        "strict_accepted_total": sum(strict_counts.values()),
        "strict_rejected": strict_rejected,
        "strict_pending": strict_pending,
        "strict_objects_with_success": sum(value > 0 for value in strict_counts.values()),
        "strict_first_success_complete": all(value > 0 for value in strict_counts.values()),
        "recent_window_minutes": args.window_minutes,
        "recent_accepted": recent,
        "recent_rate_per_hour": {name: value / hours for name, value in recent.items()},
        "workers": workers,
        "active_reward_revisions": sorted(
            {
                worker["reward_revision"]
                for worker in workers.values()
                if worker.get("reward_revision")
            }
        ),
        "active_ppo_revisions": sorted(
            {
                worker["ppo_revision"]
                for worker in workers.values()
                if worker.get("ppo_revision")
            }
        ),
        "strategy_warnings": sorted(strategy_warnings),
        "complete": all(value >= marker["target_per_object"] for value in counts.values()),
        "generated_at": time.time(),
        "gpu": _gpu_snapshot(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
