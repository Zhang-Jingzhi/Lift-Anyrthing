"""Locked Omniverse launcher settings for concurrent v2 workers."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

from .contracts import KIT_SETTINGS_PROFILE, SINGLE_GPU_KIT_ARGS

AUTO_ROOT_DIRNAME = "xhand_rl_embedded_kit"
KEEP_KIT_ROOT_ENV = "XHAND_KEEP_KIT_ROOT"


def _force_rmtree(path: Path) -> None:
    """Remove a Kit portable root, including its mode-000 cache directories.

    Kit leaves some cache directories (for example
    ``data/documents/Kit/shared/screenshots``) with no permission bits, which
    makes a plain rmtree stop partway and strand the rest of the tree.  Restore
    traversal rights first: os.walk descends into a directory only after the
    entry that names it has been yielded, so chmod-ing it here is enough for
    the walk itself to reach the level below.
    """
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    for parent, dirnames, _ in os.walk(path, topdown=True, onerror=lambda _error: None):
        for name in dirnames:
            try:
                os.chmod(os.path.join(parent, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _process_is_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").is_dir()


def sweep_stale_auto_roots(base: Path) -> int:
    """Delete auto-generated roots whose owning process is gone.

    An auto root is named ``<label>_<pid>`` and is never reused, so a root
    whose pid is no longer running was stranded by an exited or killed
    process.  A live pid is always skipped, so a recycled pid can only ever
    cause a directory to be kept, never removed while it is in use.
    """
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return 0
    removed = 0
    for entry in entries:
        suffix = entry.name.rpartition("_")[2]
        if not suffix.isdigit() or not entry.is_dir():
            continue
        try:
            if _process_is_alive(int(suffix)):
                continue
        except (OSError, ValueError):
            continue
        _force_rmtree(entry)
        removed += 1
    return removed


def keep_kit_roots() -> bool:
    """True when the caller asked for every Kit root to be left on disk.

    The flag covers both the exit cleanup and the stale sweep, so a root kept
    for debugging is not removed by the next launch either.
    """
    return os.environ.get(KEEP_KIT_ROOT_ENV) == "1"


def configure_isolated_kit(args: Namespace, *, label: str) -> Path:
    requested = getattr(args, "kit_portable_root", None)
    auto_generated = requested is None
    if auto_generated:
        base = Path(tempfile.gettempdir()) / AUTO_ROOT_DIRNAME
        base.mkdir(parents=True, exist_ok=True)
        if not keep_kit_roots():
            sweep_stale_auto_roots(base)
        requested = base / f"{label}_{os.getpid()}"
    portable_root = Path(requested).expanduser().resolve()
    if any(character.isspace() for character in str(portable_root)):
        raise ValueError("Kit portable root must not contain whitespace")
    portable_root.mkdir(parents=True, exist_ok=True)
    args.kit_portable_root = portable_root
    args.kit_settings_profile = KIT_SETTINGS_PROFILE
    args.multi_gpu = False
    args.kit_args = f"{SINGLE_GPU_KIT_ARGS} --portable-root {portable_root}"
    if auto_generated and not keep_kit_roots():
        atexit.register(_force_rmtree, portable_root)
    return portable_root
