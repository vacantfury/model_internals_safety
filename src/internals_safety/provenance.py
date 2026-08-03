"""Run provenance — the record that makes a number re-derivable.

Canonical schema and rationale: `.claude/skills/reproducible-run-logging`. This
module is that skill's implementation, shared by every phase entrypoint rather
than pasted into each.

**The one rule that carries the weight: a git hash is only honest if the working
tree was clean when the run started.** With uncommitted edits at launch, the
recorded hash points at code that is not what ran, and the record lies quietly.
So `guard_working_tree` is not advisory on the machine that produces results.

Posture, per the skill and specific to this project: cluster runs **refuse**
unless `--allow-dirty`, local CPU/MPS debugging runs **warn and record**. Two
reasons this project is strict. Cluster runs pass the family's experiment-run
approval gate, so they are few, deliberate and expensive; and the paper's
central object is a *regime label* built by combining four measurements on one
prompt, so a silent code change between measurements corrupts the label rather
than merely perturbing a number.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internals_safety.paths import PROJECT_ROOT

# Devices whose runs produce numbers the paper might report. Anything else is a
# laptop debugging run, where blocking on a dirty tree buys nothing.
RESULT_BEARING_DEVICES = frozenset({"cuda"})


class DirtyWorkingTree(RuntimeError):
    """Raised when a result-bearing run is launched from an uncommitted tree."""


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT, check=False
        )
    except FileNotFoundError:  # pragma: no cover - git absent
        return ""
    return completed.stdout.strip()


def git_hash() -> str:
    return _git("rev-parse", "HEAD")


def is_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def file_digest(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def guard_working_tree(device: str, allow_dirty: bool = False) -> bool:
    """Enforce the dirty-tree posture. Returns whether the tree is dirty.

    Raises `DirtyWorkingTree` for a result-bearing device unless `allow_dirty`;
    warns to stderr otherwise. Never silently proceeds.
    """
    dirty = is_dirty()
    if not dirty:
        return False
    if device in RESULT_BEARING_DEVICES and not allow_dirty:
        raise DirtyWorkingTree(
            "working tree has uncommitted changes, so the git hash this run would "
            f"record does not describe the code that runs (device={device}). Commit "
            "first, or pass --allow-dirty to record the diff alongside the results."
        )
    print(
        f"WARNING: uncommitted changes at run start (device={device}); "
        "the diff is recorded in results.json alongside the hash.",
        file=sys.stderr,
    )
    return True


def capture_provenance(
    config: dict[str, Any],
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The run record's provenance half, per the skill's schema.

    `config` should be the *resolved* config the run actually used — snapshot the
    frozen pydantic objects with `.model_dump()`; re-reading the YAML would
    record the file rather than the run.
    """
    dirty = is_dirty()
    record: dict[str, Any] = {
        "git_hash": git_hash(),
        "git_dirty": dirty,
        "config": config,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_lock": {
            "path": "uv.lock",
            "sha256_16": file_digest(PROJECT_ROOT / "uv.lock"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if dirty:
        # The whole diff, not a path: the record has to stay self-contained when
        # results are rsynced down from the cluster without the tree.
        record["git_diff"] = _git("diff")
    if extra:
        record.update(extra)
    return record


def write_results(directory: Path, record: dict[str, Any]) -> Path:
    """Write `results.json` into a run directory, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
