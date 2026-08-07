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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from internals_safety.measurements.contract import Reading, reportable_only, withheld_summary
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


# The run-record schema's version. Adopted 2026-08-06 from the guardrail sibling
# (`pipeline_convergence.md` §b), which has carried one since its first record
# while ours had none.
#
# ⚠️ Every record written before this date lacks the field, and that is exactly
# what it is for: a reader holding an old record can now tell it is old instead
# of assuming the current shape and finding a missing key at the worst moment.
# Absent means "pre-versioning", never "version 1".
#
# definitional: a schema identity, not a tunable — it changes when the shape
# changes, by hand, in the same commit. Tuning path: none; a swept schema version
# is a contradiction.
#
# v2 (2026-08-07, structure review): `config.pilot` became `config.corpus` when
# `conf/pilot.yaml` was renamed. Nothing in the repo reads that key, so no reader
# broke — which is exactly why the bump is worth making rather than skipping. A
# field renamed silently under a version that did not move is how a version field
# stops being trustworthy, and it only has to happen once.
SCHEMA_VERSION = 2


def upstream_ref(path: Path) -> dict[str, Any]:
    """Content-pin a file this run CONSUMED — lineage, not provenance.

    Adopted from the sibling's `upstream_ref{source_dir, results_sha256}`, and it
    closes a live gap rather than a hypothetical one: `scripts/rescore_ability.py`
    and `scripts/rebaseline_pilot.py` both read a PRIOR run's `cells.jsonl` and
    emit new numbers, with nothing in the output saying which bytes they read.
    That drift has already bitten once — `rescore_ability`'s self-check went red
    because `cells.jsonl` predated instrument fixes #1/#2, a fact established by
    inference rather than mechanically.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {"source": str(path), "sha256": digest, "exists": path.exists()}


def write_results(
    directory: Path,
    record: dict[str, Any],
    status: str = "success",
    warnings: Sequence[str] = (),
) -> Path:
    """Write `results.json` into a run directory, creating it if needed.

    `status` and `warnings` are recorded rather than printed (sibling parity,
    `pipeline_convergence.md` §b). Ours went to stderr, where a SLURM job's
    warnings end up in a `.err` file nobody opens beside the results — so a run
    that completed with a known caveat was indistinguishable from a clean one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        **record,
        "status": status,
        "warnings": list(warnings),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_run_record(
    directory: Path,
    record: dict[str, Any],
    readings: Sequence[Reading] = (),
    status: str = "success",
    warnings: Sequence[str] = (),
    primary_metric: str | None = None,
) -> Path:
    """`write_results` plus the typed-reading half of the schema.

    **Why this exists** (`pipeline_architecture.md` §4, answer 4). The provenance
    half of the run record has been schematised since this module was written;
    the RESULTS half never was — each script hand-built its own summary dict, and
    the two had already drifted. This is the seam that stops that: the
    reportable/withheld split is **computed** from the readings rather than
    curated by whoever wrote the script.

    The record gains two sections:

    - `readings` — every `Reading`, with its evidence, whether or not it is
      reportable. Nothing is dropped at write time.
    - `withheld` — instrument -> the reasons it failed, from
      `why_not_reportable()`.

    **`withheld` is the load-bearing one.** A run that reports three numbers and
    silently discards nine reads as "not significant" is indistinguishable from
    one that measured three things, and a withheld cell with no stated reason is
    how an instrument defect hides as a null result. Recording the complement is
    what makes a null honest.

    Passing no readings is allowed and writes neither section — a run whose
    instruments do not yet emit `Reading`s is not forced to fake them.
    """
    if readings:
        record = {
            **record,
            "readings": [asdict(reading) for reading in readings],
            "withheld": withheld_summary(readings),
            "n_reportable": len(reportable_only(readings)),
        }
    if primary_metric is not None:
        # Which number is the headline. Cheap, and meaningful here in a way it is
        # not for the sibling: naming a primary metric that is NOT in
        # `reportable_only` is a contradiction the record now makes visible.
        record = {**record, "primary_metric": primary_metric}
    return write_results(directory, record, status=status, warnings=warnings)
