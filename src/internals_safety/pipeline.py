"""The run spine both papers share.

**Canonical design: `text_docs/shared/pipeline_architecture.md` §3.3.** Do not
re-derive the selection rule here; it is settled and it is short:

    the spine holds anything whose absence in ONE script would be a defect.

That is not a generality argument, it is one incident stated as a rule. The
length null reached `as6_guard_probe.py` on 2026-08-05 and
`phase0_regime_map.py` on 2026-08-06, so for a day the guard side carried a
mandatory control the target side lacked — **for a confound measured on the
target side.** Two scripts implementing the same sequence in their own words is
what made that representable.

**What is deliberately NOT here.** The instrument roster and the combination
step. AS-5 assigns four-regime cells, AS-6 assigns guard cells, and they run
different measurements; forcing those into the spine is how a shared runner
becomes a framework. Each script keeps its own `run_family` and passes it in.

**Form: plain functions, no class.** There is no `Pipeline` object, no lifecycle,
no config-driven runner, and adding one would be the failure mode this module was
weighed against. A file of functions is a module; that is the whole of it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from internals_safety.data import Prompt, prompt_set
from internals_safety.paths import OUTPUTS_DIR, run_dir


def add_common_arguments(parser: argparse.ArgumentParser, *, default_n_prompts: int) -> None:
    """The arguments every phase entrypoint must accept.

    Shared because each one encodes a rule rather than a preference, and a script
    that quietly lacked one would be a defect rather than a variant: `--dry-run`
    is how the approval gate gets its estimate without spending, `--allow-dirty`
    is the escape hatch on the provenance guard, `--allow-cpu` exists so a batch
    job cannot silently leave an allocated GPU idle, and `--outputs-dir` is what
    lets the cluster write to scratch instead of a small repo volume.
    """
    parser.add_argument("--families", nargs="+", default=None, help="default: every configured rung")
    parser.add_argument("--n-prompts", type=int, default=default_n_prompts)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the work this run would do and exit; loads no model and needs no keys",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record the diff and run anyway (refused by default on result-bearing devices)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run inside a SLURM job (refused by default: a batch job "
        "on CPU means a GPU was allocated and left idle)",
    )
    parser.add_argument("--refresh-activations", action="store_true", help="ignore the capture cache")
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="root for activations/ and runs/ (default: the repo's outputs/); "
        "point it at cluster scratch when the repo tree is on a small volume",
    )


def load_contrast_sets(
    harmful_set: str, harmless_set: str, n_prompts: int
) -> tuple[list[Prompt], list[Prompt]]:
    """The probe's two classes, with the size check that must never be skipped.

    Unequal classes would shift every AUROC by the base rate rather than by the
    signal, and the shift is invisible in the reported number. Raising here is
    the point: both scripts fit probes on these sets, so a mismatch is a defect
    in either, not a variant of one.
    """
    harmful = prompt_set(harmful_set, limit=n_prompts)
    harmless = prompt_set(harmless_set, limit=n_prompts)
    if len(harmful) != len(harmless):
        raise SystemExit(
            f"contrast sets differ in size ({len(harmful)} vs {len(harmless)}); the probe's "
            "classes must be matched — see conf/pilot.yaml"
        )
    return harmful, harmless


def resolve_run_paths(
    phase: str, name: str, run_name: str | None, outputs_dir: str | None
) -> tuple[Path, Path, str]:
    """`(run directory, activations dir, run name)`, directory created.

    The run name defaults to a UTC timestamp rather than a local one so runs
    launched from a laptop and from the cluster sort together.
    """
    outputs = Path(outputs_dir) if outputs_dir else OUTPUTS_DIR
    resolved = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = run_dir(phase, name, resolved, runs_dir=outputs / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    return directory, outputs / "activations", resolved


def run_families(
    families: Sequence[str],
    directory: Path,
    run_one: Callable[[str], dict[str, Any]],
    report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Drive the per-family loop with a crash-durable checkpoint after each rung.

    `run_one(family)` returns `{"cells": [...], "summary": {...}}`; `report` gets
    the summary for whatever the paper wants printed. Returns the summaries and
    the total elapsed seconds.

    **The checkpoint is why this is in the spine.** Both halves were earned by a
    real loss: the comprehension-band sweep was killed at the 8 h wall having
    COMPLETED `zero_width` on two models and recovered nothing — `cells.jsonl`
    was 0 bytes because Python's buffer had not filled, and `results.json` is
    only written after the loop. So each rung is written, flushed AND fsynced
    before the next begins: a SIGKILL at the wall does not run buffers out, and
    on a networked scratch filesystem a flush alone can still sit in the page
    cache. A rung that finished must survive the job that did not, or every
    wall-clock kill silently re-buys work already paid for.

    `elapsed_seconds` is stamped on every summary because `conf/cost.yaml`'s
    throughput numbers are assumptions with exactly one tuning path — a real run
    measuring them. Gather-and-cover: the instrumentation ships with the knob.
    """
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (directory / "cells.jsonl").open("w", encoding="utf-8") as handle, (
        directory / "summaries.partial.jsonl"
    ).open("w", encoding="utf-8") as partial_handle:
        for family in families:
            print(f"\n=== {family}", flush=True)
            family_started = time.perf_counter()
            result = run_one(family)

            for cell in result["cells"]:
                handle.write(json.dumps(cell, ensure_ascii=False) + "\n")
            _checkpoint(handle)

            summary = result["summary"]
            summary["elapsed_seconds"] = round(time.perf_counter() - family_started, 1)
            summaries.append(summary)
            partial_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            _checkpoint(partial_handle)

            if report is not None:
                report(result)
            print(f"    ({summary['elapsed_seconds'] / 60:.1f} min)", flush=True)
    return summaries, time.perf_counter() - started


def _checkpoint(handle) -> None:
    """flush + fsync. Both, always — see `run_families`."""
    handle.flush()
    os.fsync(handle.fileno())


def select_known(requested: Iterable[str] | None, available: Iterable[str], *, label: str) -> list[str]:
    """Validate a requested subset against what is configured, or take all of it.

    Fails on an unknown name rather than silently running a shorter ladder: a
    typo'd `--families` would otherwise produce a complete-looking run whose map
    is missing a rung nobody notices is absent.
    """
    known = list(available)
    if requested is None:
        return known
    chosen = list(requested)
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise SystemExit(f"unknown {label} {unknown}; configured: {sorted(known)}")
    return chosen
