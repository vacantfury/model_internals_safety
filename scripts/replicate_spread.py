#!/usr/bin/env python3
"""How far does a reported cell move when the SAME run is repeated? (TODO 84 #8.)

**The question, and why the paper cannot answer it today.** Table 1 reports one
refusal rate per (model, condition). Several runs on disk measured that same
cell, and the paper says neither which run it quotes nor how far the others sit
from it. A reviewer looking at the artifacts sees replicates and one published
number; the honest response is to publish the range.

**What it measures, and why the answer is not zero.** Generation is greedy
(`models/generate.py` defaults `do_sample=False`), the corpus digest is
identical across these runs, and the judge runs at temperature 0 — so a naive
reading predicts byte-identical replicates. They are not. Two routes move a
cell, and this script separates them by comparing the stored response text:

* **generation** — the same prompt to the same weights returns different text.
  Batch composition changes padding and reduction order, one greedy argmax
  flips, and the continuation diverges. Nothing here is sampled.
* **judge** — the verdict differs on response text that is byte-identical.
  Temperature 0 is not determinism.

The product is a MEASURED reproducibility floor for every rate in the paper, to
sit beside the bootstrap noise null, which is a model of sampling noise and
cannot see either route.

**Reads cached run records only — no model, no judge, no GPU, no spend.**
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from internals_safety.paths import RUNS_DIR


@dataclass(frozen=True)
class Replicate:
    """One run's reading of one (model, condition) cell."""

    run: str
    harmful_refusal_rate: float
    benign_refusal_rate: float

    @property
    def gap(self) -> float:
        return self.harmful_refusal_rate - self.benign_refusal_rate


@dataclass
class Reproducibility:
    """Response- and verdict-level agreement over every pair of replicates."""

    n_compared: int = 0  # plumbing: counter seed
    n_identical_response: int = 0  # plumbing: counter seed
    n_flip_same_text: int = 0  # plumbing: counter seed
    n_flip_diff_text: int = 0  # plumbing: counter seed

    @property
    def response_agreement(self) -> float | None:
        if self.n_compared == 0:
            return None
        return self.n_identical_response / self.n_compared

    @property
    def verdict_flip_rate(self) -> float | None:
        if self.n_compared == 0:
            return None
        return (self.n_flip_same_text + self.n_flip_diff_text) / self.n_compared


@dataclass
class Cell:
    """Every replicate of one (model, condition), plus their agreement."""

    model: str
    condition: str
    replicates: list[Replicate] = field(default_factory=list)
    reproducibility: Reproducibility = field(default_factory=Reproducibility)

    @property
    def gap_range(self) -> tuple[float, float] | None:
        if not self.replicates:
            return None
        gaps = [r.gap for r in self.replicates]
        return min(gaps), max(gaps)

    @property
    def gap_spread(self) -> float | None:
        """Max minus min across replicates — the number the paper owes.

        Reported as a RANGE and never as a standard error: these are whole-run
        repeats of one deterministic-by-construction pipeline, not draws from a
        population, so a parametric summary would name a distribution that does
        not exist.
        """
        span = self.gap_range
        return None if span is None else span[1] - span[0]


def _readings(record: dict) -> list[dict]:
    return record.get("readings") or []


def _load_cells(run_dir: Path, condition: str) -> dict[str, dict]:
    path = run_dir / "cells.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["prompt_id"]: row for row in rows if row.get("family") == condition}


def collect(run_dirs: list[Path]) -> dict[tuple[str, str], Cell]:
    """Group every run's behaviour readings into (model, condition) cells."""
    cells: dict[tuple[str, str], Cell] = {}
    per_run: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for run_dir in run_dirs:
        record_path = run_dir / "results.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        model = (record.get("plan") or {}).get("model") or run_dir.parent.name
        for reading in _readings(record):
            if reading.get("instrument") != "behavior":
                continue
            detail = reading.get("detail") or {}
            harmful, benign = detail.get("refusal_rate"), detail.get("benign_arm_refusal_rate")
            if harmful is None or benign is None:
                continue
            condition = detail.get("family")
            if condition is None:
                continue
            key = (model, condition)
            cells.setdefault(key, Cell(model=model, condition=condition))
            cells[key].replicates.append(
                Replicate(run=run_dir.name, harmful_refusal_rate=harmful, benign_refusal_rate=benign)
            )
            per_run[key].append(run_dir)

    for key, dirs in per_run.items():
        _, condition = key
        agreement = cells[key].reproducibility
        for left, right in itertools.combinations(sorted(set(dirs)), 2):
            a, b = _load_cells(left, condition), _load_cells(right, condition)
            for prompt_id in sorted(set(a) & set(b)):
                same_text = a[prompt_id].get("attack_response") == b[prompt_id].get("attack_response")
                flipped = bool(a[prompt_id].get("refused")) != bool(b[prompt_id].get("refused"))
                agreement.n_compared += 1
                agreement.n_identical_response += int(same_text)
                if flipped:
                    if same_text:
                        agreement.n_flip_same_text += 1
                    else:
                        agreement.n_flip_diff_text += 1
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="*",
        help="run directories; defaults to every phase-0 run on disk",
    )
    parser.add_argument("--condition", default=None, help="restrict to one encoding family")
    parser.add_argument(
        "--min-replicates",
        type=int,
        # definitional: below this there is no range to report, only a value.
        # Tuning path: none needed — 2 is the arity of a comparison, not a
        # calibrated cut, and raising it can only hide cells.
        default=2,
        help="only report cells measured by at least this many runs",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the table here")
    args = parser.parse_args(argv)

    run_dirs = args.run_dirs or sorted(p.parent for p in (RUNS_DIR / "phase0").glob("*/*/results.json"))
    cells = collect(list(run_dirs))

    rows = {
        f"{cell.model}/{cell.condition}": {
            "model": cell.model,
            "condition": cell.condition,
            "n_replicates": len(cell.replicates),
            "gap_min": cell.gap_range[0],
            "gap_max": cell.gap_range[1],
            "gap_spread": cell.gap_spread,
            "response_agreement": cell.reproducibility.response_agreement,
            "verdict_flip_rate": cell.reproducibility.verdict_flip_rate,
            "flips_on_identical_text": cell.reproducibility.n_flip_same_text,
            "flips_on_different_text": cell.reproducibility.n_flip_diff_text,
            "replicates": [
                {"run": r.run, "harmful": r.harmful_refusal_rate, "benign": r.benign_refusal_rate, "gap": r.gap}
                for r in sorted(cell.replicates, key=lambda r: r.run)
            ],
        }
        for cell in cells.values()
        if len(cell.replicates) >= args.min_replicates
        and (args.condition is None or cell.condition == args.condition)
    }

    header = f"{'model':22s} {'condition':18s} {'reps':>4s} {'gap range':>16s} {'spread':>7s} {'resp=':>6s} {'flip':>6s}"
    print(header)
    for name in sorted(rows):
        row = rows[name]
        agreement = row["response_agreement"]
        flip = row["verdict_flip_rate"]
        print(
            f"{row['model'][:22]:22s} {row['condition'][:18]:18s} {row['n_replicates']:4d} "
            f"[{row['gap_min']:+.3f}, {row['gap_max']:+.3f}] {row['gap_spread']:7.3f} "
            f"{'n/a' if agreement is None else f'{agreement:.1%}':>6s} "
            f"{'n/a' if flip is None else f'{flip:.2%}':>6s}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
