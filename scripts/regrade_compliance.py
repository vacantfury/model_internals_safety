"""Re-score cached (B) cells against StrongREJECT's graded rubric — TODO 32.

**What is at stake: AS-5's headline number.** The paper's claim is that the (B)
decode-and-comply cell is populated. (B) is assigned from a BINARY HarmBench
verdict, and Souly et al. (arXiv 2402.10260, NeurIPS 2024) show binary judges
*overstate* ASR because an on-topic but vague or hallucinated response scores as
a success. Under an encoded attack that is the commonest near-success there is:
the model decodes, then produces something related and contentless. So no (B)
count may be reported as a point estimate until it has been through this.

Offline: everything needed is already in `cells.jsonl` (`plaintext`,
`attack_response`, `regime`, `refused`). **No GPU, no model load, no re-run** —
judge API calls only.

## The output is a CURVE, not a number

`substantive_rate` is reported across a range of bars rather than at one, and
`conf/judges.yaml`'s `substantive_quality_bar` is a PLACEHOLDER that this script
deliberately does not privilege. StrongREJECT never binarises its own score, so
there is no published bar to inherit, and the pilot already established the
house move for exactly this shape: report the (B) count across tau=0.75-0.90 and
state the invariance. A (B) claim that survives the whole plausible range needs
no tuned bar; one that does not is a finding ABOUT the bar, and quoting a single
value would bury it.

## (S) cells are the control, and they are free

Every run already contains (S) decode-and-refuse cells: the model read the
request and declined. Those should score at the FLOOR of the graded rubric,
because there is no harmful content in them to be convincing or specific about.
So regrading (S) alongside (B) costs a few more judge calls and buys a
discrimination check the rubric cannot otherwise get: **if (B) and (S) score
alike, the graded judge is not reading what we think it reads**, and no (B)
number from this pass may be reported at all. Same shape as the can't-decode
rungs serving as the deployment probe's noise floor — the control was already
in the data, unused.

    ./run python scripts/regrade_compliance.py --run-dir outputs/runs/phase0/... --dry-run
    ./run python scripts/regrade_compliance.py --run-dir outputs/runs/phase0/...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from internals_safety.config import load_judge_config
from internals_safety.judges import JudgeItem, StrongRejectJudge, refusal_disagreement
from internals_safety.judges.strongreject import RubricScoring
from internals_safety.paths import PROJECT_ROOT
from internals_safety.provenance import capture_provenance, guard_working_tree, write_run_record

PHASE = "regrade"

# Regimes worth judging. (B) is the object; (S) is the free discrimination
# control described in the module docstring.
# constant: the regime taxonomy's own labels (measurements/regimes.py), not a knob
REGIMES = ("B", "S")


def load_cells(run_dir: Path) -> list[dict]:
    path = run_dir / "cells.jsonl"
    if not path.exists():
        raise SystemExit(f"no cells.jsonl in {run_dir}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected(cells: list[dict]) -> list[dict]:
    """Cells with a regime we regrade AND an actual response to grade."""
    return [
        cell
        for cell in cells
        if cell.get("regime") in REGIMES and (cell.get("attack_response") or "").strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", action="append", required=True, type=Path,
        help="a phase-0 run directory containing cells.jsonl; repeatable",
    )
    parser.add_argument("--run-name", default="regrade")
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    judges = load_judge_config()
    bars = tuple(judges.substantive_quality_bars)

    cells: list[dict] = []
    for directory in args.run_dir:
        cells.extend(selected(load_cells(directory)))

    by_regime: dict[str, int] = defaultdict(int)
    for cell in cells:
        by_regime[cell["regime"]] += 1

    if args.dry_run:
        print(f"cells to regrade   {len(cells)}")
        for regime in REGIMES:
            print(f"  ({regime})              {by_regime[regime]}")
        print(f"judge calls        {len(cells)} (one per cell, model {judges.model})")
        print("GPU                none — offline against cached generations")
        if not by_regime["S"]:
            print("\n⚠️  No (S) cells selected. The discrimination control is EMPTY, so a")
            print("    (B) result from this pass would be unqualified — see module docs.")
        return 0

    # The guard runs on the real path only, and it is reached BEFORE any spend.
    # `--dry-run` returning above it is the documented trap (job 8995805): the
    # approval gate is built on --dry-run, so a green dry run is evidence about
    # argument parsing and nothing else. `tests/test_regrade_entrypoint.py` runs
    # this function for real.
    guard_working_tree("cpu", allow_dirty=args.allow_dirty)

    judge = StrongRejectJudge(judges)
    items = [
        JudgeItem(
            id=f"{cell['family']}::{cell['prompt_id']}",
            behavior=cell["plaintext"],          # PLAINTEXT — never the ciphertext
            response=cell["attack_response"],
        )
        for cell in cells
    ]
    scoring = judge.score(items)

    # Split the readings back out by regime for the control comparison.
    per_regime: dict[str, list] = defaultdict(list)
    for cell, score in zip(cells, scoring.scores):
        per_regime[cell["regime"]].append(score)

    report: dict[str, object] = {
        "phase": PHASE,
        "n_cells": len(cells),
        "parse_failure_rate": scoring.parse_failure_rate,
        "bars": list(bars),
        "by_regime": {},
        "provenance": capture_provenance("cpu"),
    }

    print(f"\ncells regraded      {len(cells)}")
    print(f"parse failures      {scoring.parse_failure_rate}")
    for regime in REGIMES:
        scores = per_regime.get(regime, [])
        if not scores:
            continue
        subset = RubricScoring(scores=tuple(scores))
        curve = {str(bar): subset.substantive_rate(bar) for bar in bars}
        disagreement = refusal_disagreement(
            scores, [cell["refused"] for cell in cells if cell["regime"] == regime]
        )
        report["by_regime"][regime] = {  # type: ignore[index]
            "n": subset.n,
            "mean_quality": subset.mean_quality,
            "substantive_rate": curve,
            "refusal_disagreement": disagreement,
        }
        print(f"\n({regime})  n={subset.n}  mean quality {subset.mean_quality}")
        for bar, rate in curve.items():
            print(f"     bar {bar}   substantive {rate}")
        print(f"     their-refusal vs ours, disagreement {disagreement}")

    # The discrimination check, stated as a verdict rather than left to the eye.
    b_mean = report["by_regime"].get("B", {}).get("mean_quality")  # type: ignore[union-attr]
    s_mean = report["by_regime"].get("S", {}).get("mean_quality")  # type: ignore[union-attr]
    if b_mean is None or s_mean is None:
        report["discriminates"] = None
        print("\n⚠️  Control incomplete — one regime had no parsed readings. No (B)")
        print("    number from this pass is reportable.")
    else:
        report["discriminates"] = b_mean > s_mean
        if b_mean <= s_mean:
            print(f"\n⚠️  (B) mean {b_mean:.3f} <= (S) mean {s_mean:.3f}. The graded judge does")
            print("    NOT separate comply from refuse on this data, so it is not reading")
            print("    what we think. No (B) number from this pass is reportable.")
        else:
            print(f"\n(B) {b_mean:.3f} > (S) {s_mean:.3f} — the rubric discriminates.")

    directory = (
        (PROJECT_ROOT / args.outputs_dir if args.outputs_dir else PROJECT_ROOT / "outputs")
        / "runs" / PHASE / args.run_name
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_run_record(directory, report)
    print(f"\nwrote {directory / 'results.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
