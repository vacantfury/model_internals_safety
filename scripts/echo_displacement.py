#!/usr/bin/env python3
"""How far did the ECHO route move the harm gap? (TODO 67, the screen that binds.)

**The question, and why it is not the one first asked.** The refusal judge counts
an echo as a refusal — its own prompt instructs it to. The obvious control is
therefore to show it a bare ciphertext and check it answers "not refused"; that
is `scripts/refusal_judge_control.py`, it ran on 2026-08-10, and it measured
**1559/1560 = 0.999**. Which settles nothing usable: the judge is behaving as
documented, at full strength, and being a property of the JUDGE that number is
identical on every rung, model and run. A gate built on it withholds the clean
conditions exactly as hard as the contaminated ones.

Contamination is susceptibility x EXPOSURE. This script measures the product, as
the displacement of the reported quantity: recompute the harm gap using only
cells that did not echo, and report how far the number moves. Design and the
both-ways sign argument: `measurements/refusal_control.py`.

**Reads cached cells only — no model, no judge, no GPU, no spend.** It needs BOTH
arms, so it requires `benign_cells.jsonl` alongside `cells.jsonl`; runs predating
that file cannot be screened retroactively and are reported as such rather than
scored on the harmful arm alone. A one-armed displacement would be a different
statistic wearing this one's name.

Live runs do not need this script — `phase0_regime_map.py` computes the same
screen inline from records already in hand, for free. This exists for the runs
that predate the wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internals_safety.measurements.refusal_control import summarize_exposure


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exposures(run_dir: Path) -> dict[str, dict]:
    harmful, benign = load(run_dir / "cells.jsonl"), load(run_dir / "benign_cells.jsonl")
    if not harmful or not benign:
        return {}
    rows: dict[str, dict] = {}
    for family in sorted({cell["family"] for cell in harmful} & {cell["family"] for cell in benign}):
        h = [cell for cell in harmful if cell["family"] == family]
        b = [cell for cell in benign if cell["family"] == family]
        exposure = summarize_exposure(
            family,
            harmful_refused=[bool(cell.get("refused")) for cell in h],
            harmful_echoed=[bool(cell.get("echoed_ciphertext")) for cell in h],
            benign_refused=[bool(cell.get("refused")) for cell in b],
            benign_echoed=[bool(cell.get("echoed_ciphertext")) for cell in b],
        )
        rows[family] = {
            "n_harmful": exposure.n_harmful,
            "n_benign": exposure.n_benign,
            "n_harmful_clean": exposure.n_harmful_clean,
            "n_benign_clean": exposure.n_benign_clean,
            "gap": exposure.gap,
            "clean_gap": exposure.clean_gap,
            "displacement": exposure.displacement,
            "bar": exposure.bar,
            "clears": exposure.clears(),
        }
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories to screen")
    parser.add_argument("--json", type=Path, default=None, help="write the table here")
    args = parser.parse_args(argv)

    everything: dict[str, dict] = {}
    unscreenable: list[Path] = []
    print(
        f"{'run':46s} {'rung':18s} {'gap':>7s} {'clean':>7s} {'displ':>7s} "
        f"{'bar':>6s} {'screen':>8s}"
    )
    for run_dir in args.run_dirs:
        rows = exposures(run_dir)
        if not rows:
            unscreenable.append(run_dir)
            continue
        everything[str(run_dir)] = rows
        tag = "/".join(run_dir.parts[-2:])[:46]
        for family, row in rows.items():
            verdict = {True: "CLEARS", False: "FAILS", None: "unmeas"}[row["clears"]]
            gap, clean = row["gap"], row["clean_gap"]
            displacement, bar = row["displacement"], row["bar"]
            print(
                f"{tag:46s} {family:18s} "
                f"{'  n/a' if gap is None else f'{gap:+7.2f}'} "
                f"{'  n/a' if clean is None else f'{clean:+7.2f}'} "
                f"{'  n/a' if displacement is None else f'{displacement:7.3f}'} "
                f"{'  n/a' if bar is None else f'{bar:6.3f}'} {verdict:>8s}"
            )

    if unscreenable:
        print(
            f"\n⚠️ {len(unscreenable)} run(s) carry no `benign_cells.jsonl` and CANNOT be "
            "screened — the gap stays withheld there, which is the correct reading of "
            "a control that was never measured. Not scored on the harmful arm alone:"
        )
        for run_dir in unscreenable:
            print(f"   {run_dir}")

    if args.json:
        args.json.write_text(json.dumps(everything, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
