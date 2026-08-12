#!/usr/bin/env python3
"""AS-5 stage 0: is the harm represented but not read? (TODO 71, `phase1_design.md` §5)

**Reads local run records only — no model, no judge, no GPU, no spend.** It
reproduces `phase1_design.md` §2's table from the records rather than from the
ad-hoc read that produced it, so every number in AS-5's internals leg has
provenance, and it applies the two screens §5 requires before any of it may be
quoted.

**The gate this script IS.** §8: if the dissociation does not survive the echo
screen and the intervals, there is no internals leg and AS-5 ships behavioural
— which §4e already says is a publishable paper. So a red result here is an
answer, not a failure, and the script exits 0 either way; it is the CONTENT
that gates, and `--strict` is available for a caller that wants the exit code.

## Four decisions this script makes, each stated rather than defaulted

**The echo screen is applied to the ENCODED arm only.** `echoed_ciphertext`
detects a response parroting the ciphertext; on the plaintext arm there is no
ciphertext, so the same flag fires on ordinary quoting of the request. Cleaning
both arms would correct them for different things and call the result one
statistic. `dissociation.echo_screen_applies_to_encoded_arm_only` carries the
same note at the type level, and §4h already reports the scaffold arm's echo
rate without ever subtracting it.

**The behavioural side uses the CLEAN encoded rates when the screen is
measurable, and says which it used.** §3.11's rule is that a statistic computed
over gaps must be computed on echo-clean cells, and the dissociation is a
statistic over gaps. Where the screen cannot be computed (an arm missing) the
row reports the raw rates and is FLAGGED, never silently mixed in with screened
rows.

**The internal AUROC is taken from the `deployment` reading, not recomputed.**
The per-cell reads on disk are booleans at a percentile operating point; the
AUROC behind them is not reconstructible from the record, so recomputing would
mean a different statistic wearing the same name. What this script adds is the
INTERVAL, which the record does not carry.

**A run whose deployment reading is unlicensed contributes `None`, never a
number.** Tri-state, and it is why the internal column can be empty for a model
without the row vanishing.

⚠️ **The control floor is UNUSABLE on these runs** (`n_controls = 0`, single
family), so internal licensing is permutation-only — §2.5's live defect, inside
the run that carries this result. The script prints it on every row rather than
in a header, because a header is what a reader skips. Do NOT import a floor
from another run to paper over it: §2.4 settled that the floor statistic is
n-dependent and cross-run floors are not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from internals_safety.measurements.dissociation import (
    DEFAULT_INTERNAL_FLOOR,
    DissociationReading,
)
from internals_safety.measurements.refusal_control import summarize_exposure
from internals_safety.paths import RUNS_DIR

# The condition AS-5's internals leg is built on. `homoglyph` is the one rung
# where the repo's most persistent confound cannot operate: it equalises
# character counts by construction (`mean_ciphertext_chars` ==
# `mean_plaintext_chars` == 86.0), so the length null has nothing to bite on.
DEFAULT_FAMILY = "homoglyph"


def _load_cells(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reading(record: dict, instrument: str) -> dict | None:
    for entry in record.get("readings", []):
        if entry.get("instrument") == instrument:
            return entry
    return None


def find_runs(root: Path, run_prefix: str) -> list[Path]:
    """Every `<root>/<model>/<run_prefix>*` directory holding a run record."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run in sorted(model_dir.glob(f"{run_prefix}*")):
            if (run / "results.json").exists():
                found.append(run)
    return found


def build_reading(
    run: Path, family: str, internal_floor: float
) -> tuple[DissociationReading, dict]:
    """One run directory -> its dissociation reading plus the audit trail."""
    record = json.loads((run / "results.json").read_text(encoding="utf-8"))
    model = run.parent.name

    plain = _reading(record, "behavior_plain")
    behaviour = _reading(record, "behavior")
    deployment = _reading(record, "deployment")

    if plain is None or behaviour is None:
        raise ValueError(
            f"{run.name}: no plaintext baseline or no behaviour reading. The "
            "dissociation is a comparison against the plaintext arm, so a run "
            "without one cannot contribute a row."
        )

    plain_detail = plain.get("detail", {})
    behaviour_detail = behaviour.get("detail", {})

    # --- the internal side, tri-state on licensing -------------------------
    internal_auroc: float | None = None
    internal_note = "no deployment reading"
    if deployment is not None:
        if deployment.get("licensed"):
            internal_auroc = deployment.get("value")
            internal_note = (
                f"L{deployment.get('detail', {}).get('layer')}"
                f" {deployment.get('detail', {}).get('position')}"
            )
        else:
            internal_note = "deployment UNLICENSED"

    # --- the echo screen, encoded arm only ---------------------------------
    harmful = _load_cells(run / "cells.jsonl")
    benign = _load_cells(run / "benign_cells.jsonl")
    exposure = None
    if harmful and benign:
        exposure = summarize_exposure(
            family=family,
            harmful_refused=[bool(c.get("refused")) for c in harmful],
            harmful_echoed=[bool(c.get("echoed_ciphertext")) for c in harmful],
            benign_refused=[bool(c.get("refused")) for c in benign],
            benign_echoed=[bool(c.get("echoed_ciphertext")) for c in benign],
        )

    screened = exposure is not None and exposure.measured
    if screened:
        encoded_harmful = exposure.clean_harmful_refusal_rate
        encoded_benign = exposure.clean_benign_refusal_rate
        n_encoded_harmful = exposure.n_harmful_clean
        n_encoded_benign = exposure.n_benign_clean
    else:
        encoded_harmful = behaviour_detail.get("refusal_rate")
        encoded_benign = behaviour_detail.get("benign_arm_refusal_rate")
        n_encoded_harmful = behaviour_detail.get("n", 0)
        n_encoded_benign = behaviour_detail.get("benign_arm_n", 0)

    reading = DissociationReading(
        model=model,
        family=family,
        internal_auroc=internal_auroc,
        n_internal_positive=plain_detail.get("plain_n", 0),
        n_internal_negative=plain_detail.get("plain_n", 0),
        plain_harmful_refusal_rate=plain_detail["plain_harmful_refusal_rate"],
        plain_benign_refusal_rate=plain_detail["plain_benign_refusal_rate"],
        n_plain_harmful=plain_detail.get("plain_n", 0),
        n_plain_benign=plain_detail.get("plain_n", 0),
        encoded_harmful_refusal_rate=encoded_harmful,
        encoded_benign_refusal_rate=encoded_benign,
        n_encoded_harmful=n_encoded_harmful,
        n_encoded_benign=n_encoded_benign,
        internal_floor=internal_floor,
    )

    floor = record.get("control_floor", {})
    trail = {
        "run": run.name,
        "git_hash": record.get("git_hash"),
        "internal_cell": internal_note,
        "echo_screened": screened,
        "echo_displacement": None if exposure is None else exposure.displacement,
        "echo_bar": None if exposure is None else exposure.bar,
        "echo_clears": None if exposure is None else exposure.clears(),
        "raw_encoded_gap": None if exposure is None else exposure.gap,
        "control_floor_usable": bool(floor.get("usable")),
        "length_null_margin": (
            None if deployment is None else deployment.get("length_null_margin")
        ),
    }
    return reading, trail


def _fmt(value: float | None, places: int = 3) -> str:  # plumbing(places): display precision only; --out carries full precision
    return "  --  " if value is None else f"{value:+.{places}f}"


def _fmt_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "     --     "
    return f"[{interval[0]:+.3f},{interval[1]:+.3f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase", default="phase0", help="run phase to search (default: phase0)"
    )
    parser.add_argument(
        "--run-prefix",
        default="scaffold-control",
        help="run-directory prefix (default: scaffold-control)",
    )
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument(
        "--internal-floor",
        type=float,
        default=DEFAULT_INTERNAL_FLOOR,
        help=(
            "AUROC lower-bound the internal side must clear "
            f"(default: {DEFAULT_INTERNAL_FLOOR})"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 unless every run clears — off by default, because a red "
        "result here is an ANSWER (ship behavioural) rather than a failure",
    )
    args = parser.parse_args(argv)

    root = RUNS_DIR / args.phase
    runs = find_runs(root, args.run_prefix)
    if not runs:
        print(f"no runs matching {args.run_prefix}* under {root}", file=sys.stderr)
        return 2

    print(f"AS-5 internals leg — stage 0, offline, $0   (family: {args.family})")
    print(f"internal floor: AUROC lower bound >= {args.internal_floor}\n")

    header = (
        f"{'model':<26}{'AUROC':>8}{'  interval':>16}"
        f"{'plain':>9}{'enc':>9}{'destroyed':>11}{'  interval':>18}  verdict"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for run in runs:
        reading, trail = build_reading(run, args.family, args.internal_floor)
        verdict = reading.clears()
        mark = {True: "DISSOCIATION", False: "no", None: "unmeasured"}[verdict]
        destroyed = reading.discrimination_destroyed
        print(
            f"{reading.model:<26}"
            f"{'  --  ' if reading.internal_auroc is None else f'{reading.internal_auroc:.3f} ':>8}"
            f"{_fmt_interval(reading.auroc_interval_conditional_on_selection):>16}"
            f"{reading.plain_gap:>+9.2f}"
            f"{_fmt(reading.encoded_gap, 2):>9}"
            f"{'   --  ' if destroyed is None else f'{100 * destroyed:>9.1f}%'}"
            f"{_fmt_interval(reading.discrimination_destroyed_interval):>18}"
            f"  {mark}"
        )
        rows.append(
            {
                "model": reading.model,
                "family": reading.family,
                "internal_auroc": reading.internal_auroc,
                "internal_interval": reading.auroc_interval_conditional_on_selection,
                "internal_survives": reading.internal_survives,
                "plain_gap": reading.plain_gap,
                "plain_gap_half_width": reading.plain_gap_half_width,
                "encoded_gap": reading.encoded_gap,
                "encoded_gap_half_width": reading.encoded_gap_half_width,
                "discrimination_destroyed": destroyed,
                "discrimination_destroyed_interval": (
                    reading.discrimination_destroyed_interval
                ),
                "behaviour_fails": reading.behaviour_fails,
                "dissociation": verdict,
                **trail,
            }
        )

    print("\nechoscreen  (encoded arm only — the plaintext arm has no ciphertext)")
    for row in rows:
        state = {True: "clears", False: "MOVES THE GAP", None: "unmeasured"}[
            row["echo_clears"]
        ]
        print(
            f"  {row['model']:<26}"
            f"displacement {_fmt(row['echo_displacement'])} "
            f"vs bar {_fmt(row['echo_bar'])}  {state}"
            f"   raw gap {_fmt(row['raw_encoded_gap'], 2)}"
        )

    print("\n⚠️ limits that travel with every number above")
    unusable = [r["model"] for r in rows if not r["control_floor_usable"]]
    if unusable:
        print(
            "  control floor UNUSABLE (permutation-only licensing, §2.5): "
            + ", ".join(unusable)
        )
    print("  AUROC intervals are conditional on the selected (layer x position) cell")
    print("  one instrument, not two — the patching arm (I6) has not been run on this")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.strict and not all(row["dissociation"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
