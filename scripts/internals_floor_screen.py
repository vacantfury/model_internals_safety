#!/usr/bin/env python3
"""AS-5 stage 1(b), answered offline: screen the internals leg against a real floor.

**Reads local run records and cached cells only — no model, no judge, no GPU,
no spend.** Seconds.

## What this closes

`phase1_design.md` §5.1, TODO 71 and the session board all carry the same live
limit: *the control floor is UNUSABLE on all four scaffold-control runs
(`n_controls = 0`), so the internals leg's licensing is permutation-only* —
§2.5's defect, inside the run that carries the result — and closing it is
stage 1(b), a GPU job.

**It is not a GPU job.** The limit is true of those RUNS and does not follow for
the MEASUREMENT. `deployment` is deterministic given (model, family, corpus,
cached activations, probe config): it fits on the plain contrast and transfers
to the encoded arm, with no sampling anywhere. The identical reading — same
corpus digest, same cached tensors, same selected cell, same value to full
precision — also sits in runs that carried can't-decode rungs and therefore
have a floor.

This script finds those witness runs, PROVES the identity rather than assuming
it, and screens the leg against each witness's own floor.

## Why this does not violate §2.4

§2.4 forbids carrying a floor derived at one n to a number measured somewhere
else, because the max statistic moves with n. Nothing here does that. The
witness run's floor screens the witness run's OWN reading; the leg's claim to
that verdict is that its reading and the witness's are the same measurement, not
two measurements that resemble each other. `measurements/floor_witness.py` is
where that check lives — in the spine rather than here, because a second script
screening a floor without it would be a defect, which is the spine's own
selection rule.

⚠️ **A `bound` floor stays a bound.** The grade is printed on every row.
Reporting a bound as though it were a distribution is exactly the error §2.2's
table caused, and a witness does not upgrade anything.

## The floor is RE-DERIVED, not read

`results.json` carries a `control_floor` computed at run time. This script
recomputes it from the witness's own per-family transfer AUROCs and ability
recomputed from `cells.jsonl` under the settled cuts — the same discipline
`control_floor.py` and `guard_control_floor.py` follow, and for the same reason:
recorded ability in the older phase-0 runs predates instrument fixes #1/#2. Both
values are printed. **A divergence is a finding, not noise**, so the script says
so loudly instead of preferring one silently.

    uv run python scripts/internals_floor_screen.py \\
        --out outputs/analysis/internals_floor_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_floor import ability_by_family  # noqa: E402

from internals_safety.config import load_measurements_config  # noqa: E402
from internals_safety.measurements.control_floor import AbilitySource  # noqa: E402
from internals_safety.measurements.control_floor import derive as derive_control_floor  # noqa: E402
from internals_safety.measurements.control_floor import sigma_bounds  # noqa: E402
from internals_safety.measurements.floor_witness import Provenance, WitnessScreen  # noqa: E402
from internals_safety.paths import RUNS_DIR  # noqa: E402

DEFAULT_FAMILY = "homoglyph"
DEFAULT_LEG_PREFIX = "scaffold-control"


def _family_block(record: dict, family: str) -> dict | None:
    for block in record.get("metrics", {}).get("families", []):
        if block.get("family") == family:
            return block
    return None


def provenance(run: Path, record: dict, family: str) -> Provenance | None:
    """The identity of this run's deployment reading on `family`, or None."""
    block = _family_block(record, family)
    if block is None:
        return None
    deployment = block.get("deployment") or {}
    corpus = record.get("corpus") or {}
    acts = record.get("activations_path") or {}
    return Provenance(
        model=run.parent.name,
        family=family,
        harmful_digest=corpus.get("harmful_digest"),
        harmless_digest=corpus.get("harmless_digest"),
        n_prompts=corpus.get("n_prompts"),
        plain_harmful_activations=acts.get("plain_harmful"),
        plain_harmless_activations=acts.get("plain_harmless"),
        encoded_harmful_activations=(block.get("activations") or {}).get("encoded_harmful"),
        encoded_harmless_activations=(block.get("activations") or {}).get("encoded_harmless"),
        layer=deployment.get("layer"),
        position=deployment.get("position"),
        transfer_auroc=deployment.get("transfer_auroc"),
        run=run.name,
    )


def rederive_floor(run: Path, record: dict, measurements):
    """Recompute the witness run's floor from its own rungs. Never read it."""
    auroc: dict[str, float] = {}
    for block in record.get("metrics", {}).get("families", []):
        value = (block.get("deployment") or {}).get("transfer_auroc")
        if value is not None:
            auroc[block["family"]] = value

    cells = run / "cells.jsonl"
    ability = ability_by_family(cells, measurements.ability) if cells.exists() else {}

    model = run.parent.name
    source = AbilitySource(rates=ability, measured_on=model, screens=model)
    floor = derive_control_floor(
        auroc,
        ability=source,
        max_ability=measurements.controls.control_ability_max,
        sigma=measurements.controls.control_floor_sigma,
        min_controls=measurements.controls.control_floor_min_controls,
    )
    return floor, auroc, ability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", default="phase0")
    parser.add_argument("--leg-prefix", default=DEFAULT_LEG_PREFIX,
                        help="run prefix holding the internals leg's reading")
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 unless every model finds a witness that clears")
    args = parser.parse_args(argv)

    measurements = load_measurements_config()
    root = RUNS_DIR / args.phase
    if not root.is_dir():
        print(f"no runs under {root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        leg_runs = [
            r for r in sorted(model_dir.glob(f"{args.leg_prefix}*"))
            if (r / "results.json").exists()
        ]
        if not leg_runs:
            continue
        leg = leg_runs[0]
        leg_record = json.loads((leg / "results.json").read_text(encoding="utf-8"))
        leg_prov = provenance(leg, leg_record, args.family)
        if leg_prov is None or leg_prov.transfer_auroc is None:
            continue

        leg_floor_usable = bool((leg_record.get("control_floor") or {}).get("usable"))

        witnesses: list[dict] = []
        for candidate in sorted(model_dir.iterdir()):
            if not candidate.is_dir() or candidate == leg:
                continue
            results = candidate / "results.json"
            if not results.exists():
                continue
            record = json.loads(results.read_text(encoding="utf-8"))
            cand_prov = provenance(candidate, record, args.family)
            if cand_prov is None:
                continue
            floor, auroc, ability = rederive_floor(candidate, record, measurements)
            screen = WitnessScreen(reading=leg_prov, witness=cand_prov, floor=floor)
            recorded = (record.get("control_floor") or {}).get("value")
            lower, upper = (None, None)
            if floor.kind == "distribution":
                lower, upper = sigma_bounds(
                    auroc,
                    ability=AbilitySource(rates=ability, measured_on=candidate.parent.name,
                                          screens=candidate.parent.name),
                    max_ability=measurements.controls.control_ability_max,
                    genuine=[args.family],
                )
            witnesses.append({
                "run": candidate.name,
                "same_measurement": screen.is_same_measurement,
                "mismatched_fields": list(screen.mismatched_fields),
                "floor_value": floor.value,
                "floor_kind": floor.kind,
                "n_controls": floor.n,
                "controls": list(floor.controls),
                "floor_recorded": recorded,
                # Tri-state deliberately: a run that never computed a floor
                # (the pre-2026-08-05 records) has nothing to agree or disagree
                # with, and calling that a divergence would manufacture an
                # alarm out of an absence — the same unmeasured-is-not-negative
                # rule this repo enforces on every other axis.
                "floor_matches_record": (
                    None if recorded is None or floor.value is None
                    else abs(recorded - floor.value) < 1e-9
                ),
                "verdict": screen.verdict,
                "margin": screen.margin,
                "reason": screen.reason,
                "sigma_window": [lower, upper],
            })

        usable = [w for w in witnesses if w["same_measurement"] and w["floor_value"] is not None]
        best = None
        if usable:
            # Prefer a distribution floor over a bound; then the most controls.
            best = sorted(
                usable,
                key=lambda w: (w["floor_kind"] != "distribution", -w["n_controls"]),
            )[0]
        rows.append({
            "model": leg_prov.model,
            "leg_run": leg.name,
            "auroc": leg_prov.transfer_auroc,
            "leg_floor_usable": leg_floor_usable,
            "best_witness": best,
            "witnesses": witnesses,
        })

    print(f"AS-5 internals leg — stage 1(b) screened OFFLINE, $0   (family: {args.family})")
    print("the leg's own runs carry no floor; these are the SAME reading in runs that do\n")
    header = (f"{'model':<24}{'AUROC':>8}{'floor':>9}{'grade':>15}{'n':>4}"
              f"{'margin':>9}  witness run")
    print(header)
    print("-" * (len(header) + 22))
    for row in rows:
        best = row["best_witness"]
        if best is None:
            print(f"{row['model']:<24}{row['auroc']:>8.4f}{'  --':>9}"
                  f"{'NO WITNESS':>15}{'-':>4}{'  --':>9}  none of "
                  f"{len(row['witnesses'])} candidates is the same measurement")
            continue
        mark = {True: "clears", False: "BELOW FLOOR", None: "unjudged"}[best["verdict"]]
        print(f"{row['model']:<24}{row['auroc']:>8.4f}{best['floor_value']:>9.4f}"
              f"{best['floor_kind']:>15}{best['n_controls']:>4}"
              f"{best['margin']:>+9.4f}  {best['run'][:34]}  {mark}")

    print("\nfloor re-derivation vs the value each witness recorded")
    for row in rows:
        best = row["best_witness"]
        if best is None:
            continue
        state = {True: "agrees", False: "!! DIVERGES",
                 None: "witness recorded none — nothing to compare"}[
            best["floor_matches_record"]]
        recorded = ("--" if best["floor_recorded"] is None
                    else f"{best['floor_recorded']:.4f}")
        print(f"  {row['model']:<24}recomputed {best['floor_value']:.4f}  "
              f"recorded {recorded}  {state}")
        print(f"      controls: {', '.join(best['controls'])}")
        if best["floor_kind"] == "distribution":
            lower, upper = best["sigma_window"]
            window = ("could not be computed" if lower is None or upper is None
                      else f"[{lower:.3f}, {upper:.3f})")
            configured = measurements.controls.control_floor_sigma
            valid = (lower is not None and upper is not None
                     and lower <= configured < upper)
            print(f"      sigma window {window} vs configured {configured}"
                  f"  {'VALID' if valid else '!! CONFIGURED SIGMA OUTSIDE THE WINDOW'}")

    print("\n⚠️ what this does and does not close")
    print("  CLOSED: permutation-only licensing (§2.5) for every model with a witness")
    print("  NOT closed: a `bound` grade is still a bound — §2.4, the max statistic")
    print("              moves with n, and a witness never upgrades a grade")
    print("  NOT closed: AUROC intervals conditional on the selected cell (§4.9)")
    print("  NOT closed: one instrument, not two — I6's patching arm has not run")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.strict and not all(
        row["best_witness"] and row["best_witness"]["verdict"] for row in rows
    ):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
