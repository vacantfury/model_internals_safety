"""Screen re-licensed rungs against the control-calibrated floor — both statistics.

CPU-only, no GPU, no model, no judge, no keys. Reads what `relicense_probes.py`
wrote plus a run's cached `cells.jsonl`, and answers the one question permutation
licensing cannot: **does this rung's deployment probe read decoded content, or
does it read the same surface features it reads on rungs the model cannot decode
at all?**

## Why this is separate from `recalibrate_deployment.py`

That script derives a floor and applies it while re-reading a run's per-cell
operating point; it owns the *read*. This one owns the *screen*, over an
already-licensed set, and it exists because the screen's statistic turned out to
be the thing in question (`instrument_layer.md` §2.4).

## The defect it found, and the rule that replaced it

The floor used to be **max** over the control rungs. That is the binding
constraint when there are two of them, and an upward-biased estimator once there
are eleven — max is monotone non-decreasing in the control-set size, so a rung's
pass/fail depended on how many rungs the model happened to be unable to decode in
that run. Measured 2026-08-07: the same instrument on the same models moved
0.656 -> 0.674 (Llama) and 0.671 -> 0.683 (Qwen) purely from 2 controls to 11/10.

**SETTLED 2026-08-07 (owner go): the rule is `mean + sigma*SD`**, derived in
`measurements/control_floor.py` and configured in `conf/measurements.yaml`. This
script still prints BOTH — the retired statistic beside the adopted one, each
with its control-set size, plus the **sigma window** within which no conclusion
changes. A floor without its n is not interpretable, and a rung the two
statistics split on is inside the noise of the choice and is reported as
measurable under neither.

## The control set is DERIVED, never listed

Ability is recomputed from the cached restatement text under the CURRENT settled
cuts, exactly as `rebaseline_pilot.py` does — never read from `cells.jsonl`'s
stored `ability`, which predates instrument fixes #1/#2 and is stale by 439 cells
on this repo's own data. A hand-maintained list of "the inert rungs" would be the
same defect one level up: the control set is a measurement, not a constant.

Usage:

    uv run python scripts/control_floor.py \
        --relicense outputs/analysis/relicense_20260807 \
        --run-dir outputs/runs/phase0/llama3_1_8b_instruct/phase0-20260802 \
        --run-dir outputs/runs/phase0/qwen2_5_7b_instruct/phase0-20260802
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from internals_safety.config import load_measurements_config
from internals_safety.encodings.recovery import score_recovery
from internals_safety.measurements.control_floor import AbilitySource
from internals_safety.measurements.control_floor import derive as derive_control_floor
from internals_safety.measurements.control_floor import sigma_bounds


def ability_by_family(cells_path: Path, cuts) -> dict[str, float]:
    """Fraction of cells whose plaintext was recovered, per rung — recomputed.

    Same three routes to recovery as the settled ability binary (exact/contains,
    similarity with a content-overlap veto, order-blind overlap for word
    permutations), because a control set built on a retired rule would quietly
    admit rungs the model can in fact read.
    """
    hits: dict[str, list[bool]] = {}
    with cells_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            cell = json.loads(line)
            score = score_recovery(
                cell["plaintext"], cell.get("restate_response") or "", cell.get("ciphertext")
            )
            hits.setdefault(cell["family"], []).append(
                score.is_recovered(
                    cuts.similarity_threshold,
                    cuts.content_overlap_threshold,
                    cuts.order_blind_overlap_threshold,
                )
            )
    return {family: sum(v) / len(v) for family, v in hits.items() if v}


def transfer_by_family(relicense_dir: Path, model: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(relicense_dir.glob(f"{model}__*.json")):
        for row in json.loads(path.read_text()):
            out[row["family"]] = row["deployment_max_transfer_auroc"]
    return out


def floors(control_values: list[float], sigma: float) -> dict[str, float | None]:
    """Both floor statistics side by side, for the comparison this script reports.

    The DECISION now lives in `measurements/control_floor.derive`; this stays as
    the diagnostic view, because showing the retired statistic next to the
    adopted one is how a reader sees that the two differ and by how much.

    `None`, not a default: with no controls there is nothing to judge a reading
    against, and with one there is no spread. A single-control SD of 0.0 would
    turn mean+kSD into "any reading above the one control passes".
    """
    n = len(control_values)
    return {
        "n": n,
        "max": max(control_values) if n >= 1 else None,
        "mean": statistics.mean(control_values) if n >= 1 else None,
        "mean_plus_2sd": (
            statistics.mean(control_values) + sigma * statistics.stdev(control_values)
            if n >= 2
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relicense", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, action="append")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    measurements = load_measurements_config()
    cuts = measurements.ability
    max_ability = measurements.controls.control_ability_max

    # Several run dirs may belong to ONE model: the Qwen phase-0 head run was
    # killed at the 8h wall and a tail run completed the ladder, so its rungs'
    # ability is split across two dirs. Merging is not a convenience — reading
    # only the head dir left `zero_width` and `reverse_words` with no ability
    # measurement at all, i.e. the two rungs the screen exists to judge.
    by_model: dict[str, list[Path]] = {}
    for run_dir in args.run_dir:
        by_model.setdefault(run_dir.parent.name, []).append(run_dir)

    report: dict = {"control_ability_max": max_ability, "models": {}}
    for model, run_dirs in by_model.items():
        auroc = transfer_by_family(args.relicense, model)
        if not auroc:
            print(f"!! no re-licensing rows for {model} under {args.relicense}")
            continue
        ability: dict[str, float] = {}
        for run_dir in run_dirs:
            cells = run_dir / "cells.jsonl"
            if not cells.exists():
                print(f"!! {cells} does not exist — skipped")
                continue
            for family, rate in ability_by_family(cells, cuts).items():
                # Later dirs win: the tail run re-captured what the killed head
                # run left partial, so its measurement is the live one.
                ability[family] = rate
        missing = sorted(set(auroc) - set(ability))

        adopted = derive_control_floor(
            auroc,
            ability=AbilitySource(rates=ability, measured_on=model, screens=model),
            max_ability=max_ability,
            sigma=measurements.controls.control_floor_sigma,
            min_controls=measurements.controls.control_floor_min_controls,
        )
        controls = list(adopted.controls)
        stats = floors([auroc[f] for f in controls],
                       sigma=measurements.controls.control_floor_sigma)

        print(f"\n{'=' * 78}\n{model}\n{'=' * 78}")
        print(f"control set (ability <= {max_ability}): n={stats['n']}  {', '.join(controls)}")
        if missing:
            # Loud: a rung with no ability measurement cannot be placed on either
            # side of the screen, and silently treating it as non-control would
            # let an unmeasured rung set the floor.
            print(f"!! no ability measurement for: {', '.join(missing)} — excluded from the control set")
        if stats["mean_plus_2sd"] is None:
            print("!! control set too small for a distributional floor — max only")
        else:
            print(f"floor(max)        = {stats['max']:.4f}   <- RETIRED 2026-08-07, grows with n")
            print(f"floor(mean+{measurements.controls.control_floor_sigma:g}SD)   = "
                  f"{adopted.value:.4f}   <- ADOPTED ({adopted.kind}, n={adopted.n})")
            low, high = sigma_bounds(auroc,
                                     ability=AbilitySource(rates=ability, measured_on=model,
                                                           screens=model),
                                     max_ability=max_ability,
                                     genuine=[f for f in ("zero_width", "reverse_words") if f in auroc])
            if low is not None:
                window = f"sigma window: >= {low:.3f} (no control passes)"
                window += f", <= {high:.3f} (genuine rungs still pass)" if high else ""
                print(window)
                if high is not None and low >= high:
                    print("!! THE WINDOW HAS CROSSED — this screen has no valid sigma. "
                          "Do not pick one; the instrument cannot separate these rungs.")

        print(f"\n{'rung':<20}{'ability':>8}{'AUROC':>9}  {'vs max':>9}  {'vs m+2SD':>9}")
        rows = []
        for family in sorted(auroc, key=lambda f: -auroc[f]):
            value = auroc[family]
            by_max = value > stats["max"] and family not in controls
            by_sd = (
                stats["mean_plus_2sd"] is not None
                and value > stats["mean_plus_2sd"]
                and family not in controls
            )
            tag = "CONTROL" if family in controls else ""
            print(
                f"{family:<20}{ability.get(family, float('nan')):>8.2f}{value:>9.4f}"
                f"  {('CLEARS' if by_max else tag or '-'):>9}"
                f"  {('CLEARS' if by_sd else tag or '-'):>9}"
            )
            rows.append(
                {
                    "family": family,
                    "ability_rate": ability.get(family),
                    "transfer_auroc": value,
                    "is_control": family in controls,
                    "clears_max": by_max,
                    "clears_mean_plus_2sd": by_sd,
                }
            )

        disagree = [r["family"] for r in rows if r["clears_max"] != r["clears_mean_plus_2sd"]]
        if disagree:
            # The whole reason both are printed. A rung the two statistics split
            # on is inside the noise of the choice and must not be reported as
            # measurable under either.
            print(f"\n!! the two floors DISAGREE on: {', '.join(disagree)}")
            print("   Such a rung is not measurable — do not report it under whichever rule passes it.")
        report["models"][model] = {"floors": stats, "controls": controls, "rungs": rows}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
