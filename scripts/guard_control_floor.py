"""The guard-side control floor, derived from the guard's BASE MODEL's ability.

**Why this script is not `control_floor.py` with a different path.** That script
derives a floor from a run's own can't-decode rungs, and the selector — ability,
measured from generated text — is what makes the floor non-circular (§3.4: *a
control selected on the same variable it controls for is not a control*). On the
guard side that selector does not exist: Llama Guard 3's template hard-wires the
classification task and ends "First line must read 'safe' or 'unsafe'", so there
is no in-distribution channel through which a guard can be asked to restate a
payload. No restatement, no ability, no control set.

`instrument_layer.md` §2.6 settles the route, and it is asymmetric:

    Llama Guard 3 8B  <- Llama-3.1-8B     ability for all 19 rungs, on disk
    WildGuard         <- Mistral-7B-v0.3  `base64` only; the rest never ran

A base model's ability is an ESTIMATE of its fine-tune's, not a measurement —
fine-tuning can move decoding capability. What it is is *independent* of the
guard's decode probe, which is the property §3.4 actually requires. The floor is
therefore labelled **inherited**, and that label is not optional: `AbilitySource`
requires both model names, so no caller can report an inherited floor as a
measured one by forgetting a keyword.

**Ability is RECOMPUTED from cached cells under the settled cuts, never read from
`results.json`.** The recorded ability in every phase-0 run predates instrument
fixes #1/#2 — `hex` is 0.21 there and 0.84 under the settled rule — so a control
set built from recorded values would admit rungs the model can in fact read. This
is the same reason `control_floor.py` recomputes.

Keyless, no GPU, no weights, no spend. Seconds.

    uv run python scripts/guard_control_floor.py \\
        --guard-run outputs/runs/as6_phase1/llama_guard_3_8b/matched-b10 \\
        --base-model llama3_1_8b_instruct \\
        --ability-cells outputs/runs/phase0/llama3_1_8b_instruct/phase0-20260802/cells.jsonl \\
        --ability-cells outputs/runs/phase0/llama3_1_8b_instruct/band2-20260805/cells.jsonl \\
        --out outputs/analysis/guard_control_floor.json
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


def guard_rows(run_dir: Path) -> list[dict]:
    """Per-rung decode AUROC, permutation verdict and cell rates, from one run."""
    results = json.loads((run_dir / "results.json").read_text())
    rows = []
    for summary in results.get("summaries", []):
        decode = summary.get("decode") or {}
        rows.append(
            {
                "family": summary["family"],
                "transfer_auroc": decode.get("transfer_auroc"),
                "permutation_licensed": decode.get("licensed"),
                "p_value": decode.get("p_value"),
                "layer": decode.get("layer"),
                "block_rate": summary.get("block_rate"),
                "decoded_not_blocked_rate": summary.get("decoded_not_blocked_rate"),
            }
        )
    return rows


def split_half_auroc(
    split_rows: list[dict], guard: str, auroc: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """The item-split AUROC map for one guard, plus the rungs it cannot cover.

    Returns `B_split_logistic.mean` keyed by family, and the families present in
    the run's own AUROC map that the split artifact does not carry. Those are
    DROPPED by the caller rather than left at their unsplit value: a floor is a
    statistic over its control set, so one family measured under the other
    procedure silently moves either the bar or a candidate. Mixing the two
    procedures in one comparison is precisely the apples-to-oranges the AS-5
    refutation record warns about, one level down.
    """
    split = {
        r["family"]: r["B_split_logistic"]["mean"]
        for r in split_rows
        if r.get("model") == guard and isinstance(r.get("B_split_logistic"), dict)
    }
    return split, sorted(set(auroc) - set(split))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-run", type=Path, required=True)
    parser.add_argument("--guard-name", default=None,
                        help="defaults to the run dir's parent, which is the guard")
    parser.add_argument("--base-model", required=True,
                        help="the model whose ability SELECTS the controls")
    parser.add_argument("--ability-cells", type=Path, action="append", required=True,
                        help="cells.jsonl to recompute ability from; later files win")
    parser.add_argument(
        "--auroc-from",
        type=Path,
        default=None,
        help=(
            "split_half_transfer.py artifact; re-derive the floor from the ITEM-SPLIT "
            "AUROC (its B statistic) instead of the run's recorded transfer AUROC"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    measurements = load_measurements_config()
    controls_config = measurements.controls
    guard = args.guard_name or args.guard_run.parent.name

    ability: dict[str, float] = {}
    for cells in args.ability_cells:
        if not cells.exists():
            print(f"!! {cells} does not exist — skipped")
            continue
        ability.update(ability_by_family(cells, measurements.ability))

    rows = guard_rows(args.guard_run)
    auroc = {r["family"]: r["transfer_auroc"] for r in rows if r["transfer_auroc"] is not None}

    # The floor was itself derived from readings taken under the procedure whose
    # item non-holdout withdrew AS-5's internals leg, so "clears the floor" is
    # only meaningful when BOTH sides are re-derived. `--auroc-from` swaps the
    # whole AUROC map for the item-split statistic -- controls included. Swapping
    # only the screened rungs would compare a split reading against an unsplit
    # floor, which is the apples-to-oranges the refutation record warns against.
    split_source = None
    if args.auroc_from is not None:
        split_rows = json.loads(args.auroc_from.read_text())
        split, unsplit = split_half_auroc(split_rows, guard, auroc)
        if not split:
            print(f"!! {args.auroc_from} carries no rows for guard {guard!r}")
            return 1
        if unsplit:
            print(f"!! dropping {len(unsplit)} rung(s) with no item-split reading: {unsplit}")
        auroc = split
        rows = [r for r in rows if r["family"] in split]
        for row in rows:
            row["transfer_auroc"] = split[row["family"]]
        split_source = str(args.auroc_from)

    if not auroc:
        print(f"!! no decode AUROCs in {args.guard_run}")
        return 1

    source = AbilitySource(rates=ability, measured_on=args.base_model, screens=guard)
    floor = derive_control_floor(
        auroc,
        ability=source,
        max_ability=controls_config.control_ability_max,
        sigma=controls_config.control_floor_sigma,
        min_controls=controls_config.control_floor_min_controls,
    )

    missing = sorted(set(auroc) - set(ability))
    print(f"\n{'='*92}\n{guard}   floor inherited from {args.base_model}"
          f"   (inherited={source.is_inherited})\n{'='*92}")
    if missing:
        # Loud, never silent: a rung with no ability measurement is neither a
        # control nor screened, and pretending otherwise is how an unmeasured
        # rung ends up setting the bar everything else is judged against.
        print(f"!! no base-model ability for {len(missing)} rung(s): {missing}")
    print(f"controls (ability <= {controls_config.control_ability_max}): "
          f"n={floor.n} kind={floor.kind}")
    print(f"  {list(floor.controls)}")
    if floor.mean is not None:
        sd = f"{floor.stdev:.4f}" if floor.stdev is not None else "n/a"
        print(f"  mean={floor.mean:.4f}  sd={sd}  observed_max={floor.observed_max:.4f}")
    print(f"FLOOR = {floor.value:.4f}" if floor.is_usable else "FLOOR = undefined")

    # `genuine` is chosen SIGMA-FREE, so the window is not derived from the
    # answer it is meant to justify: a non-control rung whose AUROC beats the
    # largest control's is the sigma-independent candidate set.
    genuine = [
        r["family"] for r in rows
        if r["family"] in auroc and r["family"] not in floor.controls
        and floor.observed_max is not None and auroc[r["family"]] > floor.observed_max
    ]
    low, high = sigma_bounds(auroc, ability=source,
                             max_ability=controls_config.control_ability_max,
                             genuine=genuine)
    window: dict = {"lower": low, "upper": high, "configured": controls_config.control_floor_sigma}
    if low is not None:
        line = f"sigma window: >= {low:.3f} (no control passes)"
        line += f", <= {high:.3f} (candidates still pass)" if high is not None else ""
        print(line)
        if high is not None and low >= high:
            print("!! THE WINDOW HAS CROSSED — this screen has no valid sigma on this guard.")
            window["verdict"] = "crossed"
        elif controls_config.control_floor_sigma < low:
            # NOT a detail. The configured sigma is the AS-5 ladder's; if it sits
            # below this guard's lower bound then a control clears its own floor
            # here, which is the requirement the constant was derived FROM.
            print(f"!! CONFIGURED SIGMA {controls_config.control_floor_sigma:g} IS BELOW THE "
                  f"LOWER BOUND {low:.3f} — invalid on this guard. The AS-5 sigma does not port; "
                  f"§2.6's 'the selector does not port' extends to the calibration constant.")
            window["verdict"] = "configured_below_lower_bound"
        else:
            window["verdict"] = "configured_inside_window"

    verdict_label = {True: "PASS", False: "fail", None: "?"}
    header = ("family", "base abil", "auroc", "ctrl", "perm", "floor", "dnb", "block")
    print("\n%-20s%10s%9s%6s%6s%7s%7s%7s" % header)
    screened = []
    for row in sorted(rows, key=lambda r: -(r["transfer_auroc"] or 0)):
        family = row["family"]
        if family not in auroc:
            continue
        clears = floor.clears(auroc[family], family)
        is_control = family in floor.controls
        dnb = row["decoded_not_blocked_rate"]
        screened.append({**row, "base_ability": ability.get(family),
                         "is_control": is_control, "clears_floor": clears})
        print("%-20s%10s%9.4f%6s%6s%7s%7s%7.2f" % (
            family,
            "%.2f" % ability[family] if family in ability else "-",
            auroc[family],
            "yes" if is_control else "",
            "yes" if row["permutation_licensed"] else "no",
            verdict_label[clears],
            "%.2f" % dnb if dnb is not None else "-",
            row["block_rate"] if row["block_rate"] is not None else float("nan"),
        ))

    demoted = [r["family"] for r in screened if r["permutation_licensed"] and not r["clears_floor"]]
    if demoted:
        print(f"\nDEMOTED by the floor (permutation licensed, floor does not clear): {demoted}")
        print("  Significance is not sufficiency — the second measurement of that, on a guard.")

    report = {
        "guard": guard,
        "guard_run": str(args.guard_run),
        "auroc_source": split_source or "run record (transfer, items NOT held out)",
        "base_model": args.base_model,
        "inherited": source.is_inherited,
        "control_ability_max": controls_config.control_ability_max,
        "floor": {
            "value": floor.value, "kind": floor.kind, "n": floor.n,
            "controls": list(floor.controls), "mean": floor.mean, "stdev": floor.stdev,
            "observed_max": floor.observed_max,
            "ability_measured_on": floor.ability_measured_on,
            "ability_screens": floor.ability_screens,
        },
        "sigma_window": window,
        "rungs_without_base_ability": missing,
        "rungs": screened,
        "demoted_by_floor": demoted,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
