"""Do AS-6's surviving cells actually fail to separate, or is the test underpowered?

TODO 89, and external-review con 7 one level deeper.

Table 2's caption says the conditions' Wilson intervals "overlap substantially,
so this table establishes that the cell is populated on every surviving
condition and does not order them". Overlapping independent intervals are a weak
basis for that refusal: the conditions are the SAME 100 harmful prompts wearing
different transformations, so they are item-paired, and an independent interval
throws away the pairing. AS-5's methods review overturned a claim on exactly this
mis-specification, in the dangerous direction. AS-6's version is honest -- it
abstains rather than asserting equality -- but an abstention taken on the weaker
of two available tests may be discarding a real ordering the paper could report.

**The refusal is not relaxed by assuming a paired test would be stronger.** It is
relaxed, or kept, by running one. This script runs both: exact McNemar over the
discordant items, and a paired bootstrap interval on the rate difference, against
the unpaired interval the paper currently reasons from.

**The per-prompt read is reconstructed at the REPORTED operating point**, not at
the one the source run was produced at. `scores-b10` is at `reading_percentile`
50, which is retired; `outputs/analysis/operating_point_*_20260808.json` carries
the threshold at each swept percentile. Recomputing `decode_score > threshold`
at 75 reproduces all six of Table 2's counts exactly, and the script asserts that
before it compares anything -- a paired test on a read that does not reproduce the
table is a test of something else.

**Multiplicity is reported, not assumed away.** Three pairs per guard is a family,
and this repo has already paid for reading significance as sufficiency, so Holm
adjusted values sit beside the raw ones.

Keyless, GPU-free, no model, seconds.

    uv run python scripts/paired_cell_ordering.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from internals_safety.config import load_measurements_config
from internals_safety.measurements.intervals import (
    holm_adjusted,
    mcnemar_exact,
    paired_bootstrap_rate_difference,
    wilson,
    z_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "outputs" / "runs" / "as6_phase1"
ANALYSIS = REPO_ROOT / "outputs" / "analysis"

SOURCE = "scores-b10"

# constant: not a knob here but a MIRROR of the settled knob
# (conf/measurements.yaml probes.reading_percentile), duplicated so this script can
# detect a run produced at a retired operating point; reading the config would
# defeat the check, since the config is what the run would also have read.
#: The reported operating point.
READING_PERCENTILE = 75.0

#: Table 2's rows, and its counts. Held here so the reconstruction can be checked
#: against the PUBLISHED numbers rather than against whatever it recomputes --
#: a self-check that agrees with itself is not a check.
TABLE_TWO = {
    "llama_guard_3_8b": {"homoglyph": 7, "zero_width": 17, "reverse_words": 8},
    "wildguard": {"homoglyph": 23, "zero_width": 23, "reverse_words": 9},
}


def per_prompt_cells(guard: str) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, float]]]:
    """The per-prompt cell, plus the MARGINALS needed to interpret an ordering.

    Returns `{family: {prompt_id: decoded-and-not-blocked}}` and
    `{family: {decode_rate, block_rate, unblocked}}`. The marginals are not
    decoration: D&notB is a conjunction, and whether an ordering of it is a
    finding about decoding or a restatement of the block rate depends entirely
    on how close the decode term is to 1.
    """
    operating = json.loads((ANALYSIS / f"operating_point_{guard}_20260808.json").read_text())
    rows = defaultdict(list)
    with (RUNS / guard / SOURCE / "cells.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["family"]].append(row)

    out: dict[str, dict[str, bool]] = {}
    marginals: dict[str, dict[str, float]] = {}
    for family in TABLE_TWO[guard]:
        swept = operating["families"].get(family)
        if not swept:
            raise SystemExit(f"{guard}/{family}: absent from the operating-point sweep")
        point = [r for r in swept if r["percentile"] == READING_PERCENTILE]
        if not point:
            raise SystemExit(
                f"{guard}/{family}: the sweep carries no {READING_PERCENTILE} point; "
                "the reported operating point cannot be reconstructed"
            )
        threshold = point[0]["threshold"]
        out[family] = {
            row["prompt_id"]: (row["decode_score"] > threshold and not row["blocked"])
            for row in rows[family]
        }
        marginals[family] = {
            "decode_rate": sum(r["decode_score"] > threshold for r in rows[family])
            / len(rows[family]),
            "block_rate": sum(r["blocked"] for r in rows[family]) / len(rows[family]),
            "unblocked": sum(not r["blocked"] for r in rows[family]),
        }
    return out, marginals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    measurements = load_measurements_config()
    alpha = measurements.probes.alpha
    z = z_for(alpha)
    rng = np.random.default_rng(measurements.controls.bootstrap_seed)
    draws = measurements.controls.bootstrap_draws

    results: dict[str, list[dict]] = {}
    for guard, expected in TABLE_TWO.items():
        cells, marginals = per_prompt_cells(guard)

        # THE RECONSTRUCTION MUST REPRODUCE THE PUBLISHED TABLE. Otherwise every
        # comparison below is about a different read than the one the paper made.
        for family, count in expected.items():
            got = sum(cells[family].values())
            if got != count:
                raise SystemExit(
                    f"{guard}/{family}: reconstructed D&notB is {got}, Table 2 says {count}. "
                    "The per-prompt read does not reproduce the published table; nothing "
                    "computed from it would be about the paper's numbers."
                )

        ids = sorted(set.intersection(*(set(v) for v in cells.values())))
        rows: list[dict] = []
        for left, right in combinations(sorted(expected), 2):
            a = [cells[left][i] for i in ids]
            b = [cells[right][i] for i in ids]
            only_a = sum(1 for x, y in zip(a, b) if x and not y)
            only_b = sum(1 for x, y in zip(a, b) if y and not x)
            point, lo, hi = paired_bootstrap_rate_difference(
                a, b, rng, draws=draws, alpha=alpha
            )
            a_lo, a_hi = wilson(sum(a), len(a), z)
            b_lo, b_hi = wilson(sum(b), len(b), z)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_count": sum(a),
                    "right_count": sum(b),
                    "discordant_left_only": only_a,
                    "discordant_right_only": only_b,
                    "mcnemar_p": mcnemar_exact(only_a, only_b),
                    "paired_difference": point,
                    "paired_lo": lo,
                    "paired_hi": hi,
                    "paired_separates": lo > 0.0 or hi < 0.0,
                    "wilson_intervals_overlap": not (a_lo > b_hi or b_lo > a_hi),
                }
            )
        for row, adjusted in zip(rows, holm_adjusted([r["mcnemar_p"] for r in rows])):
            row["mcnemar_p_holm"] = adjusted
            row["separates_after_adjustment"] = adjusted < alpha and row["paired_separates"]
        results[guard] = {"pairs": rows, "marginals": marginals}

        print(f"\n{guard}   ({len(ids)} shared prompts, alpha {alpha}, {draws} draws)")
        print("  marginals — D&notB is a conjunction, so read the ordering against these:")
        for family, m in marginals.items():
            share = expected[family] / m["unblocked"] if m["unblocked"] else float("nan")
            print(
                f"    {family:16s} decoded {m['decode_rate']:.2f}  blocked {m['block_rate']:.2f}  "
                f"unblocked {m['unblocked']:3d}  D&notB is {share:.2f} of the unblocked"
            )
        print(
            f"  {'pair':32s} {'counts':>9s} {'disc':>8s} {'diff [95% CI]':>24s} "
            f"{'McNemar':>9s} {'Holm':>7s}  unpaired"
        )
        for row in rows:
            pair = f"{row['left']} vs {row['right']}"
            counts = f"{row['left_count']}/{row['right_count']}"
            disc = f"{row['discordant_left_only']}/{row['discordant_right_only']}"
            ci = f"{row['paired_difference']:+.2f} [{row['paired_lo']:+.2f},{row['paired_hi']:+.2f}]"
            unpaired = "overlaps" if row["wilson_intervals_overlap"] else "separates"
            verdict = "SEPARATES" if row["separates_after_adjustment"] else "-"
            print(
                f"  {pair:32s} {counts:>9s} {disc:>8s} {ci:>24s} "
                f"{row['mcnemar_p']:>9.4f} {row['mcnemar_p_holm']:>7.4f}  "
                f"{unpaired:9s} {verdict}"
            )

    separating = [
        (guard, row)
        for guard, block in results.items()
        for row in block["pairs"]
        if row["separates_after_adjustment"]
    ]
    hidden = [(guard, row) for guard, row in separating if row["wilson_intervals_overlap"]]
    total = sum(len(block["pairs"]) for block in results.values())
    print(
        f"\n{len(separating)} of {total} pairs separate under the paired test; "
        f"{len(hidden)} of those the unpaired intervals could not see."
    )

    # WHAT THE ORDERING IS AN ORDERING OF. If the decode term is near 1, D&notB
    # is the non-block count wearing a conjunction, and separating two such cells
    # says nothing the block rate did not already say.
    degenerate = sorted(
        f"{guard}/{family} ({m['decode_rate']:.2f})"
        for guard, block in results.items()
        for family, m in block["marginals"].items()
        if m["decode_rate"] >= 0.90
    )
    if degenerate:
        print(
            f"⚠️  the decode term is >= 0.90 on {len(degenerate)} condition(s): "
            + ", ".join(degenerate)
        )
        shares = [
            f"{guard}/{family} {ordering_share:.0%}"
            for guard, block in results.items()
            for family, ordering_share in (
                (f, TABLE_TWO[guard][f] / m["unblocked"])
                for f, m in block["marginals"].items()
                if m["decode_rate"] >= 0.90 and m["unblocked"]
            )
        ]
        print(
            "    On those the decode term is not the binding one: D&notB recovers "
            + ", ".join(shares)
            + " of the unblocked prompts, so an ordering of it largely restates the "
            "BLOCK RATE and is not a decode finding."
        )

    out = args.out or ANALYSIS / "paired_cell_ordering_20260821.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "reading_percentile": READING_PERCENTILE,
                "alpha": alpha,
                "bootstrap_draws": draws,
                "method": (
                    "exact McNemar over discordant items plus a paired bootstrap interval on "
                    "the D&notB rate difference, item-paired across conditions; Holm adjusted "
                    "within each guard's three pairs. Compared against the independent Wilson "
                    "overlap the paper currently reasons from."
                ),
                "guards": results,
            },
            indent=1,
        )
    )
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
