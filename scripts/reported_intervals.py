#!/usr/bin/env python3
"""An interval on EVERY rate and EVERY gap in Tables 1 and 2. (Review 3, Q6.)

**The objection, stated as the referee stated it.** Both tables report bare
point estimates, and intervals appear only in prose and only for selected
difference-of-differences. A reader cannot see which component
(harmful refusal or benign refusal) produces an observed gap change, nor how
much of any of it is resolvable at ``n=100``.

**Provenance, established before any number was computed here.** Table 1 is the
four ``plain-baseline-*`` runs and nothing else: each supplies BOTH its
plaintext and its homoglyph columns from one job, so no cross-run difference
enters the comparison. Table 2 is the four ``scaffold-control-*`` runs, each
carrying all three arms. Several other runs on disk measure the same cells to
within 0.03; they are replicates and are quantified by
``scripts/replicate_spread.py``, not mixed in here.

**Which interval, and why each is the right one.**

* **Rates** get Wilson. At ``n=100`` with counts as low as 1, the normal
  approximation's lower bound runs negative exactly where this paper's smallest
  cells sit.
* **Gaps** get the unpaired Wald difference, and unpaired is CORRECT rather
  than a compromise: the harm gap's two arms are the harmful and the benign
  corpus, which are different prompts. There is no pairing to exploit.

⚠️ **WHAT THIS CANNOT DO, and it is the referee's con 6.** The gap DECOMPOSITION
in Section 5 (template term plus character term) compares gaps ACROSS
conditions on the same items, and that contrast is genuinely paired. Computing
it requires per-item verdicts in the plaintext and scaffold arms.
``phase0_regime_map`` persists per-item records for the ENCODED arm only
(``cells.jsonl`` / ``benign_cells.jsonl``); the plaintext and scaffold arms
survive only as aggregate rates inside ``results.json`` readings. So the
decomposition's intervals are unpaired because the data to pair them was not
written down, not because unpaired was chosen. Unpaired is the CONSERVATIVE
direction here, because positive item-level correlation across conditions would
shrink the true interval, so the reported width is an upper bound. The fix is
a persistence change plus a re-run, not a re-analysis.

**Reads cached run records only. No model, no judge, no GPU, no spend.**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internals_safety.measurements.intervals import (
    unpaired_difference_interval,
    wilson,
    z_for,
)

MODEL_ORDER = (
    "llama3_1_8b_instruct",
    "qwen2_5_7b_instruct",
    "tulu3_8b",
    "mistral_7b_instruct",
)


def readings(run_dir: Path) -> dict[str, dict]:
    record = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return {r["instrument"]: (r.get("detail") or {}) for r in record.get("readings", [])}


def counts(rate: float | None, n: int) -> int | None:
    """A stored rate at n=100 recovers its count exactly; refuse to guess otherwise."""
    if rate is None:
        return None
    k = round(rate * n)
    if abs(k - rate * n) > 1e-6:
        raise ValueError(f"rate {rate} is not a whole count out of {n}")
    return k


def cells_from(run_dir: Path, which: str) -> list[tuple[str, float, float, int]]:
    """(condition, harmful_rate, benign_rate, n) for one run."""
    r = readings(run_dir)
    plain, behavior = r.get("behavior_plain", {}), r.get("behavior", {})
    out: list[tuple[str, float, float, int]] = []
    if plain:
        out.append(
            ("plaintext", plain["plain_harmful_refusal_rate"], plain["plain_benign_refusal_rate"], plain["plain_n"])
        )
    if which == "table2" and "scaffold_harmful_refusal_rate" in behavior:
        out.append(
            (
                "scaffold",
                behavior["scaffold_harmful_refusal_rate"],
                behavior["scaffold_benign_refusal_rate"],
                behavior["scaffold_n_harmful"],
            )
        )
    if behavior:
        label = "homoglyph" if which == "table1" else "encoded"
        out.append((label, behavior["refusal_rate"], behavior["benign_arm_refusal_rate"], behavior["n"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--table2-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    z = z_for(args.alpha)
    everything: dict[str, dict] = {}

    for which, runs in (("table1", args.table1_runs), ("table2", args.table2_runs)):
        print(f"\n=== {which} ===")
        print(
            f"{'model':16s} {'condition':10s} {'harmful':>20s} {'benign':>20s} {'gap':>22s}"
        )
        by_model = {r.parts[-2]: r for r in runs}
        for model in MODEL_ORDER:
            run_dir = by_model.get(model)
            if run_dir is None:
                continue
            for condition, hr, br, n in cells_from(run_dir, which):
                kh, kb = counts(hr, n), counts(br, n)
                hlo, hhi = wilson(kh, n, z)
                blo, bhi = wilson(kb, n, z)
                glo, ghi = unpaired_difference_interval(kh, n, kb, n, z)
                print(
                    f"{model[:16]:16s} {condition:10s} "
                    f"{hr:5.2f} [{hlo:.2f},{hhi:.2f}] "
                    f"{br:5.2f} [{blo:.2f},{bhi:.2f}] "
                    f"{hr - br:+6.2f} [{glo:+.2f},{ghi:+.2f}]"
                )
                everything.setdefault(which, {}).setdefault(model, {})[condition] = {
                    "n": n,
                    "harmful": hr,
                    "harmful_ci": [hlo, hhi],
                    "benign": br,
                    "benign_ci": [blo, bhi],
                    "gap": hr - br,
                    "gap_ci": [glo, ghi],
                }

    print(
        "\nEvery gap interval above is unpaired BY CONSTRUCTION: the two arms are "
        "different corpora.\nThe Section 5 decomposition is a different contrast, it "
        "IS pairable, and the per-item\nverdicts needed to pair it were never "
        "persisted for the plaintext and scaffold arms."
    )

    if args.json:
        args.json.write_text(json.dumps(everything, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
