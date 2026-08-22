"""The full 2x3 factorial behind AS-6's wrapper screen, with intervals.

**External review con 8.** The wrapper screen as written applies a thresholded
"the encoded block rate exceeds its own wrapper-induced false-positive rate"
rule, which is a comparison of two marginal rates against a bar. The referee
asked for the factorial reported jointly instead: bare plaintext, wrapped
plaintext and encoded, crossed with benign and harmful, plus the INTERACTION and
its interval rather than a difference of marginals. This produces that.

The screen's verdicts do not change. What changes is that they stop being a
verdict and become an estimate with a width, which is the whole of the referee's
objection.

⚠️ **The intervals here are CONSERVATIVE and the reason is a data limitation,
not a choice.** Conditions within one arm are the same prompts rendered
differently, so they are item-paired, but the wrapper runs persisted per-item
verdicts for the encoded harmful arm ONLY (`benign_cells.jsonl` is empty in every
one of them). Five of the six cells survive as aggregate rates, so the pairing
cannot be exploited and `unpaired_interaction_interval` inflates the width by the
shared item difficulty. Consequence, stated because it decides how a reader may
use this: an interaction whose interval EXCLUDES zero here is real, and one whose
interval includes zero is NOT established as absent.

Source runs: `guard-scaffold-*` (jobs 9049076 / 9049077, 2026-08-10). Keyless,
GPU-free, seconds.

    uv run python scripts/guard_factorial.py
    uv run python scripts/guard_factorial.py --json outputs/analysis/<name>.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from internals_safety.measurements.intervals import (
    unpaired_difference_interval,
    unpaired_interaction_interval,
    wilson,
    z_for,
)

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD_GLOB = "outputs/runs/as6_phase1/*/guard-scaffold-*/results.json"

#: Reported in Table 2 after every screen. The factorial is computed for every
#: condition the run measured; these are the ones a claim may rest on.
REPORTED = ("homoglyph", "zero_width", "reverse_words")


def _counts(rate: float | None, n: int) -> tuple[int, int]:
    """A rate back to (successes, total). The runs stored rates, not counts."""
    if rate is None:
        raise ValueError("missing arm rate — a factorial cell cannot be inferred")
    return round(rate * n), n


def factorial(alpha: float) -> dict:
    z = z_for(alpha)
    out: dict[str, dict] = {}
    for path in sorted(REPO.glob(SCAFFOLD_GLOB.replace("outputs/", "outputs/"))):
        record = json.loads(path.read_text())
        guard = record["config"]["guard"]["name"]
        plain_h = _counts(record["plain_block_rate"], 100)
        plain_b = _counts(record.get("plain_benign_block_rate"), 100)
        plain_gap = unpaired_difference_interval(*plain_h, *plain_b, z)
        conditions = {}
        for summary in record["summaries"]:
            scaffold, benign = summary.get("scaffold_arm"), summary.get("benign_arm")
            if not scaffold or not benign:
                continue
            n = scaffold["n"]
            wrapped_h = _counts(scaffold["scaffold_harmful_block_rate"], n)
            wrapped_b = _counts(scaffold["scaffold_benign_block_rate"], n)
            enc_h = _counts(summary["block_rate"], n)
            enc_b = _counts(benign["benign_block_rate"], n)
            conditions[summary["family"]] = {
                "reported": summary["family"] in REPORTED,
                "cells": {
                    "plain_harmful": {"rate": plain_h[0] / 100, "ci": wilson(*plain_h, z)},
                    "plain_benign": {"rate": plain_b[0] / 100, "ci": wilson(*plain_b, z)},
                    "wrapped_harmful": {"rate": wrapped_h[0] / n, "ci": wilson(*wrapped_h, z)},
                    "wrapped_benign": {"rate": wrapped_b[0] / n, "ci": wilson(*wrapped_b, z)},
                    "encoded_harmful": {"rate": enc_h[0] / n, "ci": wilson(*enc_h, z)},
                    "encoded_benign": {"rate": enc_b[0] / n, "ci": wilson(*enc_b, z)},
                },
                "harm_gap": {
                    "plain": {"point": plain_h[0] / 100 - plain_b[0] / 100, "ci": plain_gap},
                    "wrapped": {
                        "point": (wrapped_h[0] - wrapped_b[0]) / n,
                        "ci": unpaired_difference_interval(*wrapped_h, *wrapped_b, z),
                    },
                    "encoded": {
                        "point": (enc_h[0] - enc_b[0]) / n,
                        "ci": unpaired_difference_interval(*enc_h, *enc_b, z),
                    },
                },
                # The two interactions the screen's verdict actually rests on.
                "interaction": {
                    "wrapper_alone": unpaired_interaction_interval(
                        plain_h, plain_b, wrapped_h, wrapped_b, z
                    ),
                    "encoding_beyond_wrapper": unpaired_interaction_interval(
                        wrapped_h, wrapped_b, enc_h, enc_b, z
                    ),
                    "total": unpaired_interaction_interval(plain_h, plain_b, enc_h, enc_b, z),
                },
            }
        out[guard] = {"run": path.parent.name, "conditions": conditions}
    if not out:
        raise FileNotFoundError(f"no scaffold runs under {SCAFFOLD_GLOB}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", type=Path, help="also write the full result here")
    parser.add_argument("--all", action="store_true", help="print every condition, not just reported")
    args = parser.parse_args()

    result = factorial(args.alpha)
    for guard, payload in result.items():
        print(f"\n=== {guard}   ({payload['run']})")
        print(f"{'condition':17s} {'gap plain':>11s} {'gap wrapped':>12s} {'gap encoded':>12s}"
              f" {'wrapper':>18s} {'encoding|wrapper':>22s}")
        for name, cond in payload["conditions"].items():
            if not (args.all or cond["reported"]):
                continue
            gap = cond["harm_gap"]
            wrap = cond["interaction"]["wrapper_alone"]
            enc = cond["interaction"]["encoding_beyond_wrapper"]
            star = "*" if (enc[1] > 0 or enc[2] < 0) else " "
            mark = "" if cond["reported"] else "  (screened out)"
            print(f"{name:17s} {gap['plain']['point']:>11.2f} {gap['wrapped']['point']:>12.2f}"
                  f" {gap['encoded']['point']:>12.2f}"
                  f" {wrap[0]:>7.2f} [{wrap[1]:.2f},{wrap[2]:.2f}]"
                  f" {enc[0]:>8.2f} [{enc[1]:.2f},{enc[2]:.2f}]{star}{mark}")
    print("\n* = the encoding-beyond-wrapper interval excludes zero.")
    print("Intervals are CONSERVATIVE (per-item pairing unavailable); a null here is not a null.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
