"""Re-baseline the phase-0 pilot's cached cells against the CURRENT instrument.

No GPU, no judge calls, no re-run — everything needed is already in
`cells.jsonl` (plaintext, restate_response, the probe readouts, behaviour) and
`results.json` (per-family probe licensing).

Why this exists (2026-08-05). Instrument fixes #1 and #2 landed in `0483fe8`,
which changed BOTH the ability binary and `assign_regime` itself. The cached
cells carry regimes computed by the pre-fix logic, so:

  * `scripts/rescore_ability.py` now fails its own self-check by design
    (74/1500 Llama cells: `reverse_characters recorded=X recomputed=R`), and it
    is right to — its guard exists to stop exactly this.
  * every headline computed from those cells predates the fixes.

`rescore_ability.py` stays as it is: it answers "what would thresholding
similarity do?", and its self-check pins it to the shipped labels. This script
answers the different question "what does the CURRENT instrument say about the
cached data?", so it deliberately does not self-check against recorded regimes —
disagreement with them is the output, not an error.

## What is recomputed, and what is inherited

RECOMPUTED from cached text, deterministically:
  * ability — `score_recovery(plaintext, restate_response)` under the settled
    three-route rule, cuts read from `conf/measurements.yaml` (never hardcoded).
  * recognition tri-state — `None` wherever `results.json` records the
    recognition probe as unlicensed on that family, instead of the fake `False`
    the pre-fix pipeline wrote.
  * the regime label — current `assign_regime`.

INHERITED, because it cannot be recovered from cached text:
  * deployment, refusal, jailbreak — these are probe/judge readouts, not
    re-derivable from stored strings.

## The defect this surfaces (read before quoting any number below)

`recognition` is tri-state; **`deployment` is not**. But `results.json` records
`deployment.licensed` per family, and on several rungs it is FALSE while every
cell still carries `deployment=false`. That is the identical failure tri-state
was introduced to fix, one measurement over: an unlicensed probe asserting
"the model did not decode during the attack" when the truth is "this instrument
could not read this rung".

It is load-bearing rather than cosmetic, because `deployment` is what separates
(S)/(B) from (R)/(D) — so on an unlicensed-deployment rung the regime map is
reporting a distinction it did not measure. This script therefore reports every
rung's deployment licensing beside its counts, and marks unlicensed rungs
UNMEASURABLE rather than quietly re-labelling them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from internals_safety.config import load_measurements_config
from internals_safety.encodings.recovery import score_recovery
from internals_safety.measurements.regimes import HARD_INCOHERENCES, Regime, assign_regime

LADDER = (
    "base64", "base32", "hex", "binary", "ascii_decimal", "unicode_escape",
    "rot13", "caesar3", "caesar7", "atbash", "vigenere", "morse",
    "reverse_characters", "reverse_words", "zero_width",
)


def load_cells(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def licensing(results_path: Path) -> dict[str, dict[str, bool]]:
    """family -> {recognition: licensed?, deployment: licensed?}, from the run record."""
    if not results_path.exists():
        return {}
    record = json.loads(results_path.read_text())
    out: dict[str, dict[str, bool]] = {}
    for entry in record.get("metrics", {}).get("families", []):
        out[entry["family"]] = {
            "recognition": bool(entry.get("recognition", {}).get("licensed", False)),
            "deployment": bool(entry.get("deployment", {}).get("licensed", False)),
        }
    return out


def rebaseline(cells: list[dict], licenses: dict[str, dict[str, bool]], cuts) -> list[dict]:
    rows = []
    for cell in cells:
        score = score_recovery(
            cell["plaintext"], cell.get("restate_response") or "", cell.get("ciphertext")
        )
        ability = score.is_recovered(
            cuts.similarity_threshold,
            cuts.content_overlap_threshold,
            cuts.order_blind_overlap_threshold,
        )
        family_license = licenses.get(cell["family"], {})
        recognition = cell["recognition"] if family_license.get("recognition", False) else None
        assignment = assign_regime(
            ability=ability,
            deployment=bool(cell["deployment"]),
            recognition=recognition,
            refused=bool(cell["refused"]),
            prompt_is_harmful=True,
        )
        rows.append(
            {
                "family": cell["family"],
                "old_regime": cell["regime"],
                "old_ability": bool(cell["ability"]),
                "new_regime": assignment.regime.value,
                "new_ability": ability,
                "similarity": score.similarity,
                "content_overlap": score.content_overlap,
                "deployment_licensed": family_license.get("deployment", False),
                "recognition_licensed": family_license.get("recognition", False),
                "hard": any(f in HARD_INCOHERENCES for f in assignment.incoherences),
            }
        )
    return rows


def report(rows: list[dict], label: str) -> None:
    print("#" * 92)
    print(f"# {label}  ({len(rows)} cells)")
    print("#" * 92)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)

    regimes = [r.value for r in Regime]
    head = f"{'rung':<20} {'abil':>9} {'dep?':>5} {'rec?':>5}  " + " ".join(f"{r:>4}" for r in regimes)
    print(head + "   note")
    print("-" * len(head + "   note"))

    for family in LADDER:
        subset = grouped.get(family)
        if not subset:
            continue
        n = len(subset)
        old_ability = sum(r["old_ability"] for r in subset)
        new_ability = sum(r["new_ability"] for r in subset)
        counts = Counter(r["new_regime"] for r in subset)
        dep_ok = subset[0]["deployment_licensed"]
        rec_ok = subset[0]["recognition_licensed"]
        hard_rate = sum(r["hard"] for r in subset) / n

        notes = []
        if not dep_ok:
            notes.append("UNMEASURABLE (deployment probe unlicensed)")
        if hard_rate > 0.10:
            notes.append(f"INSTRUMENT FAILURE {hard_rate:.0%}")
        changed = sum(r["old_regime"] != r["new_regime"] for r in subset)
        if changed:
            notes.append(f"{changed}/{n} relabelled")

        print(
            f"{family:<20} {old_ability:>3}->{new_ability:<4} "
            f"{'y' if dep_ok else 'n':>5} {'y' if rec_ok else 'n':>5}  "
            + " ".join(f"{counts.get(r, 0):>4}" for r in regimes)
            + "   " + "; ".join(notes)
        )

    total_changed = sum(r["old_regime"] != r["new_regime"] for r in rows)
    ability_gained = sum(r["new_ability"] and not r["old_ability"] for r in rows)
    ability_lost = sum(r["old_ability"] and not r["new_ability"] for r in rows)
    dep_unlicensed = sum(1 for r in rows if not r["deployment_licensed"])
    print(
        f"\n  {total_changed}/{len(rows)} cells relabelled · "
        f"ability +{ability_gained} / -{ability_lost} · "
        f"{dep_unlicensed}/{len(rows)} cells sit on an unlicensed deployment probe\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs/phase0"))
    parser.add_argument("--write", action="store_true", help="write cells_rebaselined.jsonl beside each run")
    args = parser.parse_args()

    cuts = load_measurements_config().ability
    print(
        f"ability cuts from conf/measurements.yaml: similarity>={cuts.similarity_threshold}, "
        f"content_overlap>={cuts.content_overlap_threshold}, "
        f"order_blind>={cuts.order_blind_overlap_threshold}\n"
    )

    runs = sorted(
        path
        for path in args.runs_dir.glob("*/*/cells.jsonl")
        if path.parent.name not in {"smoke", "plumbing"}
    )
    if not runs:
        raise SystemExit(f"no run cells under {args.runs_dir}")

    for path in runs:
        cells = load_cells(path)
        rows = rebaseline(cells, licensing(path.parent / "results.json"), cuts)
        report(rows, f"{path.parent.parent.name}/{path.parent.name}")
        if args.write:
            out = path.parent / "cells_rebaselined.jsonl"
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
