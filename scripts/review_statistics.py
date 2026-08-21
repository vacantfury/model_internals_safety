"""Uncertainty and threshold-sensitivity for AS-6's reported cells.

Answers the parts of external-review cons 5 and 7 that the LOCAL data supports,
and is explicit about the parts it does not (see ``WHAT THIS CANNOT DO``).

Con 7: only ``decoded_not_blocked`` carried an interval; block rates, the
``blocked_without_decoding`` bound and the rest were bare point estimates.
Con 5: the paper says it reports an operating-point sweep and never shows one,
which leaves post-hoc threshold selection unexcluded from the outside.

Source is the per-prompt record, not a summary: ``scores-b10/cells.jsonl`` holds
100 harmful prompts x 19 conditions per guard with ``decode_score`` and the
per-condition ``decode_threshold``. Keyless, no GPU, no model, seconds.

WHAT THIS CANNOT DO, and why it is stated here rather than discovered later:

* **No AUROC intervals.** AUROC is a harmful-versus-benign separation and the
  benign-arm scores are not in these records.
* **No benign-arm or wrapper-arm rates, and no factorial interaction term**
  (con 8). Those runs' per-prompt records live on the cluster; only summary
  numbers reached ``phase1_map.md`` §2. They need a down-sync, not a re-run.

The sweep is therefore in SCORE space rather than percentile space. Re-deriving a
percentile threshold needs the same-condition benign distribution, which is not
local; sweeping the raw cut needs nothing further and answers the question the
referee actually asked, which is whether the reported cells are threshold-robust.

    uv run python scripts/review_statistics.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "outputs" / "runs" / "as6_phase1"
OUT = REPO_ROOT / "outputs" / "analysis"

#: The run carrying per-prompt decode scores. The sibling ``as6p1-full`` records
#: the same cells WITHOUT them, so it cannot answer con 5.
SOURCE = "scores-b10"

#: Conditions the paper reports. Everything else is (U) and is summarised only.
REPORTED = {
    "llama_guard_3_8b": ["homoglyph", "zero_width", "fullwidth", "reverse_words"],
    "wildguard": ["zero_width", "homoglyph", "reverse_words"],
}


# constant(z): 1.96 is the standard-normal two-sided 95% quantile, a property of the
# normal distribution and of the confidence level the paper reports, not a knob.
def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Chosen over normal-approximation for a reason.

    At n=100 with counts as low as 7, the normal approximation's lower bound
    runs negative and its coverage is poor exactly where this paper's smallest
    and most-quoted cells sit. Wilson stays inside [0, 1] and does not degenerate
    at zero, which matters for ``blocked_without_decoding``, whose finding IS a
    count near zero.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


#: The settled operating point (``conf/measurements.yaml`` probes.reading_percentile).
#: Duplicated as a literal ON PURPOSE: this script must be able to say "the run you
#: gave me predates the settled knob" even when run against an old checkout, and a
#: guard that reads the same file the run would have read cannot detect that.
# constant: not a knob here but a MIRROR of the settled knob
# (conf/measurements.yaml probes.reading_percentile), duplicated so this script can
# detect a run that predates it; reading the config would defeat the check.
SETTLED_READING_PERCENTILE = 75.0


def operating_point(guard: str) -> float | None:
    """The reading percentile the source run was produced at, or None."""
    record = RUNS / guard / SOURCE / "results.json"
    if not record.exists():
        return None
    data = json.loads(record.read_text())
    probes = data.get("config", {}).get("measurements", {}).get("probes", {})
    return probes.get("reading_percentile")


def threshold_dependent_is_valid(guard: str) -> tuple[bool, str]:
    """Whether a threshold-DEPENDENT quantity from this run may be reported.

    ⚠️ This guard exists because the obvious thing to do here is wrong. The local
    per-prompt records are real, complete and load cleanly, and every cell count
    derived from them is a correct count OF THAT RUN. But ``scores-b10`` was
    produced at ``reading_percentile = 50``, which was retired on 2026-08-08, and
    the paper's Table 2 comes from a later run at 75 whose records are NOT local.
    So the numbers computed here reproduce the SUPERSEDED map exactly (Llama
    Guard 8/17/12/25, matching ``phase1_map.md`` §0.6's retired table) and
    disagree with the paper by up to 17 points on one condition.

    Attaching confidence intervals to those would put an interval on a number the
    paper does not contain, which is worse than having no interval. Threshold-
    INDEPENDENT quantities are unaffected and stay reportable: a block rate does
    not consult the decode read, and the block rates here match Table 2 exactly.
    """
    found = operating_point(guard)
    if found is None:
        return False, "source run records no reading_percentile"
    if abs(found - SETTLED_READING_PERCENTILE) > 1e-9:
        return False, (f"source run is at reading_percentile={found:g}, settled is "
                       f"{SETTLED_READING_PERCENTILE:g} -- threshold-dependent "
                       "quantities need the paper's own run down-synced")
    return True, ""


def load(guard: str) -> list[dict]:
    path = RUNS / guard / SOURCE / "cells.jsonl"
    if not path.exists():
        raise SystemExit(f"missing per-prompt records: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_condition(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    return grouped


def rates(cells: list[dict]) -> dict:
    """Every rate the paper asserts for one condition, each with its interval."""
    n = len(cells)
    counts = defaultdict(int)
    for cell in cells:
        counts[cell["cell"]] += 1
    blocked = sum(1 for cell in cells if cell["blocked"])
    out = {"n": n, "blocked": blocked, "block_rate": blocked / n if n else float("nan")}
    low, high = wilson(blocked, n)
    out["block_rate_ci95"] = [low, high]
    for name in ("decoded_not_blocked", "blocked_on_content",
                 "blocked_without_decoding", "never_decoded", "unmeasured"):
        count = counts[name]
        low, high = wilson(count, n)
        out[name] = {"count": count, "rate": count / n if n else float("nan"),
                     "ci95": [low, high]}
    return out


# plumbing(steps): 41 is the resolution of a diagnostic curve, affecting only how finely
# the x-axis is sampled between the observed min and max; no reported quantity
# depends on it, so it is display granularity rather than a parameter.
def sweep(cells: list[dict], steps: int = 41) -> dict:
    """Cell counts as a function of the decode cut, over the observed range.

    The recorded ``decode_threshold`` is marked so a reader can see where the
    reported operating point sits inside the curve rather than taking it on
    trust. A cell that only exists in a narrow band of cuts is threshold-driven;
    one that persists across the range is not, and that is the distinction con 5
    asks the paper to make visible.
    """
    scores = [cell["decode_score"] for cell in cells]
    if not scores:
        return {}
    low, high = min(scores), max(scores)
    operating = cells[0]["decode_threshold"]
    points = []
    for index in range(steps):
        cut = low + (high - low) * index / (steps - 1)
        decoded_not_blocked = sum(
            1 for cell in cells if cell["decode_score"] >= cut and not cell["blocked"]
        )
        blocked_without = sum(
            1 for cell in cells if cell["decode_score"] < cut and cell["blocked"]
        )
        points.append({"cut": cut,
                       "decoded_not_blocked": decoded_not_blocked,
                       "blocked_without_decoding": blocked_without})
    return {"score_min": low, "score_max": high,
            "operating_point": operating, "points": points}


def main() -> int:
    report: dict = {"source_run": SOURCE, "guards": {}}
    for guard in sorted(REPORTED):
        grouped = by_condition(load(guard))
        valid, reason = threshold_dependent_is_valid(guard)
        entry: dict = {"conditions": {},
                       "threshold_dependent_reportable": valid,
                       "withheld_reason": reason or None,
                       "source_reading_percentile": operating_point(guard)}
        for condition, cells in sorted(grouped.items()):
            record = rates(cells)
            if not valid:
                # Fail closed, per the instrument contract: an unreportable
                # number is withheld, never emitted with a caveat elsewhere.
                for key in ("decoded_not_blocked", "blocked_on_content",
                            "blocked_without_decoding", "never_decoded", "unmeasured"):
                    record[key] = {"withheld": reason}
            if condition in REPORTED[guard]:
                record["sweep"] = sweep(cells) if valid else {"withheld": reason}
                record["reported"] = True
            entry["conditions"][condition] = record

        # The paper's cross-condition claim: blocked_without_decoding is "at
        # most 5 per 100" everywhere. Report it as a bound WITH an interval on
        # the worst condition, which is what con 7 asks for.
        if valid:
            worst = max(grouped, key=lambda c: sum(
                1 for cell in grouped[c] if cell["cell"] == "blocked_without_decoding"))
            count = sum(1 for cell in grouped[worst]
                        if cell["cell"] == "blocked_without_decoding")
            entry["blocked_without_decoding_bound"] = {
                "worst_condition": worst, "count": count, "n": len(grouped[worst]),
                "ci95": list(wilson(count, len(grouped[worst]))),
            }
        else:
            entry["blocked_without_decoding_bound"] = {"withheld": reason}
        report["guards"][guard] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "review_statistics_20260821.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for guard, entry in report["guards"].items():
        print(f"\n{guard}  (source run at reading_percentile="
              f"{entry['source_reading_percentile']})")
        if not entry["threshold_dependent_reportable"]:
            print(f"  ⚠️  THRESHOLD-DEPENDENT QUANTITIES WITHHELD: {entry['withheld_reason']}")
            print("      Block rates below do not consult the decode read and stay valid.")
        print(f"  {'condition':18s} block rate (95% CI)")
        for condition in REPORTED[guard]:
            record = entry["conditions"][condition]
            low, high = record["block_rate_ci95"]
            print(f"  {condition:18s} {record['block_rate']:.2f} [{low:.2f}-{high:.2f}]")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
