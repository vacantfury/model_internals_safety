"""Recompute every paper claim whose SET lives in a run artefact.

**Why this is a script and not a test.** `outputs/` is gitignored, so this cannot
run in the suite. Keeping it separate from `tests/test_paper_claim_integrity.py`
is deliberate: a checker that silently skipped this half while reporting the
mechanical half green would be the "--dry-run returns before the guard" shape,
which has cost this repo two queue cycles. The suite checks what a fresh clone
can check; this checks what only a machine holding the runs can.

**What it is for.** A screen adopted after a table exists never touches the
table's numbers, and a screen that later shrinks one set leaves every OTHER
sentence ranging over the old one. Both happened on 2026-08-22, one per paper,
and both were found by recomputing the claim's set from the artefact. This
mechanises that.

**One truth per claim, and it is the paper.** The ledger (`conf/claim_sets.yaml`)
holds a locating regex and a recipe name, never a third copy of the value. A
recipe not implemented here is a hard failure rather than a skip.

    uv run python scripts/claim_sets.py            # exits 1 on any mismatch
    uv run python scripts/claim_sets.py --show     # print the sets, check nothing

Keyless, GPU-free, seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from internals_safety.config import load_measurements_config
from internals_safety.paper_claims import WORD_NUMBERS

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "conf" / "claim_sets.yaml"
PAPER_DIR = REPO / "paper"
ANALYSIS = REPO / "outputs" / "analysis"

#: The echo-screen artefact. Discovered by glob, never pinned: names carry dates
#: and a path in source is a path that rots.
ECHO_GLOB = "echo_displacement_*.json"

#: The encodings the pipeline section reports. Three, and the paper says so.
REPORTED_ENCODINGS = ("fullwidth", "homoglyph", "zero_width")


def _latest_echo() -> Path:
    found = sorted(ANALYSIS.glob(ECHO_GLOB))
    found = [p for p in found if "spread" not in p.name]
    if not found:
        raise FileNotFoundError(f"no {ECHO_GLOB} under {ANALYSIS}")
    return found[-1]


def _echo_cells() -> dict[str, dict[tuple[str, str], object]]:
    """{run-kind: {(model_or_stage, rung): clears}}, `clears` TRI-STATE.

    None means the screen could not read that cell (no non-echoing cells in one
    arm). Folding it into False is the defect this repo has now recorded three
    times, so it is carried through.
    """
    raw = json.loads(_latest_echo().read_text())
    out: dict[str, dict[tuple[str, str], object]] = defaultdict(dict)
    for path, rungs in raw.items():
        name = path.split("/")[-1]
        kind = "spread" if "/spread-" in path else "ladder" if "/ladder-plain-" in path else "other"
        label = path.split("/")[-2] if kind == "spread" else name.split("_")[0].split("-")[-1]
        for rung, record in rungs.items():
            out[kind][(label, rung)] = record["clears"]
    return out


def _spread_families() -> dict[tuple[str, str], dict]:
    runs = REPO / "outputs" / "runs" / "phase0"
    out: dict[tuple[str, str], dict] = {}
    for directory in sorted(runs.glob("*/spread-*")):
        record = json.loads((directory / "results.json").read_text())
        for family in record["metrics"]["families"]:
            out[(directory.parent.name, family["family"])] = family
    return out



def _resolve_source(claim: dict) -> list[Path]:
    """The artefact a claim says it came from, as paths, or a hard failure.

    The provenance half of this file. A claim that names its source is a claim
    whose POINTER recomputes, and on 2026-08-21 a wrong pointer blocked five
    items for a day while the correct data sat on disk. `source` is a glob under
    `outputs/` so a dated filename never has to be pinned in code.
    """
    pattern = claim.get("source")
    if not pattern:
        raise ValueError(
            f"{claim['id']}: recipe requires `source:` and the ledger omits it — "
            "a provenance claim with no artefact to open is the defect this field exists to catch"
        )
    found = sorted((REPO / "outputs").glob(pattern))
    if not found:
        raise ValueError(
            f"{claim['id']}: `source: {pattern}` matched nothing under outputs/ — "
            "the pointer names an artefact that is not there"
        )
    return found


def _by_guard(paths: list[Path]) -> dict[str, dict]:
    """Guard name read from INSIDE each record, never from its filename."""
    out = {}
    for p in paths:
        record = json.loads(p.read_text())
        guard = record.get("guard") or record.get("config", {}).get("guard")
        if guard is None:
            raise ValueError(f"{p.name}: no guard field; cannot attribute this record")
        out[guard] = record
    return out


#: Table 2 of AS-6. One row per reported guard-condition pair.
_TABLE2_ROW = re.compile(
    r"&\s*\\texttt\{(?P<cond>[a-z\\_]+)\}(?:\$\^\\dagger\$)?\s*&"
    r"\s*(?P<auroc>[0-9.]+)\s*&\s*(?P<dnb>[0-9]+)\s*\["
)


def _as6_table2(kit: Path) -> list[tuple[str, str, int]]:
    """[(guard, condition, D&notB)] parsed from the kit itself.

    The paper's own set, so the check compares the paper against the artefact
    rather than against a list restated here.
    """
    body = kit.read_text()
    start = body.index(r"\label{tab:map}")
    block = body[body.rindex(r"\begin{table}", 0, start) : start]
    rows, guard = [], None
    for line in block.splitlines():
        if "multirow" in line:
            guard = "llama_guard_3_8b" if "Llama Guard" in line else "wildguard"
        m = _TABLE2_ROW.search(line)
        if m and guard:
            rows.append((guard, m.group("cond").replace("\\_", "_"), int(m.group("dnb"))))
    return rows


# --------------------------------------------------------------------------
# Recipes. CLOSED vocabulary — `conf/claim_sets.yaml` may name only these.
# --------------------------------------------------------------------------


def spread_echo_measured_cells(claim: dict) -> tuple[int, str]:
    cells = _echo_cells()["spread"]
    measured = [k for k, v in cells.items() if v is not None]
    unmeasured = [k for k, v in cells.items() if v is None]
    return len(measured), (
        f"{len(measured)} measured of {len(cells)} cells; "
        f"{len(unmeasured)} unmeasured and excluded: {sorted(unmeasured)}"
    )


def ladder_reported_cells_rejected(claim: dict) -> tuple[int, str]:
    cells = _echo_cells()["ladder"]
    reported = {k: v for k, v in cells.items() if k[1] in REPORTED_ENCODINGS}
    rejected = sorted(k for k, v in reported.items() if v is False)
    unmeasured = sorted(k for k, v in reported.items() if v is None)
    if unmeasured:
        raise ValueError(
            f"unmeasured reported cells {unmeasured} — a rejection count over a "
            "partly unmeasured set reads as a measured negative, which is the "
            "defect this file exists to catch"
        )
    return len(rejected), f"rejected {rejected}"


def ladder_reported_cells_total(claim: dict) -> tuple[int, str]:
    cells = _echo_cells()["ladder"]
    reported = [k for k in cells if k[1] in REPORTED_ENCODINGS]
    stages = sorted({k[0] for k in reported})
    return len(reported), f"{len(stages)} stages {stages} x {len(REPORTED_ENCODINGS)} encodings"


def spread_rungs_passing_every_screen(claim: dict) -> tuple[int, str]:
    """Readability, exact invertibility, and the echo screen, on ALL four models.

    The ability cut comes from the settled config rather than a literal here: a
    threshold that lives in two places can move in one of them.
    """
    cut = load_measurements_config().ability.similarity_threshold
    echo = _echo_cells()["spread"]
    families = _spread_families()
    per_rung: dict[str, list[bool]] = defaultdict(list)
    for (model, rung), family in families.items():
        per_rung[rung].append(
            family["ability_rate"] >= cut
            and family["invertibility"] == "exact"
            and echo.get((model, rung)) is True
        )
    passing = sorted(r for r, flags in per_rung.items() if all(flags) and len(flags) == 4)
    return len(passing), f"passing on all four at ability cut {cut}: {passing or '(none)'}"



# --- AS-6. These carry `source:`, so the POINTER is checked, not just the count.


def _as6_bwd_cells(claim: dict) -> list[tuple[str, str, float | None]]:
    """[(guard, condition, blocked_without_decoding_rate)] over all 19 x 2 pairs.

    The rate is None where the decode measurement is unlicensed. Tri-state is
    carried, never folded: a bound quoted over a partly unmeasured set reads as a
    measured negative, which is what section 5.8 got wrong once already.
    """
    out = []
    for path in _resolve_source(claim):
        record = json.loads(path.read_text())
        guard = record["config"]["guard"]
        guard = guard["name"] if isinstance(guard, dict) else guard
        for summary in record["summaries"]:
            out.append((guard, summary["family"], summary.get("blocked_without_decoding_rate")))
    return out


def as6_table2_provenance(claim: dict) -> tuple[int, str]:
    """Every Table 2 count, against the artefact the ledger says it came from.

    Returns the number of MISMATCHES, so 0 is the passing value. This is the
    provenance check: it fails if the named source does not contain the paper's
    numbers, which is the failure that cost a day on 2026-08-21 when a ledger
    sentence named the causal-intervention run as Table 2's source.
    """
    percentile = claim["percentile"]
    by_guard = _by_guard(_resolve_source(claim))
    mismatches, checked = [], 0
    for kit in _kits(claim["paper"]):
        label = kit.relative_to(REPO).parts[2]
        for guard, condition, asserted in _as6_table2(kit):
            # A wrong `source:` lands here, and it is the whole point of this
            # recipe, so it must read as a diagnostic and never as a traceback.
            if guard not in by_guard:
                raise ValueError(
                    f"source {claim['source']} holds no record for guard {guard!r} "
                    f"(it has {sorted(by_guard)}) — the pointer names the wrong artefact"
                )
            families = by_guard[guard].get("families")
            if not isinstance(families, dict) or condition not in families:
                raise ValueError(
                    f"source {claim['source']} has no per-condition sweep for "
                    f"{guard}/{condition} — the pointer names an artefact of the wrong KIND"
                )
            entries = {e["percentile"]: e for e in families[condition]}
            if percentile not in entries:
                mismatches.append(f"{label}/{guard}/{condition}: no p{percentile} in source")
                continue
            found = entries[percentile]["decoded_not_blocked"]
            checked += 1
            if found != asserted:
                mismatches.append(f"{label}/{guard}/{condition}: paper {asserted} vs source {found}")
    detail = f"{checked} Table 2 cells checked against {claim['source']} at p{percentile}"
    return len(mismatches), detail + ("; " + "; ".join(mismatches) if mismatches else "; all agree")


def as6_bwd_max_reported(claim: dict) -> tuple[int, str]:
    """Max blocked-without-decoding, per 100, over the pairs the paper REPORTS.

    The reported set is parsed from Table 2 rather than restated here, so the
    bound follows the table when the table changes. It did not on 2026-08-21:
    the bound still said 5, sourced from a condition the item holdout had
    withdrawn.
    """
    rates = {(g, c): r for g, c, r in _as6_bwd_cells(claim)}
    reported = {(g, c) for kit in _kits(claim["paper"]) for g, c, _ in _as6_table2(kit)}
    unmeasured = sorted(p for p in reported if rates.get(p) is None)
    if unmeasured:
        raise ValueError(
            f"reported pairs with an unlicensed decode read {unmeasured} — a bound "
            "over a partly unmeasured set reads as a measured negative"
        )
    worst = max(reported, key=lambda p: rates[p])
    return round(rates[worst] * 100), f"max over {len(reported)} reported pairs is {worst}"


def as6_bwd_zero_pairs(claim: dict) -> tuple[int, str]:
    cells = _as6_bwd_cells(claim)
    zero = [(g, c) for g, c, r in cells if r == 0.0]
    nonzero = sorted((g, c, r) for g, c, r in cells if r not in (None, 0.0))
    return len(zero), f"{len(zero)} zero; nonzero cells {nonzero}"


def as6_bwd_measurable_pairs(claim: dict) -> tuple[int, str]:
    cells = _as6_bwd_cells(claim)
    measurable = [(g, c) for g, c, r in cells if r is not None]
    return len(measurable), f"{len(measurable)} measurable of {len(cells)} pairs"


def as6_bwd_unlicensed_pairs(claim: dict) -> tuple[int, str]:
    """Unlicensed pairs, and each one's block rate, because section 5.8's
    containment argument stands or falls on those block rates being 0.00."""
    out, blocks = [], []
    for path in _resolve_source(claim):
        record = json.loads(path.read_text())
        guard = record["config"]["guard"]
        guard = guard["name"] if isinstance(guard, dict) else guard
        for summary in record["summaries"]:
            if summary.get("blocked_without_decoding_rate") is None:
                out.append((guard, summary["family"]))
                blocks.append(summary["block_rate"])
    nonzero = [b for b in blocks if b != 0.0]
    if nonzero:
        raise ValueError(
            f"unlicensed pairs with a nonzero block rate {nonzero} — section 5.8's "
            "containment argument requires every one of them to be 0.00"
        )
    return len(out), f"{len(out)} unlicensed, every block rate 0.00 (containment holds)"


def as6_length_bound_clearing_cells(claim: dict) -> tuple[int, str]:
    """How many REPORTED cells clear the monotone length-only bound.

    The paper says all six do. The set that matters is `reported`, and it has
    already moved once: the item-level holdout withdrew Llama Guard
    `fullwidth` to (U), which would have taken a "seven" sentence stale without
    touching a single number in it. So the count is recomputed from the artifact
    rather than trusted, and the total is checked too — a claim that six of six
    clear is wrong in two different ways if the denominator changes.
    """
    (source,) = _resolve_source(claim)
    data = json.loads(source.read_text())
    reported = [
        (guard, row)
        for guard, rows in data["guards"].items()
        for row in rows
        if row["reported"]
    ]
    if not reported:
        raise ValueError(
            f"{claim['id']}: the artifact marks no cell `reported` — the bound would "
            "vacuously clear on an empty set, which is the shape of a passing check "
            "that checked nothing"
        )
    clearing = [
        (guard, row)
        for guard, row in reported
        if row["observed_gap"] > row["length_only_gap_bound"]
    ]
    if len(clearing) != len(reported):
        missed = ", ".join(f"{g}/{r['family']}" for g, r in reported if (g, r) not in clearing)
        raise ValueError(
            f"{claim['id']}: {len(reported) - len(clearing)} reported cell(s) no longer "
            f"clear the length bound ({missed}); the paper's sentence asserts all of them do"
        )
    detail = ", ".join(
        f"{g}/{r['family']} {r['observed_gap'] - r['length_only_gap_bound']:+.2f}"
        for g, r in clearing
    )
    return len(clearing), f"{len(reported)} reported cells, all clearing: {detail}"


def _paired_pairs(claim: dict) -> list[dict]:
    (source,) = _resolve_source(claim)
    data = json.loads(source.read_text())
    return [row for block in data["guards"].values() for row in block["pairs"]]


def as6_paired_separating_pairs(claim: dict) -> tuple[int, str]:
    """Pairs separating under the paired test, after the Holm adjustment.

    The numerator of a "three of the six" sentence. Both halves move when the
    reported set moves: the item-level holdout has already taken one condition
    out, which would change six to three without touching either word in a
    sentence that would still read fluently.
    """
    pairs = _paired_pairs(claim)
    separating = [row for row in pairs if row["separates_after_adjustment"]]
    if any(not row["wilson_intervals_overlap"] for row in pairs):
        raise ValueError(
            f"{claim['id']}: a pair now separates under the INDEPENDENT intervals too, "
            "so the paper's claim that the paired test sees what they cannot is stale"
        )
    detail = ", ".join(
        f"{row['left']}/{row['right']} p={row['mcnemar_p_holm']:.3f}" for row in separating
    )
    return len(separating), f"{len(separating)} of {len(pairs)} pairs: {detail}"


def as6_paired_total_pairs(claim: dict) -> tuple[int, str]:
    """The denominator, checked separately so it cannot drift out of the sentence."""
    pairs = _paired_pairs(claim)
    return len(pairs), f"{len(pairs)} pairwise comparisons across both guards"


def as6_wildguard_floor_controls(claim: dict) -> tuple[int, str]:
    """How many controls WildGuard's floor is a distribution over.

    This number drifted once, silently and for a whole paper cycle: the cluster
    invocation passed a single `--ability-cells` file, so three conditions with
    measured ability 0.00 never entered the control set, and the paper printed
    the resulting floor for days. Nothing in the build could catch it, because
    the floor was internally consistent with the controls it was given. It is a
    counted claim, so it recomputes.
    """
    (source,) = _resolve_source(claim)
    floor = json.loads(source.read_text())["floor"]
    if floor["kind"] != "distribution":
        raise ValueError(
            f"{claim['id']}: the floor is a {floor['kind']}, not a distribution — "
            "a bound is reported differently and the paper's sentence would be wrong"
        )
    return floor["n"], f"{floor['n']} controls at floor {floor['value']:.4f}: {', '.join(floor['controls'])}"


RECIPES = {
    "spread_echo_measured_cells": spread_echo_measured_cells,
    "ladder_reported_cells_rejected": ladder_reported_cells_rejected,
    "ladder_reported_cells_total": ladder_reported_cells_total,
    "spread_rungs_passing_every_screen": spread_rungs_passing_every_screen,
    "as6_table2_provenance": as6_table2_provenance,
    "as6_bwd_max_reported": as6_bwd_max_reported,
    "as6_bwd_zero_pairs": as6_bwd_zero_pairs,
    "as6_bwd_measurable_pairs": as6_bwd_measurable_pairs,
    "as6_bwd_unlicensed_pairs": as6_bwd_unlicensed_pairs,
    "as6_length_bound_clearing_cells": as6_length_bound_clearing_cells,
    "as6_paired_separating_pairs": as6_paired_separating_pairs,
    "as6_paired_total_pairs": as6_paired_total_pairs,
    "as6_wildguard_floor_controls": as6_wildguard_floor_controls,
}


def _as_int(token: str, expect_word: str | None) -> int:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if token in WORD_NUMBERS:
        return WORD_NUMBERS[token]
    if expect_word is not None and token == "only":
        return WORD_NUMBERS[expect_word]
    raise ValueError(f"cannot read {token!r} as a number")


def _kits(paper: str) -> list[Path]:
    return sorted((PAPER_DIR / paper).glob("**/paper.tex"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="print the sets, check nothing")
    args = parser.parse_args()

    ledger = yaml.safe_load(LEDGER.read_text())["claims"]
    unknown = sorted({c["recipe"] for c in ledger} - set(RECIPES))
    if unknown:
        print(f"⛔ ledger names unimplemented recipes: {unknown}")
        return 1

    failures = 0
    for claim in ledger:
        try:
            value, detail = RECIPES[claim["recipe"]](claim)
        except (FileNotFoundError, ValueError) as exc:
            print(f"⛔ {claim['id']}: recomputation failed — {exc}")
            failures += 1
            continue
        print(f"\n{claim['id']}  = {value}")
        print(f"   recomputed: {detail}")
        if args.show:
            continue
        if claim.get("check") == "internal":
            # The recipe compared against the paper itself and returned a
            # mismatch count. Nothing to locate: a provenance claim has no
            # single sentence asserting a number.
            print(f"   {'✓' if value == 0 else '⛔'} internal check, {value} mismatches")
            failures += value
            continue
        kits = _kits(claim["paper"])
        if not kits:
            print(f"   ⚠️  no kits under paper/{claim['paper']}/ — claim UNCHECKED against prose")
            continue
        for kit in kits:
            flat = re.sub(r"\s+", " ", kit.read_text())
            hits = re.findall(claim["locate"], flat)
            if len(hits) != 1:
                print(f"   ⛔ {kit.relative_to(REPO)}: locate matched {len(hits)} sentences, expected 1")
                failures += 1
                continue
            asserted = _as_int(hits[0], claim.get("expect_word"))
            mark = "✓" if asserted == value else "⛔"
            print(f"   {mark} {kit.relative_to(REPO).parts[2]}: paper asserts {asserted}")
            failures += asserted != value

    print("\n" + ("⛔ MISMATCHES: %d" % failures if failures else "✓ every artefact-backed claim agrees"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
