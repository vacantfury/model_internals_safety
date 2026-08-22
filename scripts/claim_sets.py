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
    r"\s*(?P<auroc>[0-9.]+)\s*&\s*(?P<dnb>[0-9.]+)\s*\["
)


def _as6_table2(kit: Path) -> list[tuple[str, str, float]]:
    """[(guard, condition, D&notB)] parsed from the kit itself.

    The paper's own set, so the check compares the paper against the artefact
    rather than against a list restated here.

    The cell is a FLOAT since 2026-08-22: the column moved from an integer count
    under the unsplit probe to a rate averaged over the item-level holdout's
    splits. An `int()` here read the new column as zero matches and emptied the
    reported set, which surfaced as `max() iterable argument is empty` two
    recipes downstream. A parser is a claim about the table's shape.
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
            rows.append((guard, m.group("cond").replace("\\_", "_"), float(m.group("dnb"))))
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
    """Every Table 2 rate, against the artefact the ledger says it came from.

    Returns the number of MISMATCHES, so 0 is the passing value. This is the
    provenance check: it fails if the named source does not contain the paper's
    numbers, which is the failure that cost a day on 2026-08-21 when a ledger
    sentence named the causal-intervention run as Table 2's source.

    It also checks the OPERATING POINT, which is new on 2026-08-22 and is the
    day's lesson made mechanical. The recount ran at percentile 75 while another
    cell's published claim is read at 50, and nothing in the build could see
    that a table and the artefact backing it were computed at different points.
    Now a mismatched `reading_percentile` is a hard failure rather than a
    footnote nobody re-reads.
    """
    percentile = claim["percentile"]
    (source,) = _resolve_source(claim)
    records = json.loads(source.read_text())
    by_cell = {}
    for row in records:
        cells = row.get("cells_per_100")
        if cells is None:
            continue
        got = cells["reading_percentile"]
        if got != percentile:
            raise ValueError(
                f"{claim['id']}: {source.name} was computed at percentile {got} and the "
                f"ledger claims {percentile} — a table and its artefact read at different "
                "operating points cannot be compared, and the difference is invisible in print"
            )
        by_cell.setdefault((row["model"], row["family"]), cells)

    mismatches, checked = [], 0
    for kit in _kits(claim["paper"]):
        label = kit.relative_to(REPO).parts[2]
        for guard, condition, asserted in _as6_table2(kit):
            cells = by_cell.get((guard, condition))
            if cells is None:
                raise ValueError(
                    f"source {claim['source']} holds no per-prompt recount for "
                    f"{guard}/{condition} — the pointer names the wrong artefact or one "
                    "written before the recount"
                )
            found = round(cells["split"]["decoded_not_blocked"]["mean"], 1)
            checked += 1
            if found != asserted:
                mismatches.append(f"{label}/{guard}/{condition}: paper {asserted} vs source {found}")
    detail = f"{checked} Table 2 cells checked against {claim['source']} at p{percentile:g}"
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



def _guard_floor(claim: dict) -> dict:
    """The `floor` block of one guard's control-floor artefact.

    Shared by the two recipes below because the count and the value are halves
    of the same object, and a paper sentence naming one without the other is
    what let a superseded pair survive: the ledger checked "14 for WildGuard"
    while "12 controls for Llama Guard 3" sat unchecked in the same sentence.
    """
    (source,) = _resolve_source(claim)
    floor = json.loads(source.read_text())["floor"]
    if floor["kind"] != "distribution":
        raise ValueError(
            f"{claim['id']}: the floor is a {floor['kind']}, not a distribution — "
            "a bound is reported differently and the paper's sentence would be wrong"
        )
    return floor


def as6_guard_floor_value(claim: dict) -> tuple[float, str]:
    """The floor value itself, at the paper's own printed precision.

    Added 2026-08-22 after the 11-control floor pair (0.6852 / 0.6617) was
    superseded by the 14-control one (0.6803 / 0.6605) and survived for a day in
    this repo's `CLAUDE.md` and `NOW.md` while the paper and the canonical
    record were both already correct. The paper was right that time. A count
    alone would not have noticed if it had not been: 11 controls and 14 controls
    are different numbers, but a floor VALUE can drift while the count holds
    still, and the value is what a screen actually compares against.
    """
    floor = _guard_floor(claim)
    return floor["value"], f"floor {floor['value']:.6f} over {floor['n']} controls"


def as6_guard_floor_controls(claim: dict) -> tuple[int, str]:
    """How many controls a guard's floor is a distribution over.

    This number drifted once, silently and for a whole paper cycle: the cluster
    invocation passed a single `--ability-cells` file, so three conditions with
    measured ability 0.00 never entered the control set, and the paper printed
    the resulting floor for days. Nothing in the build could catch it, because
    the floor was internally consistent with the controls it was given. It is a
    counted claim, so it recomputes.
    """
    floor = _guard_floor(claim)
    return floor["n"], f"{floor['n']} controls at floor {floor['value']:.4f}: {', '.join(floor['controls'])}"


def _factorial_reported(claim: dict) -> list[tuple[str, str, list[float], list[float]]]:
    """Every cell the factorial artefact marks `reported`, with both terms.

    ⚠️ The `reported` flag is read rather than the set being rebuilt here. The
    factorial run covers 23 conditions across the two guards and only 6 are in
    Table 2, so a recipe that counted every condition would answer a different
    question in the same units — which is the failure mode this ledger exists
    for, one artefact over.
    """
    (source,) = _resolve_source(claim)
    data = json.loads(source.read_text())
    cells = []
    for guard, block in sorted(data.items()):
        for family, cell in sorted(block["conditions"].items()):
            if not cell.get("reported"):
                continue
            interaction = cell["interaction"]
            cells.append(
                (guard, family, interaction["wrapper_alone"], interaction["encoding_beyond_wrapper"])
            )
    if not cells:
        raise ValueError(f"{claim['id']}: no cell is marked reported — the flag moved or the run changed")
    return cells


def _includes_zero(term: list[float]) -> bool:
    """`[point, lo, hi]` straddling zero. Fails CLOSED on a malformed triple."""
    if len(term) != 3:
        raise ValueError(f"expected [point, lo, hi], got {term!r}")
    _, lo, hi = term
    return lo <= 0.0 <= hi


def as6_wrapper_intervals_including_zero(claim: dict) -> tuple[int, str]:
    """Reported cells whose WRAPPER-alone interval includes zero.

    The paper's point is that the wrapper term is small, so this count moving UP
    would strengthen its sentence and moving DOWN would falsify it. Either way it
    is a cardinality over a screened set and it recomputes.
    """
    cells = _factorial_reported(claim)
    hits = [(g, f) for g, f, wrapper, _ in cells if _includes_zero(wrapper)]
    detail = ", ".join(f"{g}/{f}" for g, f in hits)
    return len(hits), f"{len(hits)} of {len(cells)} reported cells include zero: {detail}"


def as6_encoding_term_separating_cells(claim: dict) -> tuple[int, str]:
    """Reported cells whose ENCODING-BEYOND-WRAPPER interval excludes zero.

    Also RAISES unless the separating cells disagree in SIGN, because the
    paper's sentence is "in opposite directions" and a count of two that both
    pointed the same way would leave a true number under a false claim.
    """
    cells = _factorial_reported(claim)
    hits = [(g, f, term) for g, f, _, term in cells if not _includes_zero(term)]
    signs = {term[0] > 0 for _, _, term in hits}
    if len(hits) >= 2 and len(signs) != 2:
        raise ValueError(
            f"{claim['id']}: {len(hits)} cells separate but all in the same direction — "
            "the paper says 'in opposite directions', which would then be false"
        )
    detail = ", ".join(f"{g}/{f} {term[0]:+.2f}" for g, f, term in hits)
    return len(hits), f"{len(hits)} separating, opposite signs: {detail}"


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
    "as6_guard_floor_controls": as6_guard_floor_controls,
    "as6_guard_floor_value": as6_guard_floor_value,
    "as6_wrapper_intervals_including_zero": as6_wrapper_intervals_including_zero,
    "as6_encoding_term_separating_cells": as6_encoding_term_separating_cells,
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


def _agrees(token: str, value: float | int, expect_word: str | None) -> tuple[str, bool]:
    """Does the paper's token agree with the recomputed value?

    Two kinds, and the DECIMAL COUNT comes from the paper rather than from a
    setting. A paper printing `0.661` is asserting a rounded quantity, so the
    artefact is rounded to the same places before comparing; a tolerance knob
    here would be a knob on how wrong a printed number may be, which is not a
    thing to tune. Counts stay exact.
    """
    token = token.strip()
    if "." in token:
        decimals = len(token.split(".", 1)[1])
        asserted = float(token)
        return token, round(float(value), decimals) == asserted
    asserted = _as_int(token, expect_word)
    return str(asserted), asserted == value


def _kits(paper: str) -> list[Path]:
    """The MAIN document of each kit. Table-parsing recipes want exactly this."""
    return sorted((PAPER_DIR / paper).glob("**/paper.tex"))


def _kit_prose(paper: str) -> list[tuple[str, str]]:
    """(kit name, whitespace-flattened prose) per kit, main AND supplement.

    A kit became TWO documents on 2026-08-22 when the appendix split out. A
    `locate` looks for its sentence once per KIT, never once per file: a
    sentence moving into the supplement is a relocation, and reading it as a
    claim that stopped being asserted would be the guard crying wolf at exactly
    the moment someone is restructuring the paper, which is when it is least
    likely to be believed.
    """
    out = []
    for main in _kits(paper):
        text = "\n".join(f.read_text() for f in sorted(main.parent.glob("*.tex")))
        out.append((main.parent.name, re.sub(r"\s+", " ", text)))
    return out


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
        # MIRRORS first: a repo file restating the same number. The paper has
        # referees and this file has none, which is why the mirror is checked
        # at all — `CLAUDE.md` is loaded into every session, so a superseded
        # value there propagates into reasoning rather than into one document.
        # Seed 2026-08-22: the 11-control floor pair (0.6852 / 0.6617) was
        # superseded by the 14-control one and survived a day in `CLAUDE.md`
        # and `NOW.md` while the paper and the canonical record were correct.
        # Its own locate, because a mirror phrases the claim its own way.
        for mirror in claim.get("mirrors", []):
            path = REPO / mirror["file"]
            if not path.exists():
                # Gitignored mirrors (NOW.md) are legitimately absent on a
                # fresh clone. Say so rather than passing quietly.
                print(f"   ⚠️  {mirror['file']} absent — mirror UNCHECKED")
                continue
            flat = re.sub(r"\s+", " ", path.read_text())
            hits = re.findall(mirror["locate"], flat)
            if len(hits) != 1:
                print(f"   ⛔ {mirror['file']}: locate matched {len(hits)}, expected 1")
                failures += 1
                continue
            asserted, agrees = _agrees(hits[0], value, claim.get("expect_word"))
            print(f"   {'✓' if agrees else '⛔'} {mirror['file']} (mirror): asserts {asserted}")
            failures += not agrees

        kits = _kit_prose(claim["paper"])
        if not kits:
            print(f"   ⚠️  no kits under paper/{claim['paper']}/ — claim UNCHECKED against prose")
            continue
        for name, flat in kits:
            hits = re.findall(claim["locate"], flat)
            if len(hits) != 1:
                print(f"   ⛔ {name}: locate matched {len(hits)} sentences, expected 1")
                failures += 1
                continue
            asserted, agrees = _agrees(hits[0], value, claim.get("expect_word"))
            mark = "✓" if agrees else "⛔"
            print(f"   {mark} {name}: paper asserts {asserted}")
            failures += not agrees

    print("\n" + ("⛔ MISMATCHES: %d" % failures if failures else "✓ every artefact-backed claim agrees"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
