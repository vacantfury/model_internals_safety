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


# --------------------------------------------------------------------------
# Recipes. CLOSED vocabulary — `conf/claim_sets.yaml` may name only these.
# --------------------------------------------------------------------------


def spread_echo_measured_cells() -> tuple[int, str]:
    cells = _echo_cells()["spread"]
    measured = [k for k, v in cells.items() if v is not None]
    unmeasured = [k for k, v in cells.items() if v is None]
    return len(measured), (
        f"{len(measured)} measured of {len(cells)} cells; "
        f"{len(unmeasured)} unmeasured and excluded: {sorted(unmeasured)}"
    )


def ladder_reported_cells_rejected() -> tuple[int, str]:
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


def ladder_reported_cells_total() -> tuple[int, str]:
    cells = _echo_cells()["ladder"]
    reported = [k for k in cells if k[1] in REPORTED_ENCODINGS]
    stages = sorted({k[0] for k in reported})
    return len(reported), f"{len(stages)} stages {stages} x {len(REPORTED_ENCODINGS)} encodings"


def spread_rungs_passing_every_screen() -> tuple[int, str]:
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


RECIPES = {
    "spread_echo_measured_cells": spread_echo_measured_cells,
    "ladder_reported_cells_rejected": ladder_reported_cells_rejected,
    "ladder_reported_cells_total": ladder_reported_cells_total,
    "spread_rungs_passing_every_screen": spread_rungs_passing_every_screen,
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
            value, detail = RECIPES[claim["recipe"]]()
        except (FileNotFoundError, ValueError) as exc:
            print(f"⛔ {claim['id']}: recomputation failed — {exc}")
            failures += 1
            continue
        print(f"\n{claim['id']}  = {value}")
        print(f"   recomputed: {detail}")
        if args.show:
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
