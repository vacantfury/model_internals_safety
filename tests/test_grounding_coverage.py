"""Every instrument on the roster has a row in the S6b grounding record.

**Why this is a test and not a habit.** Item 27 makes the methods search a
precondition of real experiments: every instrument maps to an adopted
established method — cited, tier-labelled — or to an explicitly justified
bespoke build, and 27(c) says no further real run starts on an ungrounded one.
That rule was written into `text_docs/shared/instrument_grounding.md` on
2026-08-06 and was already incomplete the day it was written: **I0 and I4 had no
row.** The item's own wording is "I1-I6 and the shipped measurements", so I0 fell
through the phrasing, and I4 fell through because it was the one instrument not
yet finished.

Neither omission was visible. `build_status.py` reports BUILD state and says so;
the grounding record is prose and nothing reconciled the two. So the roster grew
and the record did not, which is the same failure mode `tests/test_completion.py`
exists to prevent one level over — building an instrument and forgetting the
manifest.

The check is deliberately shallow: it asserts a row EXISTS, never that the
grounding is good. Judging whether a citation supports a method is the reader's
job and cannot be automated; noticing that nobody wrote anything at all is
exactly what can be.
"""

from __future__ import annotations

import re

import pytest

from internals_safety.completion import ROSTER
from internals_safety.paths import PROJECT_ROOT

GROUNDING_DOC = PROJECT_ROOT / "text_docs" / "shared" / "instrument_grounding.md"


@pytest.fixture(scope="module")
def grounding_text() -> str:
    assert GROUNDING_DOC.exists(), f"{GROUNDING_DOC} is missing; it is the S6b decision record"
    return GROUNDING_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def table_rows(grounding_text: str) -> list[str]:
    """The rows of §1's grounding table — lines starting with `|` that are not
    the header or the separator."""
    rows = [
        line
        for line in grounding_text.splitlines()
        if line.startswith("|") and not re.fullmatch(r"\|[\s|:-]+\|", line.strip())
    ]
    assert rows, "no markdown table found in the grounding record"
    return [row for row in rows if not row.startswith("| instrument")]


@pytest.mark.parametrize("item", ROSTER, ids=lambda item: item.key)
def test_every_roster_instrument_has_a_grounding_row(item, table_rows):
    """A roster entry with no row is an instrument nobody grounded.

    Matching on the bare key (`I0`, `I4`) at a row's start rather than anywhere
    in the document, because a passing mention in prose is not a grounding
    decision — §1's table is where the decision is recorded.
    """
    pattern = re.compile(rf"^\|\s*{re.escape(item.key)}\b")
    matches = [row for row in table_rows if pattern.match(row)]
    assert matches, (
        f"{item.key} ({item.what}) is on the roster with no row in §1 of "
        f"{GROUNDING_DOC.name}. Item 27(a) requires every instrument to map to an "
        "adopted method (cited, tier-labelled) or an explicitly justified bespoke "
        "build, and 27(c) bars a real run on an ungrounded instrument."
    )


@pytest.mark.parametrize("item", ROSTER, ids=lambda item: item.key)
def test_every_grounding_row_states_a_tier_or_declares_itself_bespoke(item, table_rows):
    """A row with neither a tier nor a bespoke declaration is decoration.

    The evidence-tier law exists because uniform citation formatting makes
    unequal evidence look equal. A grounding row that cites a method without
    saying how well the world verified it reintroduces exactly that.
    """
    pattern = re.compile(rf"^\|\s*{re.escape(item.key)}\b")
    for row in (row for row in table_rows if pattern.match(row)):
        has_tier = re.search(r"\*\*\((A|B|C|D)\)\*\*", row)
        declares_bespoke = "bespoke" in row.lower()
        assert has_tier or declares_bespoke, (
            f"grounding row for {item.key} states neither an evidence tier nor that it "
            f"is bespoke:\n  {row.strip()}"
        )


def test_the_ungrounded_section_exists_and_is_not_empty(grounding_text):
    """§3 must keep saying what is NOT grounded.

    A grounding record that only lists what is grounded reads as though
    everything is, which is worse than having no record: it converts an open
    question into an apparent answer.
    """
    assert "## 3. What is still ungrounded" in grounding_text
    section = grounding_text.split("## 3. What is still ungrounded", 1)[1]
    assert section.count("- **") >= 3, "§3 lists fewer open gaps than the record claims"
