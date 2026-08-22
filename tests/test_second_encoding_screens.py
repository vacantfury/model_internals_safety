"""The two new review-3 instruments, pinned in BOTH directions.

A screen that only ever fires is a verdict with a script attached, and a screen
that never fires is decoration. Both are pinned here.

The case that matters most is Mistral on ``math_bold``: the echo screen CLEARS
it and the model cannot read it, so its harm gap of zero carries no information
about discrimination at all. Reporting that cell as a discrimination failure
would be this paper's own thesis committed against itself, so the ordering of
the two screens is a test rather than a comment.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


second_encoding = _load("second_encoding")
reported_intervals = _load("reported_intervals")


def row(*, ability: float, echo_clears: bool | None) -> dict:
    return {"ability": ability, "echo_clears": echo_clears}


class TestAnUnreadableRungIsNeverReportable:
    """⚠️ The class this replaced was called ``TestReadabilityIsCheckedBeforeEcho``
    and claimed more than it pinned. Reordering the two checks is
    behaviour-preserving for every input EXCEPT a rung failing both, so the
    ordering needs its own case; a mutation run caught the overclaim, which is
    why the doubly-failing test below exists."""

    def test_unreadable_beats_a_clearing_echo_screen(self):
        """Mistral / math_bold: echo clears, ability is 0.00, gap is meaningless."""
        assert second_encoding.verdict(row(ability=0.0, echo_clears=True), 0.75) == "unreadable"

    def test_a_rung_failing_BOTH_screens_is_labelled_unreadable(self):
        """The only input the check ORDER changes. ``echo-fails`` would imply the
        rung was readable and merely contaminated, which is a different and much
        more interesting claim than 'the model cannot read this at all'."""
        assert second_encoding.verdict(row(ability=0.0, echo_clears=False), 0.75) == "unreadable"

    def test_a_readable_rung_that_clears_is_reportable(self):
        assert second_encoding.verdict(row(ability=1.0, echo_clears=True), 0.75) == "REPORTABLE"

    def test_a_readable_rung_that_fails_the_echo_screen_is_not(self):
        assert second_encoding.verdict(row(ability=1.0, echo_clears=False), 0.75) == "echo-fails"

    def test_the_floor_is_the_floor(self):
        """Just under fails, exactly on passes. Pins the boundary in both directions."""
        assert second_encoding.verdict(row(ability=0.74, echo_clears=True), 0.75) == "unreadable"
        assert second_encoding.verdict(row(ability=0.75, echo_clears=True), 0.75) == "REPORTABLE"


class TestTheTriStateSurvives:
    """`None` means the screen could not read the cell. Folding it into False is
    the defect this repo has now fixed five times."""

    def test_unmeasured_is_not_a_failure(self):
        assert second_encoding.verdict(row(ability=1.0, echo_clears=None), 0.75) == "unmeasured"

    def test_and_it_is_not_a_pass_either(self):
        assert second_encoding.verdict(row(ability=1.0, echo_clears=None), 0.75) != "REPORTABLE"


class TestACountIsNeverGuessed:
    """A stored rate recovers its integer count or the script refuses."""

    def test_a_whole_count_recovers(self):
        assert reported_intervals.counts(0.83, 100) == 83
        assert reported_intervals.counts(0.0, 100) == 0

    def test_a_rate_that_is_not_a_whole_count_raises(self):
        with pytest.raises(ValueError, match="whole count"):
            reported_intervals.counts(0.835, 100)

    def test_a_missing_rate_stays_missing(self):
        assert reported_intervals.counts(None, 100) is None


class TestTheAbilityFloorCannotBeSilentlyDefaulted:
    """`verdict` takes the floor explicitly: a screen whose threshold has a
    default is a screen somebody will call without deciding the threshold."""

    def test_ability_floor_is_required(self):
        parameters = inspect.signature(second_encoding.verdict).parameters
        assert parameters["ability_floor"].default is inspect.Parameter.empty
