"""The instrument contract — P1..P7 as types rather than prose.

Every test here corresponds to a defect this repo actually shipped and fixed.
The contract exists so those defects become impossible to express, not so the
modules look tidy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from internals_safety.measurements.contract import (
    Instrument,
    Kind,
    Reading,
    assert_distinct_questions,
    reportable_only,
    withheld_summary,
)


def reading(**overrides) -> Reading:
    """A fully-evidenced, reportable reading, so each test varies one axis."""
    defaults = dict(
        instrument="decode_lens",
        kind="correlational",
        value=0.90,
        operating_point="probability margin over a same-condition mismatched plaintext",
        licensed=True,
        control_reading=0.10,
        control_margin=0.05,
        length_null_margin=0.24,
        selection_inside_null=True,
    )
    return Reading(**{**defaults, **overrides})


@dataclass(frozen=True)
class FakeInstrument:
    name: str
    question: str
    kind: Kind = "correlational"


class TestReportable:
    def test_a_fully_evidenced_reading_is_reportable(self):
        assert reading().reportable
        assert reading().why_not_reportable() == []

    def test_unmeasured_is_not_reportable_and_says_so(self):
        """The tri-state defect (instrument_layer 1.5). `None` must not pass by
        being neither True nor obviously False."""
        withheld = reading(licensed=None)
        assert not withheld.reportable
        assert "unmeasured" in withheld.why_not_reportable()[0]

    def test_unmeasured_is_distinguished_from_measured_negative(self):
        """These are different findings and conflating them is what made the
        cipher band's uniform (R) an artefact of a silent False."""
        assert reading(licensed=None).why_not_reportable() != reading(licensed=False).why_not_reportable()

    def test_a_missing_negative_control_disqualifies_rather_than_passes(self):
        """P2 fails CLOSED: 'we did not run the control' is not 'the control passed'."""
        missing = reading(control_reading=None)
        assert not missing.reportable
        assert any("no negative control" in why for why in missing.why_not_reportable())

    def test_reading_loudly_on_the_negative_control_disqualifies(self):
        loud = reading(value=0.30, control_reading=0.29)
        assert not loud.clears_controls
        assert not loud.reportable

    def test_a_missing_length_null_disqualifies_rather_than_passes(self):
        """P3. Character length separates the corpora at AUROC 0.654 and every
        encoder preserves it, so an uncontrolled number is uninterpretable."""
        missing = reading(length_null_margin=None)
        assert not missing.reportable
        assert any("no length null" in why for why in missing.why_not_reportable())

    def test_a_number_inside_the_length_null_disqualifies(self):
        inside = reading(length_null_margin=-0.01)
        assert not inside.clears_length_null
        assert any("inside the length null" in why for why in inside.why_not_reportable())

    def test_selection_outside_the_null_disqualifies(self):
        """P7 — an uncorrected argmax over ~33 cells is a multiple comparison."""
        assert not reading(selection_inside_null=False).reportable

    def test_every_failing_axis_is_named_at_once(self):
        """A run record must say WHY a cell was withheld; 'not reportable' with
        no reason is how an instrument defect hides as a null result."""
        broken = reading(
            licensed=False, control_reading=None, length_null_margin=None,
            selection_inside_null=False,
        )
        assert len(broken.why_not_reportable()) == 4


class TestOperatingPoint:
    def test_the_operating_point_travels_with_the_number(self):
        """The band run's hard incoherence was traced to an operating point
        stated only in a config comment that never reached the label."""
        assert "margin" in reading().operating_point


class TestKindSeparation:
    def test_kind_is_stored_not_inferred(self):
        """P5. Whether a claim is causal is about what the measurement DID."""
        assert reading(instrument="causal_license", kind="causal").kind == "causal"

    def test_correlational_and_causal_readings_coexist_without_merging(self):
        readings = [reading(), reading(instrument="causal_license", kind="causal")]
        assert {r.kind for r in readings} == {"correlational", "causal"}


class TestDistinctQuestions:
    def test_distinct_questions_pass(self):
        assert_distinct_questions([
            FakeInstrument("decode_lens", "did the model decode the ciphertext in situ"),
            FakeInstrument("entropy_dynamics", "does the uncertainty profile mark this input"),
        ])

    def test_two_instruments_with_one_question_are_rejected(self):
        """The roster's value is independent failure modes; two instruments with
        the same question are one instrument counted twice."""
        with pytest.raises(ValueError, match="P1 violated"):
            assert_distinct_questions([
                FakeInstrument("probe_a", "is harm linearly decodable here"),
                FakeInstrument("probe_b", "Is Harm Linearly Decodable Here"),
            ])

    def test_an_empty_roster_is_vacuously_fine(self):
        assert_distinct_questions([])

    def test_the_protocol_is_structural(self):
        assert isinstance(FakeInstrument("x", "y"), Instrument)


class TestAggregation:
    def test_reportable_only_keeps_the_quotable_subset(self):
        good, bad = reading(), reading(instrument="trajectory", licensed=None)
        assert reportable_only([good, bad]) == [good]

    def test_withheld_summary_explains_every_exclusion(self):
        """A table of what was measured is only honest beside a table of what
        was measured and thrown away."""
        summary = withheld_summary([
            reading(),
            reading(instrument="trajectory", licensed=None),
            reading(instrument="entropy", length_null_margin=None),
        ])
        assert set(summary) == {"trajectory", "entropy"}
        assert all(reasons for reasons in summary.values())
