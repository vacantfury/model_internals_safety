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
    gate_per_prompt,
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


def null_reading(**overrides) -> Reading:
    """An ability-0 rung: value at the floor, control at the floor, scorer works.

    Deliberately shaped like the real case — `tag_block` and `reverse_characters`
    read 0.00 against a mismatched-pairing control that also reads 0.00, and their
    `identity_rate` is 1.0 on all 54 cached conditions.
    """
    defaults = dict(
        instrument="ability",
        kind="correlational",
        value=0.0,
        operating_point="fraction of prompts recovered under the three-route rule",
        licensed=True,
        control_reading=0.0,
        control_margin=0.03,
        claim="null",
        sensitivity=1.0,
        sensitivity_floor=1.0,
    )
    return Reading(**{**defaults, **overrides})


class TestNullClaims:
    """TODO 42 — the axis that licenses an absence.

    The old contract asked only about specificity, which is unanswerable for a
    reading whose value EQUALS its negative control by construction. Those are
    the ability-0 rungs, and they calibrate three other instruments, so the gap
    was load-bearing rather than pedantic.
    """

    def test_an_ability_zero_rung_with_a_working_scorer_is_reportable(self):
        """The case the whole change exists for. Under the old contract this was
        permanently withheld, which meant the calibration rungs for three
        instruments could not themselves be quoted."""
        assert null_reading().reportable
        assert null_reading().why_not_reportable() == []

    def test_an_absence_with_no_sensitivity_arm_is_refused(self):
        """A broken scorer and a model that cannot decode produce the identical
        0.00, so silence alone is not evidence of anything."""
        missing = null_reading(sensitivity=None)
        assert not missing.reportable
        assert any("COULD have fired" in why for why in missing.why_not_reportable())

    def test_a_scorer_that_fails_on_this_rung_cannot_support_an_absence(self):
        """`normalize` NFKC-folds and strips zero-width characters, so a rung with
        an unusual character set can break the scorer where no other rung would
        reveal it — which is exactly a rung whose 0.00 means nothing."""
        broken = null_reading(sensitivity=0.62)
        assert not broken.clears_sensitivity
        assert not broken.reportable
        assert any("below the floor" in why for why in broken.why_not_reportable())

    def test_a_NaN_sensitivity_fails_closed(self):
        assert not null_reading(sensitivity=float("nan")).reportable

    # ---- the dodge, and the guard that closes it ---------------------------

    def test_labelling_a_SIGNAL_as_a_null_is_refused(self):
        """⚠️ The property that makes a DECLARED claim safe.

        If `null` were a free pass out of P2 and P3, any instrument could dodge
        its own negative control by declaring the permissive direction. A reading
        that BEATS its control is not an absence — it is a positive reading
        wearing the wrong label, and it is refused on that ground.
        """
        dodge = null_reading(value=0.90, sensitivity=1.0)
        assert dodge.clears_controls
        assert not dodge.claim_is_coherent
        assert not dodge.reportable
        assert any("mislabelled" in why for why in dodge.why_not_reportable())

    def test_the_dodge_would_otherwise_have_worked(self):
        """Pins the motivation: the same numbers with no coherence guard satisfy
        every remaining null-path condition, so the guard is what is doing the
        work rather than some other axis catching it incidentally."""
        dodge = null_reading(value=0.90)
        assert dodge.licensed is not None and dodge.clears_sensitivity

    def test_the_same_reading_declared_POSITIVE_is_judged_the_strict_way(self):
        """Direction changes which evidence is demanded, never whether evidence
        is demanded. Declared positive, the identical numbers now owe P3 and P7."""
        honest = null_reading(value=0.90, claim="positive")
        assert not honest.reportable
        assert any("length null" in why for why in honest.why_not_reportable())

    # ---- what a null claim does NOT owe, and why -------------------------

    def test_an_absence_does_not_owe_the_length_null(self):
        """P3 is an INFLATION control: it asks whether an observed signal is
        really length. A value at the floor has no signal to explain away, and
        length cannot manufacture an absence."""
        assert null_reading(length_null_margin=None).reportable

    def test_an_absence_does_not_owe_P7(self):
        """Selection over a grid inflates a maximum, so a null surviving an
        uncorrected search is CONSERVATIVE — 'even the best layer found
        nothing' — rather than suspect."""
        assert null_reading(selection_inside_null=False).reportable

    def test_unlicensed_is_SUPPORTING_evidence_for_an_absence(self):
        """`licensed=False` means no signal above the instrument's own null,
        which is what an absence asserts. Disqualifying it would make the null
        path unusable by exactly the probe instruments that need it."""
        assert null_reading(licensed=False).reportable

    def test_unmeasured_still_disqualifies_an_absence(self):
        """The one thing that fails on BOTH paths. 'I could not read this' never
        becomes 'there is nothing here' — that is the original tri-state defect,
        and the null path must not reintroduce it."""
        unmeasured = null_reading(licensed=None)
        assert not unmeasured.reportable
        assert any("unmeasured" in why for why in unmeasured.why_not_reportable())

    # ---- direction is declared, and defaults to the strict route ----------

    def test_the_default_claim_is_positive(self):
        """A caller who never thought about direction gets the demanding route."""
        assert reading().claim == "positive"

    def test_a_positive_claim_ignores_the_sensitivity_arm(self):
        """Sensitivity is not a second way to pass P2. A positive claim that
        reads loudly on its control stays refused however well it can fire."""
        loud = reading(value=0.30, control_reading=0.29, sensitivity=1.0, sensitivity_floor=1.0)
        assert not loud.reportable

    def test_the_reasons_are_direction_specific(self):
        """Telling an absence it lacks a specificity margin sends a reader after
        evidence that cannot exist for it."""
        why = null_reading(sensitivity=None).why_not_reportable()
        assert not any("(P2)" in reason or "(P3)" in reason or "(P7)" in reason for reason in why)

    def test_a_reportable_absence_gates_its_per_prompt_axis_through(self):
        """The granularity join must respect the new path too, or the rungs this
        change licensed would still be dropped one level down."""
        assert gate_per_prompt(null_reading(), [False, False]) == [False, False]


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
