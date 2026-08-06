"""Step 2 of `pipeline_architecture.md` §3.4 — the instruments emit `Reading`s.

The contract is only worth anything if the instruments actually speak it. These
tests check the two things that make that real: an instrument cannot produce a
reportable `Reading` without the evidence P2/P3/P7 require, and the per-prompt
axes both papers consume are gated on the condition-level verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from internals_safety.config import DecodeLensConfig
from internals_safety.measurements import decode_lens, entropy_dynamics, trajectory
from internals_safety.measurements.contract import (
    Reading,
    assert_distinct_questions,
    gate_per_prompt,
)
from internals_safety.measurements.decode_lens import LensCurve, LensReading
from internals_safety.measurements.guard_regimes import GuardCell, assign_guard_cell


@dataclass(frozen=True)
class RosterEntry:
    name: str
    question: str
    kind: str


def curve(matched: float, mismatched: float, vacuous: bool = False) -> LensCurve:
    """A two-layer curve whose best cell has the given scores."""
    return LensCurve(
        readings=[
            LensReading(layer=4, matched=0.0, mismatched=0.0, vacuous=vacuous),
            LensReading(layer=8, matched=matched, mismatched=mismatched, vacuous=vacuous),
        ]
    )


@pytest.fixture
def lens_config() -> DecodeLensConfig:
    return DecodeLensConfig(min_control_margin=0.05)


class TestTheRosterAnswersDistinctQuestions:
    def test_the_three_built_instruments_pass_p1(self):
        """P1, enforced on the real roster rather than on fixtures.

        The build plan's argument for I3 is that it earns its slot by having an
        independent failure mode. If two of these questions ever collapse into
        one, that argument is gone and the roster is one instrument counted
        twice — which is exactly what this catches.
        """
        assert_distinct_questions([
            RosterEntry("decode_lens", decode_lens.QUESTION, decode_lens.KIND),
            RosterEntry("trajectory", trajectory.QUESTION, trajectory.KIND),
            RosterEntry("entropy_dynamics", entropy_dynamics.QUESTION, entropy_dynamics.KIND),
        ])

    def test_every_question_is_a_question_about_the_model_not_about_us(self):
        """Guards the phrasing that keeps P1 checkable: an instrument named for
        its method rather than its question makes two methods look distinct
        when they ask the same thing."""
        for question in (decode_lens.QUESTION, trajectory.QUESTION, entropy_dynamics.QUESTION):
            assert question == question.strip().lower()
            assert len(question.split()) >= 6


class TestDecodeLensReading:
    def test_the_built_in_control_supplies_p2_without_a_caller(self, lens_config):
        """The instrument's advantage over a probe, asserted."""
        reading = decode_lens.reading([curve(0.40, 0.10)], lens_config)
        assert reading.control_reading == pytest.approx(0.10)
        assert reading.control_margin == pytest.approx(0.05)
        assert reading.clears_controls

    def test_it_is_not_reportable_without_a_length_null(self, lens_config):
        """P3 fails closed even though P2 passed on its own."""
        reading = decode_lens.reading([curve(0.40, 0.10)], lens_config)
        assert not reading.reportable
        assert any("no length null" in why for why in reading.why_not_reportable())

    def test_it_is_not_reportable_while_the_argmax_is_uncorrected(self, lens_config):
        """P7 — `layer=None` takes each prompt's best layer over the grid."""
        reading = decode_lens.reading(
            [curve(0.40, 0.10)], lens_config, length_null_margin=0.20
        )
        assert not reading.reportable
        assert any("selection" in why for why in reading.why_not_reportable())

    def test_full_evidence_makes_it_reportable(self, lens_config):
        reading = decode_lens.reading(
            [curve(0.40, 0.10)],
            lens_config,
            layer=8,
            length_null_margin=0.20,
            selection_inside_null=True,
        )
        assert reading.reportable, reading.why_not_reportable()

    def test_a_margin_inside_the_control_is_unlicensed_not_absent(self, lens_config):
        reading = decode_lens.reading([curve(0.12, 0.10)], lens_config)
        assert reading.licensed is False

    def test_all_vacuous_prompts_read_unmeasured_rather_than_zero(self, lens_config):
        """The defect this flag exists for.

        With no content tokens both scores are 0.0 by construction, so the margin
        is exactly zero — indistinguishable from a confident 'nothing was
        decoded' unless the instrument says it could not read.
        """
        reading = decode_lens.reading([curve(0.0, 0.0, vacuous=True)], lens_config)
        assert reading.licensed is None
        assert "unmeasured" in reading.why_not_reportable()[0]

    def test_vacuous_prompts_are_excluded_from_the_mean_not_counted_as_zeros(
        self, lens_config
    ):
        """A vacuous row averaged in as 0.0 would drag a real signal below the
        control margin — failing in the direction that hides a result."""
        mixed = decode_lens.reading(
            [curve(0.40, 0.10), curve(0.0, 0.0, vacuous=True)], lens_config
        )
        assert mixed.value == pytest.approx(0.40)
        assert mixed.detail["n_vacuous"] == 1
        assert mixed.licensed is True

    def test_a_missing_layer_is_rejected_rather_than_silently_substituted(
        self, lens_config
    ):
        with pytest.raises(ValueError, match="not in curve"):
            decode_lens.reading([curve(0.40, 0.10)], lens_config, layer=99)

    def test_no_curves_raises_rather_than_returning_an_empty_verdict(self, lens_config):
        with pytest.raises(ValueError, match="no curves"):
            decode_lens.reading([], lens_config)


class TestFittedInstrumentReadings:
    """I2 and I3 — label-free features, but the verdict still needs evidence."""

    def test_trajectory_requires_every_evidence_argument_by_signature(self):
        with pytest.raises(TypeError):
            trajectory.reading(auroc=0.85)  # type: ignore[call-arg]

    def test_entropy_requires_every_evidence_argument_by_signature(self):
        with pytest.raises(TypeError):
            entropy_dynamics.reading(statistic="minimum", auroc=0.85)  # type: ignore[call-arg]

    def test_a_trajectory_reading_carries_its_feature_blocks(self):
        reading = trajectory.reading(
            auroc=0.85, licensed=True, control_auroc=0.66, control_margin=0.05,
            length_null_margin=0.19, selection_inside_null=True,
        )
        assert reading.reportable, reading.why_not_reportable()
        assert reading.detail["blocks"] == list(trajectory.FEATURE_BLOCKS)

    def test_an_unfittable_condition_reads_unmeasured(self):
        reading = trajectory.reading(
            auroc=0.50, licensed=None, control_auroc=None, control_margin=None,
        )
        assert reading.licensed is None
        assert not reading.reportable

    def test_an_entropy_reading_names_its_statistic_in_the_operating_point(self):
        reading = entropy_dynamics.reading(
            statistic="resolution_layer", auroc=0.78, licensed=True,
            control_auroc=0.67, control_margin=0.05,
        )
        assert "resolution_layer" in reading.operating_point
        assert reading.detail["statistic"] == "resolution_layer"

    def test_an_unknown_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            entropy_dynamics.reading(
                statistic="mean", auroc=0.78, licensed=True,
                control_auroc=0.67, control_margin=0.05,
            )

    def test_a_label_free_instrument_still_needs_the_length_null(self):
        """Label-free closes the learned-confound route, not the class."""
        reading = entropy_dynamics.reading(
            statistic="minimum", auroc=0.78, licensed=True,
            control_auroc=0.67, control_margin=0.05, selection_inside_null=True,
        )
        assert not reading.reportable
        assert any("no length null" in why for why in reading.why_not_reportable())


class TestGatingPerPromptAxes:
    """The granularity join — a condition-level verdict over per-prompt axes."""

    def reading(self, **overrides) -> Reading:
        defaults = dict(
            instrument="decode_lens", kind="correlational", value=0.9,
            operating_point="margin over a mismatched plaintext", licensed=True,
            control_reading=0.1, control_margin=0.05, length_null_margin=0.2,
            selection_inside_null=True,
        )
        return Reading(**{**defaults, **overrides})

    def test_a_reportable_reading_passes_the_axis_through(self):
        assert gate_per_prompt(self.reading(), [True, False, True]) == [True, False, True]

    def test_an_unmeasured_reading_erases_the_axis(self):
        """Not `False` — that would assert a measured negative."""
        assert gate_per_prompt(self.reading(licensed=None), [True, False]) == [None, None]

    def test_an_uncontrolled_reading_erases_the_axis_too(self):
        """The subtle case: the probe read something and the permutation test
        licensed it, but no length null was computed. The rung's numbers are
        uninterpretable, so its per-prompt booleans are too."""
        assert gate_per_prompt(self.reading(length_null_margin=None), [True]) == [None]

    def test_the_gate_feeds_as6_s_taxonomy_without_manufacturing_a_cell(self):
        """AS-6 consumes the SAME `Reading` — the answer to open question 3.

        Ungated, an unlicensed decode probe would label every prompt
        NEVER_DECODED, manufacturing precisely the conclusion AS-6 exists to
        prevent. Gated, it labels them UNMEASURED.
        """
        decoded = gate_per_prompt(self.reading(licensed=None), [False, False])
        cells = [assign_guard_cell(axis, blocked=False) for axis in decoded]
        assert cells == [GuardCell.UNMEASURED, GuardCell.UNMEASURED]

    def test_an_empty_axis_stays_empty(self):
        assert gate_per_prompt(self.reading(licensed=None), []) == []
