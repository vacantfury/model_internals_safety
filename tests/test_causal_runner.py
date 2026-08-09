"""The write-side causal runner (`measurements/causal.py`) and its plug-in point."""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import CausalLicenseConfig
from internals_safety.measurements.causal import (
    QUESTION,
    CausalRun,
    forward_passes,
    reading,
    resolve_refusal_tokens,
)
from internals_safety.measurements.causal_license import CausalEvidence

CONFIG = CausalLicenseConfig()


class FakeTokenizer:
    """Encodes each whitespace-delimited word as one id, so a multi-word opening
    is genuinely multi-token — the case `resolve_refusal_tokens` must refuse."""

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(word)) % 50000 for word in text.split()]


class FakeLoaded:
    tokenizer = FakeTokenizer()


def test_refusal_openings_resolve_through_the_live_tokenizer():
    ids = resolve_refusal_tokens(FakeLoaded(), ["I", "As"])
    assert len(ids) == 2
    assert all(isinstance(value, int) for value in ids)


def test_a_multi_token_opening_fails_loud_rather_than_truncating():
    """A silent truncation would score the probability of a word fragment and
    call it refusal — the same class of defect as reading a verdict at the wrong
    position, which cost this repo a run once already."""
    with pytest.raises(ValueError, match="not 1"):
        resolve_refusal_tokens(FakeLoaded(), ["I cannot"])


def test_no_openings_is_refused_rather_than_scoring_zero():
    with pytest.raises(ValueError):
        resolve_refusal_tokens(FakeLoaded(), [])


def test_the_cost_function_is_two_baselines_plus_three_per_candidate():
    """The approval gate prices the run from this, so it is asserted rather than
    left to a comment."""
    assert forward_passes(0) == 2
    assert forward_passes(1) == 5
    assert forward_passes(14) == 44


def evidence(layer=4, bypass=0.5, induce=0.2, kl=0.01) -> CausalEvidence:
    return CausalEvidence(
        layer=layer,
        position="instruction_final",
        behaviour="refusal_opening",
        behaviour_before=1.0,
        behaviour_after_ablation=1.0 - bypass,
        harmless_behaviour_before=0.0,
        harmless_behaviour_after_addition=induce,
        kl=kl,
    )


def run_of(*items) -> CausalRun:
    return CausalRun(
        evidence=tuple(items),
        behaviour="refusal_opening",
        behaviour_before=1.0,
        harmless_behaviour_before=0.0,
        n_harmful=10,
        n_harmless=10,
    )


def test_the_reading_is_labelled_CAUSAL_so_P5_can_keep_the_kinds_apart():
    result = reading(run_of(evidence()), CONFIG, n_layers=32)
    assert result.kind == "causal"
    assert result.instrument == "causal_license"


def test_no_eligible_direction_is_a_MEASURED_zero_not_an_unmeasured_one():
    """"No direction is causally effective" and "we did not run the causal test"
    are different states, and the tri-state discipline is about not merging
    them. A filtered-out sweep is the former."""
    # Fails every criterion: last-layer, huge KL, no induction, no bypass.
    result = reading(run_of(evidence(layer=31, bypass=0.0, induce=0.0, kl=9.9)), CONFIG, n_layers=32)
    assert result.licensed is True
    assert result.value == 0.0
    assert result.detail["n_eligible"] == 0


class TestAnEmptyFilterIsDiagnosable:
    """⚠️ Paid for by three runs that produced an uninterpretable reading.

    The gate returned `n_eligible: 0` on Llama-3.1-8B-Instruct (runs 9007219 /
    9008632) while its best candidate bypassed refusal by 0.737 over the
    matched-norm null, on a `refusal_before` of 0.949 — a bypass fraction near
    0.78 against a 0.5 bar. The bypass criterion passed comfortably, something
    else rejected every candidate, and the record could not say which. The same
    empty set was then produced against a content guard (run 9033528) and was
    one reading away from being written up as a fact about the guard.
    """

    def llama_shaped(self, kl=9.9, induce=0.2):
        """The real run's numbers: a direction that removes 78% of refusal."""
        return CausalEvidence(
            layer=16,
            position="instruction_final",
            behaviour="refusal_opening",
            behaviour_before=0.9489,
            behaviour_after_ablation=0.9489 - 0.7367,
            harmless_behaviour_before=0.1480,
            harmless_behaviour_after_addition=0.1480 + induce,
            kl=kl,
        )

    def test_the_bypass_criterion_is_NOT_what_rejected_the_real_direction(self):
        """Pinned because it is the fact that makes the zero a defect rather
        than a result: the direction cleared the bar we set."""
        assert self.llama_shaped().bypass_fraction > CONFIG.min_bypass_fraction

    def test_a_filtered_out_WORKING_direction_is_unmeasured_not_a_measured_zero(self):
        result = reading(run_of(self.llama_shaped()), CONFIG, n_layers=32)
        assert result.detail["n_eligible"] == 0
        assert result.licensed is None, (
            "a direction that removed 78% of refusal was discarded on a secondary "
            "criterion — reporting that as 'no direction is causally effective' is "
            "the silent zero this repo has now fixed five times"
        )
        assert result.detail["filtered_out_a_working_direction"] is True

    def test_nothing_acting_is_STILL_a_measured_zero(self):
        """The boundary. Only the working-direction case flips to unmeasured;
        a sweep where no intervention did anything remains a real negative, or
        the fix would have thrown away the finding it was protecting."""
        result = reading(run_of(evidence(bypass=0.0, induce=0.0, kl=9.9)), CONFIG, n_layers=32)
        assert result.licensed is True
        assert result.value == 0.0
        assert result.detail["filtered_out_a_working_direction"] is False

    def test_the_attrition_NAMES_the_criterion(self):
        """`n_eligible: 0` is the same number for opposite findings. The
        criterion is what tells them apart."""
        by_kl = reading(run_of(self.llama_shaped(kl=9.9)), CONFIG, n_layers=32)
        assert by_kl.detail["attrition"] == {"kl": 1}

        by_induce = reading(run_of(self.llama_shaped(kl=0.01, induce=-0.5)), CONFIG, n_layers=32)
        assert by_induce.detail["attrition"] == {"induce": 1}

        by_bypass = reading(run_of(evidence(bypass=0.0, kl=0.01)), CONFIG, n_layers=32)
        assert by_bypass.detail["attrition"] == {"bypass": 1}

    def test_max_bypass_fraction_reports_over_ALL_candidates_not_the_eligible_ones(self):
        """The eligible set is empty in every case this class covers, so a
        maximum over it would be 0.0 and say nothing."""
        result = reading(run_of(self.llama_shaped()), CONFIG, n_layers=32)
        assert result.detail["max_bypass_fraction"] == pytest.approx(0.7764, abs=1e-3)

    def test_every_candidate_is_recorded_so_a_null_is_rediagnosable_offline(self):
        """The first three runs cost a queue cycle each to re-ask. 13 rows of
        small floats is the whole fix."""
        result = reading(
            run_of(self.llama_shaped(), evidence(bypass=0.0)), CONFIG, n_layers=32
        )
        rows = result.detail["candidates"]
        assert len(rows) == 2
        assert {row["discarded_for"] for row in rows} == {"kl", "bypass"}
        assert all("bypass_fraction" in row and "kl" in row for row in rows)


class TestTheControlFieldsDoNotCrossTwoSelections:
    """`control_reading` was `value - null_margin`, and those come from
    different candidates whenever the filter empties: `value` from the eligible
    set, the margin from the raw best the null was drawn on. On the real run it
    printed -0.737 as "what the control read"."""

    def test_the_control_reading_is_the_nulls_own_mean(self):
        result = reading(
            run_of(evidence(bypass=0.8)),
            CONFIG,
            n_layers=32,
            null_margin=0.7,
            null_p_value=0.05,
            null_observed=0.8,
        )
        assert result.control_reading == pytest.approx(0.1)  # 0.8 observed - 0.7 margin

    def test_omitting_the_observed_statistic_drops_the_control_rather_than_faking_it(self):
        """Fails closed: a control field computed from a margin alone is the
        incoherence being fixed, so it is withheld instead of guessed."""
        result = reading(
            run_of(evidence(bypass=0.8)), CONFIG, n_layers=32, null_margin=0.7, null_p_value=0.05
        )
        assert result.control_reading is None
        assert not result.clears_controls


class TestTheClaimDirectionFollowsTheEligibleSet:
    def test_an_empty_filter_declares_a_NULL_claim(self):
        """So the contract asks for SENSITIVITY — could this gate fire when a
        direction does exist — rather than for a length null. The first three
        runs were all withheld for 'no length null (P3)', which is the wrong
        question to send a reader after."""
        result = reading(run_of(evidence(bypass=0.0, kl=9.9)), CONFIG, n_layers=32)
        assert result.claim == "null"
        assert "sensitivity" in " ".join(result.why_not_reportable()).lower()

    def test_a_surviving_direction_declares_a_POSITIVE_claim(self):
        result = reading(run_of(evidence(bypass=0.8, induce=0.2, kl=0.01)), CONFIG, n_layers=32)
        assert result.claim == "positive"
        assert result.detail["n_eligible"] == 1


def test_selection_is_only_inside_the_null_when_a_null_was_actually_drawn():
    """The sweep takes a maximum over candidates, so P7 is unmet until the
    random-direction null covers that selection."""
    without = reading(run_of(evidence()), CONFIG, n_layers=32)
    assert not without.selection_inside_null
    assert "selection was not inside the null" in " ".join(without.why_not_reportable())

    with_null = reading(
        run_of(evidence()), CONFIG, n_layers=32, null_margin=0.3, null_p_value=0.01
    )
    assert with_null.selection_inside_null


def test_the_reading_states_that_refusal_is_a_probability_not_log_odds():
    """The one divergence from Arditi that makes our bypass magnitudes
    incomparable to theirs must be visible at the operating point, not only in a
    module docstring."""
    result = reading(run_of(evidence()), CONFIG, n_layers=32)
    assert "PROBABILITY" in result.operating_point
    assert "log-odds" in result.operating_point


def test_the_question_is_distinct_from_every_correlational_instrument():
    from internals_safety.measurements import ability, decode_lens, entropy_dynamics, trajectory

    others = {
        ability.QUESTION,
        decode_lens.QUESTION,
        entropy_dynamics.QUESTION,
        trajectory.QUESTION,
    }
    assert QUESTION not in others


def test_ablation_and_addition_actually_change_the_logits(tiny_model):
    """The interventions were built and orphaned for a day; this is the check
    that they do something when driven through the runner's own path."""
    from internals_safety.measurements.causal import _final_logits
    from internals_safety.models.interventions import ablate_direction, add_direction

    prompts = ["hello there", "goodbye now"]
    baseline = _final_logits(tiny_model, prompts, batch_size=2)

    hidden = baseline.shape[-1]
    width = tiny_model.model.config.hidden_size
    direction = torch.zeros(width)
    direction[0] = 1.0

    with ablate_direction(tiny_model, direction):
        ablated = _final_logits(tiny_model, prompts, batch_size=2)
    with add_direction(tiny_model, direction, layer=0, coefficient=5.0):
        added = _final_logits(tiny_model, prompts, batch_size=2)

    assert baseline.shape == ablated.shape == added.shape == (2, 1, hidden)
    assert not torch.allclose(baseline, added)


def test_degenerate_cells_are_filtered_rather_than_crashing_in_a_hook():
    """`difference_in_means` returns a ZERO vector where the classes coincide,
    and `ablate_direction` refuses to project one out. Found end to end
    2026-08-06: layer-0 `resid_pre` is the raw embedding, so a fixture whose
    tokenizer maps every content word to [UNK] has no direction there at all."""
    from internals_safety.measurements.causal import viable_directions
    from internals_safety.probes.directions import Direction

    def direction(vector):
        return Direction(
            vector=vector, layer=0, position="last", n_positive=4, n_negative=4, raw_norm=0.0
        )

    real = direction(torch.tensor([1.0, 0.0]))
    degenerate = direction(torch.zeros(2))
    assert viable_directions([real, degenerate]) == [real]


def test_only_EXACT_degeneracy_is_filtered_so_no_threshold_is_invented():
    """A merely weak direction is the causal criteria's business — they judge it
    on its own bypass and induce evidence. A norm cut here would be a second,
    unfounded gate on the same question."""
    from internals_safety.measurements.causal import viable_directions
    from internals_safety.probes.directions import Direction

    tiny = Direction(
        vector=torch.tensor([1e-9, 0.0]), layer=1, position="last",
        n_positive=4, n_negative=4, raw_norm=1e-9,
    )
    assert viable_directions([tiny]) == [tiny]


def test_an_unrunnable_causal_test_is_unmeasured_not_a_measured_zero():
    """The tri-state distinction, one instrument further on. "No direction is
    causally effective" is a result; "no direction existed to intervene on" is
    the instrument failing to read, and reporting 0.0 for it would assert that
    harm is causally unmediated on the strength of a cache that never produced a
    direction."""
    from internals_safety.measurements.causal import unmeasured_reading

    result = unmeasured_reading(CONFIG, "all candidate cells were degenerate")
    assert result.licensed is None
    assert result.value != result.value  # NaN
    assert not result.reportable
    assert "unmeasured" in " ".join(result.why_not_reportable())
