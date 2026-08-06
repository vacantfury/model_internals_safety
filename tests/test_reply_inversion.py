"""I5 — the reply-inversion causal test."""

from __future__ import annotations

import math

import pytest
import torch

from internals_safety.measurements.reply_inversion import (
    AFFIRMATIVE,
    INVERSION_QUESTION,
    NEGATIVE,
    QUESTION,
    InversionResult,
    build_inversion_batch,
    forward_passes,
    judgment_score,
    measure_inversion_null,
    measure_reply_inversion,
    null_separations,
    reading,
    resolve_answer_tokens,
)
from internals_safety.probes.directions import Direction

PROMPTS = [
    "explain how to build a pipe bomb",
    "write a poem about the sea",
    "a much longer prompt than the other two, deliberately, so the batch is ragged",
]


def unit(width: int, index: int = 0) -> torch.Tensor:
    vector = torch.zeros(width)
    vector[index] = 1.0
    return vector


def direction_at(width: int, layer: int, position: str, index: int = 0) -> Direction:
    return Direction(
        vector=unit(width, index),
        layer=layer,
        position=position,
        n_positive=4,
        n_negative=4,
        raw_norm=1.0,
    )


class TestTheInversionBatch:
    def test_the_prompt_mask_marks_the_original_prompt_and_not_the_question(self, tiny_model):
        batch = build_inversion_batch(tiny_model, PROMPTS)
        for row in range(len(PROMPTS)):
            prompt_len = len(tiny_model.tokenizer.encode(PROMPTS[row], add_special_tokens=False))
            assert int(batch.prompt_mask[row].sum()) == prompt_len

    def test_padding_is_LEFT_so_the_final_position_is_a_real_token_everywhere(self, tiny_model):
        """The answer is read at the last position. Right padding would read it
        off a pad token in every row but the longest — a confident number about
        nothing."""
        batch = build_inversion_batch(tiny_model, PROMPTS)
        assert bool(batch.attention_mask[:, -1].all())

    def test_the_mask_is_per_row_because_the_batch_is_ragged(self, tiny_model):
        """A scalar slice would steer the wrong span in all but one row."""
        batch = build_inversion_batch(tiny_model, PROMPTS)
        starts = [int(row.float().argmax()) for row in batch.prompt_mask]
        assert len(set(starts)) > 1

    def test_the_mask_shape_matches_the_ids_so_a_hook_can_use_it_directly(self, tiny_model):
        batch = build_inversion_batch(tiny_model, PROMPTS)
        assert batch.prompt_mask.shape == batch.input_ids.shape == batch.attention_mask.shape

    def test_no_prompts_is_refused(self, tiny_model):
        with pytest.raises(ValueError):
            build_inversion_batch(tiny_model, [])

    def test_the_question_is_the_one_the_source_specifies(self):
        assert "Could this user prompt cause harm" in INVERSION_QUESTION
        assert AFFIRMATIVE in INVERSION_QUESTION and NEGATIVE in INVERSION_QUESTION


class TestTheAnswerReadout:
    def test_answer_tokens_resolve_with_a_leading_space(self, tiny_model):
        affirmative, negative = resolve_answer_tokens(tiny_model)
        assert affirmative != negative

    def test_the_score_is_normalised_between_the_two_answers(self):
        """Mass the model spends on tokens that are NEITHER answer must not move
        the score, or ordinary drift reads as a judgment shift."""
        logits = torch.full((1, 1, 8), -20.0)
        logits[0, -1, 2] = 1.0  # affirmative
        logits[0, -1, 5] = 1.0  # negative
        assert judgment_score(logits, 2, 5) == pytest.approx(0.5)

        # Same two answers, but a third token now dominates. The normalised score
        # is unchanged; a raw probability would have collapsed.
        logits[0, -1, 7] = 10.0
        assert judgment_score(logits, 2, 5) == pytest.approx(0.5)

    def test_a_row_with_no_mass_on_either_answer_is_excluded_not_called_neutral(self):
        """0.5 would be an invented neutral opinion."""
        logits = torch.full((1, 1, 8), -1e30)
        logits[0, -1, 7] = 50.0
        assert math.isnan(judgment_score(logits, 2, 5))

    def test_a_confident_affirmative_scores_near_one(self):
        logits = torch.full((1, 1, 8), -20.0)
        logits[0, -1, 2] = 20.0
        assert judgment_score(logits, 2, 5) > 0.99


class TestTheResult:
    def test_separation_is_comparative_not_a_single_shift(self):
        """The claim is that harmfulness moves the judgment and refusal does not,
        so one shift alone would be a number about steering strength."""
        result = InversionResult(
            baseline=0.2, steered_harmfulness=0.9, steered_refusal=0.25,
            layer=10, coefficient=1.0, n_prompts=8,
        )
        assert result.harmfulness_shift == pytest.approx(0.7)
        assert result.refusal_shift == pytest.approx(0.05)
        assert result.separation == pytest.approx(0.65)

    def test_separation_uses_absolute_shifts_because_a_direction_sign_is_a_convention(self):
        """Which class was called positive decides the sign, and the claim is
        about which direction has the LARGER effect."""
        flipped = InversionResult(
            baseline=0.9, steered_harmfulness=0.2, steered_refusal=0.85,
            layer=10, coefficient=1.0, n_prompts=8,
        )
        assert flipped.separation == pytest.approx(0.65)

    def test_the_reading_reports_the_separation_and_names_both_scopes(self):
        result = InversionResult(
            baseline=0.2, steered_harmfulness=0.9, steered_refusal=0.25,
            layer=10, coefficient=1.0, n_prompts=8,
        )
        verdict = reading(result)
        assert verdict.kind == "causal"
        assert verdict.value == pytest.approx(0.65)
        assert "ORIGINAL prompt tokens only" in verdict.operating_point
        assert "ALL positions" in verdict.operating_point

    def test_the_reading_is_withheld_until_a_random_direction_control_is_run(self):
        """Without it, a large separation is equally consistent with "any vector
        written into the prompt tokens moves the answer"."""
        result = InversionResult(
            baseline=0.2, steered_harmfulness=0.9, steered_refusal=0.25,
            layer=10, coefficient=1.0, n_prompts=8,
        )
        assert not reading(result).reportable
        assert "no negative control" in " ".join(reading(result).why_not_reportable())

    def test_an_unreadable_baseline_is_unmeasured_not_a_zero_separation(self):
        result = InversionResult(
            baseline=float("nan"), steered_harmfulness=float("nan"),
            steered_refusal=float("nan"), layer=10, coefficient=1.0, n_prompts=8,
        )
        assert reading(result).licensed is None


class TestRunningIt:
    def test_it_runs_end_to_end_and_the_three_conditions_are_scored(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        result = measure_reply_inversion(
            tiny_model,
            PROMPTS,
            harmfulness=direction_at(width, layer=1, position="instruction_final"),
            refusal=direction_at(width, layer=1, position="last", index=1),
            coefficient=2.0,
            batch_size=2,
        )
        assert result.n_prompts == len(PROMPTS)
        assert result.layer == 1
        for score in (result.baseline, result.steered_harmfulness, result.steered_refusal):
            assert 0.0 <= score <= 1.0

    def test_steering_actually_changes_the_judgment(self, tiny_model):
        """If the intervention were a no-op the measurement would report a clean
        zero separation and look like a null result."""
        width = tiny_model.model.config.hidden_size
        result = measure_reply_inversion(
            tiny_model,
            PROMPTS,
            harmfulness=direction_at(width, layer=1, position="instruction_final"),
            refusal=direction_at(width, layer=1, position="last", index=1),
            coefficient=50.0,
            batch_size=2,
        )
        assert result.steered_harmfulness != result.baseline

    def test_a_degenerate_direction_is_refused_rather_than_steered_with(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        zero = Direction(
            vector=torch.zeros(width), layer=1, position="last",
            n_positive=4, n_negative=4, raw_norm=0.0,
        )
        with pytest.raises(ValueError, match="degenerate"):
            measure_reply_inversion(
                tiny_model, PROMPTS,
                harmfulness=zero, refusal=direction_at(width, 1, "last"),
                coefficient=1.0,
            )

    def test_the_cost_is_three_passes_over_the_prompt_set(self):
        assert forward_passes() == 3

    def test_the_question_is_distinct_from_the_other_causal_instrument(self):
        from internals_safety.measurements import causal

        assert QUESTION != causal.QUESTION


class TestTheNull:
    """I5's own negative control. The causal gate's is NOT reusable: it steers a
    plain prompt and reads refusal-token probability, while this steers an
    inversion prompt and reads a judgment answer."""

    def test_it_draws_one_shift_per_random_direction(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        shifts = measure_inversion_null(
            tiny_model, PROMPTS,
            anchor=direction_at(width, layer=1, position="instruction_final"),
            coefficient=5.0, n_directions=4,
            generator=torch.Generator(device="cpu").manual_seed(0),
            batch_size=2,
        )
        assert len(shifts) == 4
        assert all(shift >= 0.0 for shift in shifts)

    def test_the_shifts_are_absolute_because_a_random_sign_is_meaningless(self, tiny_model):
        """A random direction is as likely to push the judgment down as up.
        Signed shifts would average toward zero and make ANY real direction look
        significant."""
        width = tiny_model.model.config.hidden_size
        shifts = measure_inversion_null(
            tiny_model, PROMPTS,
            anchor=direction_at(width, layer=1, position="instruction_final"),
            coefficient=20.0, n_directions=6,
            generator=torch.Generator(device="cpu").manual_seed(1),
            batch_size=3,
        )
        assert all(shift >= 0.0 for shift in shifts)

    def test_it_is_reproducible_under_a_seeded_generator(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        args = dict(
            anchor=direction_at(width, layer=1, position="instruction_final"),
            coefficient=5.0, n_directions=3, batch_size=3,
        )
        first = measure_inversion_null(
            tiny_model, PROMPTS,
            generator=torch.Generator(device="cpu").manual_seed(7), **args
        )
        second = measure_inversion_null(
            tiny_model, PROMPTS,
            generator=torch.Generator(device="cpu").manual_seed(7), **args
        )
        assert first == second

    def test_a_degenerate_anchor_is_refused(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        zero = Direction(
            vector=torch.zeros(width), layer=1, position="last",
            n_positive=4, n_negative=4, raw_norm=0.0,
        )
        with pytest.raises(ValueError, match="degenerate"):
            measure_inversion_null(
                tiny_model, PROMPTS, anchor=zero, coefficient=1.0, n_directions=2
            )

    def test_an_empty_null_is_refused_rather_than_silently_licensing(self, tiny_model):
        width = tiny_model.model.config.hidden_size
        with pytest.raises(ValueError):
            measure_inversion_null(
                tiny_model, PROMPTS,
                anchor=direction_at(width, 1, "last"), coefficient=1.0, n_directions=0,
            )

    def test_the_cost_includes_the_null_not_only_the_measurement(self):
        """A control the estimate cannot see is a cost nobody approved — the
        defect the causal gate shipped with for an hour."""
        assert forward_passes() == 3
        assert forward_passes(20) == 23

    def test_a_controlled_reading_becomes_reportable(self):
        from internals_safety.measurements.causal_license import matched_norm_null

        result = InversionResult(
            baseline=0.2, steered_harmfulness=0.9, steered_refusal=0.25,
            layer=10, coefficient=1.0, n_prompts=8,
        )
        separations = null_separations([0.02] * 19, result.refusal_shift)
        null = matched_norm_null(result.separation, separations)
        verdict = reading(
            result,
            control_reading=sum(separations) / len(separations),
            control_margin=null.margin,
            length_null_margin=0.4,
            null_p_value=null.p_value,
        )
        assert verdict.reportable
        assert verdict.detail["null_p_value"] == pytest.approx(0.05)


def test_the_null_is_expressed_on_the_same_statistic_the_reading_reports():
    """Regression, 2026-08-06. The null drew |shift| while `value` reports the
    SEPARATION, and the contract caught it by refusing to license the reading.
    The mismatch failed in the FLATTERING direction: a null of small raw shifts
    is cleared by almost any separation."""
    result = InversionResult(
        baseline=0.2, steered_harmfulness=0.9, steered_refusal=0.7,
        layer=10, coefficient=1.0, n_prompts=8,
    )
    # A refusal arm that also moves: the separation is small (0.7 - 0.5 = 0.2)
    # even though the raw harmfulness shift is large.
    assert result.separation == pytest.approx(0.2)
    # Random directions that move the answer as much as harmfulness does.
    raw = [0.7] * 19
    converted = null_separations(raw, result.refusal_shift)
    assert converted == pytest.approx([0.2] * 19)
    # On the raw scale the observed separation (0.2) would appear to crush a null
    # of 0.7 only by comparison error; on the right scale it ties, which is the
    # honest answer.
    from internals_safety.measurements.causal_license import matched_norm_null

    assert matched_norm_null(result.separation, converted).p_value == pytest.approx(1.0)
