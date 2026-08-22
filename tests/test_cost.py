"""Cost-model tests — the arithmetic behind the experiment-run approval gate.

Hermetic: `census_phase0` takes a `count_tokens` callable, so these run with a
stub tokenizer and never download anything. What they pin is the reasoning the
gate depends on — that encoding inflation reaches the estimate, that the ranges
cross the right way, and that a job which would blow the partition's wall says
so rather than reporting a large number and leaving the reader to compare.
"""

from __future__ import annotations

import pytest

from internals_safety.config import AbilityConfig, BehaviorConfig, MeasurementsConfig
from internals_safety.cost import (
    HardwareProfile,
    JudgeCostConfig,
    SchedulerLimits,
    TokenCensus,
    census_phase0,
    estimate,
    format_range,
    load_cost_config,
)
from internals_safety.data import Prompt
from internals_safety.encodings.registry import load_ladder

MEASUREMENTS = MeasurementsConfig(
    ability=AbilityConfig(max_new_tokens=100),
    behavior=BehaviorConfig(max_new_tokens=200),
)
HARDWARE = HardwareProfile(
    label="test GPU",
    vram_gb=40,
    prefill_tokens_per_s=(1000, 2000),
    decode_tokens_per_s=(100, 200),
    load_seconds=0,
    # A fixture models the STRICTEST real implementation: `measured` is required
    # with no default precisely so nobody can forget to say whether a card's
    # rate was fitted, and a fixture that omitted it would be testing a laxer
    # type than the one that ships.
    measured=True,
)
SCHEDULER = SchedulerLimits(
    partition="gpu", max_wall_clock_hours=8, gpus_per_job=1, max_concurrent_jobs=4
)
JUDGE = JudgeCostConfig(chars_per_token=4, output_tokens_per_verdict=(100, 500))


def count_chars(text: str) -> int:
    """Stand-in tokenizer: one token per character. Keeps the relative inflation
    between rungs real (it is a property of the ciphertext, not the tokenizer)
    while staying hermetic."""
    return len(text)


@pytest.fixture
def prompts():
    harmful = [Prompt(id=f"h{i}", text=f"harmful request {i}", category="c") for i in range(4)]
    harmless = [Prompt(id=f"b{i}", text=f"benign request {i}", category="c") for i in range(4)]
    return harmful, harmless


class TestCensus:
    def test_decode_tokens_are_the_configured_budget(self, prompts):
        harmful, harmless = prompts
        census = census_phase0(
            count_chars, harmful, harmless, load_ladder(), ["base64", "rot13"], MEASUREMENTS, 500
        )
        # 2 families x [4 harmful x (100 restate + 200 attack)
        #               + 4 BENIGN x 200 attack].
        #
        # The benign term was missing until 2026-08-07 (TODO 61).
        # `behavior_control` claimed to cost judge calls only, on the grounds
        # that the benign arm is already captured — but capture is prefill-only
        # and the control calls `measure_behavior`, which GENERATES. It was
        # buying a second generation pass no estimate showed.
        # ... PLUS the model-level plain baseline: 8 prompts (both arms) x 200,
        # charged ONCE rather than per family, since the plaintext denominator
        # does not depend on which rung ran (evidence_and_story.md 4c).
        # ... PLUS the SCAFFOLD control (2026-08-09): both arms again, PER
        # FAMILY, because each rung's wrapper names its own encoding. It is the
        # first control that scales with the ladder, which is why it is written
        # inside the `2 * (...)` term rather than beside the plain baseline.
        # ... PLUS the BENIGN ARM'S RESTATE pass (2026-08-21, defect (11)):
        # 4 benign x 100, per family. Comprehension used to be a harmful-arm-only
        # measurement, so a harm gap and a comprehension gap were not separable.
        # It is the THIRD control to land in the entrypoint and need adding here,
        # after the benign attack pass and the scaffold arm, and the pattern is
        # the point: the census does not track the code, so each one is a
        # deliberate edit or a cost nobody approved.
        assert census.decode_tokens == (
            2 * (4 * 300 + 4 * 200 + 4 * 100 + (4 + 4) * 200) + (4 + 4) * 200
        )

    def test_two_judges_per_attack_response(self, prompts):
        harmful, harmless = prompts
        census = census_phase0(
            count_chars, harmful, harmless, load_ladder(), ["base64", "rot13"], MEASUREMENTS, 500
        )
        # 2 judges x 2 families x (4 harmful + 4 benign), PLUS the model-level
        # plain baseline's 2 judges x (4 harmful + 4 benign). The benign arm is
        # measurement #4's mandatory control; the plain baseline is the
        # plaintext denominator (evidence_and_story.md 4c), also mandatory, and
        # also judged by the same two. PLUS the SCAFFOLD control (2026-08-09),
        # per family like the encoded arm and unlike the plain baseline: 2
        # judges x 2 families x (4 harmful + 4 benign).
        assert census.judge_calls == (
            2 * 2 * (4 + 4)  # encoded, both arms
            + 2 * 2 * (4 + 4)  # scaffold, both arms
            + 2 * (4 + 4)  # plain baseline, model-level
        )

    def test_the_benign_control_arm_is_priced_at_all(self, prompts):
        """⚠️ Pinned as its own test because its absence was invisible for as
        long as the control was opt-in. Halving the corpus's benign half must
        move BOTH the decode budget and the judge count; if either is unmoved,
        the control has gone unpriced again."""
        harmful, harmless = prompts
        full = census_phase0(
            count_chars, harmful, harmless, load_ladder(), ["base64"], MEASUREMENTS, 500
        )
        halved = census_phase0(
            count_chars, harmful, harmless[:2], load_ladder(), ["base64"], MEASUREMENTS, 500
        )
        assert halved.judge_calls < full.judge_calls
        assert halved.decode_tokens < full.decode_tokens

    def test_the_scaffold_control_is_priced_and_SCALES_WITH_THE_LADDER(self, prompts):
        """⚠️ The sibling guard, and the stronger one, because this control is
        the first whose cost grows with the sweep.

        The benign arm is per-rung too, but the plain baseline is not, and the
        estimate has already been wrong once in each direction. Adding a rung
        must move the scaffold's share; if the per-rung delta does not include
        two full arms of generation and judging, the control has gone unpriced
        in the way that only shows up on a 15-rung sweep — where it is 3,000
        generations, not 200.
        """
        harmful, harmless = prompts
        ladder = load_ladder()
        one = census_phase0(
            count_chars, harmful, harmless, ladder, ["base64"], MEASUREMENTS, 500
        )
        two = census_phase0(
            count_chars, harmful, harmless, ladder, ["base64", "rot13"], MEASUREMENTS, 500
        )
        # One extra rung buys: encoded (4 restate + 4 attack + 4 benign) AND
        # scaffold (4 harmful + 4 benign). Without the scaffold the judge delta
        # would be 2 * (4 + 4) = 16; with it, 32.
        assert two.judge_calls - one.judge_calls == 32
        # Same for decode: 4*300 + 4*200 encoded, plus 8*200 scaffold.
        # 4 * 100 is the benign restate pass, which scales with the ladder for
        # the same reason the scaffold control does: it is per rung, not per model.
        assert two.decode_tokens - one.decode_tokens == (
            4 * 300 + 4 * 200 + 4 * 100
        ) + 8 * 200

    def test_inflation_reaches_the_estimate(self, prompts):
        """The load-bearing property: the ciphertext drives prompt length, so a
        cost model built on pass counts alone would be wrong by an order of
        magnitude on exactly the rungs that cost the most. Asserted on the
        per-rung prompt length rather than total prefill, because prefill also
        carries the family-independent plain captures, which dilute the ratio."""
        harmful, harmless = prompts
        ladder = load_ladder()
        binary = census_phase0(
            count_chars, harmful, harmless, ladder, ["binary"], MEASUREMENTS, 500
        )
        words = census_phase0(
            count_chars, harmful, harmless, ladder, ["reverse_words"], MEASUREMENTS, 500
        )
        assert (
            binary.per_family_mean_prompt_tokens["binary"]
            > 2 * words.per_family_mean_prompt_tokens["reverse_words"]
        )
        assert binary.prefill_tokens > words.prefill_tokens
        # Decode is a fixed budget, so only prefill moves with the rung.
        assert binary.decode_tokens == words.decode_tokens

    def test_the_longest_prompt_is_reported(self, prompts):
        """A rung whose prompts exceed the model's context is silently truncated
        and every regime in it is garbage, so the census surfaces the maximum."""
        harmful, harmless = prompts
        census = census_phase0(
            count_chars, harmful, harmless, load_ladder(), ["binary"], MEASUREMENTS, 500
        )
        assert census.max_prompt_tokens >= max(census.per_family_mean_prompt_tokens.values())

    def test_more_families_cost_more(self, prompts):
        harmful, harmless = prompts
        ladder = load_ladder()
        one = census_phase0(count_chars, harmful, harmless, ladder, ["hex"], MEASUREMENTS, 500)
        two = census_phase0(
            count_chars, harmful, harmless, ladder, ["hex", "base64"], MEASUREMENTS, 500
        )
        assert two.prefill_tokens > one.prefill_tokens
        # NOT 2x: the plain baseline is model-level, so it is charged once
        # whatever the ladder size. Doubling the rungs doubles only the
        # per-family term -- which is the property that makes the baseline cheap
        # enough to be mandatory, so it is asserted rather than approximated.
        per_family = one.judge_calls - 2 * (4 + 4)
        assert two.judge_calls == one.judge_calls + per_family


class TestEstimate:
    def test_slow_hardware_gives_the_pessimistic_end(self):
        """The ranges cross: the LOW end of hours comes from the HIGH end of
        tokens per second. Getting this backwards would report the optimistic
        case as the worst case, which is the direction that breaks a gate."""
        census = TokenCensus(
            prefill_tokens=2000, decode_tokens=200, judge_calls=0, judge_input_chars=0
        )
        result = estimate(census, HARDWARE, SCHEDULER, JUDGE, 0.25, 2.00)
        assert result.prefill_hours[0] == pytest.approx(2000 / 2000 / 3600)
        assert result.prefill_hours[1] == pytest.approx(2000 / 1000 / 3600)
        assert result.gpu_hours[0] < result.gpu_hours[1]

    def test_decode_dominates_a_generation_heavy_run(self):
        census = TokenCensus(
            prefill_tokens=1000, decode_tokens=1000, judge_calls=0, judge_input_chars=0
        )
        result = estimate(census, HARDWARE, SCHEDULER, JUDGE, 0.25, 2.00)
        assert result.decode_hours[1] > result.prefill_hours[1]

    def test_a_run_over_the_partition_wall_says_so(self):
        census = TokenCensus(
            prefill_tokens=0, decode_tokens=100 * 3600 * 9, judge_calls=0, judge_input_chars=0
        )
        result = estimate(census, HARDWARE, SCHEDULER, JUDGE, 0.25, 2.00)
        assert result.gpu_hours[1] > SCHEDULER.max_wall_clock_hours
        assert not result.fits_wall_clock

    def test_reasoning_tokens_are_charged_as_output(self):
        """gpt-5-mini bills hidden reasoning tokens as output. Ignoring them
        would understate the judge bill several-fold, so the range's high end
        must actually respond to the assumption."""
        census = TokenCensus(
            prefill_tokens=0, decode_tokens=0, judge_calls=1000, judge_input_chars=0
        )
        result = estimate(census, HARDWARE, SCHEDULER, JUDGE, 0.25, 2.00)
        assert result.judge_usd[0] == pytest.approx(1000 * 100 * 2.00 / 1_000_000)
        assert result.judge_usd[1] == pytest.approx(1000 * 500 * 2.00 / 1_000_000)

    def test_free_gpus_leave_only_the_judge_bill(self):
        census = TokenCensus(
            prefill_tokens=1000, decode_tokens=1000, judge_calls=10, judge_input_chars=400
        )
        result = estimate(census, HARDWARE, SCHEDULER, JUDGE, 0.25, 2.00)
        assert result.gpu_usd == (0.0, 0.0)
        assert result.total_usd == result.judge_usd

    def test_a_priced_gpu_reaches_the_total(self):
        paid = SchedulerLimits(
            partition="cloud",
            max_wall_clock_hours=24,
            gpus_per_job=1,
            max_concurrent_jobs=1,
            gpu_usd_per_hour=2.0,
        )
        census = TokenCensus(
            prefill_tokens=0, decode_tokens=360000, judge_calls=0, judge_input_chars=0
        )
        result = estimate(census, HARDWARE, paid, JUDGE, 0.25, 2.00)
        assert result.gpu_usd[1] == pytest.approx(result.gpu_hours[1] * 2.0)
        assert result.total_usd[1] > result.judge_usd[1]


class TestScaling:
    def test_a_capture_only_phase_costs_no_decode(self):
        census = TokenCensus(
            prefill_tokens=1000, decode_tokens=1000, judge_calls=10, judge_input_chars=400
        )
        scaled = census.scaled(prefill=5.0, decode=0.0, judge=0.0)
        assert scaled.prefill_tokens == 5000
        assert scaled.decode_tokens == 0
        assert scaled.judge_calls == 0


class TestShippedConfig:
    def test_the_repo_config_loads_and_covers_the_pilot_hardware(self):
        config = load_cost_config()
        assert "a100_40gb" in config.hardware and "v100_32gb" in config.hardware
        assert config.scheduler.max_wall_clock_hours > 0
        # Free cluster: the money line of the gate is judge spend only.
        assert config.scheduler.gpu_usd_per_hour == 0.0

    def test_every_phase_projection_names_itself(self):
        for name, scale in load_cost_config().phases.items():
            assert scale.description, f"{name} has no description"


def test_format_range_is_readable():
    assert format_range((1.234, 5.678)) == "1.2-5.7"
    assert format_range((1.0, 2.0), places=2) == "1.00-2.00"
