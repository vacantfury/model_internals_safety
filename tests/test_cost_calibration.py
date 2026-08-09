"""The approval gate's throughput knob must bracket the runs already measured.

**Why a test and not a comment.** `decode_tokens_per_s` has now been wrong twice
in opposite directions — [300, 1200] assumed (7.5-17x too fast, so the pilot was
costed at a quarter of its true time) and [40, 71] fitted to a run whose slowness
was a since-fixed defect (3.3x too slow). Both were documented in `conf/cost.yaml`
at the time. A comment does not fail.

The invariant is the one an approval gate actually needs: **the configured range
must contain every completed run's implied rate for the shape it is used on**.
Too fast under-predicts and gets a job killed at the wall; too slow inflates
every estimate the owner is asked to approve.

Hermetic — the rates below are recorded measurements, not a re-run.
"""

from __future__ import annotations

import pytest

from internals_safety.cost import load_cost_config

# Implied effective rates, `census.decode_tokens / measured wall clock`, from
# `outputs/analysis/cost_recalibration_20260808.json` (13 runs) plus the four
# held-out `spread_*` runs whose records live only on the cluster.
# Shape: a ladder sweep of >= 3 rungs at n >= 50, which is what the gate costs.
IN_SHAPE_RATES = {
    "dissociation_tulu3": 233.1, "naturalness_band_llama": 234.4,
    "decode_lens_floor": 235.4, "ladder_tulu3_rlvr": 236.2,
    "ladder_tulu3_dpo": 237.4, "benign_arm_qwen": 260.3,
    "dissociation_mistral": 260.9, "ladder_plain_rlvr": 262.0,
    "ladder_plain_dpo": 270.8, "ladder_tulu3_sft": 274.6,
    "causal_sweep": 274.9, "decode_lens_real": 285.7, "ladder_plain_sft": 305.1,
}
# Held out from the fit because their run records are not local. 998,400 decode
# tokens against 1:12-1:22 of job wall clock (sacct 2026-08-08), less the
# configured 300 s load. This is the number the floor is actually set to.
SPREAD_HELD_OUT = 216.0

# Shapes the estimator does NOT model, kept as data rather than deleted: it has
# no per-run fixed-cost term, so these amortise probe fitting and judge setup
# over different amounts of work. They must NOT constrain the range.
OUT_OF_SHAPE_RATES = {
    "smoke": 92.5, "plain_baseline_tulu3": 408.6, "plain_baseline_llama": 448.1,
    "plain_baseline_qwen": 461.2, "plain_baseline_mistral": 490.2,
}


@pytest.fixture(scope="module")
def measured_profile():
    return load_cost_config().hardware["h200_144gb"]


class TestTheRangeBracketsWhatHasBeenMeasured:
    def test_no_in_shape_run_is_faster_than_the_ceiling(self, measured_profile):
        """A run above the ceiling means the gate over-predicts its time."""
        low, high = measured_profile.decode_tokens_per_s
        too_fast = {k: v for k, v in IN_SHAPE_RATES.items() if v > high}
        assert not too_fast, f"faster than the configured ceiling {high}: {too_fast}"

    def test_no_in_shape_run_is_slower_than_the_floor(self, measured_profile):
        """A run below the floor means the gate UNDER-predicts — the wall-kill
        direction, and the one the pilot got wrong in 2026-08-02."""
        low, _ = measured_profile.decode_tokens_per_s
        too_slow = {k: v for k, v in IN_SHAPE_RATES.items() if v < low}
        assert not too_slow, f"slower than the configured floor {low}: {too_slow}"

    def test_the_held_out_spread_runs_are_covered_too(self, measured_profile):
        """The fit saw 13 local runs; the floor is set by the four it could not.

        Fitting only to what happened to be on this laptop would have put the
        floor at 233 — 8% too fast for a run that had already happened.
        """
        low, _ = measured_profile.decode_tokens_per_s
        assert low <= SPREAD_HELD_OUT


class TestTheRangeIsNotVacuouslyWide:
    def test_it_is_tighter_than_a_factor_of_two(self, measured_profile):
        """A range wide enough to contain anything approves anything.

        The retired [40, 71] spanned 1.78x while sitting entirely outside the
        truth, which is the combination worth guarding: wide AND wrong.
        """
        low, high = measured_profile.decode_tokens_per_s
        assert high / low < 2.0

    def test_the_out_of_shape_runs_do_NOT_constrain_it(self, measured_profile):
        """Deliberate, and stated: including them would span 5.3x.

        `smoke` (n=8) and the 1-rung plaintext baselines are not slower or
        faster hardware — they amortise fixed cost differently. Averaging them
        in would produce a range no gate could act on.
        """
        low, high = measured_profile.decode_tokens_per_s
        assert min(OUT_OF_SHAPE_RATES.values()) < low
        assert max(OUT_OF_SHAPE_RATES.values()) > high
        span_if_included = max(OUT_OF_SHAPE_RATES.values()) / min(OUT_OF_SHAPE_RATES.values())
        assert span_if_included > 5.0


def test_the_retired_ranges_would_both_fail_this(measured_profile):
    """Mutation: the guard must reject the two values that were actually shipped.

    A test that only passes on the current value proves nothing about whether it
    would have caught the mistakes it exists to catch.
    """
    for retired in ((300.0, 1200.0), (40.0, 71.0)):
        low, high = retired
        covers = all(low <= rate <= high for rate in IN_SHAPE_RATES.values())
        assert not covers, f"{retired} would have passed, so this test guards nothing"
