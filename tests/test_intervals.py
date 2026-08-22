"""The interval estimators, pinned in BOTH directions.

A test suite for uncertainty is easy to write so that it only ever confirms.
Every class below therefore carries a case the estimator must FAIL to report as
resolved, because an interval that always excludes zero is a verdict with a
statistic attached.

The paired-versus-unpaired distinction is the one that has already cost this
repo a claim (`as5/evidence_and_story.md` §4p), so it is pinned by construction
rather than by a tolerance: perfectly-correlated conditions must give a paired
interval strictly narrower than the unpaired one on the same data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from internals_safety.measurements.intervals import (
    holm_adjusted,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_bootstrap_rate_difference,
    unpaired_difference_interval,
    wilson,
    z_for,
)

Z = z_for(0.05)

DRAWS = 4000
ALPHA = 0.05


class TestWilson:
    def test_it_stays_inside_the_unit_interval_at_the_boundary(self):
        """The reason it was chosen over the normal approximation."""
        lo, hi = wilson(0, 100, Z)
        assert lo == 0.0
        assert 0.0 < hi < 0.05
        lo, hi = wilson(100, 100, Z)
        # `approx`, not `== 1.0`: the closed form lands an epsilon under the
        # clamp at p=1, and tightening the estimator to hit the bound exactly
        # would be changing an estimator to satisfy a test.
        assert hi == pytest.approx(1.0)
        assert 0.95 < lo < 1.0

    def test_it_does_not_degenerate_at_zero(self):
        """A count of zero must still produce a non-empty upper bound."""
        _, hi = wilson(0, 20, Z)
        assert hi > 0.1, "a zero count out of 20 is not evidence of impossibility"

    def test_it_brackets_the_point_estimate(self):
        lo, hi = wilson(30, 100, Z)
        assert lo < 0.30 < hi

    def test_it_narrows_with_n(self):
        narrow = wilson(300, 1000, Z)
        wide = wilson(30, 100, Z)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_an_empty_sample_is_nan_not_zero(self):
        lo, hi = wilson(0, 0, Z)
        assert math.isnan(lo) and math.isnan(hi)


class TestUnpairedDifference:
    def test_identical_arms_give_an_interval_containing_zero(self):
        lo, hi = unpaired_difference_interval(50, 100, 50, 100, Z)
        assert lo < 0.0 < hi

    def test_a_large_separation_excludes_zero(self):
        lo, hi = unpaired_difference_interval(95, 100, 20, 100, Z)
        assert lo > 0.0

    def test_the_sign_is_preserved_rather_than_clipped(self):
        """A reversal must read as negative, never be flattened to zero."""
        lo, hi = unpaired_difference_interval(20, 100, 95, 100, Z)
        assert hi < 0.0

    def test_an_empty_arm_is_nan(self):
        lo, hi = unpaired_difference_interval(0, 0, 5, 10, Z)
        assert math.isnan(lo) and math.isnan(hi)


class TestPairedBootstrap:
    def test_two_identical_conditions_report_no_difference(self):
        """The direction that matters: it must be able to say 'nothing here'."""
        rng = np.random.default_rng(0)
        harmful = [True] * 80 + [False] * 20
        benign = [True] * 30 + [False] * 70
        point, lo, hi = paired_bootstrap_difference(
            (harmful, benign), (harmful, benign), rng, draws=DRAWS, alpha=ALPHA
        )
        assert point == 0.0
        assert lo == 0.0 and hi == 0.0, "identical inputs cannot manufacture spread"

    def test_a_real_shift_excludes_zero(self):
        rng = np.random.default_rng(0)
        a = ([True] * 90 + [False] * 10, [True] * 80 + [False] * 20)  # gap +0.10
        b = ([True] * 90 + [False] * 10, [True] * 20 + [False] * 80)  # gap +0.70
        point, lo, hi = paired_bootstrap_difference(
            a, b, rng, draws=DRAWS, alpha=ALPHA
        )
        assert point == pytest.approx(-0.60, abs=1e-9)
        assert hi < 0.0

    def test_it_is_narrower_than_the_unpaired_interval_on_shared_items(self):
        """Pairing is not cosmetic: it must actually buy resolution.

        Both conditions answer the same items and differ on a handful, which is
        the pipeline table's shape. Treating the arms as independent samples
        widens the interval enough to hide the difference.
        """
        rng = np.random.default_rng(1)
        harmful = [True] * 95 + [False] * 5
        benign_a = [True] * 55 + [False] * 45
        benign_b = [True] * 45 + [False] * 55
        _, lo, hi = paired_bootstrap_difference(
            (harmful, benign_a), (harmful, benign_b), rng, draws=DRAWS, alpha=ALPHA
        )
        paired_width = hi - lo
        # The same contrast built from two independent gap intervals.
        g_a = unpaired_difference_interval(95, 100, 55, 100, Z)
        g_b = unpaired_difference_interval(95, 100, 45, 100, Z)
        independent_width = (g_a[1] - g_a[0]) + (g_b[1] - g_b[0])
        assert paired_width < independent_width

    def test_mismatched_item_counts_raise_rather_than_silently_unpair(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same items"):
            paired_bootstrap_difference(
                ([True] * 10, [True] * 10),
                ([True] * 9, [True] * 10),
                rng,
                draws=10,
                alpha=ALPHA,
            )

    def test_draws_and_alpha_cannot_be_reached_by_omission(self):
        rng = np.random.default_rng(0)
        with pytest.raises(TypeError):
            paired_bootstrap_difference(
                ([True], [True]), ([True], [True]), rng
            )  # type: ignore[call-arg]


class TestMcNemar:
    def test_no_discordance_is_no_evidence(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_symmetric_discordance_is_no_evidence(self):
        assert mcnemar_exact(10, 10) == pytest.approx(1.0)

    def test_all_flips_one_way_is_significant(self):
        """Seven discordant items all running one way, the §4p case."""
        assert mcnemar_exact(7, 0) == pytest.approx(2 / 128)

    def test_it_is_symmetric_in_its_arguments(self):
        assert mcnemar_exact(3, 11) == mcnemar_exact(11, 3)

    def test_a_p_value_never_exceeds_one(self):
        assert mcnemar_exact(1, 1) <= 1.0


class TestTheUnpairedInteractionInterval:
    """The difference-of-differences estimator used for AS-6's factorial.

    It exists only because the guard wrapper runs persisted per-item verdicts
    for one of six cells, so the correct paired estimator cannot be reached.
    Its conservatism is the property under test: a claim that an interaction is
    ABSENT must not be read off it.
    """

    def test_it_recovers_a_known_difference_of_differences(self):
        from internals_safety.measurements.intervals import unpaired_interaction_interval, z_for

        # (0.98 - 0.23) - (0.92 - 0.39) = 0.75 - 0.53 = 0.22
        point, lo, hi = unpaired_interaction_interval(
            (98, 100), (23, 100), (92, 100), (39, 100), z_for(0.05)
        )
        assert point == pytest.approx(0.22, abs=1e-9)
        assert lo < point < hi

    def test_it_is_wider_than_the_paired_estimator_on_identical_data(self):
        """The whole reason it is second choice, asserted rather than assumed.

        Perfectly correlated items are the case where pairing buys the most, so
        if the unpaired interval were ever narrower the estimator would be
        understating and the docstring's safe-direction promise would be false.
        """
        import numpy as np

        from internals_safety.measurements.intervals import (
            paired_bootstrap_difference,
            unpaired_interaction_interval,
            z_for,
        )

        rng = np.random.default_rng(0)
        a_h = [True] * 90 + [False] * 10
        a_b = [True] * 30 + [False] * 70
        b_h = [True] * 80 + [False] * 20
        b_b = [True] * 40 + [False] * 60
        _, p_lo, p_hi = paired_bootstrap_difference(
            (a_h, a_b), (b_h, b_b), rng, draws=4000, alpha=0.05
        )
        _, u_lo, u_hi = unpaired_interaction_interval(
            (90, 100), (30, 100), (80, 100), (40, 100), z_for(0.05)
        )
        assert (u_hi - u_lo) > (p_hi - p_lo)

    def test_a_zero_denominator_is_nan_and_never_a_zero_effect(self):
        from internals_safety.measurements.intervals import unpaired_interaction_interval, z_for

        point, lo, hi = unpaired_interaction_interval(
            (0, 0), (1, 10), (1, 10), (1, 10), z_for(0.05)
        )
        assert all(v != v for v in (point, lo, hi))  # NaN, not 0.0

    def test_sign_reversals_survive_rather_than_being_clipped(self):
        """WildGuard homoglyph's interaction is NEGATIVE: the encoding recovers
        discrimination relative to the wrapper. Clipping at zero would erase the
        most surprising cell in the table."""
        from internals_safety.measurements.intervals import unpaired_interaction_interval, z_for

        point, lo, hi = unpaired_interaction_interval(
            (99, 100), (63, 100), (75, 100), (23, 100), z_for(0.05)
        )
        assert point < 0
        assert hi < 0


class TestThePairedRateDifference:
    """The single-arm paired contrast, whose whole point is that it is narrower."""

    def test_it_recovers_a_known_difference(self):
        rng = np.random.default_rng(0)
        a = [True] * 30 + [False] * 70
        b = [True] * 10 + [False] * 90
        point, lo, hi = paired_bootstrap_rate_difference(a, b, rng, draws=2000, alpha=0.05)
        assert point == pytest.approx(0.20)
        assert lo < 0.20 < hi

    def test_it_is_narrower_than_treating_the_arms_as_independent(self):
        """The justification for the estimator, tested rather than asserted.

        Two conditions that agree on most items: the paired interval sees the
        agreement, an independent one cannot and pays for it in width.
        """
        rng = np.random.default_rng(1)
        a = [i % 10 < 3 for i in range(100)]
        b = [i % 10 < 2 for i in range(100)]
        _, lo, hi = paired_bootstrap_rate_difference(a, b, rng, draws=20000, alpha=0.05)
        paired_width = hi - lo

        z = z_for(0.05)
        a_lo, a_hi = wilson(sum(a), len(a), z)
        b_lo, b_hi = wilson(sum(b), len(b), z)
        independent_width = (a_hi - a_lo) + (b_hi - b_lo)
        assert paired_width < independent_width

    def test_mismatched_item_counts_raise_rather_than_pairing_by_position(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same items"):
            paired_bootstrap_rate_difference([True, False], [True], rng, draws=10, alpha=0.05)

    def test_an_empty_arm_is_nan_not_zero(self):
        rng = np.random.default_rng(0)
        point, lo, hi = paired_bootstrap_rate_difference([], [], rng, draws=10, alpha=0.05)
        assert math.isnan(point) and math.isnan(lo) and math.isnan(hi)


class TestHolmAdjustment:
    """Order preservation and monotonicity, the two ways this goes quietly wrong."""

    def test_it_preserves_input_order(self):
        # Sorted: 0.01, 0.03, 0.04 at indices 1, 2, 0. Steps are 3*0.01 = 0.03,
        # 2*0.03 = 0.06, 1*0.04 = 0.04 which the running maximum lifts to 0.06.
        assert holm_adjusted([0.04, 0.01, 0.03]) == pytest.approx(
            [0.06, 0.03, 0.06], abs=1e-9
        )

    def test_it_is_monotone_non_decreasing_in_rank(self):
        """A step-down procedure that is not enforced monotone can invert a pair."""
        raw = [0.001, 0.02, 0.021, 0.9]
        adjusted = holm_adjusted(raw)
        by_rank = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
        assert by_rank == sorted(by_rank)

    def test_a_single_test_is_unchanged(self):
        assert holm_adjusted([0.03]) == pytest.approx([0.03])

    def test_it_never_exceeds_one(self):
        assert holm_adjusted([0.6, 0.7, 0.8]) == pytest.approx([1.0, 1.0, 1.0])
