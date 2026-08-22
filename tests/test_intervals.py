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
    mcnemar_exact,
    paired_bootstrap_difference,
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
