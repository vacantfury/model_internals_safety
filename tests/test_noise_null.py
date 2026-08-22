"""The cross-model spread null, independent and paired (TODO 84 #3, 2026-08-21).

The paper's four models answer the SAME 100 prompts. Drawing each model's rate
as an independent binomial ignores that, and the difference is not cosmetic: on
the harmful arm it is what separated "inside the null, we cannot distinguish
them" from "outside it". Both nulls stay in the tree because one condition ---
plaintext --- has no per-item verdicts on disk, and a caller that silently
substituted one for the other would be the same defect one level up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from figure_arm_inversion import noise_null, paired_noise_null  # noqa: E402

DRAWS = 4000  # plumbing: simulation size for the tests, not a reported quantity


def rng():
    return np.random.default_rng(0)


class TestThePairedNullIsNarrower:
    def test_shared_item_difficulty_shrinks_the_spread(self):
        """The whole point. Same mean rate, less drift between identical models."""
        items = np.array([[True] * 60 + [False] * 40] * 4)
        independent = noise_null([0.6] * 4, 100, rng(), draws=DRAWS)
        paired = paired_noise_null(items, rng(), draws=DRAWS)
        assert paired[0] < independent[0]

    def test_perfectly_agreeing_models_have_a_ZERO_null(self):
        """Deterministic items leave nothing for identical models to differ on.

        The other direction, and the reason the paired null bites: where every
        item is 0 or 1 for everyone, an independent binomial still invents a
        spread out of a rate that was never uncertain.
        """
        items = np.array([[True] * 50 + [False] * 50] * 4)
        median, low, high = paired_noise_null(items, rng(), draws=DRAWS)
        assert median == pytest.approx(0.0)
        assert high == pytest.approx(0.0)
        assert noise_null([0.5] * 4, 100, rng(), draws=DRAWS)[0] > 0.0

    def test_total_disagreement_gives_the_WIDEST_null(self):
        """Items at difficulty 0.5 carry maximal variance — the conservative end."""
        half = np.array([[True] * 50 + [False] * 50, [False] * 50 + [True] * 50] * 2)
        agreeing = np.array([[True] * 50 + [False] * 50] * 4)
        assert paired_noise_null(half, rng(), draws=DRAWS)[0] > paired_noise_null(
            agreeing, rng(), draws=DRAWS
        )[0]


class TestBothNullsBehave:
    def test_the_interval_brackets_the_median(self):
        items = np.array([[True] * 60 + [False] * 40] * 4)
        median, low, high = paired_noise_null(items, rng(), draws=DRAWS)
        assert low <= median <= high

    def test_the_same_seed_reproduces(self):
        items = np.array([[True] * 60 + [False] * 40] * 4)
        assert paired_noise_null(items, rng(), draws=DRAWS) == paired_noise_null(
            items, rng(), draws=DRAWS
        )

    def test_draws_is_keyword_only_with_no_default(self):
        import inspect

        for fn in (noise_null, paired_noise_null):
            parameter = inspect.signature(fn).parameters["draws"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
            assert parameter.default is inspect.Parameter.empty, fn.__name__
