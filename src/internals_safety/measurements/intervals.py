"""Uncertainty for reported RATES, and for the contrasts built out of them.

Both papers report rates and differences of rates. Until 2026-08-22 the interval
machinery for them lived as a private helper inside one script, and the external
review's most-repeated objection was that whole tables carry point estimates with
no uncertainty at all. A second script needing the same helper is the rule of two
arriving, so it lands in the library rather than being copied: **the spine holds
anything whose absence in ONE script would be a defect** (`pipeline_architecture.md`
§3.4, the rule the length-null incident produced).

Three estimators, and choosing between them is not cosmetic here.

* `wilson` for a single rate.
* `unpaired_difference_interval` for the harm gap, because its two arms are
  DIFFERENT prompt sets. There is no item pairing to exploit and pretending
  otherwise would understate the width.
* `paired_bootstrap_difference` for a contrast between two conditions measured on
  the SAME items, which is every stage-to-stage comparison in the pipeline table.
  Independent variances there are too wide, and this repo has already had one
  claim overturned by exactly that mistake in the opposite direction (§4p: an
  unpaired null hid a real effect).

⚠️ **An interval is not a test, and a wide one is not a null.** Two of the
review's objections are that the paper calls an unresolved contrast "nothing".
These functions quantify; whether a number may be reported at all is
`contract.py`'s question and stays there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np


def z_for(alpha: float) -> float:
    """The two-sided standard-normal quantile for a confidence level.

    Every estimator below takes `z` explicitly and none defaults it. That is the
    config-discipline rule applied honestly rather than satisfied with a comment:
    a hard-coded 1.96 IS a confidence level chosen in source, and this repo
    already configures `measurements.probes.alpha`. Deriving it means changing
    alpha in YAML changes every reported interval, which is what a knob is for.

    `NormalDist` is stdlib, so this adds no dependency; scipy is not declared in
    `pyproject.toml` and a library module must not import what a fresh clone may
    not have.

    `refusal_control.Z_95` stays where it is and is NOT re-exported here: that
    module is a pure combination module barred from importing measurement
    siblings (`tests/test_package_structure.py`), and reaching across for a
    constant would trade a duplicated number for a broken invariant.
    """
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson(successes: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval. Chosen over normal-approximation for a reason.

    At n=100 with counts as low as 7, the normal approximation's lower bound
    runs negative and its coverage is poor exactly where this paper's smallest
    and most-quoted cells sit. Wilson stays inside [0, 1] and does not degenerate
    at zero, which matters for a finding that IS a count near zero.

    Moved here from `scripts/review_statistics.py` on 2026-08-22 when the
    pipeline table became its second caller. `z` lost its default in the same
    move; see `z_for`.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def unpaired_difference_interval(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    z: float,
) -> tuple[float, float]:
    """Wald interval on ``p_a - p_b`` for two INDEPENDENT samples.

    The harm gap's arms are the harmful and benign corpora: different prompts,
    no pairing available. The interval is deliberately allowed outside
    ``[-1, 1]``-free clipping only at the logical bounds, since a difference of
    proportions legitimately spans negative values and clipping the sign would
    hide a reversal.
    """
    if total_a == 0 or total_b == 0:
        return (float("nan"), float("nan"))
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    half = z * math.sqrt(p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b)
    return (max(-1.0, p_a - p_b - half), min(1.0, p_a - p_b + half))


def paired_bootstrap_difference(
    arm_a: Sequence[Sequence[bool]],
    arm_b: Sequence[Sequence[bool]],
    rng: np.random.Generator,
    *,
    draws: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Percentile interval for a contrast between two ITEM-PAIRED conditions.

    Each argument is a pair of per-item verdict vectors ``(harmful, benign)``
    for one condition, and the statistic is
    ``(mean(a_harmful) - mean(a_benign)) - (mean(b_harmful) - mean(b_benign))``.
    Items are resampled ONCE per draw and the same resampled indices are used in
    both conditions, which is what makes this paired: a stage-to-stage
    comparison in the pipeline runs the identical 100 harmful and 100 benign
    prompts through two checkpoints, so item difficulty is shared and must not
    be resampled independently on each side.

    `draws` and `alpha` are KEYWORD-ONLY with no default, following this repo's
    rule that a statistical knob a caller could forget is a knob that will be
    forgotten. Returns ``(point, lo, hi)``.
    """
    a_harmful, a_benign = (np.asarray(v, dtype=float) for v in arm_a)
    b_harmful, b_benign = (np.asarray(v, dtype=float) for v in arm_b)
    if a_harmful.shape != b_harmful.shape or a_benign.shape != b_benign.shape:
        raise ValueError(
            "paired bootstrap needs the same items in both conditions; got "
            f"harmful {a_harmful.shape} vs {b_harmful.shape} and "
            f"benign {a_benign.shape} vs {b_benign.shape}"
        )
    if a_harmful.size == 0 or a_benign.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = float(
        (a_harmful.mean() - a_benign.mean()) - (b_harmful.mean() - b_benign.mean())
    )
    h_idx = rng.integers(0, a_harmful.size, size=(draws, a_harmful.size))
    b_idx = rng.integers(0, a_benign.size, size=(draws, a_benign.size))
    stat = (a_harmful[h_idx].mean(axis=1) - a_benign[b_idx].mean(axis=1)) - (
        b_harmful[h_idx].mean(axis=1) - b_benign[b_idx].mean(axis=1)
    )
    lo, hi = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two DISCORDANT counts.

    Concordant items carry no information about a difference between paired
    conditions, so they never enter. Under the null each discordant item is a
    fair coin, which makes this a two-sided binomial test on ``only_a`` out of
    ``only_a + only_b``.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)
