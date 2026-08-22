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


def unpaired_interaction_interval(
    a_harmful: tuple[int, int],
    a_benign: tuple[int, int],
    b_harmful: tuple[int, int],
    b_benign: tuple[int, int],
    z: float,
) -> tuple[float, float, float]:
    """Wald interval on a difference of differences from FOUR independent samples.

    The statistic is the same one `paired_bootstrap_difference` computes,
    ``(p_a_harmful - p_a_benign) - (p_b_harmful - p_b_benign)``: how much the
    harm gap moves between two conditions. Each argument is
    ``(successes, total)``.

    **Use this only when the per-item verdicts are unavailable.** Conditions
    within one arm are the SAME prompts rendered differently, so they are
    genuinely item-paired and the paired bootstrap is the correct estimator.
    This one treats all four cells as independent, which inflates the variance
    by the shared item difficulty and therefore returns a WIDER interval than
    the truth. That is the safe direction for a claim that an interaction is
    nonzero, and the wrong direction for a claim that one is absent, so a null
    read off this estimator is not a null.

    It exists because the guard wrapper runs persisted per-item records for the
    encoded harmful arm only; the other five cells of the factorial survive as
    aggregate rates. Recovering the pairing means re-running with per-item
    persistence on every arm, not a different formula.
    """
    cells = (a_harmful, a_benign, b_harmful, b_benign)
    if any(total == 0 for _, total in cells):
        return (float("nan"), float("nan"), float("nan"))
    p = [s / n for s, n in cells]
    point = (p[0] - p[1]) - (p[2] - p[3])
    variance = sum(pi * (1 - pi) / n for pi, (_, n) in zip(p, cells))
    half = z * math.sqrt(variance)
    return (point, max(-2.0, point - half), min(2.0, point + half))


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


def paired_bootstrap_rate_difference(
    arm_a: Sequence[bool],
    arm_b: Sequence[bool],
    rng: np.random.Generator,
    *,
    draws: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Percentile interval for ONE rate minus another, over the SAME items.

    Distinct from `paired_bootstrap_difference`, which contrasts two
    *harmful-minus-benign gaps*. Here each arm is a single per-item verdict
    vector and the statistic is ``mean(a) - mean(b)``. AS-6's surviving
    conditions are the identical 100 harmful prompts wearing different
    transformations, so item difficulty is shared and resampling the two sides
    independently would widen the interval by variance that is not there.

    One resampled index vector serves both arms per draw. That is the whole
    difference from an unpaired interval, and it is the reason a comparison the
    unpaired form must abstain on can separate here.

    `draws` and `alpha` are KEYWORD-ONLY with no default, following this repo's
    rule that a statistical knob a caller could forget is a knob that will be
    forgotten. Returns ``(point, lo, hi)``.
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            "a paired rate difference needs the same items in both conditions; got "
            f"{a.shape} vs {b.shape}"
        )
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = float(a.mean() - b.mean())
    idx = rng.integers(0, a.size, size=(draws, a.size))
    stat = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, order preserved.

    Reported alongside the raw values rather than instead of them. A family of
    pairwise comparisons over the same conditions is exactly where an unadjusted
    p-value invites the reading it cannot support, and this repo has already
    paid once for treating significance as sufficiency.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    running = 0.0
    out = [0.0] * len(p_values)
    for rank, index in enumerate(order):
        adjusted = (len(p_values) - rank) * p_values[index]
        running = max(running, adjusted)
        out[index] = min(1.0, running)
    return out


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
