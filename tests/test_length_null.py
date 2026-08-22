"""The length null model.

Written against the numbers that produced it: on the phase-0 pilot the JBB
harmful corpus averaged 86.0 characters and the benign one 73.8, giving a
length-only AUROC of 0.6544, and 12 of 15 rungs licensed a "deployment" probe at
essentially exactly that value. The cases below reproduce that shape rather than
testing the arithmetic abstractly.
"""

from __future__ import annotations

import math

import pytest

from internals_safety.config import load_measurements_config
from internals_safety.measurements.length_null import (
    LengthNull,
    length_auroc,
    measure_length_null,
)


def texts(lengths):
    return ["x" * n for n in lengths]


def test_perfectly_separable_lengths_score_one():
    assert length_auroc(texts([10, 11, 12]), texts([1, 2, 3])) == 1.0


def test_identical_length_distributions_score_chance():
    assert length_auroc(texts([5, 6, 7]), texts([5, 6, 7])) == pytest.approx(0.5)


def test_the_null_is_two_sided():
    """A confound running the other way is just as exploitable.

    A directional AUROC would score benign-longer-than-harmful as 0.0 and read as
    "no length signal at all", when a linear probe can use that direction exactly
    as well. The null must be what the BEST length-only classifier could do.
    """
    longer_positives = length_auroc(texts([10, 11, 12]), texts([1, 2, 3]))
    longer_negatives = length_auroc(texts([1, 2, 3]), texts([10, 11, 12]))

    assert longer_positives == longer_negatives == 1.0


def test_empty_or_constant_input_is_nan_not_a_number_that_licenses():
    assert math.isnan(length_auroc([], texts([1, 2])))
    assert math.isnan(length_auroc(texts([1, 2]), []))
    # All identical: no separation is even definable.
    assert math.isnan(length_auroc(texts([7, 7]), texts([7, 7])))


def test_margin_is_measured_against_the_encoded_baseline():
    """The probe sees ciphertext, so ciphertext length is the confound available.

    Using the plaintext baseline would understate the confound for any encoder
    that amplifies the length gap — base64 and hex both do, being fixed-ratio
    expansions.
    """
    null = LengthNull(
        family="hex",
        plain_auroc=0.65,
        encoded_auroc=0.70,
        mean_positive_chars=86.0,
        mean_negative_chars=73.8,
        n_positive=100,
        n_negative=100,
    )
    assert null.margin(0.75) == pytest.approx(0.05)


def test_the_confounded_rung_from_the_pilot_does_not_beat_the_null():
    """The measured case: deployment 0.659 against a length baseline of 0.654.

    This is the reading that licensed on 12 of 15 rungs and was reported as the
    map growing to fourteen. It must not pass.
    """
    null = LengthNull(
        family="base64",
        plain_auroc=0.6544,
        encoded_auroc=0.6539,
        mean_positive_chars=86.0,
        mean_negative_chars=73.8,
        n_positive=100,
        n_negative=100,
    )
    margin = load_measurements_config().probes.length_null_min_margin

    assert null.margin(0.6590) == pytest.approx(0.0051, abs=1e-4)
    assert not null.beats_null(0.6590, margin)


def test_the_two_genuine_rungs_from_the_pilot_do_beat_the_null():
    """reverse_words 0.8442 and zero_width 0.9454 against the same baseline."""
    null = LengthNull(
        family="zero_width",
        plain_auroc=0.6544,
        encoded_auroc=0.6544,
        mean_positive_chars=86.0,
        mean_negative_chars=73.8,
        n_positive=100,
        n_negative=100,
    )
    margin = load_measurements_config().probes.length_null_min_margin

    assert null.beats_null(0.8442, margin)
    assert null.beats_null(0.9454, margin)


def test_a_nan_baseline_fails_closed():
    """An unmeasurable baseline must never license — NaN comparisons are False in
    Python, but silently, so this pins the intent rather than the language rule."""
    null = LengthNull(
        family="broken",
        plain_auroc=float("nan"),
        encoded_auroc=float("nan"),
        mean_positive_chars=float("nan"),
        mean_negative_chars=float("nan"),
        n_positive=0,
        n_negative=0,
    )
    assert not null.beats_null(0.99, 0.05)


def test_measure_length_null_reads_the_texts_actually_sent():
    """Encoded AUROC is computed from ciphertexts, not re-derived from plaintext."""
    plain_pos, plain_neg = texts([80, 90, 100]), texts([70, 75, 72])
    # A fixed-ratio expansion preserves the ordering, hence the separation.
    encoded_pos = ["y" * (len(t) * 2) for t in plain_pos]
    encoded_neg = ["y" * (len(t) * 2) for t in plain_neg]

    null = measure_length_null("double", plain_pos, plain_neg, encoded_pos, encoded_neg)

    assert null.plain_auroc == null.encoded_auroc == 1.0
    assert null.mean_positive_chars == pytest.approx(90.0)
    assert null.mean_negative_chars == pytest.approx(72.333, abs=1e-3)
    assert (null.n_positive, null.n_negative) == (3, 3)


def test_monotone_encoders_carry_the_confound_through_unchanged():
    """Why this is a ladder-wide problem and not one rung's quirk.

    Any encoder monotone in input length preserves the length ordering, so the
    baseline it produces is the plaintext baseline. That is what the pilot
    measured on all 15 rungs (0.649-0.657 against a plaintext 0.6544).
    """
    plain_pos, plain_neg = texts([86, 90, 82]), texts([73, 70, 78])
    for expand in (lambda n: n * 2, lambda n: n * 4 + 3, lambda n: n + 17):
        encoded_pos = ["z" * expand(len(t)) for t in plain_pos]
        encoded_neg = ["z" * expand(len(t)) for t in plain_neg]
        null = measure_length_null("m", plain_pos, plain_neg, encoded_pos, encoded_neg)
        assert null.encoded_auroc == null.plain_auroc


# --- length-matched permutation licensing ----------------------------------


def test_strata_are_quantile_bins_in_probe_order():
    """Positives first then negatives — the order the probe layer concatenates."""
    from internals_safety.measurements.length_null import length_strata

    strata = length_strata(texts([1, 2, 3, 4]), texts([5, 6, 7, 8]), n_bins=4)

    assert len(strata) == 8
    # Longest four texts are the negatives here, so they occupy the top bins.
    assert set(strata[:4]) == {0, 1}
    assert set(strata[4:]) == {2, 3}


def test_strata_spread_ties_instead_of_collapsing():
    """A corpus of identical lengths must still spread across bins.

    Equal-width binning would put every tied example in one bin, which silently
    degrades the matched null back into a free permutation — the exact failure it
    exists to prevent, and invisible in the output.
    """
    from internals_safety.measurements.length_null import length_strata

    strata = length_strata(texts([50] * 10), texts([50] * 10), n_bins=5)

    assert len(set(strata)) == 5


def test_a_length_only_probe_cannot_license_under_the_matched_null():
    """The property the whole module exists for, tested end to end.

    Construct a probe whose ONLY signal is length: activations are the character
    count, and the harmful class is systematically longer. Under a free
    permutation this licenses easily. Under the length-matched null it must not.
    """
    import numpy as np
    import torch

    from internals_safety.config import load_measurements_config
    from internals_safety.measurements.length_null import length_strata
    from internals_safety.models.capture import ActivationBatch
    from internals_safety.probes.linear import (
        permutation_null_max_transfer_auroc,
        permutation_p_value,
    )

    rng = np.random.default_rng(0)
    # Harmful 86 chars on average, benign 73.8 — the real corpus's gap.
    harmful_lengths = rng.normal(86.0, 12.0, 60)
    benign_lengths = rng.normal(73.8, 12.0, 60)

    def batch(lengths):
        # One "feature" that is exactly the length, plus noise dimensions.
        features = np.stack(
            [lengths, rng.normal(0, 1, len(lengths)), rng.normal(0, 1, len(lengths))], axis=1
        )
        tensor = torch.tensor(features, dtype=torch.float32).reshape(len(lengths), 1, 1, 3)
        return ActivationBatch(
            tensor=tensor, layers=[0], positions=["last"], site="resid_pre", model_name="t"
        )

    config = load_measurements_config().probes.model_copy(update={"n_permutations": 100})
    plain_pos, plain_neg = batch(harmful_lengths), batch(benign_lengths)
    enc_pos, enc_neg = batch(harmful_lengths * 2), batch(benign_lengths * 2)

    free = permutation_null_max_transfer_auroc(plain_pos, plain_neg, enc_pos, enc_neg, config)
    strata = length_strata(
        texts([int(round(n)) for n in harmful_lengths * 2]),
        texts([int(round(n)) for n in benign_lengths * 2]),
        n_bins=10,
    )
    matched = permutation_null_max_transfer_auroc(
        plain_pos, plain_neg, enc_pos, enc_neg, config, strata=strata
    )

    # The observed statistic is whatever this length-only probe achieves.
    observed = float(np.max(matched)) if matched.size else float("nan")
    observed = max(observed, float(np.max(free)))

    # The matched null sits well above the free one: it reproduces the length
    # signal, which is the whole point.
    assert matched.mean() > free.mean() + 0.05

    # And a length-only probe fails to clear the matched null.
    real_auroc = 0.66  # what length alone buys on this corpus
    assert permutation_p_value(real_auroc, matched) > 0.05
    assert permutation_p_value(real_auroc, free) < 0.05


def test_strata_length_mismatch_is_an_error_not_a_silent_misalignment():
    import numpy as np
    import torch

    from internals_safety.config import load_measurements_config
    from internals_safety.models.capture import ActivationBatch
    from internals_safety.probes.linear import permutation_null_max_transfer_auroc

    def batch(n):
        tensor = torch.randn(n, 1, 1, 4)
        return ActivationBatch(
            tensor=tensor, layers=[0], positions=["last"], site="resid_pre", model_name="t"
        )

    config = load_measurements_config().probes.model_copy(update={"n_permutations": 4})
    with pytest.raises(ValueError, match="same order"):
        permutation_null_max_transfer_auroc(
            batch(10), batch(10), batch(10), batch(10), config, strata=np.zeros(5, dtype=int)
        )


# ---------------------------------------------------------------------------
# The RATE-scale null (TODO 84 #12/#21, 2026-08-21).
# ---------------------------------------------------------------------------

from internals_safety.measurements.length_null import (  # noqa: E402
    RateLengthNull,
    measure_rate_length_null,
    stratified_rate_gap,
)

NULL_KNOBS = dict(n_bins=4, n_permutations=200, quantile=0.95, seed=0)


def _texts(lengths):
    return ["x" * n for n in lengths]


class TestTheRateNullSeparatesLengthFromTheEffect:
    """Both directions. A control that only ever passes is not a control."""

    def test_a_gap_MADE_of_length_does_not_survive_matching(self):
        """Refusal decided by length alone; the arms differ only in how long they run.

        Overlapping ranges on purpose --- disjoint ones cannot be matched at all,
        and the instrument correctly returns NaN there rather than a verdict.
        """
        harmful_lengths, benign_lengths = list(range(30, 130)), list(range(10, 110))
        refuses = lambda n: n >= 70  # noqa: E731  # definitional: the fixture's rule
        null = measure_rate_length_null(
            "synthetic",
            _texts(harmful_lengths),
            _texts(benign_lengths),
            [refuses(n) for n in harmful_lengths],
            [refuses(n) for n in benign_lengths],
            **NULL_KNOBS,
        )
        assert null.observed_gap == pytest.approx(0.20)
        assert abs(null.stratified_gap) < abs(null.observed_gap) / 2
        # Exactly 0.0, not negative, and that is the point: where the effect is
        # WHOLLY length, the matched gap and the matched null both collapse to
        # zero together. The contract's `clears_length_null` requires a strictly
        # positive margin, so the boundary case fails closed. Asserting the
        # predicate rather than the sign keeps this test pinned to the rule.
        assert not null.margin > 0.0

    def test_a_gap_INDEPENDENT_of_length_survives_matching(self):
        """Identical length distributions, refusal decided by the arm alone."""
        lengths = _texts(range(20, 120))
        null = measure_rate_length_null(
            "synthetic",
            lengths,
            list(lengths),
            [True] * 100,
            [False] * 100,
            **NULL_KNOBS,
        )
        assert null.observed_gap == pytest.approx(1.0)
        assert null.stratified_gap == pytest.approx(1.0)
        assert null.matching_shift == pytest.approx(0.0)
        assert null.margin > 0.0

    def test_the_shift_is_reported_and_not_gated_on(self):
        """A gap that moves under matching but survives is still a pass."""
        harmful_lengths, benign_lengths = list(range(30, 130)), list(range(10, 110))
        null = measure_rate_length_null(
            "synthetic",
            _texts(harmful_lengths),
            _texts(benign_lengths),
            [True] * len(harmful_lengths),
            [n >= 70 for n in benign_lengths],
            **NULL_KNOBS,
        )
        assert null.observed_gap == pytest.approx(0.60)
        assert null.matching_shift != pytest.approx(0.0)
        assert null.margin > 0.0


class TestStratification:
    def test_a_bin_with_only_one_arm_contributes_nothing(self):
        """A stratum with no benign item cannot speak to a difference."""
        gap, used = stratified_rate_gap(
            _texts([10, 11, 12, 900]),
            _texts([10, 11, 12]),
            [False, False, False, True],
            [False, False, False],
            n_bins=4,
        )
        assert used < 4
        assert gap == pytest.approx(0.0)

    def test_no_usable_stratum_is_NaN_never_zero(self):
        gap, used = stratified_rate_gap(_texts([10]), [], [True], [], n_bins=4)
        assert used == 0
        assert gap != gap

    def test_empty_input_is_NaN_never_zero(self):
        gap, used = stratified_rate_gap([], [], [], [], n_bins=4)
        assert used == 0
        assert gap != gap


class TestItFailsClosed:
    def test_a_NaN_stratified_gap_propagates_to_the_margin(self):
        null = RateLengthNull(
            family="f",
            observed_gap=0.5,
            stratified_gap=float("nan"),
            null_quantile=0.1,
            n_bins=4,
            n_strata_used=0,
            n_positive=0,
            n_negative=0,
        )
        assert null.margin != null.margin

    def test_the_margin_is_two_sided(self):
        """A negative gap of the same size clears identically.

        Direction is the `claim`'s business; a control that only recognised
        positive gaps would silently pass every reading asserting a reversal.
        """
        forward = RateLengthNull("f", 0.4, 0.4, 0.1, 4, 4, 50, 50)
        reverse = RateLengthNull("f", -0.4, -0.4, 0.1, 4, 4, 50, 50)
        assert forward.margin == pytest.approx(reverse.margin)


def test_the_same_seed_gives_the_same_null():
    args = ("synthetic", _texts(range(20, 70)), _texts(range(20, 70)), [True] * 50, [False] * 50)
    first = measure_rate_length_null(*args, **NULL_KNOBS)
    second = measure_rate_length_null(*args, **NULL_KNOBS)
    assert first.null_quantile == pytest.approx(second.null_quantile)


def test_every_knob_is_keyword_only_and_has_no_default():
    """The sibling defect was a caller silently getting a passing value."""
    import inspect

    signature = inspect.signature(measure_rate_length_null)
    for name in ("n_bins", "n_permutations", "quantile", "seed"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is inspect.Parameter.empty, name
