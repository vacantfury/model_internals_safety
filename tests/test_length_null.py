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
