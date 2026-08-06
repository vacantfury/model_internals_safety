"""The pairing primitives every negative control is built on.

Moved here from `test_decode_lens.py` on 2026-08-06 when `ability_control` became
the second caller — the rule has one home, so its tests do too.
"""

from __future__ import annotations

import pytest

from internals_safety.pairing import (
    derange,
    derangement,
    quantile_strata,
    stratified_derangement,
)


def test_derangement_leaves_no_row_scored_against_itself():
    items = ["a", "b", "c", "d"]
    assert all(new != old for new, old in zip(derange(items), items))


def test_derangement_is_a_permutation_not_a_resample():
    items = ["a", "b", "c", "d"]
    assert sorted(derange(items)) == sorted(items)


def test_derangement_needs_two_items():
    with pytest.raises(ValueError):
        derange(["only"])


def test_the_derangement_fixes_no_point_at_any_size():
    """The property the whole control rests on, asserted rather than assumed.

    A fixed point silently scores a row against itself, which inflates the
    control reading and shrinks the margin toward zero — failing in the direction
    that hides a real result.
    """
    for size in range(2, 40):
        assert all(index != partner for index, partner in enumerate(derangement(size)))


def test_quantile_bins_spread_ties_instead_of_collapsing():
    """Equal lengths must still spread, or a matched null degrades to a free one."""
    strata = quantile_strata([50.0] * 20, n_bins=5)
    assert len(set(strata)) == 5
    assert sorted(strata) == sorted([bin_index for bin_index in range(5) for _ in range(4)])


def test_quantile_bins_are_ordered_by_value():
    strata = quantile_strata([10.0, 90.0, 20.0, 80.0], n_bins=2)
    assert strata[0] == strata[2] == 0
    assert strata[1] == strata[3] == 1


def test_quantile_bins_handle_an_empty_input():
    assert quantile_strata([], n_bins=4) == []


def test_a_stratified_derangement_pairs_only_within_a_stratum():
    values = [1.0, 2.0, 100.0, 101.0]
    partners = stratified_derangement(values, n_bins=2)
    # Two bins of two: the short pair with each other, the long likewise.
    assert {0, 1} == {0, partners[0]}
    assert {2, 3} == {2, partners[2]}


def test_a_stratified_derangement_still_fixes_no_point():
    values = [float(index) for index in range(30)]
    partners = stratified_derangement(values, n_bins=5)
    assert all(index != partner for index, partner in enumerate(partners) if partner is not None)


def test_a_singleton_stratum_is_dropped_rather_than_paired_across_bins():
    """The extreme-length items are exactly the ones a cross-bin pairing would
    mislabel as matched, so they are returned unpairable instead."""
    partners = stratified_derangement([1.0, 2.0, 3.0], n_bins=3)
    assert partners == [None, None, None]


def test_every_stratum_member_gets_a_partner_when_the_bin_has_two():
    partners = stratified_derangement([1.0, 1.0, 2.0, 2.0], n_bins=2)
    assert all(partner is not None for partner in partners)
