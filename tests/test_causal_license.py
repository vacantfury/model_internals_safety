"""Causal licensing — the criteria our correlational licensing has no analogue for."""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import CausalLicenseConfig
from internals_safety.measurements.causal_license import (
    CausalEvidence,
    is_discarded,
    kl_divergence,
    licenses,
    refusal_probability,
    select_direction,
)

N_LAYERS = 32


def evidence(**overrides) -> CausalEvidence:
    """A candidate that passes every criterion, so each test varies one thing."""
    defaults = dict(
        layer=18,
        position="instruction_final",
        behaviour="refusal_opening",
        behaviour_before=0.90,
        behaviour_after_ablation=0.10,
        harmless_behaviour_before=0.02,
        harmless_behaviour_after_addition=0.70,
        kl=0.01,
    )
    return CausalEvidence(**{**defaults, **overrides})


def test_kl_is_zero_for_identical_distributions():
    logits = torch.randn(4, 6, 20)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_when_the_intervention_moves_mass():
    baseline = torch.zeros(2, 3, 10)
    intervened = torch.zeros(2, 3, 10)
    intervened[:, -1, 0] = 5.0
    assert kl_divergence(baseline, intervened) > 0.0


def test_kl_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        kl_divergence(torch.zeros(2, 3, 10), torch.zeros(2, 4, 10))


def test_refusal_probability_reads_the_named_tokens():
    logits = torch.full((1, 2, 6), -20.0)
    logits[0, -1, 3] = 20.0
    assert refusal_probability(logits, [3]) == pytest.approx(1.0, abs=1e-4)
    assert refusal_probability(logits, [0, 1]) == pytest.approx(0.0, abs=1e-4)


def test_refusal_probability_refuses_an_empty_token_set():
    """Would be identically zero — a silent zero reading as 'never refuses'."""
    with pytest.raises(ValueError, match="no refusal tokens"):
        refusal_probability(torch.zeros(1, 1, 5), [])


def test_bypass_and_induce_scores_are_signed_differences():
    candidate = evidence()
    assert candidate.bypass_score == pytest.approx(0.80)
    assert candidate.induce_score == pytest.approx(0.68)


def test_bypass_fraction_is_scale_free():
    """The cipher band refuses at ~100% and the comprehension band does not, so
    an absolute drop would be strict on one and vacuous on the other."""
    assert evidence(behaviour_before=1.00, behaviour_after_ablation=0.50).bypass_fraction == pytest.approx(0.5)
    assert evidence(behaviour_before=0.40, behaviour_after_ablation=0.20).bypass_fraction == pytest.approx(0.5)


def test_bypass_fraction_is_zero_when_nothing_was_refused():
    """Nothing was released because nothing was held — not a division by zero."""
    assert evidence(behaviour_before=0.0, behaviour_after_ablation=0.0).bypass_fraction == 0.0


def test_a_direction_that_releases_nothing_is_discarded():
    """OURS, not Arditi's. Their filter only NaN-checks the bypass score because
    for them it is a sort key over a pool known to contain a real direction. As
    an upstream GATE it has to bind."""
    config = CausalLicenseConfig()
    assert is_discarded(
        evidence(behaviour_before=0.90, behaviour_after_ablation=0.88), N_LAYERS, config
    )


def test_a_good_candidate_survives_the_filter():
    assert not is_discarded(evidence(), N_LAYERS, CausalLicenseConfig())


def test_a_high_kl_candidate_is_discarded():
    """This is the criterion that screens the length confound: a direction whose
    removal damages harmless behaviour was not carrying refusal specifically."""
    config = CausalLicenseConfig()
    assert is_discarded(evidence(kl=config.kl_threshold + 0.01), N_LAYERS, config)


def test_a_candidate_that_does_not_induce_refusal_is_discarded():
    assert is_discarded(evidence(harmless_behaviour_after_addition=0.0), N_LAYERS, CausalLicenseConfig())


def test_late_layer_candidates_are_pruned():
    """Past 80% of depth the residual stream is committed to tokens, so an
    effect there is downstream of the computation being claimed about."""
    config = CausalLicenseConfig()
    cut = int(N_LAYERS * (1.0 - config.prune_layer_percentage))
    assert is_discarded(evidence(layer=cut), N_LAYERS, config)
    assert not is_discarded(evidence(layer=cut - 1), N_LAYERS, config)


def test_nan_evidence_is_discarded_not_propagated():
    """An intervention that produced no number is not evidence; letting NaN
    through would let the sort put it first."""
    assert is_discarded(evidence(kl=float("nan")), N_LAYERS, CausalLicenseConfig())


def test_selection_takes_the_best_bypass_among_eligible_candidates():
    weak = evidence(layer=10, behaviour_after_ablation=0.60)
    strong = evidence(layer=12, behaviour_after_ablation=0.05)
    ineligible = evidence(layer=14, behaviour_after_ablation=0.00, kl=5.0)
    chosen = select_direction([weak, strong, ineligible], N_LAYERS, CausalLicenseConfig())
    assert chosen is strong


def test_selection_raises_when_everything_is_filtered_out():
    """An empty filter is a FINDING — the correlational probe found a separation
    no intervention can act on. Returning the least-bad candidate would erase it."""
    with pytest.raises(ValueError, match="no direction passes"):
        select_direction([evidence(kl=9.0), evidence(kl=8.0)], N_LAYERS, CausalLicenseConfig())


def test_selection_raises_on_an_empty_candidate_list():
    with pytest.raises(ValueError, match="no candidate directions"):
        select_direction([], N_LAYERS, CausalLicenseConfig())


def test_licenses_is_true_when_any_candidate_survives():
    config = CausalLicenseConfig()
    assert licenses([evidence(kl=9.0), evidence()], N_LAYERS, config)
    assert not licenses([evidence(kl=9.0)], N_LAYERS, config)


def test_the_length_confound_scenario_is_rejected():
    """The case that motivated this module, as a regression test.

    A direction that separates harmful from benign by character length: it
    classifies well, so the permutation test licenses it — but ablating it does
    not release refusal, and it does not induce refusal when added. Our
    correlational licensing has no way to see this.
    """
    length_like = evidence(
        behaviour_before=0.90,
        behaviour_after_ablation=0.88,     # ablation released nothing
        harmless_behaviour_after_addition=0.02,  # adding it induced nothing
        harmless_behaviour_before=0.02,
        kl=0.005,                        # and it was harmless to remove
    )
    config = CausalLicenseConfig()
    assert is_discarded(length_like, N_LAYERS, config)
    assert not licenses([length_like], N_LAYERS, config)
