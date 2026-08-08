"""I4's pre-gate — and the only property that matters is that it FAILS.

A gate that passes everything is the same as no gate, and this one guards a
specific silent failure: Llama Scope is trained on Llama-3.1-8B-**Base** while
our target is **Instruct**, and a base-trained dictionary applied to an
instruct-tune returns a perfectly normal-looking reconstruction and perfectly
normal-looking feature activations. So the tests below are mostly adversarial —
dictionaries that should NOT be licensed, and the gate refusing them.

Every SAE here is a test double satisfying the two-method protocol. That is the
point of the protocol: no download, no GPU, no 256-checkpoint suite.
"""

from __future__ import annotations

import math

import pytest
import torch

from internals_safety.config import SAEConfig, load_measurements_config
from internals_safety.measurements.sae_reconstruction import (
    RandomDictionary,
    ReconstructionQuality,
    l0_sparsity,
    measure_reconstruction,
    reading,
    unmeasured_reading,
    variance_explained,
)


class PerfectSAE:
    """The identity — an upper bound no real dictionary reaches."""

    def encode(self, activations):
        return activations

    def decode(self, features):
        return features


class DeadSAE:
    """Returns zeros. Reconstructs nothing, and must never be licensed."""

    def encode(self, activations):
        return torch.zeros_like(activations)

    def decode(self, features):
        return features


class MeanSAE:
    """Emits the batch mean and nothing else.

    ⚠️ The most important negative case. It has a genuinely low MSE on
    low-variance activations and would look respectable on any absolute error
    bar — but it carries zero information about the individual prompt. This is
    why `variance_explained` is CENTRED: the mean is free, so this must score 0.
    """

    def encode(self, activations):
        mean = activations.mean(dim=tuple(range(activations.ndim - 1)), keepdim=True)
        return mean.expand_as(activations)

    def decode(self, features):
        return features


def _quality(**overrides):
    base = dict(
        layer=1, n_prompts=8, mse=0.01, variance_explained=0.9, l0=32.0,
        kl_sae=0.1, kl_ablated=5.0, control_variance_explained=0.2,
    )
    return ReconstructionQuality(**{**base, **overrides})


class TestVarianceExplained:
    def test_a_perfect_reconstruction_explains_everything(self):
        x = torch.randn(20, 8)
        assert variance_explained(x, x) == pytest.approx(1.0)

    def test_emitting_the_mean_explains_nothing(self):
        """The mean is free, so a dictionary that only reproduces it scores 0,
        not something flattering."""
        x = torch.randn(20, 8)
        assert variance_explained(x, x.mean(dim=0, keepdim=True).expand_as(x)) == pytest.approx(0.0)

    def test_a_worse_than_mean_reconstruction_goes_negative(self):
        """Not clamped: 'worse than doing nothing' is a real outcome for an
        out-of-distribution dictionary and must be visible as one."""
        x = torch.randn(20, 8)
        assert variance_explained(x, -x) < 0.0

    def test_a_constant_batch_is_undefined_rather_than_perfect(self):
        x = torch.ones(20, 8)
        assert math.isnan(variance_explained(x, x))


class TestSparsityIsReported:
    def test_l0_counts_active_features(self):
        features = torch.zeros(4, 100)
        features[:, :7] = 1.0
        assert l0_sparsity(features) == pytest.approx(7.0)

    def test_the_random_control_honours_its_k(self):
        """A DENSE control would reconstruct better for reasons unrelated to
        having learned features, making it trivially easy to beat."""
        generator = torch.Generator().manual_seed(0)
        control = RandomDictionary(d_model=16, n_features=128, k=5, generator=generator)
        assert l0_sparsity(control.encode(torch.randn(4, 16))) == pytest.approx(5.0)

    def test_the_control_round_trip_has_the_right_shape(self):
        generator = torch.Generator().manual_seed(0)
        control = RandomDictionary(d_model=16, n_features=128, k=5, generator=generator)
        x = torch.randn(4, 16)
        assert control.decode(control.encode(x)).shape == x.shape


class TestTheGate:
    """What is and is not admissible."""

    def test_a_good_dictionary_passes(self):
        config = load_measurements_config().sae
        result = reading(_quality(), config)
        assert result.licensed is True
        assert result.value == pytest.approx(1.0 - 0.1 / 5.0)

    def test_a_dictionary_no_better_than_deleting_the_layer_fails(self):
        config = load_measurements_config().sae
        assert reading(_quality(kl_sae=5.0), config).licensed is False

    def test_a_dictionary_WORSE_than_deleting_the_layer_reads_negative(self):
        """Not clamped. An out-of-distribution dictionary can actively harm the
        output distribution, and rounding that to 0 would hide the finding."""
        result = reading(_quality(kl_sae=8.0), load_measurements_config().sae)
        assert result.value < 0.0
        assert result.licensed is False

    def test_good_downstream_KL_does_not_excuse_poor_variance(self):
        config = load_measurements_config().sae
        assert reading(_quality(variance_explained=0.10), config).licensed is False

    def test_a_dictionary_its_random_control_matches_is_not_licensed(self):
        """If a matched-shape random basis reconstructs as well, 'the SAE
        reconstructs our model' is a fact about linear algebra."""
        config = load_measurements_config().sae
        assert reading(_quality(control_variance_explained=0.95), config).licensed is False

    def test_no_control_at_all_is_not_licensed(self):
        config = load_measurements_config().sae
        assert reading(_quality(control_variance_explained=None), config).licensed is False

    def test_a_layer_with_no_downstream_contribution_is_UNMEASURED_not_passed(self):
        """⚠️ Tri-state again. If deleting the layer changes nothing, there is no
        contribution to recover and the ratio is undefined — reporting 1.0 would
        credit the dictionary for a layer that does not matter."""
        result = reading(_quality(kl_ablated=0.0), load_measurements_config().sae)
        assert result.licensed is None
        assert math.isnan(result.value)

    def test_the_reading_records_what_the_dictionary_was_trained_on(self):
        """`trained_on` vs `evaluated_on` IS the pre-gate's subject; a record
        carrying only one of the pair cannot answer its own question."""
        detail = reading(_quality(), load_measurements_config().sae).detail
        assert "trained_on" in detail and "evaluated_on" in detail
        assert detail["l0"] == pytest.approx(32.0)

    def test_the_operating_point_says_what_zero_and_negative_mean(self):
        text = reading(_quality(), load_measurements_config().sae).operating_point
        assert "deleting the layer" in text

    def test_an_unmeasured_reading_states_its_reason(self):
        assert "reason" in unmeasured_reading("no SAE supplied").detail


class TestEndToEndOnARealModel:
    """Through the tiny model, so the substitution hooks genuinely run."""

    @pytest.fixture
    def prompts(self):
        return ["the cat sat", "a dog ran fast", "birds fly"]

    @pytest.fixture
    def config(self):
        return SAEConfig(trained_on="tiny-test", min_kl_recovered=0.8, min_transfer_ratio=0.8)

    def test_the_identity_dictionary_recovers_everything(self, tiny_model, prompts, config):
        """The sanity check the gate rests on: substituting an exact round trip
        must leave the output distribution untouched, so KL recovered is ~1."""
        quality = measure_reconstruction(
            tiny_model, PerfectSAE(), prompts, layer=1, config=config, batch_size=8
        )
        assert quality.variance_explained == pytest.approx(1.0, abs=1e-4)
        assert quality.kl_sae == pytest.approx(0.0, abs=1e-4)
        if quality.kl_recovered is not None:
            assert quality.kl_recovered == pytest.approx(1.0, abs=1e-3)

    def test_a_dead_dictionary_is_indistinguishable_from_ablating_the_layer(
        self, tiny_model, prompts, config
    ):
        """It writes zeros, which is exactly what the ablation arm writes — so
        KL recovered must be ~0, and the gate must refuse it."""
        quality = measure_reconstruction(
            tiny_model, DeadSAE(), prompts, layer=1, config=config, batch_size=8
        )
        if quality.kl_recovered is not None:
            assert quality.kl_recovered == pytest.approx(0.0, abs=1e-3)
        assert reading(quality, config).licensed is not True

    def test_the_mean_dictionary_explains_no_variance(self, tiny_model, prompts, config):
        quality = measure_reconstruction(
            tiny_model, MeanSAE(), prompts, layer=1, config=config, batch_size=8
        )
        assert quality.variance_explained < 0.5
        assert reading(quality, config).licensed is not True

    def test_a_shape_changing_dictionary_fails_with_the_shapes_named(
        self, tiny_model, prompts, config
    ):
        """Found while writing these tests: a dictionary that reduces over the
        batch surfaces as a broadcast error inside
        `scaled_dot_product_attention`, which says nothing about the dictionary
        that caused it."""

        class Reducing:
            def encode(self, activations):
                return activations.mean(dim=0, keepdim=True)

            def decode(self, features):
                return features

        with pytest.raises(ValueError, match="changed shape"):
            measure_reconstruction(
                tiny_model, Reducing(), prompts, layer=1, config=config, batch_size=8
            )

    def test_the_random_control_runs_and_is_recorded(self, tiny_model, prompts, config):
        generator = torch.Generator().manual_seed(0)
        control = RandomDictionary(
            d_model=tiny_model.d_model, n_features=tiny_model.d_model * 4, k=4,
            generator=generator,
        )
        quality = measure_reconstruction(
            tiny_model, PerfectSAE(), prompts, layer=1, config=config,
            control=control, batch_size=8,
        )
        assert quality.control_variance_explained is not None
        assert quality.control_margin is not None
