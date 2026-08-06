"""I3 — entropy dynamics, and the lens seam it stands on."""

from __future__ import annotations

import math

import pytest
import torch

from internals_safety.measurements.entropy_dynamics import (
    entropy,
    measure_entropy_dynamics,
)
from internals_safety.models.capture import capture_activations
from internals_safety.models.lens import final_norm, lens_distribution, unembed
from internals_safety.models.loader import prepare_prompts, tokenize_batch


class TestEntropy:
    def test_a_uniform_distribution_has_maximum_entropy(self):
        assert float(entropy(torch.zeros(8))) == pytest.approx(math.log(8), abs=1e-5)

    def test_a_point_mass_has_zero_entropy(self):
        logits = torch.full((6,), -1e4)
        logits[2] = 1e4
        assert float(entropy(logits)) == pytest.approx(0.0, abs=1e-5)

    def test_extreme_logits_do_not_produce_nan(self):
        """Computed from log-softmax so p*log p -> 0 instead of 0 * -inf -> nan."""
        logits = torch.tensor([1e4, -1e4, -1e4, -1e4])
        assert torch.isfinite(entropy(logits)).all()

    def test_entropy_reduces_the_last_dimension(self):
        assert entropy(torch.randn(3, 5, 20)).shape == (3, 5)


class TestLensSeam:
    def test_the_final_norm_is_found(self, tiny_model):
        assert final_norm(tiny_model) is not None

    def test_unembedding_the_last_hidden_state_reproduces_the_model_logits(
        self, tiny_model, messages
    ):
        """The load-bearing test for the whole seam.

        capture.py takes resid_post before the final norm, so a captured state is
        NOT what the unembedding expects. If this fails, the norm is being
        skipped and every entropy reported here is miscalibrated in a way that
        still looks like a valid distribution.
        """
        prompts = prepare_prompts(tiny_model, messages, positions=["last"])
        inputs = tokenize_batch(tiny_model, prompts)
        with torch.inference_mode():
            output = tiny_model.model(**inputs, output_hidden_states=True, use_cache=False)

        final_state = output.hidden_states[-1][:, -1, :]
        torch.testing.assert_close(
            unembed(tiny_model, final_state),
            output.logits[:, -1, :].float(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_the_norm_is_actually_applied(self, tiny_model):
        """Non-vacuous guard, via RMSNorm's scale invariance.

        A first attempt compared normed against un-normed logits on a real
        hidden state and PASSED WRONGLY: this fixture's RMSNorm weights are near
        1 and its states are near unit scale, so the norm is nearly the identity
        and the two agreed to four decimals. Scaling the input separates them by
        construction instead — RMSNorm divides out the scale, so a normed readout
        is unchanged by a 10x input while an un-normed one scales with it.
        """
        state = torch.randn(2, tiny_model.d_model)
        torch.testing.assert_close(
            unembed(tiny_model, state), unembed(tiny_model, state * 10.0),
            rtol=1e-3, atol=1e-3,
        )
        with torch.inference_mode():
            small = tiny_model.model.lm_head(state).float()
            large = tiny_model.model.lm_head(state * 10.0).float()
        assert not torch.allclose(small, large, rtol=1e-2, atol=1e-2)

    def test_the_lens_distribution_sums_to_one(self, tiny_model):
        probabilities = lens_distribution(tiny_model, torch.randn(4, tiny_model.d_model))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4), rtol=1e-4, atol=1e-4)

    def test_a_wrong_width_state_is_rejected(self, tiny_model):
        with pytest.raises(ValueError, match="expected d_model"):
            unembed(tiny_model, torch.randn(2, tiny_model.d_model + 1))


class TestEntropyDynamics:
    @pytest.fixture
    def profile(self, tiny_model, messages):
        prompts = prepare_prompts(tiny_model, messages, positions=["instruction_final", "last"])
        captured = capture_activations(
            tiny_model, prompts, layers="all", positions=["instruction_final", "last"]
        )
        return measure_entropy_dynamics(tiny_model, captured, "last", batch_size=2)

    def test_one_entropy_per_prompt_per_layer(self, profile, tiny_model, messages):
        assert profile.entropies.shape == (len(messages), tiny_model.n_layers)
        assert profile.n_prompts == len(messages)

    def test_entropies_are_finite_and_within_the_vocabulary_bound(self, profile, tiny_model):
        assert torch.isfinite(profile.entropies).all()
        assert (profile.entropies >= 0).all()
        ceiling = math.log(tiny_model.model.config.vocab_size)
        assert (profile.entropies <= ceiling + 1e-4).all()

    def test_chunking_does_not_change_the_answer(self, tiny_model, messages):
        """A statistic that depends on batch composition is not a statistic."""
        prompts = prepare_prompts(tiny_model, messages, positions=["last"])
        captured = capture_activations(tiny_model, prompts, layers="all", positions=["last"])
        one = measure_entropy_dynamics(tiny_model, captured, "last", batch_size=1)
        many = measure_entropy_dynamics(tiny_model, captured, "last", batch_size=64)
        torch.testing.assert_close(one.entropies, many.entropies, rtol=1e-4, atol=1e-4)

    def test_the_resolution_layer_is_the_steepest_drop(self):
        from internals_safety.measurements.entropy_dynamics import EntropyProfile

        profile = EntropyProfile(
            position="last",
            layers=[0, 1, 2, 3],
            # drops between consecutive layers: 0.1, 3.0, 0.2 -> steepest is index 1
            entropies=torch.tensor([[5.0, 4.9, 1.9, 1.7]]),
        )
        assert int(profile.resolution_layer()[0]) == 1
        assert float(profile.total_drop()[0]) == pytest.approx(3.3)
        assert float(profile.minimum[0]) == pytest.approx(1.7)

    def test_a_resolution_layer_needs_two_layers(self):
        from internals_safety.measurements.entropy_dynamics import EntropyProfile

        profile = EntropyProfile(position="last", layers=[0], entropies=torch.tensor([[2.0]]))
        with pytest.raises(ValueError, match="at least two captured layers"):
            profile.resolution_layer()

    def test_an_uncaptured_position_is_rejected(self, tiny_model, messages):
        prompts = prepare_prompts(tiny_model, messages, positions=["last"])
        captured = capture_activations(tiny_model, prompts, layers="all", positions=["last"])
        with pytest.raises(ValueError, match="not captured"):
            measure_entropy_dynamics(tiny_model, captured, "instruction_final")

    def test_a_zero_batch_size_is_rejected(self, tiny_model, messages):
        prompts = prepare_prompts(tiny_model, messages, positions=["last"])
        captured = capture_activations(tiny_model, prompts, layers="all", positions=["last"])
        with pytest.raises(ValueError, match="at least 1"):
            measure_entropy_dynamics(tiny_model, captured, "last", batch_size=0)
