"""The Llama Scope loader — three properties that would each silently corrupt I4.

Built against a SYNTHETIC checkpoint, so no download and no network. The real
checkpoint's shape and settings were read off the artifact when the loader was
written (`fnlp/Llama3_1-8B-Base-LXR-8x`, layer 15) and the fixture below mirrors
them exactly; the numbers in the module docstring are from that read.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from internals_safety.models.sae_loader import LlamaScopeSAE, load_llama_scope_sae

D_MODEL, D_SAE = 16, 64

HYPERPARAMS = {
    "hook_point_in": "blocks.15.hook_resid_post",
    "act_fn": "jumprelu",
    "jump_relu_threshold": 0.35546875,
    "apply_decoder_bias_to_pre_encoder": False,
    "norm_activation": "dataset-wise",
    "dataset_average_activation_norm": {"in": 10.8125, "out": 10.8125},
    "d_model": D_MODEL,
    "d_sae": D_SAE,
    "top_k": 50,
    "expansion_factor": 8,
}


@pytest.fixture
def checkpoint(tmp_path):
    def build(**overrides):
        hyper = {**HYPERPARAMS, **overrides}
        generator = torch.Generator().manual_seed(0)
        weights = {
            "encoder.weight": torch.randn(D_SAE, D_MODEL, generator=generator),
            "encoder.bias": torch.zeros(D_SAE),
            "decoder.weight": torch.randn(D_MODEL, D_SAE, generator=generator),
            "decoder.bias": torch.zeros(D_MODEL),
        }
        w = tmp_path / f"{len(list(tmp_path.iterdir()))}.safetensors"
        h = w.with_suffix(".json")
        save_file(weights, str(w))
        h.write_text(json.dumps(hyper))
        return w, h

    return build


class TestTheLayerMapping:
    """⚠️ Trap 1: Llama Scope names an SAE by the block whose resid_POST it reads,
    and we capture resid_PRE. Off by one does not error — it reconstructs a
    neighbouring layer badly and reads as 'the dictionary does not transfer',
    which is the pre-gate's own verdict wearing our bug."""

    def test_resid_post_of_block_N_is_our_resid_pre_of_block_N_plus_1(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        assert sae.hook_point == "blocks.15.hook_resid_post"
        assert sae.our_layer == 16

    def test_an_unrecognised_hook_point_REFUSES_rather_than_defaulting(self, checkpoint):
        sae = load_llama_scope_sae(
            *checkpoint(hook_point_in="blocks.15.hook_mlp_out"), repo_id="test"
        )
        with pytest.raises(ValueError, match="unrecognised hook point"):
            _ = sae.our_layer


class TestTheActivationFunction:
    """⚠️ Trap 2: `top_k: 50` sits in the same file as `act_fn: jumprelu` — a
    training-schedule leftover. A TopK forward gives exactly 50 active features
    where the real dictionary gives however many clear the threshold."""

    def test_it_is_jumprelu_not_topk(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(8, D_MODEL) * 3.0
        active = (sae.encode(x) > 0).float().sum(dim=-1)
        assert not torch.all(active == sae.nominal_top_k)

    def test_values_above_the_threshold_pass_through_UNCHANGED(self, checkpoint):
        """JumpReLU is not ReLU: it does not shrink what it keeps."""
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(4, D_MODEL) * 3.0
        features = sae.encode(x)
        raw = sae._normalise(x) @ sae.encoder_weight.T + sae.encoder_bias
        kept = features > 0
        assert torch.allclose(features[kept], raw[kept])

    def test_values_below_the_threshold_are_zero_not_small(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        features = sae.encode(torch.randn(4, D_MODEL))
        assert torch.all((features == 0) | (features > sae.jump_relu_threshold))

    def test_the_nominal_k_is_carried_but_never_used_to_select(self, checkpoint):
        """Carried only so a caller can SEE it is not the activation function —
        the control's sparsity must match the observed L0 instead."""
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        assert sae.nominal_top_k == 50
        assert sae.observed_l0(torch.randn(8, D_MODEL) * 3.0) != 50


class TestTheNormalisation:
    """⚠️ Trap 3: the SAE was fitted on activations rescaled to average norm
    sqrt(d_model). Feeding raw activations skips a ~5.9x scale factor."""

    def test_encoding_scales_into_the_fitted_space(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(4, D_MODEL)
        x = x / x.norm(dim=-1, keepdim=True) * sae.input_norm
        assert sae._normalise(x).norm(dim=-1).mean() == pytest.approx(D_MODEL**0.5, rel=1e-4)

    def test_decoding_undoes_it_so_the_round_trip_is_in_MODEL_units(self, checkpoint):
        """Otherwise the substitution hook writes activations ~5.9x too large and
        the KL gate measures a scale error rather than a dictionary."""
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(4, D_MODEL)
        assert torch.allclose(sae._denormalise(sae._normalise(x)), x, atol=1e-4)


class TestItFailsLoudOnADifferentDICTIONARY:
    """A checkpoint that differs in these is not unsupported-and-approximated —
    it is a different function, and approximating it would produce a confident
    reconstruction number for something nobody implemented."""

    def test_a_non_jumprelu_checkpoint_is_refused(self, checkpoint):
        with pytest.raises(ValueError, match="not 'jumprelu'"):
            load_llama_scope_sae(*checkpoint(act_fn="topk"), repo_id="test")

    def test_a_pre_encoder_decoder_bias_is_refused(self, checkpoint):
        with pytest.raises(ValueError, match="decoder bias before encoding"):
            load_llama_scope_sae(
                *checkpoint(apply_decoder_bias_to_pre_encoder=True), repo_id="test"
            )

    def test_a_different_normalisation_is_refused(self, checkpoint):
        with pytest.raises(ValueError, match="norm_activation"):
            load_llama_scope_sae(*checkpoint(norm_activation="token-wise"), repo_id="test")

    def test_a_checkpoint_missing_a_tensor_is_refused(self, tmp_path):
        w, h = tmp_path / "w.safetensors", tmp_path / "h.json"
        save_file({"encoder.weight": torch.randn(D_SAE, D_MODEL)}, str(w))
        h.write_text(json.dumps(HYPERPARAMS))
        with pytest.raises(ValueError, match="missing"):
            load_llama_scope_sae(w, h, repo_id="test")


class TestItSatisfiesTheProtocol:
    def test_the_round_trip_preserves_shape(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(6, D_MODEL)
        assert sae.decode(sae.encode(x)).shape == x.shape

    def test_it_records_what_it_was_trained_on(self, checkpoint):
        """`trained_on` vs the model it is read on IS the pre-gate's subject."""
        sae = load_llama_scope_sae(*checkpoint(), repo_id="fnlp/Llama3_1-8B-Base-L15R-8x")
        assert "Base" in sae.trained_on

    def test_it_is_stateless_so_both_pre_gate_arms_can_share_one(self, checkpoint):
        sae = load_llama_scope_sae(*checkpoint(), repo_id="test")
        x = torch.randn(4, D_MODEL)
        assert torch.allclose(sae.encode(x), sae.encode(x))
        with pytest.raises((AttributeError, TypeError)):
            sae.trained_on = "mutated"  # frozen
