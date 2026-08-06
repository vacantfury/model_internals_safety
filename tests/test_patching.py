"""The write side of the capture stack.

These pin the mechanism, not the semantics: that a patched state lands at exactly
the site and position it was captured from, that patching is confined to that
one cell, and that the failure modes fail loudly. Whether a patched state
*decodes* to anything needs real weights and lives in the slow suite — a
randomly-initialised 3-layer model has nothing to decode.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.models.capture import capture_activations
from internals_safety.models.loader import prepare_prompts
from internals_safety.models.patching import _replace_at, patch_residual


def _logits(loaded, prompts):
    from internals_safety.models.loader import tokenize_batch

    with torch.inference_mode():
        return loaded.model(**tokenize_batch(loaded, prompts), use_cache=False).logits


def test_patching_a_state_back_into_its_own_pass_is_a_no_op(tiny_model, messages):
    """The round trip that makes read and write the same site.

    Capture at (layer, position), patch that exact tensor back in, and nothing
    may change. If this fails, the hook points have drifted apart and every
    intervention would be writing somewhere the diagnosis never read.
    """
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    baseline = _logits(tiny_model, prompts)

    captured = capture_activations(tiny_model, prompts, layers=[1], positions=["last"])
    with patch_residual(tiny_model, layer=1, position=-1, vectors=captured.select(1, "last")):
        patched = _logits(tiny_model, prompts)

    torch.testing.assert_close(baseline, patched, rtol=1e-4, atol=1e-4)


def test_patching_a_different_state_changes_the_logits(tiny_model, messages):
    """The complement — otherwise the no-op test above would pass vacuously."""
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    baseline = _logits(tiny_model, prompts)

    captured = capture_activations(tiny_model, prompts, layers=[1], positions=["last"])
    scrambled = captured.select(1, "last").flip(0)
    with patch_residual(tiny_model, layer=1, position=-1, vectors=scrambled):
        patched = _logits(tiny_model, prompts)

    assert not torch.allclose(baseline, patched, rtol=1e-3, atol=1e-3)


def test_patching_leaves_earlier_positions_untouched(tiny_model, messages):
    """A patch at the last position must not disturb instruction_final."""
    prompts = prepare_prompts(tiny_model, messages, positions=["instruction_final", "last"])
    before = capture_activations(tiny_model, prompts, layers=[2], positions=["instruction_final"])

    noise = torch.randn_like(before.select(2, "instruction_final"))
    with patch_residual(tiny_model, layer=0, position=-1, vectors=noise):
        after = capture_activations(tiny_model, prompts, layers=[2], positions=["instruction_final"])

    # Patching layer 0's last position cannot reach back to an earlier position
    # at layer 2: attention is causal, so nothing at the end informs the middle.
    torch.testing.assert_close(
        before.select(2, "instruction_final"), after.select(2, "instruction_final"),
        rtol=1e-4, atol=1e-4,
    )


def test_hooks_are_removed_after_the_context_exits(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    noise = torch.randn(len(messages), tiny_model.d_model)

    with patch_residual(tiny_model, layer=1, position=-1, vectors=noise):
        pass
    after = _logits(tiny_model, prompts)
    baseline = _logits(tiny_model, prompts)

    torch.testing.assert_close(after, baseline, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "position, vectors_shape, expected",
    [
        (0, (3, 32), "negative offset"),
        (-1, (2, 32), "batch mismatch"),
        (-1, (3, 16), "d_model mismatch"),
    ],
)
def test_replace_at_rejects_mismatches(position, vectors_shape, expected):
    hidden = torch.zeros(3, 5, 32)
    with pytest.raises(ValueError, match=expected):
        _replace_at(hidden, position, torch.zeros(*vectors_shape))


def test_replace_at_rejects_a_position_before_the_sequence_start():
    with pytest.raises(ValueError, match="before the start"):
        _replace_at(torch.zeros(1, 2, 4), -5, torch.zeros(1, 4))


def test_patch_residual_rejects_an_out_of_range_layer(tiny_model):
    with pytest.raises(ValueError, match="outside"):
        with patch_residual(tiny_model, layer=99, position=-1, vectors=torch.zeros(1, 32)):
            pass


def test_patch_residual_refuses_resid_post_at_the_final_layer(tiny_model):
    """Fails closed rather than patching through the final norm a second time.

    Patchscopes handles this with `skip_final_ln`; we do not implement that path,
    so the honest behaviour is to refuse rather than to silently return a number
    computed through an extra normalisation.
    """
    last = tiny_model.n_layers - 1
    with pytest.raises(ValueError, match="final-layer-norm"):
        with patch_residual(
            tiny_model, layer=last, position=-1, vectors=torch.zeros(1, 32), site="resid_post"
        ):
            pass


def test_patch_residual_rejects_a_non_2d_vector_block(tiny_model):
    with pytest.raises(ValueError, match=r"\[n_prompts, d_model\]"):
        with patch_residual(tiny_model, layer=0, position=-1, vectors=torch.zeros(3, 2, 32)):
            pass
