"""Capture tests.

Two of these are load-bearing for the science rather than merely for the code:

* `test_resid_pre_matches_hidden_states` / `test_resid_post_matches_hidden_states`
  pin what the hook points actually are. Every direction, probe and
  intervention in the paper is defined at a named site; if the hooks drifted
  one layer or one side, every result would still *compute* and be wrong.
* `test_capture_is_invariant_to_batch_size` pins that an activation depends on
  its prompt and nothing else. A padding or position bug would make a measured
  quantity depend on batch composition.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.models.capture import ActivationBatch, capture_activations, resolve_layers
from internals_safety.models.loader import prepare_prompts, tokenize_batch


def _reference_hidden_states(tiny_model, prompts):
    inputs = tokenize_batch(tiny_model, prompts)
    with torch.inference_mode():
        output = tiny_model.model(**inputs, use_cache=False, output_hidden_states=True)
    return inputs["input_ids"].shape[1], output.hidden_states


def _assert_matches(batch, prompts, seq_len, hidden_states, layer_shift: int):
    for layer_position, layer in enumerate(batch.layers):
        for position_index, position in enumerate(batch.positions):
            for prompt_index, prompt in enumerate(prompts):
                token_index = seq_len + prompt.positions[position]
                expected = hidden_states[layer + layer_shift][prompt_index, token_index]
                torch.testing.assert_close(
                    batch.tensor[prompt_index, layer_position, position_index],
                    expected.to("cpu", torch.float32),
                    atol=1e-5,
                    rtol=1e-4,
                )


def test_resid_pre_matches_hidden_states(tiny_model, messages):
    """hidden_states[i] is the input to layer i — that is what resid_pre means."""
    prompts = prepare_prompts(tiny_model, messages)
    seq_len, hidden_states = _reference_hidden_states(tiny_model, prompts)
    batch = capture_activations(tiny_model, prompts, site="resid_pre", batch_size=len(prompts))
    _assert_matches(batch, prompts, seq_len, hidden_states, layer_shift=0)


def test_resid_post_matches_hidden_states(tiny_model, messages):
    """hidden_states[i + 1] is the output of layer i — that is resid_post.

    Final layer excluded on purpose: see the next test.
    """
    prompts = prepare_prompts(tiny_model, messages)
    seq_len, hidden_states = _reference_hidden_states(tiny_model, prompts)
    non_final = list(range(tiny_model.n_layers - 1))
    batch = capture_activations(
        tiny_model, prompts, layers=non_final, site="resid_post", batch_size=len(prompts)
    )
    _assert_matches(batch, prompts, seq_len, hidden_states, layer_shift=1)


def test_final_layer_resid_post_is_pre_norm(tiny_model, messages):
    """`hidden_states[-1]` is post-final-norm; our capture is not, deliberately.

    Pinned rather than left to a docstring because it is a live footgun: a
    reader who validates the last layer against `output_hidden_states` will see
    a mismatch and conclude the hooks are wrong. Consistency across layers is
    what the probes need, so we keep the raw residual stream at every layer and
    apply no norm anywhere.
    """
    prompts = prepare_prompts(tiny_model, messages)
    seq_len, hidden_states = _reference_hidden_states(tiny_model, prompts)
    last = tiny_model.n_layers - 1
    batch = capture_activations(
        tiny_model, prompts, layers=[last], site="resid_post", batch_size=len(prompts)
    )

    captured = batch.select(layer=last, position="last")
    reference = hidden_states[-1][:, seq_len - 1, :].to("cpu", torch.float32)
    assert not torch.allclose(captured, reference, atol=1e-5)
    normed = tiny_model.model.model.norm(captured.to(tiny_model.dtype)).to(torch.float32)
    torch.testing.assert_close(normed, reference, atol=1e-5, rtol=1e-4)


def test_capture_is_invariant_to_batch_size(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    one_at_a_time = capture_activations(tiny_model, prompts, batch_size=1)
    all_together = capture_activations(tiny_model, prompts, batch_size=len(prompts))
    torch.testing.assert_close(one_at_a_time.tensor, all_together.tensor, atol=1e-5, rtol=1e-4)


def test_capture_shape_and_metadata(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    batch = capture_activations(tiny_model, prompts, layers=[0, 2], positions=["last"])
    assert batch.tensor.shape == (len(messages), 2, 1, tiny_model.d_model)
    assert batch.tensor.dtype is torch.float32
    assert batch.layers == [0, 2]
    assert batch.positions == ["last"]
    assert batch.site == "resid_pre"
    assert batch.model_name == "tiny_test_model"
    assert batch.user_messages == messages


def test_select_returns_one_cell(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    batch = capture_activations(tiny_model, prompts)
    cell = batch.select(layer=1, position="last")
    assert cell.shape == (len(messages), tiny_model.d_model)
    torch.testing.assert_close(cell, batch.tensor[:, 1, batch.positions.index("last"), :])


def test_positions_differ_between_readout_sites(tiny_model, messages):
    """instruction_final and last must not silently collapse to the same vector."""
    prompts = prepare_prompts(tiny_model, messages)
    batch = capture_activations(tiny_model, prompts)
    instruction = batch.select(layer=2, position="instruction_final")
    final = batch.select(layer=2, position="last")
    assert not torch.allclose(instruction, final)


def test_save_and_load_roundtrip(tiny_model, messages, tmp_path):
    prompts = prepare_prompts(tiny_model, messages)
    batch = capture_activations(tiny_model, prompts)
    path = batch.save(tmp_path / "activations" / "batch.pt")
    reloaded = ActivationBatch.load(path)

    torch.testing.assert_close(reloaded.tensor, batch.tensor)
    assert reloaded.layers == batch.layers
    assert reloaded.positions == batch.positions
    assert reloaded.site == batch.site
    assert reloaded.user_messages == batch.user_messages


def test_resolve_layers(tiny_model):
    assert resolve_layers(tiny_model, None) == [0, 1, 2]
    assert resolve_layers(tiny_model, [0, 2]) == [0, 2]
    with pytest.raises(ValueError, match="outside"):
        resolve_layers(tiny_model, [0, 7])


def test_unprepared_position_is_rejected(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    with pytest.raises(ValueError, match="instruction_final"):
        capture_activations(tiny_model, prompts, positions=["instruction_final"])


def test_hooks_are_removed_after_capture(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    capture_activations(tiny_model, prompts)
    for layer in tiny_model.layers:
        assert not layer._forward_pre_hooks
        assert not layer._forward_hooks
