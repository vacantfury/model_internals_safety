"""Directional ablation and activation addition — the causal write operations."""

from __future__ import annotations

import pytest
import torch

from internals_safety.models.capture import capture_activations
from internals_safety.models.interventions import (
    ablate_direction,
    add_direction,
    project_out,
)
from internals_safety.models.loader import prepare_prompts, tokenize_batch


def _unit(d_model: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(d_model, generator=generator)
    return vector / vector.norm()


def _logits(loaded, prompts):
    with torch.inference_mode():
        return loaded.model(**tokenize_batch(loaded, prompts), use_cache=False).logits


def test_projection_removes_the_component_along_the_direction():
    direction = _unit(8)
    hidden = torch.randn(3, 5, 8)
    residual = project_out(hidden, direction)
    torch.testing.assert_close(
        residual @ direction, torch.zeros(3, 5), rtol=1e-5, atol=1e-5
    )


def test_projection_is_idempotent():
    """Ablating twice must equal ablating once — otherwise repeated application
    across layers would compound into something that is not a projection."""
    direction = _unit(8, seed=3)
    hidden = torch.randn(2, 4, 8)
    once = project_out(hidden, direction)
    torch.testing.assert_close(once, project_out(once, direction), rtol=1e-5, atol=1e-5)


def test_projection_leaves_orthogonal_content_untouched():
    direction = torch.zeros(4)
    direction[0] = 1.0
    hidden = torch.zeros(1, 1, 4)
    hidden[0, 0, 1:] = torch.tensor([2.0, 3.0, 4.0])
    torch.testing.assert_close(project_out(hidden, direction), hidden)


def test_projection_rejects_a_non_unit_direction():
    with pytest.raises(ValueError, match="unit-norm"):
        project_out(torch.randn(1, 1, 4), torch.tensor([3.0, 0.0, 0.0, 0.0]))


def test_projection_rejects_a_zero_direction():
    with pytest.raises(ValueError, match="zero direction"):
        project_out(torch.randn(1, 1, 4), torch.zeros(4))


def test_ablation_changes_the_logits(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    baseline = _logits(tiny_model, prompts)
    with ablate_direction(tiny_model, _unit(tiny_model.d_model)):
        ablated = _logits(tiny_model, prompts)
    assert not torch.allclose(baseline, ablated, rtol=1e-3, atol=1e-3)


def test_ablation_removes_the_direction_from_every_captured_layer(tiny_model, messages):
    """The necessity claim is about all layers, so the projection must hold at
    all of them — a partial ablation would test something much weaker."""
    direction = _unit(tiny_model.d_model, seed=5)
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    with ablate_direction(tiny_model, direction):
        captured = capture_activations(
            tiny_model, prompts, layers="all", positions=["last"], site="resid_post"
        )
    for layer in captured.layers:
        projections = captured.select(layer, "last") @ direction
        torch.testing.assert_close(
            projections, torch.zeros_like(projections), rtol=1e-4, atol=1e-4
        )


def test_ablation_can_be_restricted_to_named_layers(tiny_model, messages):
    direction = _unit(tiny_model.d_model, seed=7)
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    with ablate_direction(tiny_model, direction, layers=[0]):
        captured = capture_activations(
            tiny_model, prompts, layers="all", positions=["last"], site="resid_post"
        )
    cleared = captured.select(0, "last") @ direction
    torch.testing.assert_close(cleared, torch.zeros_like(cleared), rtol=1e-4, atol=1e-4)
    # A later layer re-introduces a component, so the restriction is real.
    assert captured.select(tiny_model.n_layers - 1, "last").matmul(direction).abs().max() > 1e-4


def test_ablation_hooks_are_removed_on_exit(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    baseline = _logits(tiny_model, prompts)
    with ablate_direction(tiny_model, _unit(tiny_model.d_model)):
        pass
    torch.testing.assert_close(baseline, _logits(tiny_model, prompts), rtol=1e-5, atol=1e-5)


def test_ablation_rejects_an_out_of_range_layer(tiny_model):
    with pytest.raises(ValueError, match="outside"):
        with ablate_direction(tiny_model, _unit(tiny_model.d_model), layers=[0, 99]):
            pass


def test_addition_with_zero_coefficient_is_a_no_op(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    baseline = _logits(tiny_model, prompts)
    with add_direction(tiny_model, _unit(tiny_model.d_model), layer=1, coefficient=0.0):
        added = _logits(tiny_model, prompts)
    torch.testing.assert_close(baseline, added, rtol=1e-5, atol=1e-5)


def test_addition_shifts_the_residual_stream_by_the_coefficient(tiny_model, messages):
    direction = _unit(tiny_model.d_model, seed=11)
    prompts = prepare_prompts(tiny_model, messages, positions=["last"])
    before = capture_activations(
        tiny_model, prompts, layers=[1], positions=["last"], site="resid_post"
    )
    with add_direction(tiny_model, direction, layer=1, coefficient=2.5):
        after = capture_activations(
            tiny_model, prompts, layers=[1], positions=["last"], site="resid_post"
        )
    delta = after.select(1, "last") - before.select(1, "last")
    torch.testing.assert_close(
        delta @ direction, torch.full((len(messages),), 2.5), rtol=1e-3, atol=1e-3
    )


def test_addition_rejects_a_non_unit_direction(tiny_model):
    with pytest.raises(ValueError, match="unit-norm"):
        with add_direction(tiny_model, torch.zeros(tiny_model.d_model), layer=0, coefficient=1.0):
            pass


def test_addition_rejects_an_out_of_range_layer(tiny_model):
    with pytest.raises(ValueError, match="outside"):
        with add_direction(tiny_model, _unit(tiny_model.d_model), layer=99, coefficient=1.0):
            pass
