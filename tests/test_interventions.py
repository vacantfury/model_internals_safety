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


class TestTheSteeringPositionConvention:
    """TODO 33, settled 2026-08-07 — and the item's own framing was corrected.

    It was filed reading CAA alone and said "ours contaminates the input, CAA's
    does not", implying ours was the outlier. Reading the CLOSER source showed
    otherwise: Arditi et al. — the paper this estimator was ported from — add at
    every position with no mask, exactly as we do. CAA is the one that differs,
    and their choice is tied to a multiple-choice format.

    The settled rule is per CONDITION: plain prompts every-position (Arditi, and
    what the causal gate runs on), encoded prompts instruction-end-onward (CAA),
    because only there could writing the direction into the prompt change the
    DECODE and confound the very thing AS-5 measures.
    """

    def test_the_default_steers_every_position(self, tiny_model):
        """Matching Arditi. Pinned so a later 'tidy-up' cannot silently make the
        default CAA-shaped and change what the causal gate measures."""
        direction = _unit(tiny_model.d_model)
        batch = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with add_direction(tiny_model, direction, layer=0, coefficient=3.0):
            steered = tiny_model.model(input_ids=batch).logits
        with add_direction(tiny_model, direction, layer=0, coefficient=3.0,
                           mask=torch.ones((2, 3), dtype=torch.bool)):
            explicit_all = tiny_model.model(input_ids=batch).logits
        assert torch.allclose(steered, explicit_all)

    def test_a_mask_of_all_False_is_not_the_same_as_no_mask(self, tiny_model):
        """The tri-state lesson one more time: `None` means EVERY position, and
        an all-False mask means NONE. If those collapsed, a caller that built an
        empty mask would silently get the strongest intervention available."""
        direction = _unit(tiny_model.d_model)
        batch = torch.tensor([[1, 2, 3]])
        with add_direction(tiny_model, direction, layer=0, coefficient=5.0,
                           mask=torch.zeros((1, 3), dtype=torch.bool)):
            none_steered = tiny_model.model(input_ids=batch).logits
        with add_direction(tiny_model, direction, layer=0, coefficient=5.0):
            all_steered = tiny_model.model(input_ids=batch).logits
        clean = tiny_model.model(input_ids=batch).logits
        assert torch.allclose(none_steered, clean)
        assert not torch.allclose(all_steered, clean)


class TestFromInstructionEnd:
    """The CAA-convention mask builder, built at settle time so phase 2 does not
    re-derive it from the paper."""

    def test_it_marks_a_suffix_of_every_row(self, tiny_model):
        from internals_safety.models.interventions import from_instruction_end

        mask = from_instruction_end(tiny_model, ["alpha", "beta gamma delta"])
        assert mask.dtype == torch.bool
        assert mask.shape[0] == 2
        for row in mask:
            marked = row.nonzero().flatten()
            assert len(marked) > 0, "no position steered — the mask would be a no-op"
            # Contiguous, and running to the end: CAA steers from a boundary ONWARD.
            assert marked[-1].item() == mask.shape[1] - 1
            assert torch.equal(marked, torch.arange(marked[0].item(), mask.shape[1]))

    def test_it_leaves_the_instruction_itself_unsteered(self, tiny_model):
        """The whole point — the prompt's own tokens must not be written into."""
        from internals_safety.models.interventions import from_instruction_end

        mask = from_instruction_end(tiny_model, ["a reasonably long instruction here"])
        assert not mask.all(), "steering every position is the convention this exists to avoid"

    def test_a_width_shorter_than_the_longest_prompt_is_refused(self, tiny_model):
        """A truncated mask steers the wrong span and still returns a number."""
        from internals_safety.models.interventions import from_instruction_end

        with pytest.raises(ValueError, match="shorter than the longest"):
            from_instruction_end(tiny_model, ["a much longer instruction than one token"], width=2)

    def test_padding_is_LEFT_so_the_final_position_is_always_real(self, tiny_model):
        """Matches the capture spine and build_inversion_batch. With ragged rows
        and RIGHT padding the last steered token would be padding in short rows."""
        from internals_safety.models.interventions import from_instruction_end

        mask = from_instruction_end(tiny_model, ["x", "a considerably longer instruction"])
        assert mask[:, -1].all()
