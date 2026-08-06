"""I2 — processing-trajectory features.

Geometry, invariances and failure modes on constructed tensors, where the right
answer is known by hand. Whether trajectory features BEAT single-cell probes on
our rungs is the validation gate (build plan §3.2), needs the cached band-run
activations, and is not a unit test.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.measurements.trajectory import (
    FEATURE_BLOCKS,
    feature_matrix,
    feature_names,
    trajectory,
)
from internals_safety.models.capture import ActivationBatch


def batch_from(states: torch.Tensor, position: str = "last") -> ActivationBatch:
    """[n_prompts, n_layers, d_model] -> a one-position ActivationBatch."""
    n_prompts, n_layers, _ = states.shape
    return ActivationBatch(
        tensor=states.unsqueeze(2),
        layers=list(range(n_layers)),
        positions=[position],
        site="resid_pre",
        model_name="constructed",
        user_messages=[f"p{i}" for i in range(n_prompts)],
    )


def test_norms_are_read_per_layer():
    states = torch.zeros(1, 3, 4)
    states[0, 0, 0] = 3.0
    states[0, 1, 0] = 5.0
    states[0, 2, 0] = 1.0
    traj = trajectory(batch_from(states), "last")
    torch.testing.assert_close(traj.norms, torch.tensor([[3.0, 5.0, 1.0]]))


def test_step_norms_measure_movement_between_consecutive_layers():
    states = torch.zeros(1, 3, 2)
    states[0, 1, 0] = 4.0            # step of 4 from layer 0 to 1
    states[0, 2] = torch.tensor([4.0, 3.0])  # step of 3 from layer 1 to 2
    traj = trajectory(batch_from(states), "last")
    torch.testing.assert_close(traj.step_norms, torch.tensor([[4.0, 3.0]]))


def test_a_straight_trajectory_has_cosine_one():
    """Travelling in a constant direction: consecutive steps agree exactly."""
    states = torch.stack([torch.tensor([float(i), 0.0]) for i in range(4)]).unsqueeze(0)
    traj = trajectory(batch_from(states), "last")
    torch.testing.assert_close(traj.step_cosines, torch.ones(1, 2))


def test_a_reversing_trajectory_has_cosine_minus_one():
    states = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    traj = trajectory(batch_from(states), "last")
    torch.testing.assert_close(traj.step_cosines, -torch.ones(1, 1))


def test_a_right_angle_turn_has_cosine_zero():
    states = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    traj = trajectory(batch_from(states), "last")
    torch.testing.assert_close(traj.step_cosines, torch.zeros(1, 1))


def test_a_zero_step_gives_cosine_zero_not_nan():
    """A stalled layer must not poison every downstream statistic with NaN.

    0.0 is the honest encoding: there is no direction to agree with.
    """
    states = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    traj = trajectory(batch_from(states), "last")
    assert torch.isfinite(traj.step_cosines).all()
    torch.testing.assert_close(traj.step_cosines, torch.zeros(1, 1))


def test_features_are_rotation_invariant():
    """Norms, step norms and turning angles are geometric — a rotation of the
    whole residual space must not change any of them. If this fails, a feature
    is reading basis coordinates rather than geometry."""
    torch.manual_seed(0)
    states = torch.randn(3, 5, 6)
    rotation, _ = torch.linalg.qr(torch.randn(6, 6))

    plain = trajectory(batch_from(states), "last")
    rotated = trajectory(batch_from(states @ rotation), "last")

    torch.testing.assert_close(plain.norms, rotated.norms, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(plain.step_norms, rotated.step_norms, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(plain.step_cosines, rotated.step_cosines, rtol=1e-4, atol=1e-4)


def test_peak_step_layer_finds_the_largest_move():
    """The trajectory analogue of the layer-index diagnostic: where the
    representation changes most is where the work is being done."""
    states = torch.zeros(1, 4, 2)
    states[0, 1, 0] = 1.0
    states[0, 2, 0] = 9.0   # the big move is layer 1 -> 2, i.e. step index 1
    states[0, 3, 0] = 9.5
    traj = trajectory(batch_from(states), "last")
    assert int(traj.peak_step_layer()[0]) == 1


def test_feature_matrix_width_matches_the_blocks():
    torch.manual_seed(1)
    traj = trajectory(batch_from(torch.randn(2, 6, 4)), "last")
    assert feature_matrix(traj).shape == (2, 6 + 5 + 4)
    assert feature_matrix(traj, blocks=("norms",)).shape == (2, 6)


def test_feature_names_line_up_with_the_matrix():
    """Without labels a trajectory probe's coefficients are uninterpretable, and
    WHICH layer carries the signal is the validity check."""
    torch.manual_seed(2)
    traj = trajectory(batch_from(torch.randn(2, 5, 3)), "last")
    for blocks in [FEATURE_BLOCKS, ("norms",), ("step_norms", "step_cosines")]:
        assert len(feature_names(traj, blocks)) == feature_matrix(traj, blocks).shape[1]


def test_feature_names_record_the_layer_transition():
    torch.manual_seed(3)
    traj = trajectory(batch_from(torch.randn(1, 4, 3)), "last")
    names = feature_names(traj, ("step_norms",))
    assert names == ["step@L0->L1", "step@L1->L2", "step@L2->L3"]


def test_single_block_selection_is_available_for_ablation():
    """If the whole signal is carried by norms that is a much weaker result than
    one carried by turning angles, and fitting each block alone is the only way
    to tell."""
    torch.manual_seed(4)
    traj = trajectory(batch_from(torch.randn(3, 5, 4)), "last")
    widths = {block: feature_matrix(traj, (block,)).shape[1] for block in FEATURE_BLOCKS}
    assert widths == {"norms": 5, "step_norms": 4, "step_cosines": 3}


def test_an_unknown_block_is_rejected():
    torch.manual_seed(5)
    traj = trajectory(batch_from(torch.randn(1, 4, 2)), "last")
    with pytest.raises(ValueError, match="unknown feature blocks"):
        feature_matrix(traj, ("curvature",))


def test_no_blocks_is_rejected():
    torch.manual_seed(6)
    traj = trajectory(batch_from(torch.randn(1, 4, 2)), "last")
    with pytest.raises(ValueError, match="no feature blocks"):
        feature_matrix(traj, ())


def test_too_few_layers_is_rejected_rather_than_returning_empty_features():
    """An empty tensor would be silently treated downstream as a feature."""
    with pytest.raises(ValueError, match="at least 3 captured layers"):
        trajectory(batch_from(torch.randn(2, 2, 4)), "last")


def test_an_uncaptured_position_is_rejected():
    with pytest.raises(ValueError, match="not captured"):
        trajectory(batch_from(torch.randn(2, 4, 3)), "instruction_final")


def test_trajectory_works_on_a_real_capture(tiny_model, messages):
    """End to end against the capture layer, so the shapes cannot drift apart."""
    from internals_safety.models.capture import capture_activations
    from internals_safety.models.loader import prepare_prompts

    prompts = prepare_prompts(tiny_model, messages, positions=["instruction_final", "last"])
    captured = capture_activations(tiny_model, prompts, layers="all", positions=["instruction_final", "last"])
    traj = trajectory(captured, "instruction_final")

    assert traj.n_prompts == len(messages)
    assert traj.norms.shape == (len(messages), tiny_model.n_layers)
    assert traj.step_norms.shape == (len(messages), tiny_model.n_layers - 1)
    assert traj.step_cosines.shape == (len(messages), tiny_model.n_layers - 2)
    assert torch.isfinite(feature_matrix(traj)).all()
