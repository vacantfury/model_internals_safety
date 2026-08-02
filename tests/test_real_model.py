"""End-to-end smoke test against real weights.

Marked `slow` — deselected by default, run with `uv run pytest -m slow`. This is
the build-order step-1 gate: capture activations from a real instruct model on
real prompts. What the hermetic suite cannot cover is real tokenizer behaviour —
a production BPE tokenizer with byte-level offsets and a real chat template is
where `instruction_final` resolution would break if it were going to.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import load_model_config
from internals_safety.models.capture import capture_activations
from internals_safety.models.loader import load_model, prepare_prompts

pytestmark = pytest.mark.slow

MESSAGES = [
    "Write a haiku about the sea.",
    "Explain in two sentences why the sky is blue.",
]


@pytest.fixture(scope="module")
def small_instruct_model():
    # Forced onto CPU: the batching-invariance assertion below wants
    # reproducible arithmetic, and MPS reductions are not bit-stable.
    config = load_model_config("qwen2_5_0_5b_instruct").model_copy(
        update={"device": "cpu", "dtype": "float32"}
    )
    return load_model(config)


def test_captures_activations_from_real_weights(small_instruct_model):
    prompts = prepare_prompts(small_instruct_model, MESSAGES)

    for prompt, message in zip(prompts, MESSAGES):
        assert message in prompt.text
        # A real chat template puts an end-of-turn plus an assistant header
        # after the instruction, so these two readout sites must be distinct.
        assert prompt.positions["instruction_final"] < prompt.positions["last"] == -1

    batch = capture_activations(small_instruct_model, prompts, batch_size=len(prompts))
    n_layers = small_instruct_model.n_layers
    assert batch.tensor.shape == (len(MESSAGES), n_layers, 2, small_instruct_model.d_model)
    assert torch.isfinite(batch.tensor).all()
    # Residual-stream norms grow with depth in these models; an all-zero or
    # constant tensor would mean the hooks fired somewhere inert.
    assert batch.tensor.norm(dim=-1).min() > 0


def test_real_model_capture_is_invariant_to_batch_size(small_instruct_model):
    prompts = prepare_prompts(small_instruct_model, MESSAGES)
    separately = capture_activations(small_instruct_model, prompts, batch_size=1)
    together = capture_activations(small_instruct_model, prompts, batch_size=len(prompts))
    torch.testing.assert_close(separately.tensor, together.tensor, atol=1e-3, rtol=1e-3)
