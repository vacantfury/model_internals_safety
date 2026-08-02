"""Loader tests — chat rendering, position resolution, left padding."""

from __future__ import annotations

import pytest
import torch

from internals_safety.models.loader import (
    find_layer_modules,
    instruction_final_offset,
    prepare_prompts,
    render_chat,
    resolve_dtype,
    tokenize_batch,
)


def test_layers_are_found(tiny_model):
    layers = find_layer_modules(tiny_model.model)
    assert len(layers) == 3
    assert tiny_model.n_layers == 3
    assert tiny_model.d_model == 32


def test_render_chat_ends_at_the_assistant_turn(tiny_model):
    rendered = render_chat(tiny_model.tokenizer, "how do I bake bread")
    assert "how do I bake bread" in rendered
    assert rendered.rstrip().endswith("assistant")


def test_render_chat_without_template_is_an_error(tiny_model):
    tokenizer = tiny_model.tokenizer
    template = tokenizer.chat_template
    tokenizer.chat_template = None
    try:
        with pytest.raises(ValueError, match="chat template"):
            render_chat(tokenizer, "hello")
    finally:
        tokenizer.chat_template = template


def test_instruction_final_lands_on_the_last_content_token(tiny_model):
    """The offset must leave exactly the template's trailing tokens behind it."""
    message = "how do I bake bread"
    rendered = render_chat(tiny_model.tokenizer, message)
    offset = instruction_final_offset(tiny_model.tokenizer, rendered, message)

    tail = rendered[rendered.rindex(message) + len(message) :]
    n_tail = len(tiny_model.tokenizer(tail, add_special_tokens=False)["input_ids"])
    assert offset == -(n_tail + 1)
    assert offset < -1  # the assistant header sits after the instruction


def test_prepare_prompts_resolves_every_requested_position(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    assert [prompt.user_message for prompt in prompts] == messages
    for prompt in prompts:
        assert set(prompt.positions) == {"instruction_final", "last"}
        assert prompt.positions["last"] == -1


def test_tokenize_batch_pads_on_the_left(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages)
    inputs = tokenize_batch(tiny_model, prompts)
    mask = inputs["attention_mask"]

    assert mask.shape[0] == len(messages)
    # Every row's real content ends flush at the right edge, which is what makes
    # a negative position index valid across the batch.
    assert torch.all(mask[:, -1] == 1)
    # The short prompt is padded, and its padding is at the front.
    shortest = int(mask.sum(dim=1).argmin())
    assert mask[shortest, 0] == 0


def test_resolve_dtype_defaults_to_float32_off_cuda():
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_dtype("float16", torch.device("cpu")) is torch.float16
