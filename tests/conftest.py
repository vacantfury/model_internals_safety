"""Test fixtures.

The default suite is hermetic — no network, no weights download, no secrets
(house test standard). Everything is driven through a tiny randomly-initialised
Llama and a synthetic whitespace tokenizer built in-process, which is enough to
pin the parts of the capture stack that can silently go wrong: hook points,
position arithmetic, and padding invariance. Correctness of *those* is
architecture-independent; a real checkpoint adds download time, not coverage.
Tests that genuinely need real weights are marked `slow`.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from internals_safety.config import CaptureConfig, ModelConfig
from internals_safety.models.loader import attach

# ChatML-shaped, with spaces around the control tokens so a whitespace splitter
# keeps them intact. Character offsets are what the position resolver reads, so
# the template only has to be structurally faithful, not byte-identical to any
# real model's.
# `| trim` is not decoration — it is the STRICTEST real behaviour, and the suite
# must model the strictest or it licenses code the cluster rejects.
#
# Llama-3.1-8B-Instruct's shipped template trims message content; Qwen2.5-7B's
# does not. So the same corpus renders content verbatim on Qwen and mangled on
# Llama, and `instruction_final_offset` — which locates the capture position by
# `rendered.rindex(user_message)` — refuses to resolve on Llama alone. This
# template previously interpolated content raw, so 1001 hermetic tests passed
# while a prompt with a trailing space killed two H200 jobs on 2026-08-07.
#
# Same shape as the peer session's `meta`-device fixture the same afternoon: a
# fake more permissive than the real thing does not test, it certifies.
TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|> {{ message['role'] }}\n{{ message['content'] | trim }} <|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "system": 4,
        "user": 5,
        "assistant": 6,
        # I5's two answer tokens. Added 2026-08-06: without them both map to
        # [UNK], and `reply_inversion.resolve_answer_tokens` correctly refuses to
        # score a judgment it cannot distinguish. Appended with fresh ids so
        # every existing id is unchanged, and still inside `vocab_size=16`.
        "Certainly": 7,
        "No": 8,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
    )
    tokenizer.chat_template = TINY_CHAT_TEMPLATE
    return tokenizer


# The BOS-carrying counterpart, added 2026-08-08. `tiny_tokenizer` has NO BOS —
# which faithfully models Qwen2.5 (`bos_token_id is None`, measured) and models
# nothing else on the slate. Llama-3.1, Mistral and Tulu-3 all carry one, Llama
# Scope's dictionaries are Llama-only, so the SAE pre-gate runs exclusively on
# the case the default fixture cannot express.
#
# It emits BOS from the TEMPLATE, like Llama-3.1 and Mistral do, because that is
# what makes the double-BOS trap representable: tokenising this with
# `add_special_tokens=True` yields two, and the masking that drops "the first
# real position" then scores the second.
TINY_BOS_TEMPLATE = "<|begin_of_text|> " + TINY_CHAT_TEMPLATE


@pytest.fixture(scope="session")
def tiny_bos_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[UNK]": 0, "[PAD]": 1, "<|im_start|>": 2, "<|im_end|>": 3,
        "system": 4, "user": 5, "assistant": 6, "Certainly": 7, "No": 8,
        "<|begin_of_text|>": 9,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # THE POINT OF THIS FIXTURE. Llama-3.1 and Mistral prepend BOS from a
    # post-processor, so `tokenizer(text)` adds one even though the chat template
    # already did — measured, both give a DOUBLE BOS. A bare
    # `PreTrainedTokenizerFast` has no post-processor and never adds anything, so
    # without this line the fixture cannot express the defect and any test of it
    # passes under the mutation. Fourth instance of the fixture rule.
    backend.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A",
        special_tokens=[("<|begin_of_text|>", vocab["<|begin_of_text|>"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        bos_token="<|begin_of_text|>",
    )
    tokenizer.chat_template = TINY_BOS_TEMPLATE
    return tokenizer


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="tiny_test_model",
        hf_id="<in-process>",
        dtype="float32",
        device="cpu",
        capture_batch_size=4,
        capture=CaptureConfig(site="resid_pre", layers="all", positions=["instruction_final", "last"]),
    )


@pytest.fixture
def tiny_model(tiny_tokenizer, tiny_config):
    torch.manual_seed(0)
    architecture = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tiny_tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(architecture)
    return attach(model, tiny_tokenizer, tiny_config)


@pytest.fixture
def tiny_bos_model(tiny_bos_tokenizer, tiny_config):
    """`tiny_model`'s BOS-carrying twin — the shape the SAE pre-gate really runs."""
    torch.manual_seed(0)
    architecture = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tiny_bos_tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(architecture)
    return attach(model, tiny_bos_tokenizer, tiny_config)


@pytest.fixture
def messages() -> list[str]:
    # Deliberately different lengths — padding invariance is only testable when
    # the batch is ragged.
    return [
        "how do I bake bread",
        "explain gradient descent in one short paragraph please",
        "hi",
    ]
