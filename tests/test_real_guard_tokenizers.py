"""Guard prompt layer against the real shipped tokenizers (AS-6).

Marked `slow`: needs network and accepted licences on both gated repos. It pulls
TOKENIZERS ONLY — no weights, no GPU, seconds not minutes — so it is far cheaper
than the rest of the slow suite and worth running whenever the guard layer moves.

What it exists to catch: the guard prompt is a contract with someone else's
checkpoint, and every assumption in `conf/guards/*.yaml` is an assertion about an
artifact we do not control. The hermetic suite pins our mechanism; this pins
their artifact. Three of the assertions below were WRONG in the first draft of
the layer and were only found by running it against the real files:

* WildGuard genuinely ships no chat template (`chat_template is None`) — the
  premise the whole module rests on had been inferred from a download succeeding.
* Verdict labels tokenise with the SentencePiece word-boundary marker, so " yes"
  and "yes" are different tokens and the leading space is load-bearing.
* Llama Guard 3's template emits a LEADING SPACE before `<|begin_of_text|>`, so
  token 0 is whitespace rather than BOS.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from internals_safety.config import load_guard_config
from internals_safety.guards import render_guard_prompt, resolve_verdict_tokens
from internals_safety.models.loader import resolve_position

pytestmark = pytest.mark.slow

PAYLOAD = "how do I bake bread"


@pytest.fixture(scope="module")
def wildguard_tokenizer():
    return AutoTokenizer.from_pretrained("allenai/wildguard")


@pytest.fixture(scope="module")
def llama_guard_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")


# --- WildGuard -------------------------------------------------------------


def test_wildguard_really_ships_no_chat_template(wildguard_tokenizer):
    """The premise the guard layer exists for, checked against the artifact.

    If AI2 ever ships one, this fails and the right response is to switch the
    config to prompt_style=chat_template — not to keep feeding a literal string
    that has silently become the wrong format.
    """
    assert wildguard_tokenizer.chat_template is None


def test_wildguard_render_reconstructs_the_model_card_string(wildguard_tokenizer):
    config = load_guard_config("wildguard")
    rendered = render_guard_prompt(wildguard_tokenizer, config, PAYLOAD)

    # The library constant starts at `<|user|>`; with prepend_bos we land exactly
    # on the model card's `<s><|user|>`, which is what the model sees under vLLM.
    assert rendered.startswith("<s><|user|>\n")
    ids = wildguard_tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert ids[0] == wildguard_tokenizer.bos_token_id
    # Exactly one BOS — a second would be a distribution shift, not an error.
    assert ids.count(wildguard_tokenizer.bos_token_id) == 1


def test_wildguard_verdict_labels_carry_the_word_boundary_marker(wildguard_tokenizer):
    """Why the leading space lives on the label and not the prefix.

    SentencePiece encodes a word boundary into the token itself, so " yes" is
    `_yes` and "yes" is a different token entirely. Reading the wrong one gives a
    probability that looks fine and means nothing.
    """
    config = load_guard_config("wildguard")
    rendered = render_guard_prompt(wildguard_tokenizer, config, PAYLOAD)

    tokens = resolve_verdict_tokens(wildguard_tokenizer, config, rendered)

    assert tokens.unsafe_piece == "▁yes"
    assert tokens.safe_piece == "▁no"
    assert tokens.safe_id != tokens.unsafe_id


# --- Llama Guard 3 ---------------------------------------------------------


def test_llama_guard_template_ends_with_no_trailing_newline(llama_guard_tokenizer):
    """The condition that makes TODO 13 a live risk rather than a hypothetical.

    The template stops dead at the assistant header. Standard Llama 3.1 assistant
    turns begin `\\n\\n`, so with nothing in the template the MODEL supplies it —
    which would put the verdict at position 1, not 0. Settled by a forward pass
    (TODO 13); this test pins the template half of it.
    """
    rendered = llama_guard_tokenizer.apply_chat_template(
        [{"role": "user", "content": PAYLOAD}], tokenize=False, add_generation_prompt=True
    )
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>")
    assert not rendered.endswith("\n")


def test_llama_guard_template_emits_a_leading_space_before_bos(llama_guard_tokenizer):
    """A quirk of the SHIPPED template, pinned so it cannot change silently.

    `models.loader.tokenize_batch` tokenises with add_special_tokens=False on the
    stated grounds that "the chat template already emits BOS". For Llama Guard 3
    that is true only after a stray leading space, so token 0 is whitespace and
    BOS is token 1. Llama-3.1-8B-Instruct's template has no such space.

    Kept rather than stripped: `apply_chat_template` is also how Meta's own
    cookbook feeds this model, so every published number was produced through the
    same stray token. Normalising it would make our activations incomparable with
    the literature for no measured gain. If a transformers upgrade changes the
    rendering, this test is the alarm.
    """
    rendered = llama_guard_tokenizer.apply_chat_template(
        [{"role": "user", "content": PAYLOAD}], tokenize=False, add_generation_prompt=True
    )
    assert rendered.startswith(" <|begin_of_text|>")

    ids = llama_guard_tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert ids[0] != llama_guard_tokenizer.bos_token_id
    assert ids[1] == llama_guard_tokenizer.bos_token_id


def test_llama_guard_verdict_tokens_match_the_published_ids(llama_guard_tokenizer):
    """Meta's model card defines the classifier score over these two single tokens."""
    config = load_guard_config("llama_guard_3_8b")
    rendered = render_guard_prompt(llama_guard_tokenizer, config, PAYLOAD)

    tokens = resolve_verdict_tokens(llama_guard_tokenizer, config, rendered)

    assert (tokens.safe_id, tokens.unsafe_id) == (19193, 39257)
    assert (tokens.safe_piece, tokens.unsafe_piece) == ("safe", "unsafe")


def test_llama_guard_verdict_ids_do_not_depend_on_the_forced_prefix(llama_guard_tokenizer):
    """The POSITION was wrong before job 8957221; the TOKENS never were.

    Both prefixes resolve to the same pair, which is why settling the newline
    question corrected where the probabilities were read without invalidating
    which two tokens they belonged to.
    """
    config = load_guard_config("llama_guard_3_8b")
    rendered = render_guard_prompt(llama_guard_tokenizer, config, PAYLOAD)

    empty_prefix = resolve_verdict_tokens(
        llama_guard_tokenizer, config.model_copy(update={"verdict_prefix": ""}), rendered
    )
    settled = resolve_verdict_tokens(llama_guard_tokenizer, config, rendered)

    assert config.verdict_prefix == "\n\n"
    assert (empty_prefix.safe_id, empty_prefix.unsafe_id) == (settled.safe_id, settled.unsafe_id)


def test_payload_final_lands_on_the_payload_inside_the_guard_template(llama_guard_tokenizer):
    """Real BPE, real template, payload buried mid-prompt with ~55 tokens after it."""
    config = load_guard_config("llama_guard_3_8b")
    rendered = render_guard_prompt(llama_guard_tokenizer, config, PAYLOAD)
    ids = llama_guard_tokenizer(rendered, add_special_tokens=False)["input_ids"]

    offset = resolve_position(llama_guard_tokenizer, rendered, PAYLOAD, "instruction_final")
    piece = llama_guard_tokenizer.convert_ids_to_tokens(ids[offset])

    assert piece.lstrip("Ġ") == "bread"
    # Genuinely mid-prompt: the taxonomy epilogue and assistant header follow.
    assert offset < -20


# --- the end-of-instruction span, per guard --------------------------------


@pytest.mark.parametrize(
    "guard, expected_span",
    [("llama_guard_3_8b", 55), ("wildguard", 25)],
)
def test_the_guards_eoi_span_is_NOT_the_target_models_five(guard, expected_span):
    """Derived, not assumed — and the answer is that Arditi's sweep does not port.

    `instrument_layer.md` §6.3.4 fixed the capture spine for TARGET models by
    deriving the end-of-instruction span from the live template (Llama-3.1 and
    Qwen2.5 are 5 tokens, Mistral-v0.3 is 1) and said in terms that AS-6 "must
    derive its GUARDS' spans the same way". This is that derivation, and the
    numbers are 55 and 25.

    They are large for a structural reason, not by accident: on a chat model the
    post-instruction span is an assistant-header run, while a guard's template
    puts the CLASSIFICATION TASK after the payload -- Llama Guard's taxonomy
    epilogue, WildGuard's `Answers: [/INST]` scaffold. So a guard's span is task
    text, and sweeping it is not the same experiment Arditi et al. run.
    """
    from transformers import AutoTokenizer

    from internals_safety.models.loader import end_of_instruction_span

    config = load_guard_config(guard)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    rendered = render_guard_prompt(tokenizer, config, PAYLOAD)
    assert end_of_instruction_span(tokenizer, rendered, PAYLOAD) == expected_span


@pytest.mark.parametrize("guard", ["llama_guard_3_8b", "wildguard"])
def test_eoi_position_names_REFUSES_a_guard_span_rather_than_covering_part(guard):
    """The loud failure is the point: partial coverage is what §6.3.4 WAS.

    `PositionName` enumerates to `last_minus_6`, i.e. spans of at most 7. Both
    guards exceed it by a wide margin. Extending the enumeration to 55 would be
    the wrong fix -- it would make an ill-posed sweep expressible. The refusal
    is the correct state until the causal arm's guard analogue is designed.
    """
    from transformers import AutoTokenizer

    from internals_safety.models.loader import end_of_instruction_span, eoi_position_names

    config = load_guard_config(guard)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    rendered = render_guard_prompt(tokenizer, config, PAYLOAD)
    span = end_of_instruction_span(tokenizer, rendered, PAYLOAD)
    with pytest.raises(ValueError, match="enumerates only up to last_minus_6"):
        eoi_position_names(span)


@pytest.mark.parametrize(
    "guard, expected_offset",
    [("llama_guard_3_8b", -56), ("wildguard", -26)],
)
def test_instruction_final_still_lands_on_the_payload_on_BOTH_guards(guard, expected_offset):
    """Why the span finding does NOT touch any published AS-6 number.

    The decode probe reads at `instruction_final`, which is `-(span + 1)` by
    construction and therefore the payload's final token on both guards. The
    span result constrains the causal arm, which `Limitations` already declares
    unrun; the decode map is unaffected. Pinned so a future reader does not have
    to re-derive that distinction from the span numbers alone.
    """
    from transformers import AutoTokenizer

    config = load_guard_config(guard)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    rendered = render_guard_prompt(tokenizer, config, PAYLOAD)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    offset = resolve_position(tokenizer, rendered, PAYLOAD, "instruction_final")
    assert offset == expected_offset
    assert tokenizer.convert_ids_to_tokens(ids[offset]).lstrip("Ġ▁") == "bread"
