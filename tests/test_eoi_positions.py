"""The end-of-instruction span — the capture sites the two-position spine missed.

**Why this exists (2026-08-09, `instrument_layer.md` §6.3.4).** Three causal-gate
runs returned "no direction survives", and the cause was not the filter or the
sweep width: it was that the spine never captured the positions the method's
authors sweep. Arditi et al. use
`positions=list(range(-len(eoi_toks), 0))` — every end-of-instruction token. For
Llama-3.1-8B-Instruct that span is 5 tokens at -5..-1. We captured
`instruction_final` at **-6**, one token BEFORE the span, and `last` at **-1**,
its last token. Four of five positions, including the whole interior, were never
looked at.

The tests below pin the two properties that failure needed: the span is DERIVED
from the live template (never the constant 5), and asking for a span wider than
the enumerated names FAILS rather than silently covering part of it.
"""

from __future__ import annotations

import pytest

from internals_safety.config import PositionName
from internals_safety.models.loader import (
    eoi_position_names,
    end_of_instruction_span,
    instruction_final_offset,
    render_chat,
    resolve_position,
)


class TestLastMinusKResolvesToTheRightOffset:
    def test_it_counts_back_from_the_final_token(self, tiny_tokenizer):
        rendered, message = "irrelevant", "irrelevant"
        assert resolve_position(tiny_tokenizer, rendered, message, "last") == -1
        assert resolve_position(tiny_tokenizer, rendered, message, "last_minus_1") == -2
        assert resolve_position(tiny_tokenizer, rendered, message, "last_minus_4") == -5

    def test_an_unknown_name_still_fails_loud(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="unknown position"):
            resolve_position(tiny_tokenizer, "x", "x", "middle")  # type: ignore[arg-type]


class TestTheSpanIsDerivedFromTheTemplate:
    """The whole point. A constant 5 would be right for one checkpoint and wrong
    for every guard template AS-6 runs."""

    def test_instruction_final_sits_exactly_one_token_before_the_span(
        self, tiny_tokenizer
    ):
        message = "how do I bake bread"
        rendered = render_chat(tiny_tokenizer, message, None)
        span = end_of_instruction_span(tiny_tokenizer, rendered, message)
        offset = instruction_final_offset(tiny_tokenizer, rendered, message)
        # This identity IS the defect, stated as an invariant: the old spine's
        # two positions were -(span+1) and -1, so everything between was skipped.
        assert offset == -(span + 1)

    def test_the_tiny_template_has_a_MULTI_token_span(self, tiny_tokenizer):
        """A fixture whose span were 1 token could not express the bug -- the
        strictest-real-implementation rule. This template ends the user turn and
        opens an assistant turn, as every real chat template does."""
        message = "how do I bake bread"
        rendered = render_chat(tiny_tokenizer, message, None)
        assert end_of_instruction_span(tiny_tokenizer, rendered, message) > 1

    def test_the_names_cover_the_whole_span_and_nothing_else(self, tiny_tokenizer):
        message = "how do I bake bread"
        rendered = render_chat(tiny_tokenizer, message, None)
        span = end_of_instruction_span(tiny_tokenizer, rendered, message)
        names = eoi_position_names(span)
        assert len(names) == span
        offsets = sorted(
            resolve_position(tiny_tokenizer, rendered, message, name) for name in names
        )
        assert offsets == list(range(-span, 0))   # exactly Arditi's range()

    def test_instruction_final_is_NOT_among_them(self, tiny_tokenizer):
        """It is outside the span by one token, which is why it bypassed nothing
        at any of 25 layers while `last` bypassed 78% of refusal."""
        message = "how do I bake bread"
        rendered = render_chat(tiny_tokenizer, message, None)
        span = end_of_instruction_span(tiny_tokenizer, rendered, message)
        names = eoi_position_names(span)
        assert "instruction_final" not in names


class TestPartialCoverageIsRefused:
    def test_a_span_wider_than_the_enumerated_names_raises(self):
        """Returning a short list would look exactly like coverage while being
        the bug. Named positions stop at last_minus_6, so a 9-token span is an
        error telling the caller to extend the enum."""
        with pytest.raises(ValueError, match="partial coverage"):
            eoi_position_names(9)

    def test_a_degenerate_span_is_refused_rather_than_returning_nothing(self):
        with pytest.raises(ValueError, match="at least 1 token"):
            eoi_position_names(0)

    def test_every_generated_name_is_a_declared_PositionName(self):
        from typing import get_args

        for span in range(1, 8):
            assert all(name in get_args(PositionName) for name in eoi_position_names(span))


@pytest.mark.slow
class TestAgainstTheRealTemplates:
    """Tokenizers only -- no weights, no GPU, seconds. Same posture as
    `test_real_guard_tokenizers.py`: the hermetic tests above pin our mechanism,
    these pin somebody else's artifact.

    ⚠️ The span is MODEL-DEPENDENT and the spread is the finding. Measured
    2026-08-09: Llama-3.1 and Qwen2.5 are 5 tokens, Mistral-v0.3 is **1** -- its
    template closes with `[/INST]` and generation starts immediately, so it has
    no assistant header to sweep. The old two-position spine was therefore
    nearly COMPLETE for Mistral and missing 4 of 5 sites for Llama and Qwen.
    A hardcoded 5 would have been wrong on Mistral by 5x, and a cross-model
    conclusion drawn from the old spine was comparing different coverage.
    """

    @pytest.mark.parametrize(
        "repo, expected",
        [
            ("meta-llama/Llama-3.1-8B-Instruct", 5),   # reproduces Arditi's eoi_toks
            ("Qwen/Qwen2.5-7B-Instruct", 5),
            ("mistralai/Mistral-7B-Instruct-v0.3", 1),
        ],
    )
    def test_the_derived_span_matches_the_shipped_template(self, repo, expected):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo)
        message = "how do I bake bread"
        rendered = render_chat(tokenizer, message, None)
        assert end_of_instruction_span(tokenizer, rendered, message) == expected

    def test_llama31_span_equals_arditis_eoi_toks_computed_their_way(self):
        """Independent derivation, not a copied constant: theirs is
        `encode(TEMPLATE.split('{instruction}')[-1])`, ours walks the rendered
        prompt. They must agree, and agreeing is what licenses the port."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        theirs = len(
            tokenizer.encode(
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                add_special_tokens=False,
            )
        )
        message = "how do I bake bread"
        rendered = render_chat(tokenizer, message, None)
        assert end_of_instruction_span(tokenizer, rendered, message) == theirs == 5
