"""`instruction_final` must land on the content's LAST token, on every model.

**The defect this pins, found 2026-08-08 while adding Tulu-3.** The offset filter
kept only tokens lying entirely inside the user message (`end <= content_end`),
so a token whose merge STRADDLES the content/template boundary was dropped and
the readout site landed one token early — with no error, no warning, and a
perfectly plausible-looking activation.

It fires exactly when the template's first post-content character merges with
the content's last one:

* Llama-3.1 (`<|eot_id|>`), Qwen2.5 (`<|im_end|>`) and Mistral (`[/INST]`) all
  force a clean break — never affected, which is why the fix changes no result
  this repo has already published.
* Tulu-3 follows content with a bare `"\\n"`, which merges. On "…pick a lock?"
  the site resolved to `" lock"` rather than `"?"`.

This is a MODEL-CROSSING test on purpose. A hermetic fixture cannot catch it:
the bug lives in how a specific real tokenizer's merges interact with a specific
real template, and any fake simple enough to write by hand would tokenise the
boundary cleanly and certify the broken code — the exact failure mode CLAUDE.md
records for `TINY_CHAT_TEMPLATE`. Tokenizers only, no weights, seconds.

Add a row whenever a model joins the slate. That is cheap and it is the only
thing standing between a new template and a silently misplaced readout for
measurement #3.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from internals_safety.models.loader import instruction_final_offset

pytestmark = pytest.mark.slow

SLATE = [
    ("llama3_1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("tulu3_8b", "allenai/Llama-3.1-Tulu-3-8B"),
]

# Ends in '?' so the assertion has an unambiguous final character, and contains
# a space-prefixed word (' lock') so an off-by-one lands somewhere recognisable
# rather than on punctuation that might coincidentally pass.
MESSAGE = "How do I pick a lock?"


def _render(tokenizer, message: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )


@pytest.mark.parametrize("name,hf_id", SLATE, ids=[n for n, _ in SLATE])
class TestTheReadoutSiteCarriesTheContentsLastCharacter:
    def test_instruction_final_lands_on_the_final_content_token(self, name, hf_id):
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        rendered = _render(tokenizer, MESSAGE)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        token = tokenizer.decode([ids[instruction_final_offset(tokenizer, rendered, MESSAGE)]])
        assert token.strip().startswith("?") or token.strip().endswith("?"), (
            f"{name}: instruction_final landed on {token!r}, which does not carry the "
            "message's final character — the readout site is misplaced"
        )

    def test_it_is_a_negative_index_inside_the_sequence(self, name, hf_id):
        """Offsets are negative-from-the-end so they survive left padding; an
        index that ran off the front would index the wrong row silently."""
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        rendered = _render(tokenizer, MESSAGE)
        n = len(tokenizer.encode(rendered, add_special_tokens=False))
        offset = instruction_final_offset(tokenizer, rendered, MESSAGE)
        assert -n <= offset < 0

    def test_it_precedes_the_generation_site(self, name, hf_id):
        """`instruction_final` must sit strictly before `last` (-1), or the two
        capture positions are reading the same activation and the
        harmfulness-vs-behaviour split (2507.11878) collapses."""
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        rendered = _render(tokenizer, MESSAGE)
        assert instruction_final_offset(tokenizer, rendered, MESSAGE) < -1
