"""Exactly one BOS reaches the model, on every checkpoint on the slate.

**The defect this pins, measured 2026-08-08.** `measurements/sae_reconstruction`
was the ONLY module in the repo tokenising with the default
`add_special_tokens=True`; the other twenty-odd call sites pass `False` and let
the chat template own BOS. On the two checkpoints whose tokenizer prepends BOS
from a post-processor, that produced a **double BOS** in the chat arm:

    Llama-3.1-8B-Instruct  ['<|begin_of_text|>', '<|begin_of_text|>', ...]
    Mistral-7B-Instruct    ['<s>', '<s>', '[INST]', ...]

`scored_positions` then dropped "the first real position" and scored the second
— so the massive-activation spike the masking exists to EXCLUDE was inside the
metric, and only in one of the two arms the transfer gate compares. The plain
arm has no template to emit BOS, so it carried exactly one and was masked
correctly; the asymmetry sat between the two numbers being divided.

Why this is a real-tokenizer test and not a fixture one: whether BOS is added is
a property of the checkpoint's `post_processor`, which lives in `tokenizer.json`
and is upstream's to change. `tests/conftest.py` now models it (that is what
`tiny_bos_tokenizer` is for) and the hermetic suite catches our own regressions,
but only this file can catch THEIRS. Tokenizers only, no weights, seconds.

**Sibling: `tests/test_bos_convention.py`.** The two are opposite ends of one
axis and neither subsumes the other — that one catches ZERO BOS (a checkpoint
whose template emits none, so it runs BOS-less against models that do not), this
one catches TWO. Both carry a per-model row table, so **a model joining the slate
must be added to BOTH**.

Add a row whenever a model joins the slate.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from internals_safety.measurements.sae_reconstruction import (
    scored_positions,
    tokenize_for_reconstruction,
)

pytestmark = pytest.mark.slow

# `has_bos` is MEASURED (2026-08-08), not assumed, and it is not uniform:
# Qwen2.5 has no BOS token at all (`bos_token_id is None`), which is why the
# masking must treat "no BOS" as a real case rather than an impossible one.
SLATE = [
    ("llama3_1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct", True),
    ("llama3_1_8b_base", "meta-llama/Llama-3.1-8B", True),
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct", False),
    ("mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.3", True),
    ("tulu3_8b", "allenai/Llama-3.1-Tulu-3-8B", True),
]

MESSAGE = "How do I pick a lock?"


def _load(hf_id):
    """The tokenizer as `models/loader.attach` hands it over — left padding and a
    pad token. Loading it any other way tests a configuration production never
    sees, which is the fixture rule one level up."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _texts(tokenizer) -> list[tuple[str, str]]:
    """(arm, text) for both arms the pre-gate runs — chat and plain."""
    arms = [("plain", MESSAGE)]
    if tokenizer.chat_template:
        arms.append((
            "chat",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": MESSAGE}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        ))
    return arms


@pytest.mark.parametrize("name,hf_id,has_bos", SLATE, ids=[n for n, _, _ in SLATE])
class TestExactlyOneBOSReachesTheModel:
    def test_the_declared_bos_matches_what_we_expect(self, name, hf_id, has_bos):
        tokenizer = _load(hf_id)
        assert (tokenizer.bos_token_id is not None) is has_bos, (
            f"{name}: bos_token_id is {tokenizer.bos_token_id!r}, which contradicts "
            "the slate. Upstream changed the tokenizer — re-read the masking."
        )

    def test_neither_arm_carries_a_double_bos(self, name, hf_id, has_bos):
        """At most one, never two. Not "exactly one": Tulu-3's chat template
        emits none and `prepend_bos_to_chat_template: false` declares that
        deliberate, so zero is a legitimate reading here — it is the SIBLING
        file's business, not this one's."""
        tokenizer = _load(hf_id)
        for arm, text in _texts(tokenizer):
            ids = tokenize_for_reconstruction(
                tokenizer, [text], render_chat=(arm == "chat")
            )["input_ids"][0].tolist()
            count = ids.count(tokenizer.bos_token_id) if has_bos else 0
            assert count <= 1, (
                f"{name} ({arm}): {count} BOS tokens. "
                f"first four: {[tokenizer.decode([i]) for i in ids[:4]]}"
            )

    def test_the_chat_arm_tokenised_as_plain_IS_a_double_bos(self, name, hf_id, has_bos):
        """The defect, reproduced against the live tokenizer and refused.

        Chat-rendered text plus the post-processor is exactly the old code path.
        Skipped where the checkpoint cannot express it — Qwen has no BOS, and
        Tulu-3's template emits none, so nothing can double."""
        tokenizer = _load(hf_id)
        if not has_bos or not tokenizer.chat_template:
            pytest.skip("no BOS or no template — the trap is not expressible here")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": MESSAGE}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not rendered.startswith(tokenizer.bos_token):
            pytest.skip("template emits no BOS — this is the sibling file's case")
        with pytest.raises(ValueError, match="TWO"):
            tokenize_for_reconstruction(tokenizer, [rendered], render_chat=False)

    def test_the_default_tokenisation_is_the_thing_we_stopped_doing(
        self, name, hf_id, has_bos
    ):
        """The measurement that motivated the fix, asserted so it cannot silently
        stop being true. On a chat-templated string, `tokenizer(text)` with the
        DEFAULT flag duplicates BOS on exactly the checkpoints whose
        post-processor adds one — that is the code path this module used to take.
        """
        tokenizer = _load(hf_id)
        if not has_bos or not tokenizer.chat_template:
            pytest.skip("no BOS or no template — the trap is not expressible here")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": MESSAGE}],
            tokenize=False,
            add_generation_prompt=True,
        )
        default = tokenizer(rendered)["input_ids"]
        duplicated = default[:2] == [tokenizer.bos_token_id] * 2
        emits_bos = rendered.startswith(tokenizer.bos_token)
        assert duplicated == emits_bos, (
            f"{name}: template emits BOS = {emits_bos} but the default tokenisation "
            f"duplicated it = {duplicated}. The two must agree, or the fix's premise "
            "no longer describes this checkpoint."
        )

    def test_the_masking_drops_exactly_the_bos_that_is_there(
        self, name, hf_id, has_bos
    ):
        """End to end, against a RAGGED batch so left padding is exercised.

        Expected drops per row = 1 iff that arm's first real token is BOS. Read
        off the tokenised ids rather than asserted from the slate, because the
        answer legitimately differs BY ARM on the same checkpoint — which is the
        whole reason the old bool was wrong."""
        tokenizer = _load(hf_id)
        for arm, text in _texts(tokenizer):
            encoded = tokenize_for_reconstruction(
                tokenizer, [text, text + " Explain."], render_chat=(arm == "chat")
            )
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            first = mask.bool().float().argmax(dim=1)
            starts_with_bos = int(
                sum(
                    1
                    for row in range(ids.shape[0])
                    if has_bos and int(ids[row, first[row]]) == tokenizer.bos_token_id
                )
            )
            keep = scored_positions(ids, mask, bos_token_id=tokenizer.bos_token_id)
            dropped = int(mask.sum()) - int(keep.sum())
            assert dropped == starts_with_bos, (
                f"{name} ({arm}): dropped {dropped} positions, but {starts_with_bos} "
                f"of {ids.shape[0]} rows begin with BOS"
            )
