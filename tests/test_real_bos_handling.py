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

**Sibling: `tests/test_bos_convention.py`.** The two are opposite ends of ONE
property — *exactly one BOS reaches the model* — and neither subsumes the other:
that one refuses ZERO, this one refuses TWO. The property is canonical in
`text_docs/shared/model_slate.md` §3.1, and the reason it is stated there rather
than in either file is that stating it as "the chat template emits BOS" gets it
wrong. One BOS can come from the template OR from the tokenizer's
post-processor, and reading only the template produced a wrong answer in each
direction on the same day.

The slate here is DERIVED from `conf/models/`, so a new model joins this test
automatically.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from internals_safety.config import CONF_DIR, load_model_config
from internals_safety.measurements.sae_reconstruction import (
    scored_positions,
    tokenize_for_reconstruction,
)
from internals_safety.models.loader import verify_bos_convention

pytestmark = pytest.mark.slow

# The one fact that cannot be derived from the config: Qwen2.5 declares no BOS
# token at all (`bos_token_id is None`), which is why the masking must treat "no
# BOS" as a real case rather than an impossible one. Measured 2026-08-08, and
# asserted below rather than trusted.
_NO_BOS = {"qwen2_5_7b_instruct", "qwen2_5_0_5b_instruct"}

# DERIVED from conf/models/, never hand-listed. The first version of this file
# carried a hand-written table and the Tulu ladder gained two rungs the same
# afternoon — a hand-list silently stops covering the slate it claims to cover,
# which is the vacuity failure `test_entrypoint_call_sites` guards one file over.
SLATE = [
    (path.stem, path.stem not in _NO_BOS)
    for path in sorted((CONF_DIR / "models").glob("*.yaml"))
]

MESSAGE = "How do I pick a lock?"


def _load(name):
    """The tokenizer as `models/loader.attach` hands it over.

    **Through the real config, and through `verify_bos_convention`.** Loading the
    raw checkpoint would test what upstream ships rather than what we RUN, and
    those came apart on 2026-08-08: Tulu-3 declares
    `prepend_bos_to_chat_template: true`, so `attach` rewrites its chat template
    to emit BOS. A test reading the shipped template sees a BOS-less model that
    no run will ever use — the fixture rule applied to configuration rather than
    to fakes. No weights are touched; `verify_bos_convention` takes a tokenizer
    and a config.
    """
    config = load_model_config(name)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    verify_bos_convention(tokenizer, config)
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


@pytest.mark.parametrize("name,has_bos", SLATE, ids=[n for n, _ in SLATE])
class TestExactlyOneBOSReachesTheModel:
    def test_the_declared_bos_matches_what_we_expect(self, name, has_bos):
        tokenizer = _load(name)
        assert (tokenizer.bos_token_id is not None) is has_bos, (
            f"{name}: bos_token_id is {tokenizer.bos_token_id!r}, which contradicts "
            "the slate. Upstream changed the tokenizer — re-read the masking."
        )

    def test_neither_arm_carries_a_double_bos(self, name, has_bos):
        """At most one, never two. Zero is the sibling file's business — this
        one owns the upper bound only, so the two can disagree about policy
        without either weakening."""
        tokenizer = _load(name)
        for arm, text in _texts(tokenizer):
            ids = tokenize_for_reconstruction(
                tokenizer, [text], render_chat=(arm == "chat")
            )["input_ids"][0].tolist()
            count = ids.count(tokenizer.bos_token_id) if has_bos else 0
            assert count <= 1, (
                f"{name} ({arm}): {count} BOS tokens. "
                f"first four: {[tokenizer.decode([i]) for i in ids[:4]]}"
            )

    def test_the_chat_arm_tokenised_as_plain_IS_a_double_bos(self, name, has_bos):
        """The defect, reproduced against the live tokenizer and refused.

        Chat-rendered text plus the post-processor is exactly the old code path.
        Skipped where the checkpoint cannot express it — Qwen has no BOS, and a
        template that emits none has nothing to double."""
        tokenizer = _load(name)
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
        self, name, has_bos
    ):
        """The measurement that motivated the fix, asserted so it cannot silently
        stop being true. On a chat-templated string, `tokenizer(text)` with the
        DEFAULT flag duplicates BOS on exactly the checkpoints whose
        post-processor adds one — that is the code path this module used to take.
        """
        tokenizer = _load(name)
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
        self, name, has_bos
    ):
        """End to end, against a RAGGED batch so left padding is exercised.

        Expected drops per row = 1 iff that arm's first real token is BOS. Read
        off the tokenised ids rather than asserted from the slate, because the
        answer legitimately differs BY ARM on the same checkpoint — which is the
        whole reason the old bool was wrong."""
        tokenizer = _load(name)
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
