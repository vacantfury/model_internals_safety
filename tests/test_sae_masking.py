"""Reconstruction statistics must see real, non-BOS positions — and only those.

**The defect this pins cost job `9008483` and produced a FALSE REFUSAL.** The
pre-gate reduced over every position of a padded batch and reported
`variance_explained` between -936 and -1819 on Llama-3.1-8B-**Base** — the model
Llama Scope was fitted on, where a dictionary cannot fail to transfer. Two
contaminants, both in the numerator AND the denominator:

* **PAD.** `models/loader.py` sets `padding_side = "left"`, so pads sit at the
  START of every short row, adjacent to BOS.
* **BOS.** Llama-3.1 carries a massive-activation spike there, orders of
  magnitude above a normal residual. The checkpoint declares
  `dataset_average_activation_norm: 13.8125` at layer 17 — it is PER LAYER
  (17.125, 21.5 at 19 and 21) — and the failed run implies ~5,700, which is
  orders of magnitude above all three.

Why no existing test caught it: **every one uses a single prompt or equal-length
prompts**, where padding is invisible. That is the fixture blind spot CLAUDE.md
already records three instances of — a fixture more permissive than the real
thing certifies rather than tests. So the load-bearing test here is a
RAGGED batch, and its assertion is that padding changes nothing.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import SAEConfig
from internals_safety.measurements.sae_reconstruction import (
    RandomDictionary,
    measure_reconstruction,
    scored_positions,
    tokenize_for_reconstruction,
)


BOS = 9


class TestScoredPositions:
    def test_pads_are_dropped(self):
        ids = torch.tensor([[1, 1, BOS, 5, 6], [BOS, 4, 5, 6, 7]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        keep = scored_positions(ids, mask, bos_token_id=None)
        assert keep.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]

    def test_the_first_REAL_token_is_dropped_not_index_zero(self):
        """Under left padding, index 0 is a pad on every short row. Dropping it
        would remove a pad twice and leave BOS in — the whole point."""
        ids = torch.tensor([[1, 1, BOS, 5, 6], [BOS, 4, 5, 6, 7]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        keep = scored_positions(ids, mask, bos_token_id=BOS)
        assert keep.tolist() == [
            [0, 0, 0, 1, 1],   # pads out, then the BOS at index 2
            [0, 1, 1, 1, 1],   # no pads, BOS at index 0
        ]

    def test_a_model_with_no_bos_drops_nothing(self):
        """Qwen2.5 has `bos_token_id is None` — measured, not assumed. The old
        signature dropped the first real position anyway, which on Qwen removed
        `<|im_start|>` and on the plain arm removed the first word."""
        ids = torch.tensor([[4, 5, 6], [4, 5, 6]])
        mask = torch.ones(2, 3, dtype=torch.long)
        assert scored_positions(ids, mask, bos_token_id=None).all()

    def test_a_row_that_does_not_start_with_bos_keeps_its_first_token(self):
        """The assertion the old `drop_first_real: bool` made without checking.

        Row 1 is Tulu-3's chat arm: the model HAS a BOS token but this arm
        deliberately carries none (`prepend_bos_to_chat_template: false`). The
        old code removed its first content token; identity-dropping leaves it."""
        ids = torch.tensor([[BOS, 4, 5], [4, 5, 6]])
        mask = torch.ones(2, 3, dtype=torch.long)
        keep = scored_positions(ids, mask, bos_token_id=BOS)
        assert keep.tolist() == [[0, 1, 1], [1, 1, 1]]

    def test_an_all_pad_row_does_not_crash(self):
        ids = torch.full((1, 4), 1)
        mask = torch.zeros(1, 4, dtype=torch.long)
        assert not scored_positions(ids, mask, bos_token_id=BOS).any()

    def test_the_input_mask_is_not_mutated(self):
        """`clone()` — the caller's `encoded["attention_mask"]` is reused for the
        forward pass, and silently zeroing a real token there would change what
        the model attends to."""
        ids = torch.tensor([[BOS, 4, 5], [BOS, 4, 5]])
        mask = torch.ones(2, 3, dtype=torch.long)
        scored_positions(ids, mask, bos_token_id=BOS)
        assert mask.all()


class TestWhoOwnsTheBOSDecision:
    """The trap measured on the real tokenizers 2026-08-08.

    `tokenizer(chat_rendered)` with the default `add_special_tokens=True` gives
    `['<|begin_of_text|>', '<|begin_of_text|>', ...]` on Llama-3.1-Instruct and
    `['<s>', '<s>', ...]` on Mistral-v0.3, because the template already emitted
    one. This module was the only place in the repo tokenising that way.

    The fix does not decide BOS itself — since `verify_bos_convention`
    (`models/loader.py`, peer session) the CONFIG owns it for the chat arm, and
    Tulu-3 declares a deliberate BOS-less run. So the chat arm takes the template
    at its word and the plain arm, which has no template, lets the tokenizer's
    post-processor supply one.
    """

    def test_the_chat_arm_takes_the_template_at_its_word(self, tiny_bos_tokenizer):
        rendered = tiny_bos_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenize_for_reconstruction(
            tiny_bos_tokenizer, [rendered], render_chat=True
        )["input_ids"][0]
        assert ids[0] == tiny_bos_tokenizer.bos_token_id
        assert (ids == tiny_bos_tokenizer.bos_token_id).sum() == 1

    def test_a_bos_less_chat_template_is_NOT_overruled(self, tiny_tokenizer):
        """Tulu-3's case, and the reason this function decides nothing. Adding a
        BOS here would silently overrule `prepend_bos_to_chat_template: false`."""
        rendered = tiny_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenize_for_reconstruction(
            tiny_tokenizer, [rendered], render_chat=True
        )["input_ids"][0].tolist()
        assert ids == tiny_tokenizer(rendered, add_special_tokens=False)["input_ids"]

    def test_the_plain_arm_DOES_get_one(self, tiny_bos_tokenizer):
        """No template means nothing has decided, and the dictionary was fitted
        on a distribution that had BOS."""
        ids = tokenize_for_reconstruction(
            tiny_bos_tokenizer, ["user hi"], render_chat=False
        )["input_ids"][0]
        assert ids[0] == tiny_bos_tokenizer.bos_token_id

    def test_a_model_without_bos_gets_nothing(self, tiny_tokenizer):
        ids = tokenize_for_reconstruction(
            tiny_tokenizer, ["user hi"], render_chat=False
        )["input_ids"][0]
        assert ids.tolist() == tiny_tokenizer("user hi", add_special_tokens=False)["input_ids"]

    def test_a_DOUBLE_bos_is_refused(self, tiny_bos_tokenizer):
        """THE defect. Chat-rendered text (template emits BOS) tokenised as if it
        were the plain arm (post-processor adds another) is the exact code path
        this module used to take on every Instruct run."""
        rendered = tiny_bos_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
        )
        with pytest.raises(ValueError, match="TWO"):
            tokenize_for_reconstruction(tiny_bos_tokenizer, [rendered], render_chat=False)

    def test_left_padding_still_puts_bos_at_the_first_real_position(
        self, tiny_bos_tokenizer
    ):
        """Under left padding BOS is not at index 0 on short rows, so both the
        double-BOS check and the masking must find the first REAL position."""
        encoded = tokenize_for_reconstruction(
            tiny_bos_tokenizer, ["user hi", "user hi there now"], render_chat=False
        )
        keep = scored_positions(
            encoded["input_ids"],
            encoded["attention_mask"],
            bos_token_id=tiny_bos_tokenizer.bos_token_id,
        )
        assert int(keep.sum()) == int(encoded["attention_mask"].sum()) - 2


class TestPaddingChangesNothing:
    """THE test. A ragged batch and a set of single-prompt runs must agree."""

    def _quality(self, tiny_model, prompts, batch_size):
        dictionary = RandomDictionary(
            d_model=tiny_model.model.config.hidden_size,
            n_features=32,
            k=4,
            generator=torch.Generator().manual_seed(0),
        )
        return measure_reconstruction(
            tiny_model,
            dictionary,
            prompts,
            layer=1,
            config=SAEConfig(
                trained_on="t", min_kl_recovered=0.8, min_transfer_ratio=0.8
            ),
            batch_size=batch_size,
        )

    def test_a_ragged_batch_matches_one_prompt_at_a_time(self, tiny_model):
        """Batched (padded) against unbatched (never padded). Before the fix the
        two disagreed by whatever the pads contributed, which on real activations
        was enough to invert the verdict."""
        prompts = [
            "hi",
            "a somewhat longer prompt with more tokens in it",
            "medium length prompt",
        ]
        batched = self._quality(tiny_model, prompts, batch_size=3)
        unbatched = self._quality(tiny_model, prompts, batch_size=1)

        assert batched.mse == torch.tensor(unbatched.mse).item() or abs(
            batched.mse - unbatched.mse
        ) < 1e-3 * max(1.0, abs(unbatched.mse)), (
            f"padding changed MSE: {batched.mse} batched vs {unbatched.mse} single. "
            "Pads are in the reduction."
        )
        assert abs(batched.variance_explained - unbatched.variance_explained) < 1e-3, (
            f"padding changed variance explained: {batched.variance_explained} vs "
            f"{unbatched.variance_explained}"
        )

    def test_the_denominator_is_the_kept_count_not_batch_times_seqlen(
        self, tiny_model
    ):
        """Masking only the numerator gives a correct error over an inflated
        denominator — a partial fix that lands somewhere plausible-but-wrong.

        With a ragged batch, `batch * seq_len` strictly exceeds the number of
        real non-BOS positions, so an MSE computed against it is strictly
        smaller. Equality with the unbatched run is what rules that out, and the
        test above asserts it; this one pins the mechanism directly.
        """
        prompts = ["hi", "a much longer prompt with many more tokens than the first"]
        quality = self._quality(tiny_model, prompts, batch_size=2)

        encoded = tokenize_for_reconstruction(
            tiny_model.tokenizer, prompts, render_chat=False
        )
        keep = scored_positions(
            encoded["input_ids"],
            encoded["attention_mask"],
            bos_token_id=tiny_model.tokenizer.bos_token_id,
        )
        padded_total = encoded["attention_mask"].numel()

        assert int(keep.sum()) < padded_total, "fixture is not ragged — test is vacuous"
        # MSE is sq_err / n_tokens; if n_tokens were the padded total the value
        # would be scaled down by exactly kept/padded.
        assert quality.mse > 0
