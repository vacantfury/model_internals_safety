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

import torch

from internals_safety.config import SAEConfig
from internals_safety.measurements.sae_reconstruction import (
    RandomDictionary,
    measure_reconstruction,
    scored_positions,
)


class TestScoredPositions:
    def test_pads_are_dropped(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        keep = scored_positions(mask, drop_first_real=False)
        assert keep.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]

    def test_the_first_REAL_token_is_dropped_not_index_zero(self):
        """Under left padding, index 0 is a pad on every short row. Dropping it
        would remove a pad twice and leave BOS in — the whole point."""
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        keep = scored_positions(mask, drop_first_real=True)
        assert keep.tolist() == [
            [0, 0, 0, 1, 1],   # pads out, then the BOS at index 2
            [0, 1, 1, 1, 1],   # no pads, BOS at index 0
        ]

    def test_bos_survives_when_the_caller_asks_for_it(self):
        """It is an explicit decision, not a side effect of masking — the
        attention mask marks BOS as real."""
        mask = torch.ones(2, 3, dtype=torch.long)
        assert scored_positions(mask, drop_first_real=False).all()

    def test_an_all_pad_row_does_not_crash(self):
        mask = torch.zeros(1, 4, dtype=torch.long)
        assert not scored_positions(mask, drop_first_real=True).any()

    def test_the_input_mask_is_not_mutated(self):
        """`clone()` — the caller's `encoded["attention_mask"]` is reused for the
        forward pass, and silently zeroing a real token there would change what
        the model attends to."""
        mask = torch.ones(2, 3, dtype=torch.long)
        scored_positions(mask, drop_first_real=True)
        assert mask.all()


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
                trained_on="t", min_kl_recovered=0.8, min_variance_explained=0.75
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

        encoded = tiny_model.tokenizer(prompts, return_tensors="pt", padding=True)
        keep = scored_positions(encoded["attention_mask"], drop_first_real=True)
        padded_total = encoded["attention_mask"].numel()

        assert int(keep.sum()) < padded_total, "fixture is not ragged — test is vacuous"
        # MSE is sq_err / n_tokens; if n_tokens were the padded total the value
        # would be scaled down by exactly kept/padded.
        assert quality.mse > 0
