"""The checkpoint's feature-selection rule is a QUESTION, and both answers run.

**Why this exists.** `models/sae_loader.py` derived `act_fn: jumprelu` from the
artifact — threshold magnitude, decoder-bias norm — and dismissed the `top_k: 50`
sitting in the same hyperparams file as a training-schedule leftover. That
derivation was never checked against upstream's forward pass, and job 9009119
made it the leading suspect for the pre-gate's refusal on Llama-3.1-8B-**Base**:

* the normalisation is off by only **1.4-1.7x** (a corpus offset — the declared
  norm is per layer: 13.8125 / 17.125 / 21.5), which excludes the scale
  hypothesis; while
* the round trip emits a vector **~90x too large**, and a linear decoder cannot
  turn a 1.7x input error into a 90x output error;
* but selecting **549** features where **50** were intended can, and jumprelu on
  this checkpoint selects 549.

So the two rules are both expressible on the SAME weights, and the pre-gate runs
both. A dictionary reconstructs the model it was fitted on; whichever rule does
that is the rule it was trained with. **When the run settles it, the loser is
deleted rather than left as a knob** — this is not a tunable.
"""

from __future__ import annotations

import dataclasses

import torch

from internals_safety.config import SAEConfig
from internals_safety.measurements.sae_reconstruction import measure_reconstruction, reading
from internals_safety.models.sae_loader import LlamaScopeSAE

from test_sae_scale_diagnostic import CONFIG, tiny_scope_sae


class TestTheTwoRulesAreActuallyDifferent:
    """A variant that behaves identically to the original tests nothing."""

    def _pre_activations(self, sae: LlamaScopeSAE, activations: torch.Tensor):
        return sae._normalise(activations) @ sae.encoder_weight.T + sae.encoder_bias

    def test_topk_keeps_exactly_nominal_top_k_features(self):
        sae = dataclasses.replace(
            tiny_scope_sae(8, input_norm=1.0, d_sae=64), selection="topk", nominal_top_k=5
        )
        features = sae.encode(torch.randn(4, 8, generator=torch.Generator().manual_seed(0)))
        counts = (features != 0).sum(dim=-1)
        assert counts.tolist() == [5, 5, 5, 5], (
            f"topk must fix L0 by construction, got {counts.tolist()}"
        )

    def test_jumprelu_lets_the_data_decide_the_count(self):
        """The whole difference: jumprelu's L0 is a property of the data, which
        is why it can select 549 where nominal is 50."""
        sae = dataclasses.replace(
            tiny_scope_sae(8, input_norm=1.0, d_sae=64), jump_relu_threshold=0.0
        )
        features = sae.encode(torch.randn(4, 8, generator=torch.Generator().manual_seed(0)))
        counts = (features != 0).sum(dim=-1)
        assert len(set(counts.tolist())) > 1 or counts[0] != 5, (
            "a jumprelu whose count never varies is not exercising the branch under test"
        )

    def test_the_two_rules_disagree_on_the_same_weights(self):
        """**`nominal_top_k` must BIND for this test to mean anything.**

        The first version used the fixture's default `nominal_top_k=50` against
        `d_sae=64` and failed: with a zero threshold, jumprelu keeps every
        positive pre-activation (~32 of 64), and a TopK asking for 50 of 64 --
        clamped at zero -- keeps exactly the same ~32. The rules COINCIDE
        whenever k exceeds the number of positive pre-activations, so the
        original assertion was testing a regime where there is nothing to
        distinguish. k=5 puts it in the regime the real checkpoint is in, where
        50 is far below the 549 jumprelu selects.
        """
        base = dataclasses.replace(
            tiny_scope_sae(8, input_norm=1.0, d_sae=64), nominal_top_k=5
        )
        activations = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
        jump = base.encode(activations)
        top = dataclasses.replace(base, selection="topk").encode(activations)

        assert int((jump != 0).sum(dim=-1).float().mean()) > 5, (
            "jumprelu is not selecting more than k here — the regimes coincide "
            "and the comparison is vacuous"
        )
        assert not torch.allclose(jump, top), (
            "the selection branch is not reached — replace() produced an identical encoder"
        )

    def test_topk_does_not_reconstruct_from_negative_activations(self):
        """Clamped at zero. Keeping negatives would make this a THIRD rule rather
        than the one the hyperparams file documents."""
        sae = dataclasses.replace(
            tiny_scope_sae(8, input_norm=1.0, d_sae=64), selection="topk", nominal_top_k=64
        )
        features = sae.encode(torch.randn(4, 8, generator=torch.Generator().manual_seed(2)))
        assert (features >= 0).all()

    def test_topk_does_not_ask_for_more_features_than_exist(self):
        sae = dataclasses.replace(
            tiny_scope_sae(8, input_norm=1.0, d_sae=4), selection="topk", nominal_top_k=999
        )
        assert sae.encode(torch.randn(2, 8)).shape == (2, 4)


class TestJumpreluRemainsTheDefault:
    def test_an_unmodified_checkpoint_selects_jumprelu(self):
        """The default must not silently change: every reading recorded before
        today was produced under jumprelu, and a flipped default would make old
        and new records incomparable without either changing."""
        assert tiny_scope_sae(8, input_norm=1.0).selection == "jumprelu"


class TestTheRecordSaysWhichRuleProducedIt:
    def test_the_selection_reaches_the_run_record(self, tiny_model):
        """Two readings from one run differ ONLY in this field. A record without
        it cannot say which reading it is — and both land in the same
        `results.json`."""
        d_model = tiny_model.model.config.hidden_size
        for rule in ("jumprelu", "topk"):
            sae = dataclasses.replace(
                tiny_scope_sae(d_model, input_norm=1.0), selection=rule
            )
            quality = measure_reconstruction(
                tiny_model, sae, ["hi", "a longer prompt here"],
                layer=1, config=CONFIG, batch_size=2,
            )
            assert quality.selection == rule
            assert reading(quality, CONFIG).detail["selection"] == rule

    def test_a_dictionary_with_no_selection_rule_records_none(self, tiny_model):
        """`RandomDictionary` has no such field. None means "makes no claim",
        never a default rule it never used."""
        from internals_safety.measurements.sae_reconstruction import RandomDictionary

        quality = measure_reconstruction(
            tiny_model,
            RandomDictionary(
                d_model=tiny_model.model.config.hidden_size, n_features=32, k=4,
                generator=torch.Generator().manual_seed(0),
            ),
            ["hi"], layer=1, config=CONFIG, batch_size=1,
        )
        assert quality.selection is None
