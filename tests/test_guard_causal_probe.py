"""The causal test, wired to a content guard rather than a generating model.

**Why this file exists.** The seam was added on 2026-08-09 and the full suite
went green with the count UNCHANGED — 1851 before and after — which means
nothing exercised the new path. That is the shape this repo has already been
burned by: 982 green tests while `sae_pregate.py` called a function with the
wrong signature and died twenty seconds into a queued H200 job. A green suite
that does not touch the new code certifies nothing.

Everything here is hermetic — a tiny in-process guard, no weights, no network.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.guards import resolve_verdict_tokens, verdict_probability
from internals_safety.guards.verdict import label_mass_from_logits
from internals_safety.measurements.causal import (
    BehaviourProbe,
    guard_verdict_probe,
    measure_causal_evidence,
)

# The guard fixture lives in the AS-6 test module; importing the fixture module
# rather than copying the fixture keeps ONE definition of what a tiny guard is.
from test_as6_guard_probe import tiny_guard_model  # noqa: F401


def logits_with(mass: dict[int, float], vocab: int = 258, rows: int = 3) -> torch.Tensor:
    """[rows, 1, vocab] logits whose softmax puts EXACTLY `mass` on those ids.

    ⚠️ The first version of this helper rescaled the whole vector to sum to one,
    which silently normalised the "diffuse" case straight back into the
    concentrated one and made the renormalisation test pass for the wrong
    reason. That is the fixture-more-permissive-than-reality defect this repo
    already records three instances of, committed inside the test written to
    catch it. The remainder is spread over the OTHER ids so that total label mass
    is whatever the caller asked for.
    """
    if sum(mass.values()) > 1.0:
        raise ValueError("mass over the named ids exceeds 1.0")
    remainder = 1.0 - sum(mass.values())
    others = vocab - len(mass)
    probabilities = torch.full((rows, vocab), remainder / others if others else 0.0)
    for token_id, value in mass.items():
        probabilities[:, token_id] = value
    return probabilities.log().unsqueeze(1)


class TestVerdictProbabilityIsRenormalised:
    """The design claim: the score reads the guard's PREFERENCE between labels,
    not how much of the vocabulary the labels happen to hold."""

    def test_it_is_the_ratio_not_the_raw_mass(self, tiny_guard_model):  # noqa: F811
        tokens = resolve_verdict_tokens(
            tiny_guard_model.tokenizer, tiny_guard_model.config, "CLASSIFY: x ANSWER:"
        )
        # Same 3:1 preference, wildly different total label mass.
        concentrated = logits_with({tokens.unsafe_id: 0.75, tokens.safe_id: 0.25})
        diffuse = logits_with({tokens.unsafe_id: 0.075, tokens.safe_id: 0.025})

        assert verdict_probability(concentrated, tokens) == pytest.approx(0.75, abs=1e-4)
        assert verdict_probability(diffuse, tokens) == pytest.approx(0.75, abs=1e-4)

    def test_raw_mass_WOULD_have_moved_which_is_why_this_matters(self, tiny_guard_model):  # noqa: F811
        """Mutation: the rejected implementation gives a different answer.

        A test that only pins the chosen behaviour cannot show the choice
        mattered. Raw softmax mass on the unsafe token falls 10x between the two
        cases above while the guard's preference is identical -- an over-strong
        ablation that merely flattens the distribution would read as 'much less
        unsafe' under the rejected reading.
        """
        tokens = resolve_verdict_tokens(
            tiny_guard_model.tokenizer, tiny_guard_model.config, "CLASSIFY: x ANSWER:"
        )
        concentrated = logits_with({tokens.unsafe_id: 0.75, tokens.safe_id: 0.25})
        diffuse = logits_with({tokens.unsafe_id: 0.075, tokens.safe_id: 0.025})

        raw = lambda lg: float(torch.softmax(lg[:, -1, :], dim=-1)[:, tokens.unsafe_id].mean())
        assert raw(concentrated) / raw(diffuse) == pytest.approx(10.0, rel=0.05)

    def test_it_fails_loud_when_no_mass_sits_on_either_label(self, tiny_guard_model):  # noqa: F811
        """Reading at the wrong position is a bug, never a confident 'safe'."""
        tokens = resolve_verdict_tokens(
            tiny_guard_model.tokenizer, tiny_guard_model.config, "CLASSIFY: x ANSWER:"
        )
        elsewhere = torch.full((2, 1, 258), -1e30)
        elsewhere[:, :, 100] = 0.0
        with pytest.raises(ValueError, match="wrong position"):
            verdict_probability(elsewhere, tokens)

    def test_label_mass_is_the_health_check_and_reads_near_one_when_correct(
        self, tiny_guard_model  # noqa: F811
    ):
        tokens = resolve_verdict_tokens(
            tiny_guard_model.tokenizer, tiny_guard_model.config, "CLASSIFY: x ANSWER:"
        )
        healthy = logits_with({tokens.unsafe_id: 0.6, tokens.safe_id: 0.39})
        assert label_mass_from_logits(healthy, tokens) == pytest.approx(0.99, abs=1e-3)


class TestTheProbeRendersThroughTheGuardFormat:
    def test_it_is_labelled_so_a_run_record_cannot_be_misread(self, tiny_guard_model):  # noqa: F811
        assert guard_verdict_probe(tiny_guard_model).name == "guard_unsafe_verdict"

    def test_it_reads_one_row_per_payload(self, tiny_guard_model):  # noqa: F811
        probe = guard_verdict_probe(tiny_guard_model)
        logits = probe.render_and_read(["alpha", "beta", "gamma"])
        assert logits.shape[0] == 3
        assert logits.shape[1] == 1

    def test_scoring_before_rendering_fails_loud_rather_than_guessing(
        self, tiny_guard_model  # noqa: F811
    ):
        """The tokens are resolved from the first REAL payload, so score() has
        nothing to read until render_and_read has run. That ordering is enforced
        rather than papered over with a placeholder payload, because a
        placeholder can resolve a different label token than the run reads."""
        probe = guard_verdict_probe(tiny_guard_model)
        with pytest.raises(RuntimeError, match="out of order"):
            probe.score(torch.zeros(1, 1, 258))

    def test_the_probe_carries_a_health_readout_and_the_refusal_one_does_not(
        self, tiny_guard_model  # noqa: F811
    ):
        """Guards have a read-position failure mode that generators do not, so
        only the guard probe reports label mass. `health=None` is a real state,
        not an oversight."""
        from internals_safety.measurements.causal import refusal_probe

        assert guard_verdict_probe(tiny_guard_model).health is not None
        assert refusal_probe(tiny_guard_model, [1]).health is None


class TestTheProbeIsNotOptional:
    def test_omitting_it_is_a_TypeError_not_a_silent_generator_default(self):
        """`probe` is keyword-only with no default ON PURPOSE.

        A default would let a guard run inherit the generating-model readout and
        renderer silently -- the optional-flag shape that has failed in this repo
        four times in one week (`strata`, `device`, `cross_rung_screen`,
        `bos_token_id`). The fix pattern is to make omission inexpressible.
        """
        with pytest.raises(TypeError):
            measure_causal_evidence(None, [], [], [], coefficient=1.0)  # type: ignore[call-arg]


class TestTheProbeAndTheReadPathAgree:
    def test_the_logits_reader_matches_VerdictReading_semantics(self, tiny_guard_model):  # noqa: F811
        """Two call shapes, one definition of a verdict -- pinned, not assumed.

        `read_verdicts` returns per-prompt readings; the causal path needs a
        scalar from the same tensor it hands to `kl_divergence`. Both must agree
        on which label wins, or a guard would appear to flip verdicts between the
        map and the intervention.
        """
        tokens = resolve_verdict_tokens(
            tiny_guard_model.tokenizer, tiny_guard_model.config, "CLASSIFY: x ANSWER:"
        )
        unsafe_wins = logits_with({tokens.unsafe_id: 0.6, tokens.safe_id: 0.2})
        safe_wins = logits_with({tokens.unsafe_id: 0.2, tokens.safe_id: 0.6})

        # VerdictReading.unsafe is `p_unsafe > p_safe`, i.e. renormalised > 0.5.
        assert verdict_probability(unsafe_wins, tokens) > 0.5
        assert verdict_probability(safe_wins, tokens) < 0.5


class TestBehaviourProbeShape:
    def test_a_probe_must_name_what_it_reads(self):
        """`name` reaches CausalEvidence.behaviour and therefore the run record."""
        probe = BehaviourProbe(
            name="something_specific",
            render_and_read=lambda prompts: torch.zeros(len(prompts), 1, 4),
            score=lambda logits: 0.0,
        )
        assert probe.name == "something_specific"
        with pytest.raises(TypeError):
            BehaviourProbe(render_and_read=lambda p: None, score=lambda lg: 0.0)  # type: ignore[call-arg]
