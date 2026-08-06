"""Causal licensing — does intervening on this direction change behaviour?

**The gap this closes.** Everything else in `measurements/` reads out; nothing
writes. So every licensing decision in this repo answers *"is there a real
separation here"* and none answers *"is this separation the thing"*. Those come
apart badly: raw character length separates the JBB harmful set from the benign
one at AUROC 0.654, every encoder preserves length, and the deployment probe was
found riding that signal on 12 of 15 rungs (`pilot_rebaseline.md` §5). A
permutation test cannot see it — the labels are genuinely predictable, just not
for the reason claimed.

**Method: Arditi et al., NeurIPS 2024** (arXiv 2406.11717),
`other_repos/refusal_direction/pipeline/submodules/select_direction.py`. Three
criteria, and the third is the one that matters here:

1. **Bypass** — ablate the direction; refusal on harmful prompts must drop.
2. **Induce** — add the direction; refusal on harmless prompts must rise.
3. **KL preservation** — ablating it must leave harmless behaviour intact.

A length direction fails (1) outright: removing "how long is this prompt" from
the residual stream does not make a model comply with a harmful request. That is
the screen, and we did not have one.

**Scope of this module.** Pure scoring and selection over evidence supplied by
the caller, so it is testable without weights. Producing the evidence needs
generation and a GPU; that is the runner's job, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from internals_safety.config import CausalLicenseConfig


def kl_divergence(baseline_logits: torch.Tensor, intervened_logits: torch.Tensor) -> float:
    """Mean KL(baseline || intervened) over a batch, at the final position.

    Direction matters: the baseline is the reference distribution, so this asks
    how much probability mass the intervention *moved away* from what the model
    would have done. Computed in float32 regardless of model dtype — a KL of
    ~0.1 is well inside bf16's noise at these magnitudes.
    """
    if baseline_logits.shape != intervened_logits.shape:
        raise ValueError(
            f"shape mismatch: {tuple(baseline_logits.shape)} vs {tuple(intervened_logits.shape)}"
        )
    baseline = torch.log_softmax(baseline_logits[:, -1, :].float(), dim=-1)
    intervened = torch.log_softmax(intervened_logits[:, -1, :].float(), dim=-1)
    per_row = (baseline.exp() * (baseline - intervened)).sum(dim=-1)
    return float(per_row.mean())


def refusal_probability(logits: torch.Tensor, refusal_token_ids: Sequence[int]) -> float:
    """Mean probability mass on refusal-opening tokens at the final position.

    A proxy for the behaviour, not the behaviour: it reads the first token of a
    would-be refusal rather than judging a full completion. That is what makes
    it cheap enough to run over every candidate direction. The judge layer
    (measurement #4) remains the arbiter for anything a paper reports.
    """
    if not refusal_token_ids:
        raise ValueError("no refusal tokens supplied — the score would be identically zero")
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    index = torch.tensor(sorted(set(refusal_token_ids)), device=probabilities.device)
    return float(probabilities.index_select(1, index).sum(dim=-1).mean())


@dataclass(frozen=True)
class CausalEvidence:
    """What intervening on one candidate direction did."""

    layer: int
    position: str
    # Refusal probability on HARMFUL prompts, before and after ablation.
    refusal_before: float
    refusal_after_ablation: float
    # Refusal probability on HARMLESS prompts, before and after addition.
    harmless_refusal_before: float
    harmless_refusal_after_addition: float
    # KL(baseline || ablated) on HARMLESS prompts.
    kl: float

    @property
    def bypass_score(self) -> float:
        """How much ablation reduced refusal. Higher is a better direction."""
        return self.refusal_before - self.refusal_after_ablation

    @property
    def induce_score(self) -> float:
        """How much addition increased refusal on harmless prompts."""
        return self.harmless_refusal_after_addition - self.harmless_refusal_before

    @property
    def bypass_fraction(self) -> float:
        """Bypass as a fraction of the refusal that was there to remove.

        Scale-free on purpose: the cipher band refuses at ~100% and the
        comprehension band does not, so an absolute drop would be a strict bar
        on one and a vacuous one on the other. Returns 0.0 when there was no
        refusal to begin with — nothing was released because nothing was held.
        """
        if self.refusal_before <= 0.0:
            return 0.0
        return self.bypass_score / self.refusal_before


def is_discarded(evidence: CausalEvidence, n_layers: int, config: CausalLicenseConfig) -> bool:
    """Their `filter_fn`, ported. True = this candidate is not eligible.

    NaN is discarded rather than propagated: an intervention that produced no
    number is not evidence of anything, and letting it through would let a
    sort put it first.
    """
    scores = (evidence.bypass_score, evidence.induce_score, evidence.kl)
    if any(value != value for value in scores):  # NaN
        return True
    if evidence.layer >= int(n_layers * (1.0 - config.prune_layer_percentage)):
        return True
    if evidence.kl > config.kl_threshold:
        return True
    if evidence.induce_score < config.induce_refusal_threshold:
        return True
    # OURS, not theirs — see CausalLicenseConfig.min_bypass_fraction. Their
    # filter passes the bypass score in and only NaN-checks it, because for
    # them it is the sort key over a pool known to contain a real direction.
    # Used as a GATE it has to bind, or a direction that releases nothing still
    # licenses.
    if evidence.bypass_fraction < config.min_bypass_fraction:
        return True
    return False


def select_direction(
    candidates: Sequence[CausalEvidence], n_layers: int, config: CausalLicenseConfig
) -> CausalEvidence:
    """The eligible candidate that bypasses refusal most.

    Raises when nothing survives rather than returning the least-bad option.
    An empty filter means the correlational probe found a direction that no
    intervention can act on — which is a FINDING (the separation is not the
    mechanism), and returning a direction anyway would erase it.
    """
    if not candidates:
        raise ValueError("no candidate directions supplied")
    eligible = [c for c in candidates if not is_discarded(c, n_layers, config)]
    if not eligible:
        raise ValueError(
            f"all {len(candidates)} candidate directions were filtered out — "
            "no direction passes the causal criteria; report this rather than relaxing them"
        )
    return max(eligible, key=lambda c: c.bypass_score)


def licenses(
    candidates: Sequence[CausalEvidence], n_layers: int, config: CausalLicenseConfig
) -> bool:
    """Whether ANY candidate survives — the upstream gate.

    Tri-state discipline (`instrument_layer.md` §1.5) applies one level up: a
    caller must distinguish "no direction is causally licensed" from "we did not
    run the causal test", and must not read the former as "harm is not
    represented".
    """
    return any(not is_discarded(c, n_layers, config) for c in candidates)
