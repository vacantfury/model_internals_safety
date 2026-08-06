"""Running the causal test — the write-side half of causal licensing.

`causal_license.py` scores and selects; **this module produces the evidence it
scores**, which needs a model, generation and a GPU. Same split as everywhere
else in this layer (`ability` runs, `ability_control` scores), and it is what
keeps the licensing rules testable without weights.

## The structural fact this module is shaped by

**Causal licensing is a MODEL-level gate, not a rung-level instrument**, and
that is why it plugs in where it does. Every instrument built before it — I1, I2,
I3, the four measurements — answers a question about one *rung*, so it runs
inside the family loop. This one asks *which direction may be used at all*: the
direction is fit on PLAIN harmful vs PLAIN harmless, it is the same direction for
every rung, and the answer gates the downstream reads rather than being one of
them. Wiring it behind `--instruments` inside `run_family` would have re-run the
identical computation once per rung and invited the reading that a rung has its
own causally-licensed direction. It runs once, between the plain captures and the
family loop.

## Method provenance

**Established, imported: Arditi et al., *Refusal in Language Models Is Mediated
by a Single Direction*, NeurIPS 2024** (arXiv 2406.11717, 914c; our copy is
`other_repos/refusal_direction`). Their `select_direction.py` is the source of
the three criteria and of the sweep-then-filter shape.

**Two deliberate divergences, stated because the provenance law requires
established-vs-ours on first mention:**

1. **We score refusal as a PROBABILITY; they score it as log-odds.** Their
   `refusal_score` returns `log(p) - log(1-p)` over the refusal tokens. Ours is
   the raw probability mass (`causal_license.refusal_probability`). The reason is
   not aesthetic: our gate adds `min_bypass_fraction`, a *fraction* of the
   refusal that was there to remove, and a fraction of a log-odds is undefined —
   the quantity is negative whenever refusal is under 0.5 and the ratio flips
   sign. On a probability the fraction means what it says. Consequence to carry:
   their `kl_threshold` and `induce_refusal_threshold` transfer unchanged (KL is
   on its own scale, and a threshold of 0.0 is scale-free), but **any bypass
   magnitude of ours is not comparable to a number in their paper.**
2. **Refusal tokens are resolved from strings through the live tokenizer**, not
   copied as ids. They hardcode per family (`LLAMA3_REFUSAL_TOKS = [40]  # 'I'`,
   `QWEN_REFUSAL_TOKS = [40, 2121]  # 'I', 'As'`). Copying ids across a
   checkpoint change is a silent failure — the guards layer was already bitten by
   a position assumption of exactly this kind — so `resolve_refusal_tokens` fails
   LOUD on a string that is not a single token rather than truncating it.

## What a caller must not read into a null result

`select_direction` raises when nothing survives the filter, and that is a
FINDING, not an error to route around: it means the correlational probe found a
separation that no intervention can act on. Under the tri-state discipline
(`instrument_layer.md` §1.5) "no direction is causally licensed" and "we did not
run the causal test" are different states and neither is "harm is not
represented".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from internals_safety.config import CausalLicenseConfig
from internals_safety.measurements.causal_license import (
    CausalEvidence,
    kl_divergence,
    refusal_probability,
)
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.interventions import ablate_direction, add_direction
from internals_safety.models.loader import LoadedModel
from internals_safety.probes.directions import Direction

# P1 — checked across the roster by `assert_distinct_questions`.
QUESTION = "does intervening on this direction change the model's refusal behaviour"
KIND: Kind = "causal"


def resolve_refusal_tokens(loaded: LoadedModel, openings: Sequence[str]) -> list[int]:
    """Token ids of the refusal openings, resolved through THIS tokenizer.

    Fails loud on a multi-token opening. A silent truncation would score the
    probability of a word's first fragment and call it refusal — the same class
    of defect as reading a verdict at the wrong position, which cost this repo a
    run once already.
    """
    if not openings:
        raise ValueError("no refusal openings configured — the score would be identically zero")
    ids: list[int] = []
    for opening in openings:
        encoded = loaded.tokenizer.encode(opening, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"refusal opening {opening!r} is {len(encoded)} tokens under this "
                f"tokenizer, not 1 — pick a single-token opening or the score "
                "measures a word fragment"
            )
        ids.append(int(encoded[0]))
    return ids


def viable_directions(candidates: Sequence[Direction]) -> list[Direction]:
    """Drop cells where the two classes coincide and no direction exists.

    `difference_in_means` returns a ZERO vector when `raw_norm` is 0 — it says so
    and leaves the decision to the caller — and `ablate_direction` then refuses
    to project out a zero direction. Filtering here rather than deep in a hook is
    the difference between "this cell has no direction" and a crash two layers
    down, which is how this was found (the end-to-end test, 2026-08-06).

    **Only exact degeneracy is filtered, and that is deliberate: no threshold is
    invented here.** A merely WEAK direction is not this function's business —
    the causal criteria are already the filter for weak directions, since a
    direction that is numerical noise will fail bypass and induce on its own
    evidence. Adding a norm cut would be a second, unfounded gate on the same
    question, and it would be a magic number.
    """
    return [candidate for candidate in candidates if float(candidate.vector.norm()) > 0.0]


def _final_logits(
    loaded: LoadedModel, prompts: Sequence[str], batch_size: int
) -> torch.Tensor:
    """[n_prompts, 1, vocab] — final-position logits, batched.

    Only the last position is kept: every consumer here (`refusal_probability`,
    `kl_divergence`) reads `[:, -1, :]`, and holding full sequences would be
    several GB for a number that reduces to one row per prompt.
    """
    rows: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        encoded = loaded.tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = loaded.model(**inputs).logits
        rows.append(logits[:, -1:, :].float().cpu())
    if not rows:
        raise ValueError("no prompts supplied")
    return torch.cat(rows, dim=0)


@dataclass(frozen=True)
class CausalRun:
    """Every candidate's evidence, plus the baselines they all share."""

    evidence: tuple[CausalEvidence, ...]
    refusal_before: float
    harmless_refusal_before: float
    n_harmful: int
    n_harmless: int


def forward_passes(n_candidates: int) -> int:
    """Passes over the prompt sets this measurement costs, for `--dry-run`.

    Two shared baselines plus three per candidate — ablate-on-harmful,
    add-on-harmless, ablate-on-harmless. Stated as a function rather than a
    comment because the approval gate prices the run from it, and a cost that
    lives only in prose is a cost nobody checks.
    """
    return 2 + 3 * n_candidates


def measure_causal_evidence(
    loaded: LoadedModel,
    candidates: Sequence[Direction],
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    refusal_token_ids: Sequence[int],
    coefficient: float,
    # plumbing(batch_size): throughput only — the logits read are per-prompt
    batch_size: int = 8,
) -> CausalRun:
    """Run ablation and addition for each candidate direction.

    The two baselines are computed ONCE and shared: they do not depend on the
    candidate, and recomputing them per direction would triple the cost of the
    cheapest part of the measurement for no information.

    `coefficient` scales the added direction. It is the one knob here and it is
    a real one — too small induces nothing on any direction, too large induces
    refusal on all of them, and either way the induce criterion stops binding.
    Its tuning path is the free negative control this repo already uses: sweep it
    and keep the value at which a matched-norm random direction still fails to
    induce (`causal_license.random_direction_null`).
    """
    if not candidates:
        raise ValueError("no candidate directions supplied")

    harmful_baseline = _final_logits(loaded, harmful_prompts, batch_size)
    harmless_baseline = _final_logits(loaded, harmless_prompts, batch_size)
    refusal_before = refusal_probability(harmful_baseline, refusal_token_ids)
    harmless_refusal_before = refusal_probability(harmless_baseline, refusal_token_ids)

    evidence: list[CausalEvidence] = []
    for candidate in candidates:
        # Necessity: ablate everywhere, and ask whether refusal on HARMFUL
        # prompts drops. Ablation spans all layers by design — the same
        # information is present at many sites, so a single-site ablation would
        # test a much weaker claim.
        with ablate_direction(loaded, candidate.vector):
            ablated_harmful = _final_logits(loaded, harmful_prompts, batch_size)
            ablated_harmless = _final_logits(loaded, harmless_prompts, batch_size)

        # Sufficiency: write the direction in at ONE layer and ask whether
        # refusal appears on HARMLESS prompts.
        with add_direction(loaded, candidate.vector, candidate.layer, coefficient):
            added_harmless = _final_logits(loaded, harmless_prompts, batch_size)

        evidence.append(
            CausalEvidence(
                layer=candidate.layer,
                position=candidate.position,
                refusal_before=refusal_before,
                refusal_after_ablation=refusal_probability(ablated_harmful, refusal_token_ids),
                harmless_refusal_before=harmless_refusal_before,
                harmless_refusal_after_addition=refusal_probability(
                    added_harmless, refusal_token_ids
                ),
                # KL on HARMLESS prompts: the criterion asks whether removing the
                # direction damaged ordinary behaviour, which is a question about
                # the benign distribution, never the harmful one.
                kl=kl_divergence(harmless_baseline, ablated_harmless),
            )
        )

    return CausalRun(
        evidence=tuple(evidence),
        refusal_before=refusal_before,
        harmless_refusal_before=harmless_refusal_before,
        n_harmful=len(harmful_prompts),
        n_harmless=len(harmless_prompts),
    )


def unmeasured_reading(config: CausalLicenseConfig, reason: str) -> Reading:
    """The causal test could not be run at all — `licensed=None`, never 0.0.

    **This is the distinction the whole tri-state discipline exists for, arriving
    one instrument further on.** "No direction is causally effective" is a
    measured negative and is reported by `reading` with `value=0.0`. "Every
    candidate cell was degenerate, so nothing could be intervened on" is not a
    result about the model — it is the instrument failing to read, and returning
    0.0 for it would assert that harm is causally unmediated on the strength of a
    cache that never produced a direction.
    """
    return Reading(
        instrument="causal_license",
        kind=KIND,
        value=float("nan"),
        operating_point="causal test not run — see detail.reason",
        licensed=None,
        detail={"reason": reason, "n_candidates": 0, "n_eligible": 0},
    )


def reading(
    run: CausalRun,
    config: CausalLicenseConfig,
    n_layers: int,
    *,
    null_margin: float | None = None,
    null_p_value: float | None = None,
    length_null_margin: float | None = None,
    # plumbing(n_degenerate): count of dropped candidates; zero means none were
    n_degenerate: int = 0,
) -> Reading:
    """The causal gate's condition-level verdict.

    **`kind` is `"causal"`, and P5 exists so this never merges with a
    correlational number into one "it works".** The value is the best eligible
    candidate's bypass score, or 0.0 when nothing survives the filter — which is
    reported as a licensed measurement of "no direction is causally effective",
    NOT as unmeasured. The two are different states and the tri-state discipline
    is about not confusing them.

    The negative control is the matched-norm random direction
    (`causal_license.random_direction_null`): without it, "steering worked" and
    "perturbing anything worked" are the same observation.
    """
    from internals_safety.measurements.causal_license import is_discarded

    eligible = [c for c in run.evidence if not is_discarded(c, n_layers, config)]
    best = max((c.bypass_score for c in eligible), default=0.0)
    return Reading(
        instrument="causal_license",
        kind=KIND,
        value=best,
        operating_point=(
            "largest refusal-probability drop under directional ablation among "
            "candidates surviving the Arditi filter (KL on harmless <= "
            f"{config.kl_threshold}, induced refusal >= {config.induce_refusal_threshold}, "
            f"last {config.prune_layer_percentage:.0%} of layers discarded, bypass "
            f"fraction >= {config.min_bypass_fraction}); refusal read as PROBABILITY "
            "mass on the configured openings, not as their log-odds"
        ),
        licensed=True,
        control_reading=None if null_margin is None else best - null_margin,
        control_margin=null_margin,
        length_null_margin=length_null_margin,
        # The candidate set is swept and filtered, then the maximum is taken —
        # a selection, and the null that covers it is the random-direction null
        # run over the SAME sweep. Honest only when that null was drawn.
        selection_inside_null=null_p_value is not None,
        detail={
            "n_candidates": len(run.evidence),
            "n_eligible": len(eligible),
            # Cells where the classes coincided and no direction exists. Reported
            # because a sweep that silently shrank is a sweep whose coverage the
            # reader cannot check.
            "n_degenerate": n_degenerate,
            "refusal_before": run.refusal_before,
            "harmless_refusal_before": run.harmless_refusal_before,
            "n_harmful": run.n_harmful,
            "n_harmless": run.n_harmless,
            "null_p_value": null_p_value,
        },
    )
