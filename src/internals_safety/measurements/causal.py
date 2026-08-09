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
    RandomDirectionNull,
    kl_divergence,
    matched_norm_random_direction,
    random_direction_null,
    refusal_probability,
)
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.interventions import ablate_direction, add_direction
from internals_safety.models.loader import LoadedModel, prepare_prompts
from internals_safety.probes.directions import Direction, difference_in_means

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
    # ⚠️ RENDERED through the chat template, fixed 2026-08-06. This tokenised the
    # bare instruction, while the DIRECTION being intervened on was fit on
    # activations captured from rendered prompts (`capture_or_load` renders). So
    # the intervention ran on an input the instruction-tuned model never sees —
    # and refusal is largely a chat-format behaviour, which is precisely the
    # quantity this module reads. Found while building `attribution.py`, whose
    # position offsets are counted against the rendered sequence and so could not
    # have been written the old way at all.
    rendered = [prompt.text for prompt in prepare_prompts(loaded, prompts, positions=[])]
    for start in range(0, len(rendered), batch_size):
        chunk = list(rendered[start : start + batch_size])
        encoded = loaded.tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = loaded.model(**inputs).logits
        rows.append(logits[:, -1:, :].float().cpu())
    if not rows:
        raise ValueError("no prompts supplied")
    return torch.cat(rows, dim=0)


from internals_safety.config import GuardConfig
from internals_safety.guards import (
    VerdictTokens,
    label_mass_from_logits,
    render_guard_prompt,
    resolve_verdict_tokens,
    verdict_context,
    verdict_probability,
)


@dataclass(frozen=True)
class BehaviourProbe:
    """How one model KIND is rendered and scored. The seam that makes this
    measurement work on a content guard as well as on a generating model.

    Both halves have to be injected, and the rendering half is the one that is
    easy to miss. `_final_logits` renders through the model's own chat template,
    which is correct for a generating model and WRONG for a guard: a guard's
    verdict lives after its verdict prefix, so the same function reading the
    same final position answers a different question depending on what was
    rendered. Injecting only the scoring function would have produced a guard
    measurement that ran, returned plausible numbers, and read the logits one
    position too early — the defect this repo already paid for once
    (`phase1_map.md` §5: a guard that blocks 98% reported at 0.00).

    `name` is not decoration: it is written into `CausalEvidence.behaviour` and
    therefore into the run record, so a reader never has to infer which quantity
    a number is.
    """

    name: str
    render_and_read: Callable[[Sequence[str]], torch.Tensor]
    score: Callable[[torch.Tensor], float]
    # Optional per-kind sanity number recorded alongside the result (guards use
    # label mass at the verdict position). None when the kind has no such check.
    health: Callable[[torch.Tensor], float] | None = None


def refusal_probe(
    loaded: LoadedModel,
    refusal_token_ids: Sequence[int],
    # plumbing(batch_size): throughput only — the logits read are per-prompt
    batch_size: int = 8,
) -> BehaviourProbe:
    """The generating-model probe: chat-template rendering, P(refusal opening)."""
    return BehaviourProbe(
        name="refusal_opening",
        render_and_read=lambda prompts: _final_logits(loaded, prompts, batch_size),
        score=lambda logits: refusal_probability(logits, refusal_token_ids),
    )


def guard_verdict_probe(loaded: LoadedModel, batch_size: int | None = None) -> BehaviourProbe:
    """The content-guard probe: guard rendering PLUS verdict prefix, P(unsafe).

    The rendering is the whole reason this exists as a separate probe. It goes
    through `render_guard_prompt` and then `verdict_context`, which appends the
    guard's own verdict prefix — so the final position the causal machinery reads
    is the position where the guard is about to emit its label. Reusing the
    generating-model renderer would put that read one position early on a guard
    that emits two newlines first, and the resulting numbers would look ordinary
    (`phase1_map.md` §5).

    The token pair is resolved ONCE, against the first rendered context, and then
    held fixed for every intervention. That is required for the comparison to
    mean anything: P(unsafe) before and after ablation is only a difference if it
    is the same token both times, which is the property `_check_id_stability`
    exists to defend on the read path.
    """
    config = loaded.config
    if not isinstance(config, GuardConfig):
        raise TypeError(
            f"guard_verdict_probe needs a GuardConfig, got {type(config).__name__}; "
            "load the guard via config.load_guard_config"
        )
    size = batch_size or config.capture_batch_size

    # Resolved from the FIRST REAL payload on first use, not from a dummy string
    # and not per call. `resolve_verdict_tokens` takes the rendered prompt as
    # context because a label's first token can depend on what precedes it, so a
    # placeholder could resolve a different pair than the run actually reads. And
    # resolving once matters more than it looks: P(unsafe) before and after
    # ablation is a difference only if it is the same token both times.
    resolved: dict[str, VerdictTokens] = {}

    def render_and_read(payloads: Sequence[str]) -> torch.Tensor:
        rendered = [render_guard_prompt(loaded.tokenizer, config, payload) for payload in payloads]
        contexts = [verdict_context(config, text) for text in rendered]
        if "tokens" not in resolved and rendered:
            resolved["tokens"] = resolve_verdict_tokens(loaded.tokenizer, config, rendered[0])
        rows: list[torch.Tensor] = []
        for start in range(0, len(contexts), size):
            chunk = contexts[start : start + size]
            encoded = loaded.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            )
            inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = loaded.model(**inputs).logits
            rows.append(logits[:, -1:, :].float().cpu())
        if not rows:
            raise ValueError("no payloads supplied")
        return torch.cat(rows, dim=0)

    def _tokens() -> VerdictTokens:
        if "tokens" not in resolved:
            raise RuntimeError(
                "verdict tokens are resolved on the first render_and_read call; "
                "score() was reached without one, which means the probe was used "
                "out of order"
            )
        return resolved["tokens"]

    return BehaviourProbe(
        name="guard_unsafe_verdict",
        render_and_read=render_and_read,
        score=lambda logits: verdict_probability(logits, _tokens()),
        health=lambda logits: label_mass_from_logits(logits, _tokens()),
    )


@dataclass(frozen=True)
class CausalRun:
    """Every candidate's evidence, plus the baselines they all share."""

    evidence: tuple[CausalEvidence, ...]
    behaviour: str
    behaviour_before: float
    harmless_behaviour_before: float
    n_harmful: int
    n_harmless: int
    # The probe's health number on the harmful baseline, or None if the probe
    # has none. Recorded rather than asserted: the threshold is the caller's.
    health: float | None = None


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
    *,
    probe: BehaviourProbe,
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

    harmful_baseline = probe.render_and_read(harmful_prompts)
    harmless_baseline = probe.render_and_read(harmless_prompts)
    behaviour_before = probe.score(harmful_baseline)
    harmless_behaviour_before = probe.score(harmless_baseline)

    evidence: list[CausalEvidence] = []
    for candidate in candidates:
        # Necessity: ablate everywhere, and ask whether refusal on HARMFUL
        # prompts drops. Ablation spans all layers by design — the same
        # information is present at many sites, so a single-site ablation would
        # test a much weaker claim.
        with ablate_direction(loaded, candidate.vector):
            ablated_harmful = probe.render_and_read(harmful_prompts)
            ablated_harmless = probe.render_and_read(harmless_prompts)

        # Sufficiency: write the direction in at ONE layer and ask whether the
        # behaviour appears on HARMLESS prompts.
        with add_direction(loaded, candidate.vector, candidate.layer, coefficient):
            added_harmless = probe.render_and_read(harmless_prompts)

        evidence.append(
            CausalEvidence(
                layer=candidate.layer,
                position=candidate.position,
                behaviour=probe.name,
                behaviour_before=behaviour_before,
                behaviour_after_ablation=probe.score(ablated_harmful),
                harmless_behaviour_before=harmless_behaviour_before,
                harmless_behaviour_after_addition=probe.score(added_harmless),
                # KL on HARMLESS prompts: the criterion asks whether removing the
                # direction damaged ordinary behaviour, which is a question about
                # the benign distribution, never the harmful one.
                kl=kl_divergence(harmless_baseline, ablated_harmless),
            )
        )

    return CausalRun(
        evidence=tuple(evidence),
        behaviour=probe.name,
        behaviour_before=behaviour_before,
        harmless_behaviour_before=harmless_behaviour_before,
        n_harmful=len(harmful_prompts),
        n_harmless=len(harmless_prompts),
        health=probe.health(harmful_baseline) if probe.health else None,
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
    # The statistic the null was drawn ON — the RAW best candidate's bypass
    # score, which is not `value` whenever the filter empties. Required for the
    # control fields to mean anything; see the incoherence note below.
    null_observed: float | None = None,
    length_null_margin: float | None = None,
    # plumbing(n_degenerate): count of dropped candidates; zero means none were
    n_degenerate: int = 0,
) -> Reading:
    """The causal gate's condition-level verdict.

    **`kind` is `"causal"`, and P5 exists so this never merges with a
    correlational number into one "it works".** The value is the best eligible
    candidate's bypass score.

    ⚠️ **Rewritten 2026-08-09 after this function produced an uninterpretable
    reading on all three runs it had ever done.** It previously returned
    `value=0.0` with `licensed=True` whenever the filter emptied, and its
    docstring called that "a licensed measurement of 'no direction is causally
    effective'". The data refuted that reading: on Llama-3.1-8B-Instruct (runs
    9007219 / 9008632) the filter emptied while the best candidate bypassed
    refusal by **0.737** over the matched-norm null, on a `refusal_before` of
    0.949 — a bypass fraction near 0.78 against a 0.5 bar. A direction that
    removes three quarters of a model's refusal is not "no direction is
    causally effective"; it was discarded on a SECONDARY criterion, and the
    record could not say which. Three defects, all fixed here:

    1. **The zero was silent** — the same 0.0 meant "nothing acted" and
       "everything acted and was filtered on KL". Now `detail.attrition` names
       the criterion per candidate and `detail.max_bypass_fraction` says whether
       anything acted at all. Same shape as `deployment=False` meaning
       unmeasured, one instrument further on.
    2. **The control fields crossed two candidates.** `control_reading` was
       `value - null_margin`, but the null is drawn on the raw best candidate
       while `value` comes from the eligible set. With an empty filter that
       subtraction mixed a 0.0 from one selection with a margin from another and
       printed `-0.737` as "what the control read". `null_observed` closes it:
       the control reading is now the null's own mean.
    3. **The claim was always `positive`.** So an empty filter was withheld for
       lacking a length null (P3) when the honest reason is that a null claim
       needs SENSITIVITY — proof this gate can fire when a direction does exist.
       We have no such control, and now the record says so.

    The negative control is the matched-norm random direction
    (`causal_license.random_direction_null`): without it, "steering worked" and
    "perturbing anything worked" are the same observation.
    """
    from collections import Counter

    from internals_safety.measurements.causal_license import discard_reason

    verdicts = [(c, discard_reason(c, n_layers, config)) for c in run.evidence]
    eligible = [c for c, why in verdicts if why is None]
    attrition = Counter(why for _, why in verdicts if why is not None)
    best = max((c.bypass_score for c in eligible), default=0.0)
    # Over ALL candidates, not just eligible ones: this is the number that says
    # whether any intervention acted, independent of the auxiliary filter.
    max_bypass_fraction = max((c.bypass_fraction for c in run.evidence), default=0.0)

    # THE state the rewrite exists for: every candidate rejected, while at least
    # one of them cleared the bypass bar. The gate's operational answer is
    # unchanged (no direction is licensed for downstream use), but the SCIENTIFIC
    # question — is this behaviour causally mediated by the direction — was not
    # measured, because the filter removed the evidence rather than the evidence
    # coming back empty. `licensed=None`, never 0.0.
    filtered_out_a_working_direction = (
        not eligible and max_bypass_fraction >= config.min_bypass_fraction
    )
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
        licensed=None if filtered_out_a_working_direction else True,
        # The null's OWN mean, not `value` minus a margin drawn on a different
        # candidate. `null_observed - null_margin` is by construction the mean of
        # the matched-norm ensemble.
        control_reading=(
            None
            if null_margin is None or null_observed is None
            else null_observed - null_margin
        ),
        control_margin=null_margin,
        length_null_margin=length_null_margin,
        # An empty filter asserts a negative, and the contract's null route is
        # the one that demands sensitivity rather than a length null. Declared
        # from the eligible set, never from the value — deriving direction from
        # the number is the self-licensing dodge `contract.py` names.
        claim="positive" if eligible else "null",
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
            # WHICH criterion rejected each candidate. The single field that
            # makes an empty filter diagnosable instead of a bare zero.
            "attrition": dict(attrition),
            # Did ANY intervention act, filter aside? Reported next to attrition
            # because the pair is the whole diagnosis: `bypass` attrition with a
            # low max means nothing acted; `kl` attrition with a high max means
            # directions acted and were rejected for collateral damage.
            "max_bypass_fraction": max_bypass_fraction,
            "filtered_out_a_working_direction": filtered_out_a_working_direction,
            # Every candidate, compactly. 13 rows of small floats, and it is what
            # lets a null be re-diagnosed offline instead of costing a queue
            # cycle — which is exactly what the first three runs cost.
            "candidates": [
                {
                    "layer": c.layer,
                    "position": c.position,
                    "bypass_score": c.bypass_score,
                    "bypass_fraction": c.bypass_fraction,
                    "induce_score": c.induce_score,
                    "kl": c.kl,
                    "discarded_for": why,
                }
                for c, why in verdicts
            ],
            "behaviour": run.behaviour,
            "behaviour_before": run.behaviour_before,
            "harmless_behaviour_before": run.harmless_behaviour_before,
            "n_harmful": run.n_harmful,
            "n_harmless": run.n_harmless,
            "null_p_value": null_p_value,
            "null_observed": null_observed,
        },
    )


# FAIL-SAFE DEFAULT — the live value is `causal_license.max_sweep_layers`.
MAX_CAUSAL_LAYERS = 8  # config: measurements.causal_license.max_sweep_layers


def causal_candidate_cells(
    batch, prune_layer_percentage: float, max_layers: int = MAX_CAUSAL_LAYERS
) -> list[tuple[int, str]]:
    """(layer, position) cells the gate sweeps, capped and pruned up front.

    The filter discards the last `prune_layer_percentage` of layers anyway, so
    sweeping them buys candidates that cannot survive. Pruning HERE as well as in
    the filter is deliberate — the filter is the correctness boundary and this is
    the cost boundary, and a caller reading only one of them still gets a correct
    answer.

    The stride is DERIVED from the depth so that at most `max_layers` are swept,
    rather than fixed. A fixed stride is the same cost knob only when the model
    has the depth it was chosen for; at 3 layers it selected layer 0 alone, whose
    `resid_pre` is the raw embedding before any computation.
    """
    eligible = list(batch.layers[: int(len(batch.layers) * (1.0 - prune_layer_percentage))])
    if not eligible:
        return []
    stride = max(1, -(-len(eligible) // max_layers))
    return [(layer, position) for layer in eligible[::stride] for position in batch.positions]


def run_causal_gate(
    loaded: LoadedModel,
    plain_harmful_batch,
    plain_harmless_batch,
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    *,
    probe: BehaviourProbe,
    measurements,
    batch_size: int,
) -> Reading:
    """The upstream causal gate — which direction may be used at all.

    ⚠️ **Moved out of `scripts/phase0_regime_map.py` on 2026-08-09, and the move
    IS the point.** It lived in the AS-5 entrypoint, so the guard entrypoint
    could not reach it, and the only alternatives were to copy ~100 lines or to
    ship a guard paper whose central quantity has no causal test. Both are the
    failures this repo already documents — dual truth, and an instrument built
    but unreachable. `pipeline_architecture.md` §3.5's selection rule decides it:
    the spine holds anything whose absence in ONE script would be a defect.

    TODO 28, from Arditi et al. (NeurIPS 2024): probe licensing is correlational,
    and a permutation test structurally cannot distinguish a real separation from
    the RIGHT separation. A direction separating harmful from benign by character
    length passes it — and one did, on 12 of 15 rungs. It fails this: removing
    "how long is this prompt" from the residual stream does not make a model
    comply, and does not make a guard stop flagging.

    `probe` is keyword-only with no default so the gate cannot silently score a
    generating model's refusal on a content guard: the two read different tokens
    at positions produced by different renderers.
    """
    config = measurements.causal_license
    cells = causal_candidate_cells(
        plain_harmful_batch, config.prune_layer_percentage, config.max_sweep_layers
    )
    swept = [
        difference_in_means(plain_harmful_batch, plain_harmless_batch, layer, position)
        for layer, position in cells
    ]
    # A cell where the two classes coincide yields a ZERO vector, which cannot be
    # ablated. Dropping those here keeps the failure a coverage number rather
    # than an exception raised inside a forward hook.
    candidates = viable_directions(swept)
    n_degenerate = len(swept) - len(candidates)
    if not candidates:
        return unmeasured_reading(
            config,
            f"all {len(swept)} candidate cells were degenerate — the contrast sets "
            "do not separate anywhere in the swept range, so no direction exists "
            "to intervene on",
        )

    run = measure_causal_evidence(
        loaded,
        candidates,
        harmful_prompts,
        harmless_prompts,
        probe=probe,
        coefficient=config.addition_coefficient,
        batch_size=batch_size,
    )

    # The negative control: the SAME intervention on matched-norm random
    # directions at the best candidate's own cell, so the comparison holds the
    # site fixed and moves only the direction.
    null: RandomDirectionNull | None = None
    if config.n_random_directions > 0 and run.evidence:
        best = max(run.evidence, key=lambda evidence: evidence.bypass_score)
        anchor = next(c for c in candidates if (c.layer, c.position) == (best.layer, best.position))
        generator = torch.Generator(device="cpu").manual_seed(measurements.probes.seed)
        # ONE call with every random direction, not one call each: they all sit
        # at the anchor's cell, so a single call shares the two baselines instead
        # of recomputing them N times.
        random_run = measure_causal_evidence(
            loaded,
            [
                Direction(
                    vector=matched_norm_random_direction(anchor.vector, generator),
                    layer=anchor.layer,
                    position=anchor.position,
                    n_positive=anchor.n_positive,
                    n_negative=anchor.n_negative,
                    raw_norm=anchor.raw_norm,
                )
                for _ in range(config.n_random_directions)
            ],
            harmful_prompts,
            harmless_prompts,
            probe=probe,
            coefficient=config.addition_coefficient,
            batch_size=batch_size,
        )
        null = random_direction_null(best, random_run.evidence, alpha=measurements.probes.alpha)

    return reading(
        run,
        config,
        n_layers=len(plain_harmful_batch.layers),
        null_margin=None if null is None else null.margin,
        null_p_value=None if null is None else null.p_value,
        # `null.observed` is the RAW best candidate's bypass score, which is the
        # quantity the ensemble was compared against. Passing it is what keeps
        # `control_reading` from crossing two different selections.
        null_observed=None if null is None else null.observed,
        n_degenerate=n_degenerate,
    )
