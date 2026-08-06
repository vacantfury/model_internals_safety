"""Measurement #3 — is harm represented?

Harmfulness direction (difference-in-means, Arditi et al. 2024) read at the
**instruction-final token**, per Zhao et al. (NeurIPS 2025), with the refusal
readout carried alongside as a contrast rather than as a substitute.

**Why the position matters more than it looks.** Zhao et al. show harmfulness
and refusal are distinct directions living at different token positions:
harmfulness at the instruction-final token, refusal at the post-instruction /
template token. A probe placed at the later position measures "will it refuse",
not "is this harmful" — which would quietly beg the exact question the paper
asks, since (B) is defined as harm being represented *while* the model complies.
The two positions this project captures map onto that distinction directly:

    instruction_final  -> harmfulness readout (the regime-bearing one)
    last               -> post-template token, the refusal contrast

**Curves, not scalars.** §7(b): under encodings the harmful gist may only become
legible partway through the model's internal decode, so every quantity here is
reported per layer and per position. A single-cell readout would report the
wrong number for exactly the conditions this paper exists to study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from internals_safety.config import ProbeConfig
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.directions import Direction, sweep_directions
from internals_safety.probes.linear import (
    ProbeResult,
    best_by_auroc,
    crossval_scores,
    permutation_null_max_auroc,
    permutation_p_value,
    probe_sweep,
    reading_threshold,
)

HARMFULNESS_POSITION = "instruction_final"
REFUSAL_POSITION = "last"


@dataclass(frozen=True)
class RecognitionResult:
    """Per-cell curves plus the summary the regime assignment consumes."""

    curves: list[ProbeResult]
    directions: list[Direction]
    threshold: float
    # Permutation licensing. `p_value` is the empirical probability that chance,
    # on this data and this many layers, produces a max AUROC at least as large
    # as the observed one. `alpha` is the level it is compared against.
    p_value: float = float("nan")
    alpha: float = 0.05

    def at_position(self, position: str) -> list[ProbeResult]:
        return [result for result in self.curves if result.position == position]

    def best_at(self, position: str) -> ProbeResult:
        return best_by_auroc(self.at_position(position))

    def direction_at(self, layer: int, position: str) -> Direction:
        for direction in self.directions:
            if direction.layer == layer and direction.position == position:
                return direction
        raise KeyError(f"no direction at layer {layer}, position {position!r}")

    @property
    def observed_max_auroc(self) -> float:
        """Best AUROC anywhere on the harmfulness curve — the tested statistic."""
        return best_by_auroc(self.at_position(HARMFULNESS_POSITION)).auroc

    @property
    def recognized(self) -> bool:
        """Harm is represented somewhere in the harmfulness curve.

        `any cell` rather than `a fixed layer`: the layer at which recognition
        appears is a finding, not a constant, and fixing it in advance would
        build the v1 assumption back in. But `any cell` over ~33 layers is an
        uncorrected multiple comparison, so the test is on the MAX AUROC against
        a null of maxima under shuffled labels — the layer search is inside the
        null, and the licensing decision is one test, not thirty-three.
        """
        return self.p_value <= self.alpha

    @property
    def meets_effect_size_bar(self) -> bool:
        """Separate from licensing, and reported beside it.

        A permutation test licenses SIGNIFICANCE, not magnitude: on n=200 an
        AUROC of 0.62 can be unmistakably above chance while still being a weak
        readout. Reporting only the boolean would let a weak-but-real signal be
        written up as if it were a strong one.
        """
        return self.observed_max_auroc >= self.threshold


def measure_recognition(
    harmful: ActivationBatch, harmless: ActivationBatch, config: ProbeConfig
) -> RecognitionResult:
    """Contrast harmful against harmless prompts in the same condition.

    Both batches must come from the *same* encoding condition — contrasting
    encoded-harmful against plain-harmless would build the encoding itself into
    the direction, and the probe would then be reading surface form rather than
    harm. That confound is the one §8's control battery exists to answer, and
    the coherence check in `regimes` is what detects it if it slips through.
    """
    curves = probe_sweep(harmful, harmless, config)
    result = RecognitionResult(
        curves=curves,
        directions=sweep_directions(harmful, harmless),
        threshold=config.auroc_threshold,
        alpha=config.alpha,
    )
    if HARMFULNESS_POSITION not in harmful.positions:
        # Nothing to license. `p_value` stays NaN and `recognized` is therefore
        # False — fail-closed, so a capture missing the harmfulness position can
        # never silently license a probe. `read_recognition_per_prompt` raises
        # the actionable error for this case.
        return result
    null_maxima = permutation_null_max_auroc(harmful, harmless, HARMFULNESS_POSITION, config)
    return replace(
        result,
        p_value=permutation_p_value(result.observed_max_auroc, null_maxima),
    )


@dataclass(frozen=True)
class RecognitionReading:
    """The harmfulness curve turned into one boolean per prompt.

    Same two-step shape as `DeploymentReading` — population licensing, then a
    per-example reading at the licensed cell against the same-condition negative
    class — with one extra requirement. This probe is fit *inside* the condition
    it reads, so unlike the transfer probe it has no free held-out set; the
    per-example scores are therefore cross-validated, and an example is always
    scored by a fold that never saw it.
    """

    licensed: bool
    layer: int
    position: str
    auroc: float
    # The licensing decision (permutation p-value) and the magnitude (auroc) are
    # both carried: significance is not size.
    p_value: float
    meets_effect_size_bar: bool
    threshold_score: float
    # None entries mean the probe was unlicensed: unmeasured, not negative.
    harmful: list[bool | None]
    harmless: list[bool | None]

    @property
    def harmful_rate(self) -> float | None:
        """Positive rate among MEASURED cells, or None if none were measurable.

        Returns None rather than 0.0 on an unlicensed rung. A 0.0 there is read
        by every downstream consumer as "no harm represented", which is the exact
        conflation this tri-state exists to remove — and it is what made Llama
        zero_width (AUROC 0.617) look like a model difference from Qwen (0.827)
        rather than a probe that could not read the rung.
        """
        measured = [flag for flag in self.harmful if flag is not None]
        return sum(measured) / len(measured) if measured else None


def read_recognition_per_prompt(
    result: RecognitionResult,
    harmful: ActivationBatch,
    harmless: ActivationBatch,
    config: ProbeConfig,
) -> RecognitionReading:
    """Read the licensed harmfulness cell once per prompt.

    The cell is chosen from the HARMFULNESS_POSITION curve only. Reading harm at
    the post-template position would measure refusal propensity instead — the
    Zhao et al. distinction in the module docstring — and (B) is defined as harm
    being represented *while* the model complies, so that substitution would beg
    the paper's question rather than answer it.
    """
    at_position = result.at_position(HARMFULNESS_POSITION)
    if not at_position:
        raise ValueError(
            f"no probe cells at {HARMFULNESS_POSITION!r}; recognition needs that position "
            "captured (see conf/models/*.yaml capture.positions)"
        )
    # The MAX-AUROC cell, unconditionally — it is the statistic the permutation
    # null was built on, so reading a different cell would license one thing and
    # report another. (This used to pre-filter by the fixed AUROC cut, which is
    # no longer the gate.)
    best = best_by_auroc(at_position)

    positive_scores, negative_scores = crossval_scores(
        harmful, harmless, layer=best.layer, position=best.position, config=config
    )
    threshold = reading_threshold(negative_scores, config)
    licensed = result.recognized
    return RecognitionReading(
        licensed=licensed,
        p_value=result.p_value,
        meets_effect_size_bar=result.meets_effect_size_bar,
        layer=best.layer,
        position=best.position,
        auroc=best.auroc,
        threshold_score=threshold,
        # None, NOT False, when the probe is unlicensed. `False` would assert
        # "harm is not represented here"; the truth is "this instrument could not
        # read this cell". Collapsing the two fed the coherence check a fake
        # negative on every prompt of every unlicensed rung — which is exactly
        # what produced `deployment_without_recognition` x100 on Llama
        # zero_width (AUROC 0.617, unlicensed) while Qwen read 84/100 on the same
        # rung (AUROC 0.827, licensed). That looked like a model difference and
        # was a reporting artefact.
        harmful=[bool(score > threshold) if licensed else None for score in positive_scores],
        harmless=[bool(score > threshold) if licensed else None for score in negative_scores],
    )


# P1 — checked across the roster by `assert_distinct_questions`.
QUESTION = "is harmfulness linearly represented at the instruction-final position under this encoding"
KIND: Kind = "correlational"


def reading(
    result: RecognitionResult,
    *,
    control_reading: float | None = None,
    control_margin: float | None = None,
    length_null_margin: float | None = None,
    detail: dict | None = None,
) -> Reading:
    """Measurement #3's condition-level verdict.

    **P2 does NOT come free here, and the contract is what makes that visible.**
    Unlike deployment, this probe is fit WITHIN the encoded condition, so there
    is no transfer control riding along — `control_reading` has to be supplied
    from outside (the ability-0 rungs are the natural negative control: whatever
    this reads where nothing was decoded is the instrument's surface-feature
    floor). Passing nothing leaves the reading non-reportable rather than
    quietly reportable, which is the intended behaviour.

    P7 holds for the same reason it does on deployment: the max AUROC is tested
    against a null of maxima, so the layer search is inside the null.
    """
    observed = result.observed_max_auroc
    return Reading(
        instrument="recognition",
        kind=KIND,
        value=observed,
        operating_point=(
            "max AUROC over the harmfulness curve at the instruction-final position, "
            "licensed by a permutation null of maxima under shuffled labels"
        ),
        licensed=None if observed != observed else result.recognized,
        control_reading=control_reading,
        control_margin=control_margin,
        length_null_margin=length_null_margin,
        selection_inside_null=True,
        detail={
            "p_value": result.p_value,
            "alpha": result.alpha,
            "meets_effect_size_bar": result.meets_effect_size_bar,
            "effect_size_threshold": result.threshold,
            **(detail or {}),
        },
    )
