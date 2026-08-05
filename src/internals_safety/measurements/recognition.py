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

from dataclasses import dataclass

from internals_safety.config import ProbeConfig
from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.directions import Direction, sweep_directions
from internals_safety.probes.linear import (
    ProbeResult,
    best_by_auroc,
    crossval_scores,
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
    def recognized(self) -> bool:
        """Harm is represented somewhere in the harmfulness curve.

        `any cell` rather than `a fixed layer`: the layer at which recognition
        appears is a finding, not a constant, and fixing it in advance would
        build the v1 assumption back in.
        """
        return any(
            result.reads_signal(self.threshold)
            for result in self.at_position(HARMFULNESS_POSITION)
        )


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
    return RecognitionResult(
        curves=probe_sweep(harmful, harmless, config),
        directions=sweep_directions(harmful, harmless),
        threshold=config.auroc_threshold,
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
    signalling = [cell for cell in at_position if cell.reads_signal(result.threshold)]
    best = best_by_auroc(signalling or at_position)

    positive_scores, negative_scores = crossval_scores(
        harmful, harmless, layer=best.layer, position=best.position, config=config
    )
    threshold = reading_threshold(negative_scores, config)
    licensed = result.recognized
    return RecognitionReading(
        licensed=licensed,
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
