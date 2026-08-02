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
from internals_safety.probes.linear import ProbeResult, best_by_auroc, probe_sweep

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
