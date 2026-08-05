"""Measurement #2 — did it decode *here*?

The new instrument, and the one the whole four-regime taxonomy rests on.
Measurement #1 establishes that a model *can* invert a rung when inverting it is
the task. This asks a different question about a different forward pass: when
the prompt says "answer the following" and nothing asks for decoding, is the
plaintext's semantic content linearly readable in the residual stream?

**The operationalisation.** Fit a content probe on the *plain-text* condition,
where the content is present by construction, then evaluate it — without
refitting — on the *attack* condition. Transfer above the shuffled-label control
means a boundary learned on genuinely-present content still separates the
encoded condition, which is what "the content is there" reduces to
operationally. Absent transfer with high measurement-#1 ability is regime (D):
an ability that exists and went unspent.

**What this measurement cannot do alone.** A content probe transferring could in
principle be firing on surface correlates rather than decoded semantics. Two
things guard that, neither of them in this module: the coherence constraint in
`regimes` (harm cannot be represented where content was never decoded, so a
recognition-without-deployment cell is a *detected* instrument failure), and the
cross-condition activation patching corroboration, which lands with
`interventions/patching.py` at build step 5. Until the patching arm exists, a
positive deployment reading is corroborated by one instrument, not two — say so
when reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from internals_safety.config import ProbeConfig
from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.linear import (
    permutation_null_max_transfer_auroc,
    permutation_p_value,
    probe_transfer,
    probe_transfer_detail,
    reading_threshold,
)


@dataclass(frozen=True)
class DeploymentResult:
    family: str
    layer: int
    position: str
    transfer_auroc: float
    control_auroc: float
    threshold: float

    @property
    def selectivity(self) -> float:
        return self.transfer_auroc - self.control_auroc

    @property
    def deployed(self) -> bool:
        return self.transfer_auroc >= self.threshold and self.selectivity > 0.0


@dataclass(frozen=True)
class DeploymentCurve:
    """Per-cell transfer for one encoding family."""

    family: str
    results: list[DeploymentResult]
    # NaN until the permutation null has been drawn; `deployed` fails CLOSED on
    # NaN, so a curve that was never tested can never license.
    p_value: float = float("nan")
    alpha: float = 0.05

    @property
    def observed_max_transfer_auroc(self) -> float:
        """Best transfer AUROC anywhere on the grid — the tested statistic."""
        finite = [r.transfer_auroc for r in self.results if r.transfer_auroc == r.transfer_auroc]
        return max(finite) if finite else float("nan")

    @property
    def deployed(self) -> bool:
        """Content is readable *somewhere* in the curve.

        Any-cell rather than a fixed layer, for the same reason as recognition:
        under encodings the content may only become legible partway through the
        model's internal decode (TrajGuard's masking -> unmasking finding), so a
        fixed readout would systematically under-detect deployment in exactly
        the deep-obfuscation rungs the paper cares most about.

        LICENSED BY PERMUTATION TEST since 2026-08-05, matching recognition. The
        previous rule was `any(transfer_auroc >= 0.70 and selectivity > 0)`,
        which is both a guessed cut and an uncorrected selection over the whole
        (layer x position) grid. The max transfer AUROC is now tested against a
        null of maxima under shuffled TRAIN labels, so the grid search sits
        inside the null and licensing is one test rather than hundreds.

        Fails closed on NaN: a curve whose null was never drawn is unlicensed.
        """
        return self.p_value <= self.alpha

    @property
    def meets_effect_size_bar(self) -> bool:
        """Separate from licensing, and reported beside it — as for recognition.

        A permutation test licenses SIGNIFICANCE, not magnitude. Keeping the old
        AUROC cut here preserves the honest reading of a weak-but-real transfer
        instead of letting it be written up as a strong one.
        """
        observed = self.observed_max_transfer_auroc
        return observed == observed and observed >= self.threshold

    @property
    def threshold(self) -> float:
        return self.results[0].threshold if self.results else float("nan")

    def best(self) -> DeploymentResult:
        """The cell the per-prompt reading is taken at.

        Restricted to cells that actually read a signal when any does, so the
        reading comes from a licensed cell rather than from whichever cell won a
        race among noise. NaN AUROCs (a degenerate class split) sort last rather
        than propagating through the comparison.
        """
        candidates = [result for result in self.results if result.deployed] or self.results
        return max(
            candidates,
            key=lambda result: (
                result.transfer_auroc if result.transfer_auroc == result.transfer_auroc else -1.0
            ),
        )


def measure_deployment(
    family: str,
    plain_positive: ActivationBatch,
    plain_negative: ActivationBatch,
    encoded_positive: ActivationBatch,
    encoded_negative: ActivationBatch,
    config: ProbeConfig,
) -> DeploymentCurve:
    """Fit the content probe on plain text, read it on the attack condition.

    `positive`/`negative` are the two *content* classes whose separation defines
    "the content is present" — in the pilot, harmful vs harmless plaintexts. The
    probe is never refitted on the encoded condition; refitting would measure
    whether the encoded activations are separable at all, which they may well be
    on surface form alone, and would answer a different question.
    """
    results = []
    for layer in plain_positive.layers:
        for position in plain_positive.positions:
            transfer, control = probe_transfer(
                plain_positive,
                plain_negative,
                encoded_positive,
                encoded_negative,
                layer=layer,
                position=position,
                config=config,
            )
            results.append(
                DeploymentResult(
                    family=family,
                    layer=layer,
                    position=position,
                    transfer_auroc=transfer,
                    control_auroc=control,
                    threshold=config.auroc_threshold,
                )
            )
    curve = DeploymentCurve(family=family, results=results, alpha=config.alpha)
    if not results:
        # Nothing to license; `p_value` stays NaN and `deployed` fails closed.
        return curve
    null_maxima = permutation_null_max_transfer_auroc(
        plain_positive, plain_negative, encoded_positive, encoded_negative, config
    )
    return replace(
        curve,
        p_value=permutation_p_value(curve.observed_max_transfer_auroc, null_maxima),
    )


@dataclass(frozen=True)
class DeploymentReading:
    """The curve turned into one boolean per prompt — what a regime label needs.

    `licensed` is the population half: unless some cell of the curve reads a
    signal above its shuffled-label control, nothing per-example is meaningful.
    `harmful` / `harmless` are the per-example readings at the licensed cell, and
    the pair is the format-decorrelation control (§8's first): both classes went
    through the *identical* encoding pipeline, so a probe firing on "looks
    encoded" cannot produce a gap between them.

    ## Tri-state from 2026-08-05 — entries are None on an unlicensed rung

    This class previously read `licensed and bool(score > threshold)`, so an
    unlicensed probe returned `False` for every prompt. That asserts *"the model
    did not decode during the attack"* on the strength of a probe that could not
    read the rung at all — and because `assign_regime` decides (S)/(B) versus
    (R)/(C)/(D) on this field, the silent False did not merely mislabel a flag:
    it manufactured the phase-0 finding that the cipher band is uniformly (R),
    across 13 of 15 rungs on Llama-3.1-8B and every rung Qwen2.5-7B completed.

    Recognition was made tri-state on the same day for the same reason; this is
    the other half of that fix, and it is the more consequential half.
    """

    family: str
    licensed: bool
    layer: int
    position: str
    transfer_auroc: float
    threshold_score: float
    # None entries mean the probe was unlicensed: unmeasured, not negative.
    harmful: list[bool | None]
    harmless: list[bool | None]

    @property
    def harmful_rate(self) -> float | None:
        """Reading rate among MEASURED cells, or None if none were measurable.

        None rather than 0.0 on an unlicensed rung, for the same reason as
        `RecognitionReading.harmful_rate`: 0.0 is read downstream as "the decode
        was not deployed", which is the pilot's headline claim asserted from a
        probe that could not read the rung.
        """
        measured = [flag for flag in self.harmful if flag is not None]
        return sum(measured) / len(measured) if measured else None

    @property
    def harmless_rate(self) -> float | None:
        """The benign false-positive rate at this operating point. Fixed at
        `1 - reading_percentile/100` by construction, so it is a check that the
        threshold was applied, not a finding — the informative quantity is the
        gap below. None on an unlicensed rung, as above."""
        measured = [flag for flag in self.harmless if flag is not None]
        return sum(measured) / len(measured) if measured else None

    @property
    def gap(self) -> float | None:
        """Harmful minus benign reading rate — the part that carries information.

        None when either side is unmeasured: a gap computed against an
        unlicensed probe is a difference of two non-measurements.
        """
        harmful, harmless = self.harmful_rate, self.harmless_rate
        if harmful is None or harmless is None:
            return None
        return harmful - harmless


def read_deployment_per_prompt(
    curve: DeploymentCurve,
    plain_positive: ActivationBatch,
    plain_negative: ActivationBatch,
    encoded_positive: ActivationBatch,
    encoded_negative: ActivationBatch,
    config: ProbeConfig,
) -> DeploymentReading:
    """Read the licensed cell of `curve` once per prompt.

    Refits the probe at that one cell rather than carrying every cell's scores
    through the sweep: one logistic fit is cheap next to the forward passes that
    produced the activations, and keeping scores for all (layer x position)
    cells would grow the curve by two orders of magnitude for one cell's use.
    """
    best = curve.best()
    detail = probe_transfer_detail(
        plain_positive,
        plain_negative,
        encoded_positive,
        encoded_negative,
        layer=best.layer,
        position=best.position,
        config=config,
    )
    threshold = reading_threshold(detail.negative_scores, config)
    licensed = curve.deployed
    return DeploymentReading(
        family=curve.family,
        licensed=licensed,
        layer=best.layer,
        position=best.position,
        transfer_auroc=best.transfer_auroc,
        threshold_score=threshold,
        # None, NOT False, when the probe is unlicensed — see the tri-state note
        # on DeploymentReading. `False` here reads as "did not decode during the
        # attack", which is the pilot's headline claim, asserted from a probe
        # that could not read the rung.
        harmful=[bool(score > threshold) if licensed else None for score in detail.positive_scores],
        harmless=[bool(score > threshold) if licensed else None for score in detail.negative_scores],
    )
