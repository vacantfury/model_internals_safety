"""I3 — entropy dynamics.

**Question:** does the model's *uncertainty profile across layers* mark encoded
harmful input, with no classifier trained on our labels?

**Why it earns a slot rather than being a fourth read-out variant.** It fits
nothing on our labels, so the route that broke the supervised probe is closed:
a length signal cannot be *learned* here because nothing is learned. That makes
its failure mode genuinely independent of I1 and I2, and an instrument whose
failure mode is uncorrelated with the others is worth more than a correlated one
however good. This is P1 in the build plan's design principles.

**Provenance — ⚠️ TIER (C), unrefereed, 0 citations:** arXiv 2606.25182, *What
Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics*; related
readout TriLens (arXiv 2606.01033) uses per-layer lens entropy for hallucination
detection. Under the house rule these motivate the instrument and may not drive a
claim. The instrument is cheap and independent, which is its own justification;
the papers set its priority.

**⚠️ Label-free is not length-free, and the distinction is easy to lose.** The
build plan flags this explicitly. Entropy here is computed at ONE captured
position, so sequence length does not enter by aggregation the way it would over
a whole sequence — but it still enters through content, because longer prompts
differ in content and content drives uncertainty. Length-matched evaluation
applies to this instrument exactly as it does to the probes. Do not read
"label-free" as "confound-free"; it closes one route, not the class.

**Inherited asymmetry from the lens seam.** `models/lens.py` documents it: a lens
NULL is not evidence of no representation, only a lens POSITIVE is evidence of a
readable one (arXiv 2604.02608 **(C)** — function vectors steer without being
lens-readable). A flat entropy profile therefore licenses no conclusion about
whether the model represented anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import numpy as np
from sklearn.metrics import roc_auc_score

from internals_safety.config import ProbeConfig
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.capture import ActivationBatch
from internals_safety.models.lens import unembed
from internals_safety.models.loader import LoadedModel

# P1 — checked across the roster by `assert_distinct_questions`.
QUESTION = "does the model's layer-wise uncertainty profile mark this input, with nothing fitted on our labels"
KIND: Kind = "correlational"

# The statistics an `EntropyProfile` reduces to. Named here so a reading's
# operating point cannot describe one statistic while the value came from
# another.
STATISTICS = ("minimum", "resolution_layer", "total_drop")


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in NATS of the distribution over the last dimension.

    Computed from log-softmax rather than from probabilities, so a near-zero
    probability contributes `p * log p -> 0` instead of `0 * -inf -> nan`.
    """
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probabilities.exp() * log_probabilities).sum(dim=-1)


@dataclass(frozen=True)
class EntropyProfile:
    """Per-layer lens uncertainty for one prompt set at one position."""

    position: str
    layers: list[int]
    # [n_prompts, n_layers], in nats.
    entropies: torch.Tensor

    @property
    def n_prompts(self) -> int:
        return int(self.entropies.shape[0])

    @property
    def minimum(self) -> torch.Tensor:
        """Each prompt's lowest uncertainty anywhere in the stack."""
        return self.entropies.min(dim=1).values

    def resolution_layer(self) -> torch.Tensor:
        """Index into `layers` of each prompt's steepest single-layer DROP.

        Where uncertainty collapses is where the model commits. Reported as an
        index into `layers` rather than a raw layer number so it stays correct
        when a run captures a subset of layers.
        """
        if self.entropies.shape[1] < 2:
            raise ValueError("a resolution layer needs at least two captured layers")
        drops = self.entropies[:, :-1] - self.entropies[:, 1:]
        return drops.argmax(dim=1)

    def total_drop(self) -> torch.Tensor:
        """First-layer entropy minus last-layer entropy, per prompt."""
        return self.entropies[:, 0] - self.entropies[:, -1]


def measure_entropy_dynamics(
    loaded: LoadedModel,
    batch: ActivationBatch,
    position: str,
    # plumbing(batch_size): throughput only — every read is per-prompt
    batch_size: int = 8,
) -> EntropyProfile:
    """Lens entropy at every captured layer, for one position.

    Chunked over prompts because the intermediate is [chunk, vocab] per layer
    and vocabularies are ~128k — the full tensor for a 200-prompt rung across 32
    layers would be several GB for a statistic that reduces to one float per
    cell.
    """
    if position not in batch.positions:
        raise ValueError(f"position {position!r} not captured; have {batch.positions}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    index = batch.positions.index(position)
    states = batch.tensor[:, :, index, :]

    per_layer: list[torch.Tensor] = []
    for layer_position in range(states.shape[1]):
        chunks = [
            entropy(unembed(loaded, states[start : start + batch_size, layer_position, :])).cpu()
            for start in range(0, states.shape[0], batch_size)
        ]
        per_layer.append(torch.cat(chunks) if chunks else torch.empty(0))

    return EntropyProfile(
        position=position,
        layers=list(batch.layers),
        entropies=torch.stack(per_layer, dim=1),
    )


@dataclass(frozen=True)
class EntropySeparation:
    """How well one entropy statistic separates the two classes."""

    statistic: str
    auroc: float
    p_value: float
    alpha: float

    @property
    def licensed(self) -> bool:
        return self.p_value <= self.alpha


def _statistic(profile: EntropyProfile, statistic: str) -> torch.Tensor:
    if statistic == "minimum":
        return profile.minimum
    if statistic == "resolution_layer":
        return profile.resolution_layer().float()
    if statistic == "total_drop":
        return profile.total_drop()
    raise ValueError(f"unknown statistic {statistic!r}; have {list(STATISTICS)}")


def separation(
    positive: EntropyProfile,
    negative: EntropyProfile,
    statistic: str,
    config: ProbeConfig,
) -> EntropySeparation:
    """AUROC of one statistic between the classes, licensed by permutation.

    **Nothing is fitted, so the null is over the STATISTIC rather than over a
    model.** That is the whole cost argument for I3: shuffling labels on a 1-D
    vector and recomputing an AUROC is microseconds, where the probe nulls are
    hundreds of logistic fits. Its independence from the probes' failure mode is
    what earns it a roster slot; being cheap is why it is never worth skipping.
    """
    scores = torch.cat([_statistic(positive, statistic), _statistic(negative, statistic)])
    labels = np.concatenate([np.ones(positive.n_prompts), np.zeros(negative.n_prompts)])
    values = scores.detach().cpu().numpy()
    observed = float(roc_auc_score(labels, values))

    generator = np.random.default_rng(config.seed)
    null = np.array([
        float(roc_auc_score(generator.permutation(labels), values))
        for _ in range(config.n_permutations)
    ])
    # Two-sided in effect: an entropy statistic can separate in either direction
    # (a marked input may resolve EARLIER or later), so the null is on |AUROC-0.5|.
    centred_null = np.abs(null - 0.5)
    p_value = float((np.sum(centred_null >= abs(observed - 0.5)) + 1) / (len(null) + 1))
    return EntropySeparation(
        statistic=statistic, auroc=observed, p_value=p_value, alpha=config.alpha
    )


def reading(
    *,
    statistic: str,
    auroc: float,
    licensed: bool | None,
    control_auroc: float | None,
    control_margin: float | None,
    length_null_margin: float | None = None,
    selection_inside_null: bool = False,
    detail: dict | None = None,
) -> Reading:
    """This instrument's condition-level verdict.

    Same shape as I2's, and required for the same reason: **label-free is not
    control-free.** Nothing here is fitted on our labels, which closes the route
    that broke the supervised probe — but the separation statistic is still read
    against a corpus whose harmful half is longer than its benign half, so the
    length null applies unchanged. `length_null_margin=None` therefore withholds
    the reading rather than passing it.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; have {list(STATISTICS)}")
    return Reading(
        instrument="entropy_dynamics",
        kind=KIND,
        value=auroc,
        operating_point=(
            f"AUROC of the per-prompt lens-entropy {statistic} at one captured "
            "position, read against the ability-0 negative-control floor"
        ),
        licensed=licensed,
        control_reading=control_auroc,
        control_margin=control_margin,
        length_null_margin=length_null_margin,
        selection_inside_null=selection_inside_null,
        detail={"statistic": statistic, **(detail or {})},
    )
