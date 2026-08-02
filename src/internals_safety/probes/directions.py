"""Difference-in-means directions (Arditi et al. 2024).

Established method, ported not invented: the direction separating two contrast
sets is the difference of their mean activations at a given (layer, position).
Aziz et al. (2026) use the same construction on the language axis, which is
what makes our encoding-axis results directly comparable to theirs.

Two things this module deliberately does *not* do. It does not pick "the"
layer — §7 requires per-layer x per-position curves rather than a scalar,
because under encodings the harmful gist may only become legible partway
through the model's internal decode. And it does not claim causality: a
direction that separates the classes shows harm is *represented*, not that the
representation *governs* behaviour. That claim needs the ablation and
activation-addition work at step 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from internals_safety.models.capture import ActivationBatch


@dataclass(frozen=True)
class Direction:
    """A unit vector in residual-stream space at one (layer, position)."""

    vector: torch.Tensor
    layer: int
    position: str
    n_positive: int
    n_negative: int
    # Norm of the raw difference before normalisation. Reported because a
    # near-zero difference means the two classes sit on top of each other —
    # the unit vector would then be numerical noise pointing somewhere
    # arbitrary, which a cosine against another direction would happily report
    # as a meaningful angle.
    raw_norm: float

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """[..., d_model] -> [...] scalar projections onto this direction."""
        return activations.to(self.vector.dtype) @ self.vector


def difference_in_means(
    positive: ActivationBatch, negative: ActivationBatch, layer: int, position: str
) -> Direction:
    """Direction pointing from the negative class toward the positive class."""
    positive_cell = positive.select(layer, position)
    negative_cell = negative.select(layer, position)
    if positive_cell.shape[0] == 0 or negative_cell.shape[0] == 0:
        raise ValueError("both contrast sets must be non-empty")

    difference = positive_cell.mean(dim=0) - negative_cell.mean(dim=0)
    raw_norm = float(difference.norm())
    # Guard against dividing by ~0 for a layer where the classes do not
    # separate at all; the caller sees raw_norm and can discard the cell.
    vector = difference / raw_norm if raw_norm > 0 else torch.zeros_like(difference)
    return Direction(
        vector=vector,
        layer=layer,
        position=position,
        n_positive=int(positive_cell.shape[0]),
        n_negative=int(negative_cell.shape[0]),
        raw_norm=raw_norm,
    )


def sweep_directions(
    positive: ActivationBatch, negative: ActivationBatch
) -> list[Direction]:
    """One direction per (layer, position) — the curve, not a scalar."""
    if positive.layers != negative.layers or positive.positions != negative.positions:
        raise ValueError("contrast sets must be captured at the same layers and positions")
    return [
        difference_in_means(positive, negative, layer, position)
        for layer in positive.layers
        for position in positive.positions
    ]


def projection_scores(
    batch: ActivationBatch, direction: Direction
) -> torch.Tensor:
    """Project a batch onto a direction at that direction's own (layer, position).

    These scores are H4's pre-registered currency: the hypothesis is about the
    *distribution* of projections relative to the refusal threshold, not about
    the angle between directions (which Wang et al. 2505.17306 predict will be
    near-degenerate across surface forms).
    """
    return direction.project(batch.select(direction.layer, direction.position))


def cosine_similarity(first: Direction, second: Direction) -> float:
    """Secondary check only.

    Retained because a *measured* near-degeneracy is itself reportable — it
    would say encodings move a family along a shared harmfulness axis rather
    than rotating it — but H4 is pre-registered on projection overlap precisely
    because cosine is expected to carry no variance.
    """
    return float(torch.dot(first.vector, second.vector))
