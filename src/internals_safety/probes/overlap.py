"""Projection-score distribution overlap — H4's pre-registered metric.

H4 says cross-encoding safety fine-tuning generalizes to held-out families in
proportion to representational overlap between families. **Which** overlap
metric is pre-registered here rather than chosen after seeing results, and the
choice is not cosmetic:

Direction cosine is the obvious metric and is expected to be *degenerate*. Wang
et al. (NeurIPS 2025, arXiv 2505.17306) show refusal directions are near-parallel
across languages — an English-extracted vector bypasses refusal elsewhere — and
Aziz et al. find the low-resource failure is a threshold/projection shift with
the direction intact. If that universality holds across encodings too, every
pairwise cosine sits near 1 with no variance to correlate generalization
against, and a metric with no variance cannot support or refute H4.

So the registered metric is the overlap of the *projection-score distributions*:
how far each family's harmfulness projections sit relative to the refusal
threshold. Cosine is kept as a secondary check in `directions.cosine_similarity`,
and a measured near-degeneracy there is itself a reportable finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _as_array(scores: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        return scores.detach().to("cpu", torch.float32).numpy()
    return np.asarray(scores, dtype=np.float64)


def decision_threshold(
    harmful: torch.Tensor | np.ndarray, harmless: torch.Tensor | np.ndarray
) -> float:
    """The projection value that best separates harmful from harmless.

    Youden's J over the pooled scores — the operating point a refusal decision
    would sit at if it read this direction alone. Distances are measured from
    here, so it is estimated once on the reference (plain-text) condition and
    then held fixed; re-estimating it per family would define away exactly the
    threshold shift H4 is about.
    """
    positive = _as_array(harmful)
    negative = _as_array(harmless)
    candidates = np.unique(np.concatenate([positive, negative]))
    if candidates.size == 0:
        raise ValueError("no scores to threshold")

    true_positive = (positive[None, :] >= candidates[:, None]).mean(axis=1)
    false_positive = (negative[None, :] >= candidates[:, None]).mean(axis=1)
    return float(candidates[int(np.argmax(true_positive - false_positive))])


def overlap_coefficient(
    first: torch.Tensor | np.ndarray, second: torch.Tensor | np.ndarray, n_bins: int = 50
) -> float:
    """Overlapping coefficient (OVL) of two score distributions, in [0, 1].

    Histogram-based on shared bins spanning both samples: the summed minimum of
    the two normalised densities. 1.0 = identical distributions, 0.0 = disjoint
    support. Chosen over a KDE because it makes no bandwidth assumption, and
    bandwidth would be an unjustified knob in a pre-registered metric.
    """
    a = _as_array(first)
    b = _as_array(second)
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")

    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if high == low:
        return 1.0

    edges = np.linspace(low, high, n_bins + 1)
    density_a, _ = np.histogram(a, bins=edges, density=False)
    density_b, _ = np.histogram(b, bins=edges, density=False)
    return float(np.minimum(density_a / a.size, density_b / b.size).sum())


@dataclass(frozen=True)
class ProjectionSummary:
    """One family's position relative to the refusal threshold."""

    family: str
    n: int
    mean: float
    std: float
    mean_distance_to_threshold: float
    fraction_above_threshold: float

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"{self.family:<20} n={self.n:<4} mean={self.mean:+.3f} "
            f"d(threshold)={self.mean_distance_to_threshold:+.3f} "
            f"above={self.fraction_above_threshold:.2f}"
        )


def summarize_projections(
    family: str, scores: torch.Tensor | np.ndarray, threshold: float
) -> ProjectionSummary:
    values = _as_array(scores)
    return ProjectionSummary(
        family=family,
        n=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        # Signed on purpose: the sign says which side of the refusal boundary
        # the family sits on, which is the direction of the predicted shift.
        mean_distance_to_threshold=float(values.mean() - threshold),
        fraction_above_threshold=float((values >= threshold).mean()),
    )


def pairwise_overlap(
    scores_by_family: dict[str, torch.Tensor | np.ndarray], n_bins: int = 50
) -> dict[tuple[str, str], float]:
    """Every unordered family pair -> overlap. The regressor for H4."""
    families = list(scores_by_family)
    return {
        (first, second): overlap_coefficient(
            scores_by_family[first], scores_by_family[second], n_bins
        )
        for index, first in enumerate(families)
        for second in families[index + 1 :]
    }
