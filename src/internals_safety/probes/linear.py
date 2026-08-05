"""Logistic probes with AUROC, and the selectivity control that makes them mean
something.

A probe's AUROC on its own is close to uninterpretable: a high-dimensional
linear classifier on a few hundred examples can fit a lot. The standard fix
(Hewitt & Liang's control-task selectivity, applied here to the safety setting)
is to run the *identical* probe against shuffled labels and report the gap. A
0.95 AUROC with a 0.90 control is a probe memorising the split; a 0.95 with a
0.52 control is a signal.

`probe_transfer` is the instrument measurement #2 is built on: fit on one
condition, evaluate on another, never refit. That is what "is plaintext content
readable in the attack forward pass" reduces to operationally.

## Population AUROC licenses the instrument; per-example scores read a cell

A regime label is defined per (model, family, **prompt**) — `s1_idea_check.md`
§7 — but an AUROC is a property of a population, so it cannot label one prompt.
The two-step reading below is what closes that gap, and both halves are needed:

1. **Licensing (population).** Unless the (layer, position) cell reads a signal
   above its shuffled-label control, no per-example reading from it means
   anything, and every prompt reads negative. This is what keeps a noisy probe
   from distributing cells across regimes at chance.
2. **Reading (per example).** At the licensed cell, an example's decision value
   is compared against the **negative class in the same condition**
   (`reading_threshold`), not against the boundary at zero. Encoding a prompt
   and wrapping it in an attack template shifts *both* classes together, and a
   common shift moves every raw decision value while leaving the ranking — and
   therefore the AUROC that licensed the cell — untouched. Thresholding at zero
   would let that shift decide the label; thresholding against the concurrent
   negatives cannot, because the confound is common-mode.

The percentile is an operating point, not an estimate: at the 50th, the benign
control's own positive rate is 50% by construction. What carries information is
the *gap* between the harmful and benign rates, which the pilot reports.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from threadpoolctl import threadpool_limits

from internals_safety.config import ProbeConfig
from internals_safety.models.capture import ActivationBatch

_T = TypeVar("_T")


def single_threaded_blas(function: Callable[..., _T]) -> Callable[..., _T]:
    """Pin BLAS to one thread for the duration of `function`.

    ## This is not a micro-optimisation; it is the difference between a run that
    ## finishes and a run that hits the wall.

    Every probe fit here is a *small* problem — 140 x 4096, a few dozen lbfgs
    iterations, each dominated by matrix-vector products. Multi-threaded BLAS
    parallelises those by splitting work across threads and synchronising at
    every call. At this size the synchronisation costs far more than the
    arithmetic saves, so throughput collapses as threads are added.

    Measured on the cluster's real cached activations (Llama-3.1-8B,
    plain-harmful vs plain-harmless, one layer, shuffled labels — the exact
    workload the permutation null runs):

        OMP_NUM_THREADS=1     118 ms/fit     ->  12.5 min per rung
        OMP_NUM_THREADS=4   3,500 ms/fit     ->   6.2 h  per rung
        OMP_NUM_THREADS=8   3,680 ms/fit     ->   6.5 h  per rung

    **Single-threaded is 31x faster than the 8 threads an 8-CPU allocation gets
    by default.** The 2026-08-05 comprehension-band sweep was killed at the 8 h
    wall having finished one rung of seven, with `sacct` reporting 2d14h of CPU
    time over an 8 h wall — all eight cores saturated, doing almost nothing.
    That is the signature this decorator exists to prevent.

    Why a decorator on the entry points rather than `_fit`: `crossval_scores`
    fits through sklearn's own `cross_val_predict`, which never calls `_fit`, so
    a choke-point wrapper would miss it. Pinning at the entry points also pays
    the (small) threadpoolctl overhead once per sweep instead of once per fit.

    Scope is deliberately narrow — `user_api="blas"`, and only around probe
    fitting. Torch's GPU work and its own CPU thread pool are untouched.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> _T:
        with threadpool_limits(limits=1, user_api="blas"):
            return function(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class ProbeResult:
    layer: int
    position: str
    auroc: float
    control_auroc: float
    n_train: int
    n_test: int

    @property
    def selectivity(self) -> float:
        """AUROC above the shuffled-label control. This is the reportable
        quantity; raw AUROC without it is not evidence."""
        return self.auroc - self.control_auroc

    def reads_signal(self, threshold: float) -> bool:
        return self.auroc >= threshold and self.selectivity > 0.0


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu", torch.float32).numpy()


def _fit(features: np.ndarray, labels: np.ndarray, config: ProbeConfig) -> LogisticRegression:
    model = LogisticRegression(
        C=config.regularization_c, max_iter=config.max_iter, random_state=config.seed
    )
    model.fit(features, labels)
    return model


def _auroc(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, model.decision_function(features)))


@single_threaded_blas
def fit_probe(
    features: torch.Tensor, labels: torch.Tensor, config: ProbeConfig
) -> tuple[LogisticRegression, float, float]:
    """Fit one probe with a held-out split; return it with its AUROC and control.

    The control refits on shuffled labels using the *same* split, so any gap is
    attributable to the labels rather than to the split or the sample size.
    """
    x = _to_numpy(features)
    y = _to_numpy(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_fraction, random_state=config.seed, stratify=y
    )
    model = _fit(x_train, y_train, config)
    auroc = _auroc(model, x_test, y_test)

    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(y_train)
    control = _fit(x_train, shuffled, config)
    control_auroc = _auroc(control, x_test, y_test)

    return model, auroc, control_auroc


@single_threaded_blas
def probe_sweep(
    positive: ActivationBatch, negative: ActivationBatch, config: ProbeConfig
) -> list[ProbeResult]:
    """One probe per (layer, position). Curves, not a single readout — §7(b)."""
    if positive.layers != negative.layers or positive.positions != negative.positions:
        raise ValueError("both classes must be captured at the same layers and positions")

    labels = torch.cat(
        [
            torch.ones(positive.tensor.shape[0]),
            torch.zeros(negative.tensor.shape[0]),
        ]
    )
    results = []
    for layer in positive.layers:
        for position in positive.positions:
            features = torch.cat(
                [positive.select(layer, position), negative.select(layer, position)]
            )
            _, auroc, control = fit_probe(features, labels, config)
            n_test = int(round(len(labels) * config.test_fraction))
            results.append(
                ProbeResult(
                    layer=layer,
                    position=position,
                    auroc=auroc,
                    control_auroc=control,
                    n_train=len(labels) - n_test,
                    n_test=n_test,
                )
            )
    return results


@dataclass(frozen=True)
class TransferScores:
    """A transfer probe's population metrics *and* its per-example decision values.

    The scores are out-of-sample by construction: the probe never sees the test
    condition, so every encoded example is held out without needing a split.
    """

    transfer_auroc: float
    control_auroc: float
    positive_scores: np.ndarray
    negative_scores: np.ndarray


@single_threaded_blas
def probe_transfer_detail(
    train_positive: ActivationBatch,
    train_negative: ActivationBatch,
    test_positive: ActivationBatch,
    test_negative: ActivationBatch,
    layer: int,
    position: str,
    config: ProbeConfig,
) -> TransferScores:
    """Fit on one condition, evaluate on another; keep the per-example scores.

    Never refits on the test condition — the question is whether a decision
    boundary learned where the content is *plainly* present still separates the
    condition where it would have to have been decoded.
    """
    train_features = _to_numpy(
        torch.cat([train_positive.select(layer, position), train_negative.select(layer, position)])
    )
    train_labels = np.concatenate(
        [
            np.ones(train_positive.tensor.shape[0], dtype=int),
            np.zeros(train_negative.tensor.shape[0], dtype=int),
        ]
    )
    positive_features = _to_numpy(test_positive.select(layer, position))
    negative_features = _to_numpy(test_negative.select(layer, position))
    test_features = np.concatenate([positive_features, negative_features])
    test_labels = np.concatenate(
        [
            np.ones(len(positive_features), dtype=int),
            np.zeros(len(negative_features), dtype=int),
        ]
    )

    model = _fit(train_features, train_labels, config)
    rng = np.random.default_rng(config.seed)
    control_model = _fit(train_features, rng.permutation(train_labels), config)

    return TransferScores(
        transfer_auroc=_auroc(model, test_features, test_labels),
        control_auroc=_auroc(control_model, test_features, test_labels),
        positive_scores=model.decision_function(positive_features),
        negative_scores=model.decision_function(negative_features),
    )


@single_threaded_blas
def probe_transfer(
    train_positive: ActivationBatch,
    train_negative: ActivationBatch,
    test_positive: ActivationBatch,
    test_negative: ActivationBatch,
    layer: int,
    position: str,
    config: ProbeConfig,
) -> tuple[float, float]:
    """`probe_transfer_detail`'s two population metrics: (transfer, control)."""
    detail = probe_transfer_detail(
        train_positive, train_negative, test_positive, test_negative, layer, position, config
    )
    return detail.transfer_auroc, detail.control_auroc


@single_threaded_blas
def crossval_scores(
    positive: ActivationBatch,
    negative: ActivationBatch,
    layer: int,
    position: str,
    config: ProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-sample decision values for *every* example, via stratified k-fold.

    Needed where the probe is fit inside the condition it reads — measurement #3
    contrasts harmful against harmless within one encoding condition, so unlike
    the transfer probe there is no free held-out set. Each example is scored by a
    probe that never saw it; without that, a per-prompt recognition reading would
    be reporting the probe's memory of its own training set.
    """
    features = np.concatenate(
        [
            _to_numpy(positive.select(layer, position)),
            _to_numpy(negative.select(layer, position)),
        ]
    )
    n_positive = positive.tensor.shape[0]
    labels = np.concatenate(
        [np.ones(n_positive, dtype=int), np.zeros(negative.tensor.shape[0], dtype=int)]
    )

    folds = min(config.cv_folds, int(labels.sum()), int(len(labels) - labels.sum()))
    if folds < 2:
        raise ValueError(
            f"cross-validated scoring needs >=2 examples per class, got "
            f"{int(labels.sum())} positive and {int(len(labels) - labels.sum())} negative"
        )
    scores = cross_val_predict(
        LogisticRegression(
            C=config.regularization_c, max_iter=config.max_iter, random_state=config.seed
        ),
        features,
        labels,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed),
        method="decision_function",
    )
    return scores[:n_positive], scores[n_positive:]


def reading_threshold(negative_scores: np.ndarray, config: ProbeConfig) -> float:
    """The score a positive example must beat to read positive *for that cell*.

    Taken from the negative class **in the same condition**, so that the encoding
    and its attack-template wrapper — which shift both classes together — cannot
    decide the reading. See the module docstring.
    """
    if len(negative_scores) == 0:
        raise ValueError("no negative examples to set a reading threshold from")
    return float(np.percentile(negative_scores, config.reading_percentile))


def best_by_auroc(results: list[ProbeResult]) -> ProbeResult:
    """The peak of the curve. Reported *alongside* the curve, never instead of
    it: picking the best cell post hoc is a selection effect, and the layer at
    which the signal peaks is itself a finding."""
    return max(results, key=lambda result: (result.auroc if result.auroc == result.auroc else -1))


@single_threaded_blas
def permutation_null_max_auroc(
    positive: ActivationBatch,
    negative: ActivationBatch,
    position: str,
    config: ProbeConfig,
) -> np.ndarray:
    """Null distribution of the MAX AUROC across layers, under shuffled labels.

    Why the max, and why this exists at all.

    Licensing was a fixed `auroc >= 0.70` cut applied cell-by-cell, with
    `recognized = any(cell reads signal)` over every layer. Both halves of that
    are wrong in opposite directions:

    - **The number was a guess.** `conf/measurements.yaml` always said so, and
      named this as its tuning path: "the point is the gap between the real task
      and a shuffled-label control, not the raw number." A cut of 0.70 on n=200
      is far above chance, so it silently discards real-but-weak signal — which
      is exactly the state Llama `zero_width` was left in (AUROC 0.617,
      unlicensed, on a rung with deployment 200/200).
    - **`any()` over ~33 layers is an uncorrected multiple comparison.** Simply
      lowering the cut to a per-cell significance test would make it worse: at
      alpha=0.05 over 33 cells, roughly 1.6 cells license by chance alone, so
      almost every rung would "read signal".

    Permuting the labels and recording the max AUROC over the whole curve solves
    both at once. The observed max is compared against the distribution of maxima
    that chance produces on the *same* data, sample size and layer count, so the
    selection over layers is inside the null rather than unaccounted for. That is
    family-wise error control by construction.

    Cost — the first figure here was WRONG and cost a sweep. It read "~13 ms per
    fit, minutes across the whole ladder", measured on synthetic well-conditioned
    data rather than on activations. On the real cached tensors a shuffled-label
    fit is **118 ms** single-threaded, so this null is ~12.5 min per rung and
    ~1.5 h across a 7-rung band — real, budgetable, and roughly 30x the estimate.

    Unpinned it is far worse: see `single_threaded_blas`, without which the same
    null costs 6.5 h per rung and will not fit inside an 8 h wall. This function
    is decorated accordingly; do not remove it to "parallelise".
    """
    if positive.layers != negative.layers:
        raise ValueError("both classes must be captured at the same layers")

    labels = np.concatenate(
        [np.ones(positive.tensor.shape[0], dtype=int), np.zeros(negative.tensor.shape[0], dtype=int)]
    )
    # Features are gathered once; only the labels are permuted, so the null holds
    # the representation fixed and varies nothing but the label assignment.
    per_layer = [
        _to_numpy(torch.cat([positive.select(layer, position), negative.select(layer, position)]))
        for layer in positive.layers
    ]

    rng = np.random.default_rng(config.seed)
    maxima = np.empty(config.n_permutations, dtype=float)
    for draw in range(config.n_permutations):
        shuffled = rng.permutation(labels)
        best = -np.inf
        for features in per_layer:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                shuffled,
                test_size=config.test_fraction,
                random_state=config.seed,
                stratify=shuffled,
            )
            auroc = _auroc(_fit(x_train, y_train, config), x_test, y_test)
            if not np.isnan(auroc):
                best = max(best, auroc)
        maxima[draw] = best
    return maxima


@single_threaded_blas
def permutation_null_max_transfer_auroc(
    train_positive: ActivationBatch,
    train_negative: ActivationBatch,
    test_positive: ActivationBatch,
    test_negative: ActivationBatch,
    config: ProbeConfig,
) -> np.ndarray:
    """Null distribution of the MAX TRANSFER AUROC over the (layer x position) grid.

    The deployment twin of `permutation_null_max_auroc`, added 2026-08-05 so
    deployment is licensed the same way recognition is.

    Why it was needed. Deployment licensing was `transfer_auroc >= 0.70 and
    selectivity > 0`, applied per cell with `deployed = any(cell)` over the whole
    grid — the identical pair of errors the recognition version documents above,
    except worse, because deployment sweeps layers AND positions, so the
    uncorrected selection is over a larger grid. It is also the axis every regime
    label is decided on, so an under-powered cut there does not merely weaken a
    reported number: it turns whole rungs into (U). Measured consequence — 13 of
    15 rungs on Llama-3.1-8B licensed nowhere, and `hex` missed by 0.009 (0.691
    against a 0.70 cut) on a rung whose ability is 84/100.

    Which labels are permuted, and why THOSE. The **test** labels are permuted
    and the fitted direction is the real one. The null hypothesis being tested is
    the one that matters — *the plain-fit content direction does not rank encoded
    harmful above encoded harmless* — and permuting the test labels is its exact
    restatement: under it the probe's scores are exchangeable across the two
    encoded classes.

    Shuffling the TRAIN labels instead, which is what the per-cell
    `control_auroc` does, is WRONG as a null for this statistic and was tried
    first. Fitting logistic regression on shuffled labels yields a near-constant
    classifier, so its transfer AUROC collapses onto ~0.5 with almost no
    variance; the resulting null is far too tight and licenses noise. Caught by
    `test_an_unlicensed_curve_reads_no_prompt_as_deployed`, which is exactly the
    test that exists to stop a no-signal population reading as deployed. The
    per-cell `control_auroc` remains useful as a selectivity diagnostic; it is
    just not a calibrated null.

    Cost: because the direction is fixed, each cell is fitted ONCE and every draw
    only re-scores cached predictions. That makes the whole test a few seconds
    per rung on CPU — cheaper than the recognition null, which must refit per
    draw — so no GPU is needed and cached activations suffice.
    """
    if train_positive.layers != train_negative.layers:
        raise ValueError("both training classes must be captured at the same layers")

    train_labels = np.concatenate(
        [
            np.ones(train_positive.tensor.shape[0], dtype=int),
            np.zeros(train_negative.tensor.shape[0], dtype=int),
        ]
    )
    # Score once per cell with the REAL direction; only the labels move.
    scores_per_cell: list[np.ndarray] = []
    for layer in train_positive.layers:
        for position in train_positive.positions:
            train_features = _to_numpy(
                torch.cat(
                    [
                        train_positive.select(layer, position),
                        train_negative.select(layer, position),
                    ]
                )
            )
            positive_features = _to_numpy(test_positive.select(layer, position))
            negative_features = _to_numpy(test_negative.select(layer, position))
            test_features = np.concatenate([positive_features, negative_features])
            model = _fit(train_features, train_labels, config)
            scores_per_cell.append(model.decision_function(test_features))

    n_positive = test_positive.tensor.shape[0]
    n_negative = test_negative.tensor.shape[0]
    test_labels = np.concatenate(
        [np.ones(n_positive, dtype=int), np.zeros(n_negative, dtype=int)]
    )

    rng = np.random.default_rng(config.seed)
    maxima = np.empty(config.n_permutations, dtype=float)
    for draw in range(config.n_permutations):
        shuffled = rng.permutation(test_labels)
        best = -np.inf
        for scores in scores_per_cell:
            auroc = roc_auc_score(shuffled, scores)
            if not np.isnan(auroc):
                best = max(best, auroc)
        maxima[draw] = best
    return maxima


def permutation_p_value(observed: float, null_maxima: np.ndarray) -> float:
    """Empirical p-value with the +1 correction (Phipson & Smyth 2010).

    The correction matters: a plain `mean(null >= observed)` can report p=0,
    which is not a possible p-value from a finite permutation set and would
    overstate the evidence in a paper table.
    """
    if null_maxima.size == 0:
        return float("nan")
    return float((np.sum(null_maxima >= observed) + 1) / (null_maxima.size + 1))
