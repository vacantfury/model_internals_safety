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

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from internals_safety.config import ProbeConfig
from internals_safety.models.capture import ActivationBatch


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
