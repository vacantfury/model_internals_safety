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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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


def probe_transfer(
    train_positive: ActivationBatch,
    train_negative: ActivationBatch,
    test_positive: ActivationBatch,
    test_negative: ActivationBatch,
    layer: int,
    position: str,
    config: ProbeConfig,
) -> tuple[float, float]:
    """Fit on one condition, evaluate on another. Returns (transfer, control).

    Never refits on the test condition — the question is whether a decision
    boundary learned where the content is *plainly* present still separates the
    condition where it would have to have been decoded.
    """
    train_features = torch.cat(
        [train_positive.select(layer, position), train_negative.select(layer, position)]
    )
    train_labels = np.concatenate(
        [
            np.ones(train_positive.tensor.shape[0], dtype=int),
            np.zeros(train_negative.tensor.shape[0], dtype=int),
        ]
    )
    test_features = _to_numpy(
        torch.cat([test_positive.select(layer, position), test_negative.select(layer, position)])
    )
    test_labels = np.concatenate(
        [
            np.ones(test_positive.tensor.shape[0], dtype=int),
            np.zeros(test_negative.tensor.shape[0], dtype=int),
        ]
    )

    model = _fit(_to_numpy(train_features), train_labels, config)
    transfer = _auroc(model, test_features, test_labels)

    rng = np.random.default_rng(config.seed)
    control_model = _fit(_to_numpy(train_features), rng.permutation(train_labels), config)
    control = _auroc(control_model, test_features, test_labels)

    return transfer, control


def best_by_auroc(results: list[ProbeResult]) -> ProbeResult:
    """The peak of the curve. Reported *alongside* the curve, never instead of
    it: picking the best cell post hoc is a selection effect, and the layer at
    which the signal peaks is itself a finding."""
    return max(results, key=lambda result: (result.auroc if result.auroc == result.auroc else -1))
