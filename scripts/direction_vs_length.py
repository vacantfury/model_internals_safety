#!/usr/bin/env python3
"""TODO 28's offline comparison: does a difference-in-means DIRECTION read
something the correlational probe does not, and does either beat the length null?

Item 28 adopted Arditi et al.'s (NeurIPS 2024) causal licensing criterion and
noted that our own licensing is correlational: AUROC plus a permutation null.
A permutation test cannot tell a real separation from the RIGHT separation, and
this repo has the receipt: raw character length separates the JBB harmful corpus
from the benign one at AUROC 0.654, and every encoder is monotone in length, so
the confound survives into every rung.

This script is the first, free half of that item. It fits both estimators on the
PLAIN contrast sets, where the answer is known to be near ceiling, and asks where
on the layer curve each one actually beats what raw length already buys.

BOTH ESTIMATORS ARE CROSS-VALIDATED, ON THE SAME FOLDS. That is not caution, it
is the direct lesson of 2026-08-21: `probe_transfer` fits on all 200 plaintext
items and scores the encoded versions of THOSE SAME items, which withdrew AS-5's
internals leg (held-out 0.618-0.811 against a reported 0.938-0.995). A
difference-in-means vector is a fitted object exactly like a logistic probe, so
scoring the items it was built from would repeat that defect in a script written
to investigate confounds.

TWO ASYMMETRIES, both deliberate and both against us:

  * The length null is TWO-SIDED, `max(a, 1-a)`, because a length-only classifier
    may exploit either direction. The fitted estimators are reported DIRECTIONALLY:
    they had every chance to learn their own sign, so an out-of-fold AUROC below
    0.5 is a failure rather than an inverted success. This makes the null harder
    to beat than the estimators, which is the safe way round.
  * The null is computed on the same texts, so it is the length signal available
    in this very contrast, not an imported constant.

Keyless, GPU-free, CPU-only, seconds. Needs the plain captures in the local
activation cache; they are ~105MB per arm and the link drops, so verify byte
counts against the remote after any transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from internals_safety.config import load_measurements_config  # noqa: E402
from internals_safety.measurements.length_null import length_auroc  # noqa: E402
from internals_safety.models.capture import ActivationBatch  # noqa: E402
from internals_safety.paths import ACTIVATIONS_DIR  # noqa: E402
from internals_safety.probes.directions import difference_in_means  # noqa: E402
from internals_safety.probes.linear import crossval_scores  # noqa: E402


def length_rho(scores: np.ndarray, lengths: np.ndarray) -> float:
    """|Spearman| between a cell's out-of-fold scores and raw character length.

    This is LENGTH EXPOSURE proper. A smaller margin over the length null is only
    the shadow of it: an estimator can read less total signal without being any
    more length-driven, and calling the first the second would be deriving a
    claim from a number that does not contain it. Absolute value because a
    confound running either way is equally exploitable, matching `length_auroc`'s
    two-sided construction.
    """
    usable = ~np.isnan(scores)
    # BOTH inputs must vary. Guarding only the lengths left a constant-score
    # cell (layer 0 at `last`, where every prompt ends on the same template
    # token) to reach spearmanr and warn, when it is a coverage fact.
    if (
        usable.sum() < 3
        or len(set(lengths[usable])) <= 1
        or len(set(scores[usable])) <= 1
    ):
        return float("nan")
    rho = spearmanr(scores[usable], lengths[usable]).statistic
    return float("nan") if np.isnan(rho) else abs(float(rho))


def subset(batch: ActivationBatch, index: np.ndarray) -> ActivationBatch:
    """A view of the batch over a subset of prompts, fields kept consistent.

    Built rather than hand-slicing the tensor so that `difference_in_means`
    receives the same type it receives in production. A fold that silently
    disagreed with the real call site about layer ordering would be the
    fixture rule violated inside an analysis script.
    """
    keep = torch.as_tensor(index, dtype=torch.long)
    messages = [batch.user_messages[i] for i in index] if batch.user_messages else []
    return replace(batch, tensor=batch.tensor[keep], user_messages=messages)


def held_out_direction_auroc(
    positive: ActivationBatch,
    negative: ActivationBatch,
    layer: int,
    position: str,
    folds: int,
    seed: int,
) -> tuple[float, float, int, np.ndarray]:
    """Out-of-fold AUROC for a difference-in-means direction, plus mean raw norm.

    Returns (auroc, mean_raw_norm, n_degenerate). `raw_norm` is carried because a
    near-zero difference means the classes sit on top of each other and the unit
    vector is numerical noise; `difference_in_means` documents that the caller
    must discard such a cell, and a folded fit can produce one where the full fit
    does not.
    """
    n_pos, n_neg = positive.tensor.shape[0], negative.tensor.shape[0]
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    scores = np.full(len(labels), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    norms, degenerate = [], 0

    for train, test in splitter.split(np.zeros(len(labels)), labels):
        train_pos = train[train < n_pos]
        train_neg = train[train >= n_pos] - n_pos
        direction = difference_in_means(
            subset(positive, train_pos), subset(negative, train_neg), layer, position
        )
        norms.append(direction.raw_norm)
        if direction.raw_norm <= 0:
            degenerate += 1
            continue
        for i in test:
            cell = positive if i < n_pos else negative
            row = i if i < n_pos else i - n_pos
            scores[i] = float(direction.project(cell.select(layer, position)[row]))

    usable = ~np.isnan(scores)
    if usable.sum() == 0 or len(set(labels[usable])) < 2:
        return float("nan"), float(np.mean(norms)), degenerate, scores
    return (
        float(roc_auc_score(labels[usable], scores[usable])),
        float(np.mean(norms)),
        degenerate,
        scores,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3_1_8b_instruct")
    parser.add_argument("--harmful", required=True, help="plain harmful capture stem")
    parser.add_argument("--harmless", required=True, help="plain harmless capture stem")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cache = ACTIVATIONS_DIR / args.model
    positive = ActivationBatch.load(cache / f"{args.harmful}.pt")
    negative = ActivationBatch.load(cache / f"{args.harmless}.pt")
    probe_config = load_measurements_config().probes

    null = length_auroc(positive.user_messages, negative.user_messages)
    print(f"model            {args.model}")
    print(f"n                {positive.tensor.shape[0]} harmful / {negative.tensor.shape[0]} harmless")
    print(f"length null      {null:.4f}  (two-sided, same texts)")
    print(f"folds / seed     {probe_config.cv_folds} / {probe_config.seed}")
    print()
    lengths = np.array(
        [len(t) for t in positive.user_messages] + [len(t) for t in negative.user_messages],
        dtype=float,
    )
    print(f"{'layer':>5} {'position':>17} {'dim-AUROC':>10} {'probe-AUROC':>12} "
          f"{'dim-null':>9} {'probe-null':>11} {'dim-rho':>9} {'probe-rho':>10} {'raw_norm':>8}")

    rows = []
    for position in positive.positions:
        for layer in positive.layers:
            dim_auroc, raw_norm, degenerate, dim_scores = held_out_direction_auroc(
                positive, negative, layer, position, probe_config.cv_folds, probe_config.seed
            )
            pos_scores, neg_scores = crossval_scores(
                positive, negative, layer, position, probe_config
            )
            probe_auroc = float(
                roc_auc_score(
                    np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))]),
                    np.concatenate([pos_scores, neg_scores]),
                )
            )
            probe_all = np.concatenate([pos_scores, neg_scores])
            dim_rho = length_rho(dim_scores, lengths)
            probe_rho = length_rho(probe_all, lengths)
            row = {
                "layer": layer,
                "position": position,
                "direction_auroc": dim_auroc,
                "probe_auroc": probe_auroc,
                "direction_margin": dim_auroc - null,
                "probe_margin": probe_auroc - null,
                "direction_length_rho": dim_rho,
                "probe_length_rho": probe_rho,
                "raw_norm": raw_norm,
                "degenerate_folds": degenerate,
            }
            rows.append(row)
            print(f"{layer:>5} {position:>17} {dim_auroc:>10.4f} {probe_auroc:>12.4f} "
                  f"{dim_auroc - null:>+9.4f} {probe_auroc - null:>+11.4f} "
                  f"{dim_rho:>+9.3f} {probe_rho:>+10.3f} {raw_norm:>8.2f}")

    payload = {
        "model": args.model,
        "n_positive": int(positive.tensor.shape[0]),
        "n_negative": int(negative.tensor.shape[0]),
        "length_null_auroc": null,
        "cv_folds": probe_config.cv_folds,
        "seed": probe_config.seed,
        "held_out": True,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
