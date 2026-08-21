#!/usr/bin/env python
"""Does the transfer probe read harm, or does it remember the item?

The deployment probe fits on the PLAINTEXT contrast set and evaluates on the
ENCODED one. The condition is held out; the ITEMS are not, because the encoded
corpus is the same 100 harmful and 100 benign prompts wearing a transformation
the model decodes at ability 0.86-0.98. A logistic fit at n=200 against
d_model=4096 interpolates, so it can carry item identity across the transform
and score encoded item *i* by having been told plaintext item *i* was harmful.

The signature that prompted this: `recognition` fits INSIDE the encoded
condition with a genuine cross-validated split and reads 0.66-0.86, while
`deployment` -- strictly less informed about that condition -- reads 0.94-1.00
on the identical activations. A transferred probe beating a directly fitted one
by 0.14-0.32 is not what a surviving harm direction looks like.

This script settles it offline, at the cell the run already selected, with no
GPU and no judge. Five readings per (model, family):

  A  reproduce   fit on ALL plaintext, score ALL encoded -- must match the record
  B  split       fit on plaintext items in fold A, score encoded items in fold B
  C  dim-split   same folds, but a difference-in-means direction, which has no
                 capacity to memorise items -- the estimator the paper CLAIMS
  D  dim-full    difference in means, no split, for the A-to-C decomposition
  E  plain-cv    cross-validated AUROC WITHIN plaintext -- the baseline the
                 paper never reports, so "the signal survives" has a denominator

Read B against A. If B collapses toward E's in-condition analogue, the reading
was leakage. If B holds, the leg is real and only its description was wrong.
Read C against B to see how much of any gap is the estimator rather than the
split.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from internals_safety.models.capture import ActivationBatch  # noqa: E402
from internals_safety.probes.directions import difference_in_means  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def _fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    # Same hyper-parameters as probes.linear._fit, so B differs from A only in
    # which items the probe was shown.
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    model.fit(features, labels)
    return model


def _cell(path: str, layer: int, position: str) -> np.ndarray:
    """Load one activation file, keep ONE (layer, position) slice, free the rest.

    The four files are ~105 MB each and hold [n_prompts, n_layers, n_positions,
    d_model]; we need [n_prompts, d_model]. Loading all four at once is ~420 MB
    of tensor plus torch's own footprint, which a login node kills (exit 137,
    measured). Loading one at a time keeps the peak at one file and the retained
    set at ~1.6 MB per cell. float32 is kept -- the probe is fit in float64 by
    sklearn anyway, and upcasting here would double the peak for nothing.
    """
    batch = ActivationBatch.load(Path(path))
    cell = batch.select(layer, position).clone().numpy()
    del batch
    gc.collect()
    return cell


def _stack(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([pos, neg]),
        np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]),
    )


def analyse(
    plain_h: np.ndarray,
    plain_b: np.ndarray,
    enc_h: np.ndarray,
    enc_b: np.ndarray,
    n_splits: int,
    seed: int,
) -> dict:
    # The four sets are index-paired by construction (item i of the harmful set
    # has the theme-matched benign item i, and the encoded sets are those same
    # items transformed). Splitting on a shared index therefore removes an item
    # from BOTH classes and BOTH conditions at once, which is what makes fold B
    # genuinely unseen.
    n_h, n_b = len(plain_h), len(plain_b)
    if len(enc_h) != n_h or len(enc_b) != n_b:
        raise ValueError(
            f"condition sets differ in length ({n_h}/{len(enc_h)} harmful, "
            f"{n_b}/{len(enc_b)} benign) -- the item split would not be aligned"
        )

    out: dict = {"n_harmful": n_h, "n_benign": n_b}

    # --- A: the current procedure, reproduced ---------------------------------
    train_x, train_y = _stack(plain_h, plain_b)
    test_x, test_y = _stack(enc_h, enc_b)
    model = _fit_logistic(train_x, train_y, seed)
    out["A_reproduce_no_split"] = _auroc(model.decision_function(test_x), test_y)

    # --- D: difference in means, no split -------------------------------------
    def dim_scores(fit_h: np.ndarray, fit_b: np.ndarray, score_x: np.ndarray) -> np.ndarray:
        vec = fit_h.mean(axis=0) - fit_b.mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("degenerate direction: the two classes coincide")
        return score_x @ (vec / norm)

    out["D_dim_no_split"] = _auroc(dim_scores(plain_h, plain_b, test_x), test_y)

    # --- B and C: item-split, over many random folds ---------------------------
    rng = np.random.default_rng(seed)
    b_scores, c_scores, f_scores, g_scores = [], [], [], []
    for _ in range(n_splits):
        # One shared permutation, halved -- fold A trains, fold B is scored.
        idx_h = rng.permutation(n_h)
        idx_b = rng.permutation(n_b)
        fit_h, held_h = idx_h[: n_h // 2], idx_h[n_h // 2 :]
        fit_b, held_b = idx_b[: n_b // 2], idx_b[n_b // 2 :]

        tr_x, tr_y = _stack(plain_h[fit_h], plain_b[fit_b])
        te_x, te_y = _stack(enc_h[held_h], enc_b[held_b])

        # F/G: the SAME probe, same training size, scored on the items it was
        # trained on (encoded versions of fold A). B/C above score fold B. The
        # pair isolates ITEM LEAKAGE from TRAINING-SET SIZE, which the A-vs-B
        # comparison alone cannot do: B halves the training set as well as
        # holding out the items, so part of its drop is fewer samples, not
        # memory. F minus G is leakage at fixed n.
        seen_x, seen_y = _stack(enc_h[fit_h], enc_b[fit_b])

        fitted = _fit_logistic(tr_x, tr_y, seed)
        b_scores.append(_auroc(fitted.decision_function(te_x), te_y))
        f_scores.append(_auroc(fitted.decision_function(seen_x), seen_y))
        c_scores.append(_auroc(dim_scores(plain_h[fit_h], plain_b[fit_b], te_x), te_y))
        g_scores.append(_auroc(dim_scores(plain_h[fit_h], plain_b[fit_b], seen_x), seen_y))

    for key, values in (
        ("B_split_logistic", b_scores),
        ("C_split_dim", c_scores),
        ("F_seen_logistic", f_scores),
        ("G_seen_dim", g_scores),
    ):
        arr = np.asarray(values)
        out[key] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p97_5": float(np.percentile(arr, 97.5)),
            "n_splits": int(len(arr)),
        }

    # --- E: cross-validated WITHIN plaintext -----------------------------------
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oos = np.zeros(len(train_y))
    for fit_idx, held_idx in folds.split(train_x, train_y):
        fitted = _fit_logistic(train_x[fit_idx], train_y[fit_idx], seed)
        oos[held_idx] = fitted.decision_function(train_x[held_idx])
    out["E_plain_crossval"] = _auroc(oos, train_y)

    out["leakage_gap_A_minus_B"] = out["A_reproduce_no_split"] - out["B_split_logistic"]["mean"]
    # The clean leakage estimate: same probe, same training size, seen minus unseen.
    out["leakage_at_fixed_n_logistic"] = (
        out["F_seen_logistic"]["mean"] - out["B_split_logistic"]["mean"]
    )
    out["leakage_at_fixed_n_dim"] = out["G_seen_dim"]["mean"] - out["C_split_dim"]["mean"]
    return out


@dataclasses.dataclass(frozen=True)
class Target:
    """One (record, family) cell to re-test, with its four activation files."""

    model: str
    family: str
    layer: int
    position: str
    plain_harmful: str
    plain_harmless: str
    encoded_harmful: str
    encoded_harmless: str


def _model_name(value: object) -> str:
    """The model's NAME, whether the record stored a name or the whole config.

    AS-5's phase-0 schema stores `config.model` as the full model block, so the
    original single-schema reader printed a 400-character dict where a name
    belonged. Harmless in a header line, not harmless in `--out`: the artifact is
    keyed by model, and a dict key that changes whenever an unrelated capture
    knob changes will not join back to the run it came from.
    """
    if isinstance(value, dict):
        return str(value.get("name") or value.get("hf_id") or "unknown")
    return str(value)


def _sole(directory: Path, stem: str) -> str:
    """The one cached capture named `stem`, or a refusal.

    AS-5's schema records the plain-condition paths; AS-6's records only the
    per-family encoded ones, so for a guard record the plain pair is resolved by
    glob in the same directory. An ambiguous match is an ERROR rather than a
    pick: two `plain-harmful-*.pt` files mean two different corpora were captured
    into one directory, and silently taking either would fit the probe on prompts
    the encoded set was not derived from -- which is a subtler version of exactly
    the alignment defect this script exists to measure.
    """
    matches = sorted(directory.glob(f"{stem}-*.pt"))
    if len(matches) != 1:
        found = ", ".join(m.name for m in matches) if matches else "none"
        raise SystemExit(f"{directory}: expected exactly one {stem}-*.pt, found {len(matches)}: {found}")
    return str(matches[0])


def targets_from_record(path: Path) -> list[Target]:
    """Read either run schema and return every cell it selected.

    AS-5's phase-0 record carries ONE deployment reading under `readings`; AS-6's
    guard record carries one `decode` cell per family under `summaries`. Both are
    the same measurement -- a probe fit on plaintext and read on an encoded
    condition -- so both are exposed to the item-identity leakage this script
    tests, and both must be re-testable by the same instrument.
    """
    record = json.loads(path.read_text())

    if "readings" in record:  # AS-5 phase-0
        deployment = next(
            (
                r
                for r in record["readings"]
                if isinstance(r.get("detail"), dict)
                and "layer" in r["detail"]
                and "position" in r["detail"]
            ),
            None,
        )
        if deployment is None:
            raise SystemExit(f"{path}: no reading carries a (layer, position) cell")
        detail = deployment["detail"]
        family = str(detail["family"])
        paths = record["activations_path"]
        per_family = paths["per_family"][family]
        return [
            Target(
                model=_model_name(record.get("model") or record.get("config", {}).get("model")),
                family=family,
                layer=int(detail["layer"]),
                position=str(detail["position"]),
                plain_harmful=paths["plain_harmful"],
                plain_harmless=paths["plain_harmless"],
                encoded_harmful=per_family["encoded_harmful"],
                encoded_harmless=per_family["encoded_harmless"],
            )
        ]

    if "summaries" in record:  # AS-6 phase-1 guard run
        guard = str(record.get("config", {}).get("guard", {}).get("name") or path.parent.name)
        targets = []
        for summary in record["summaries"]:
            decode = summary.get("decode") or {}
            if decode.get("layer") is None or decode.get("position") is None:
                continue  # no cell was selected, so there is nothing to re-test
            activations = summary["activations"]
            directory = Path(activations["encoded_harmful"]).parent
            targets.append(
                Target(
                    model=guard,
                    family=str(summary["family"]),
                    layer=int(decode["layer"]),
                    position=str(decode["position"]),
                    plain_harmful=_sole(directory, "plain-harmful"),
                    plain_harmless=_sole(directory, "plain-harmless"),
                    encoded_harmful=activations["encoded_harmful"],
                    encoded_harmless=activations["encoded_harmless"],
                )
            )
        if not targets:
            raise SystemExit(f"{path}: no summary carries a (layer, position) cell")
        return targets

    raise SystemExit(f"{path}: unrecognised run schema (neither `readings` nor `summaries`)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path, help="results.json files to re-read")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--family",
        action="append",
        help="restrict to these families (repeatable); default is every cell the record selected",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    wanted = set(args.family) if args.family else None
    rows = []
    for run in args.runs:
        for target in targets_from_record(run):
            if wanted is not None and target.family not in wanted:
                continue

            cells = {
                name: _cell(path, target.layer, target.position)
                for name, path in (
                    ("plain_h", target.plain_harmful),
                    ("plain_b", target.plain_harmless),
                    ("enc_h", target.encoded_harmful),
                    ("enc_b", target.encoded_harmless),
                )
            }

            result = analyse(
                cells["plain_h"],
                cells["plain_b"],
                cells["enc_h"],
                cells["enc_b"],
                n_splits=args.splits,
                seed=args.seed,
            )
            del cells
            gc.collect()
            result |= {
                "run": run.parent.name,
                "model": target.model,
                "family": target.family,
                "layer": target.layer,
                "position": target.position,
            }
            rows.append(result)

            print(f"\n=== {result['model']}  {target.family}  L{target.layer} {target.position} ===")
            print(f"  A reproduce (no split, logistic) : {result['A_reproduce_no_split']:.4f}")
            print(
                f"  B item-split     (logistic)      : {result['B_split_logistic']['mean']:.4f}"
                f"  [{result['B_split_logistic']['p2_5']:.4f}, {result['B_split_logistic']['p97_5']:.4f}]"
            )
            print(
                f"  C item-split     (diff-in-means) : {result['C_split_dim']['mean']:.4f}"
                f"  [{result['C_split_dim']['p2_5']:.4f}, {result['C_split_dim']['p97_5']:.4f}]"
            )
            print(f"  D no split       (diff-in-means) : {result['D_dim_no_split']:.4f}")
            print(f"  E plaintext cross-validated      : {result['E_plain_crossval']:.4f}")
            print(f"  F seen items     (logistic, n/2) : {result['F_seen_logistic']['mean']:.4f}")
            print(f"  G seen items     (dim, n/2)      : {result['G_seen_dim']['mean']:.4f}")
            print(
                f"  --> leakage at FIXED n (F - B)   : {result['leakage_at_fixed_n_logistic']:+.4f}"
                f"   (dim, G - C: {result['leakage_at_fixed_n_dim']:+.4f})"
            )
            print(f"  --> leakage gap A - B            : {result['leakage_gap_A_minus_B']:+.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
