"""Is the harm gap survivable under length-matched pairs? Referee con 3, answered.

An external referee asked whether the behavioural harm gap could be a length
effect, citing our own reported AUROC $0.654$ for raw character length
separating the harmful from the benign corpus, and noting that every
transformation preserves or amplifies length cues. It is a fair question and the
instrument for it already existed: `measure_rate_length_null` bins both arms by
ciphertext length and permutes the harmful/benign labels WITHIN bins, so a gap
that is really a length gap cannot survive.

It had never been run on the cross-model condition, only wired into the pipeline
entrypoint, which is the third instance in this repo of a settled rule not
reaching every caller.

**One property to read correctly before quoting the output.** A model whose gap
is already zero cannot clear any null, and that is not a failed control: there is
no effect to defend. Llama is that model under homoglyph. Reporting its
non-clearance next to the others' as though the same thing happened on four
models would be the error.

Keyless, GPU-free, seconds.

    uv run python scripts/rate_length_null.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internals_safety.config import load_measurements_config
from internals_safety.measurements.length_null import measure_rate_length_null
from internals_safety.paths import RUNS_DIR

#: Runs carrying BOTH arms' per-prompt records for the cross-model condition.
#: The `plain-baseline-*` runs hold only the harmful arm, so the null is not
#: computable there -- which is why it is these and not those.
SOURCE_PREFIX = "spread-"

CONDITION = "homoglyph"


def load_arm(path: Path, family: str) -> tuple[list[str], list[bool]]:
    texts: list[str] = []
    flags: list[bool] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["family"] != family:
                continue
            texts.append(row["ciphertext"])
            flags.append(bool(row["refused"]))
    return texts, flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=CONDITION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    conf = load_measurements_config()
    probes = conf.probes
    results = {}

    for model_dir in sorted((RUNS_DIR / "phase0").iterdir()):
        if not model_dir.is_dir():
            continue
        runs = [r for r in model_dir.iterdir() if r.name.startswith(SOURCE_PREFIX)]
        if not runs:
            continue
        run = sorted(runs)[-1]
        harmful_path = run / "cells.jsonl"
        benign_path = run / "benign_cells.jsonl"
        if not (harmful_path.exists() and benign_path.exists()):
            continue
        h_texts, h_flags = load_arm(harmful_path, args.family)
        b_texts, b_flags = load_arm(benign_path, args.family)
        if not h_texts or not b_texts:
            continue
        null = measure_rate_length_null(
            args.family,
            h_texts,
            b_texts,
            h_flags,
            b_flags,
            n_bins=probes.length_strata_bins,
            n_permutations=probes.n_permutations,
            quantile=1.0 - probes.alpha,
            seed=probes.seed,
        )
        results[model_dir.name] = {
            "run": run.name,
            "observed_gap": null.observed_gap,
            "stratified_gap": null.stratified_gap,
            "matching_shift": null.matching_shift,
            "null_quantile": null.null_quantile,
            "margin": null.margin,
            "clears": null.margin > 0.0,
            "n_bins": null.n_bins,
            "n_strata_used": null.n_strata_used,
            "n_positive": null.n_positive,
            "n_negative": null.n_negative,
        }

    print(f"condition: {args.family}   bins {probes.length_strata_bins}, "
          f"{probes.n_permutations} permutations, quantile {1.0 - probes.alpha}\n")
    print(f"{'model':26} {'gap':>7} {'matched':>8} {'shift':>7} {'null':>7} {'margin':>8}  verdict")
    for model, r in results.items():
        verdict = "CLEARS" if r["clears"] else "does not clear"
        print(
            f"{model:26} {r['observed_gap']:>+7.3f} {r['stratified_gap']:>+8.3f} "
            f"{r['matching_shift']:>+7.3f} {r['null_quantile']:>7.3f} "
            f"{r['margin']:>+8.3f}  {verdict}"
        )
    print(
        "\nA gap of ~0 cannot clear a null and that is not a failed control: "
        "there is no effect to defend."
    )

    out = args.out or (RUNS_DIR.parent / "analysis" / "rate_length_null_20260822.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"condition": args.family, "models": results}, indent=1))
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
