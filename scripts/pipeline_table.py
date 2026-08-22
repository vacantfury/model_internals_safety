"""Complete the pipeline table: raw arms, intervals, echo rates, replicate range.

**Why this exists.** Both external referees converged on one objection about
§\\ref{sec:ladder}: it reports only the plaintext gap and *gap lost*, calls a
0.41 -> 0.35 endpoint move "nothing", and shows no raw encoded arms, no
intervals, no per-cell echo rates and no repeat-run variation, while the paper
itself states that differences below roughly 0.15 are not separable at n=100. As
written the claim cannot be told apart from an underpowered one.

It is a REPORTING gap and not a measurement gap. Every number below comes from
per-prompt verdicts already on disk, so this is keyless, GPU-free, $0 and takes
seconds.

**Three properties are deliberate.**

*Runs are DISCOVERED, never hard-coded.* Names carry a timestamp and a SLURM job
id, and a path pinned in source is a path that rots. The stage label is read off
the run-name suffix (``ladder-plain-<stage>``), which is the declared thing;
the model directory is not, because the RLVR checkpoint sits in ``tulu3_8b``
while SFT and DPO sit in suffixed siblings, and inferring the stage from the
directory would encode that asymmetry as a magic map.

*The stage-to-stage contrast is PAIRED and the harm gap is NOT.* Both stages run
the identical 100 harmful and 100 benign prompts, so a stage contrast shares item
difficulty and an unpaired interval there is too wide -- the mistake that cost
this repo a claim in the other direction (§4p). The harm gap's own arms are
different corpora with no pairing to exploit, so it takes the unpaired interval.
Reaching for the wrong one of these is the single easiest error in this file.

*Echo gets a three-way sensitivity, not a verdict.* The screen drops echoing
cells because their refusal status is unknown, and `refusal_control.EchoExposure`
argues that at length. The referee's objection is different and stands anyway:
without the alternatives on the page, a reader cannot tell a behaviour difference
from a selection effect. So all three treatments are printed and the argument for
the chosen one stays where it lives.

    uv run python scripts/pipeline_table.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from internals_safety.config import load_measurements_config
from internals_safety.measurements.intervals import (
    mcnemar_exact,
    paired_bootstrap_difference,
    unpaired_difference_interval,
    wilson,
    z_for,
)
from internals_safety.measurements.refusal_control import summarize_exposure
from internals_safety.paths import RUNS_DIR

#: The run family carrying BOTH arms plus the model-level plaintext baseline.
#: The sibling `ladder-tulu3-*` runs measure the same encoded harmful cells
#: without a benign arm, which makes them replicates rather than a second source.
PRIMARY_PREFIX = "ladder-plain-"
REPLICATE_PREFIX = "ladder-tulu3-"

#: Pipeline order. Declared because it is the paper's x-axis and alphabetical
#: order would silently reverse it.
STAGE_ORDER = ["sft", "dpo", "rlvr"]

#: The encodings §\ref{sec:ladder} reports. Everything else in the run is either
#: a can't-decode control or below the echo screen, and lands in the JSON only.
REPORTED = ["fullwidth", "homoglyph", "zero_width"]


def discover(phase_dir: Path, prefix: str) -> dict[str, Path]:
    """Map stage label -> run directory, from the run-name suffix."""
    found: dict[str, Path] = {}
    for run in sorted(phase_dir.glob("*/*/")):
        name = run.name
        if not name.startswith(prefix):
            continue
        stage = name[len(prefix) :].split("_")[0]
        if stage in STAGE_ORDER:
            found[stage] = run
    return found


def read_cells(path: Path) -> dict[str, list[dict]]:
    """Per-prompt records grouped by encoding family, order preserved."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["family"]].append(row)
    return grouped


def plain_baseline(results: Path) -> dict:
    """The model-level plaintext reading, which is the table's denominator."""
    record = json.loads(results.read_text())
    for reading in record.get("readings") or []:
        if reading.get("instrument") == "behavior_plain":
            return reading["detail"]
    raise KeyError(f"no behavior_plain reading in {results}")


def rate(flags) -> float:
    flags = list(flags)
    return sum(bool(f) for f in flags) / len(flags) if flags else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    conf = load_measurements_config()
    alpha = conf.probes.alpha
    z = z_for(alpha)
    rng = np.random.default_rng(conf.controls.bootstrap_seed)
    draws = conf.controls.bootstrap_draws

    phase = RUNS_DIR / "phase0"
    primary = discover(phase, PRIMARY_PREFIX)
    replicates = discover(phase, REPLICATE_PREFIX)
    missing = [s for s in STAGE_ORDER if s not in primary]
    if missing:
        print(f"missing stages: {missing} under {phase}")
        return 1

    stages: dict[str, dict] = {}
    verdicts: dict[tuple[str, str], dict[str, list[bool]]] = {}

    for stage in STAGE_ORDER:
        run = primary[stage]
        harmful = read_cells(run / "cells.jsonl")
        benign = read_cells(run / "benign_cells.jsonl")
        base = plain_baseline(run / "results.json")
        rep = read_cells(replicates[stage] / "cells.jsonl") if stage in replicates else {}

        families: dict[str, dict] = {}
        for family in sorted(set(harmful) & set(benign)):
            h = harmful[family]
            b = benign[family]
            h_ref = [bool(r["refused"]) for r in h]
            h_echo = [bool(r["echoed_ciphertext"]) for r in h]
            b_ref = [bool(r["refused"]) for r in b]
            b_echo = [bool(r["echoed_ciphertext"]) for r in b]
            verdicts[(stage, family)] = {"harmful": h_ref, "benign": b_ref}

            exposure = summarize_exposure(family, h_ref, h_echo, b_ref, b_echo)
            gap = rate(h_ref) - rate(b_ref)

            # Echo sensitivity: the screen's treatment, plus the two the referee
            # named. "Missing" is what the paper does and what EchoExposure
            # argues for; the other two are printed so the reader can see the
            # selection effect rather than take our word for its size.
            def recode(refused, echoed, as_value):
                return [as_value if e else r for r, e in zip(refused, echoed)]

            sensitivity = {
                "echo_dropped": exposure.clean_gap,
                "echo_as_refusal": rate(recode(h_ref, h_echo, True))
                - rate(recode(b_ref, b_echo, True)),
                "echo_as_non_refusal": rate(recode(h_ref, h_echo, False))
                - rate(recode(b_ref, b_echo, False)),
            }

            # ⚠️ The paper's OWN echo screen, run against the paper's OWN table.
            # `Limitations` calls this screen required for validity, and it had
            # never been applied to these cells: §\ref{sec:ladder} predates the
            # benign arm reaching disk, which is when the screen became
            # computable at all (`instrument_layer.md` §3.11).
            entry = {
                "echo_screen_displacement": exposure.displacement,
                "echo_screen_bar": exposure.bar,
                "echo_screen_clears": exposure.clears(),
                "n_harmful": len(h_ref),
                "n_benign": len(b_ref),
                "harmful_refusal": rate(h_ref),
                "harmful_refusal_ci": wilson(sum(h_ref), len(h_ref), z),
                "benign_refusal": rate(b_ref),
                "benign_refusal_ci": wilson(sum(b_ref), len(b_ref), z),
                "gap": gap,
                "gap_ci": unpaired_difference_interval(
                    sum(h_ref), len(h_ref), sum(b_ref), len(b_ref), z
                ),
                "harmful_echo_rate": rate(h_echo),
                "benign_echo_rate": rate(b_echo),
                "n_harmful_clean": exposure.n_harmful_clean,
                "n_benign_clean": exposure.n_benign_clean,
                "clean_harmful_refusal": exposure.clean_harmful_refusal_rate,
                "clean_benign_refusal": exposure.clean_benign_refusal_rate,
                "clean_gap": exposure.clean_gap,
                "echo_sensitivity": sensitivity,
                "gap_lost": base["plain_harm_gap"] - gap,
            }
            if family in rep:
                r_ref = [bool(r["refused"]) for r in rep[family]]
                entry["replicate_harmful_refusal"] = rate(r_ref)
                entry["replicate_range"] = abs(rate(r_ref) - rate(h_ref))
            families[family] = entry

        stages[stage] = {
            "run": run.name,
            "replicate_run": replicates[stage].name if stage in replicates else None,
            "plain_harmful_refusal": base["plain_harmful_refusal_rate"],
            "plain_benign_refusal": base["plain_benign_refusal_rate"],
            "plain_gap": base["plain_harm_gap"],
            "plain_n": base.get("plain_n"),
            "families": families,
        }

    # The endpoint contrast the referee asked for, run as a PAIRED test on the
    # shared items rather than as a difference of two independent intervals.
    endpoints: dict[str, dict] = {}
    for family in REPORTED:
        first, last = (STAGE_ORDER[0], family), (STAGE_ORDER[-1], family)
        if first not in verdicts or last not in verdicts:
            continue
        a, b = verdicts[first], verdicts[last]
        point, lo, hi = paired_bootstrap_difference(
            (a["harmful"], a["benign"]),
            (b["harmful"], b["benign"]),
            rng,
            draws=draws,
            alpha=alpha,
        )
        only_a = sum(1 for x, y in zip(a["harmful"], b["harmful"]) if x and not y)
        only_b = sum(1 for x, y in zip(a["harmful"], b["harmful"]) if y and not x)
        endpoints[family] = {
            "delta_gap_sft_minus_rlvr": point,
            "ci": [lo, hi],
            "excludes_zero": (lo > 0) or (hi < 0),
            "harmful_arm_discordant_sft_only": only_a,
            "harmful_arm_discordant_rlvr_only": only_b,
            "harmful_arm_mcnemar_p": mcnemar_exact(only_a, only_b),
        }

    print(f"stages: {[stages[s]['run'] for s in STAGE_ORDER]}")
    print(f"bootstrap: {draws} draws, seed {conf.controls.bootstrap_seed}, alpha {alpha}\n")
    header = f"{'stage':6} {'encoding':11} {'harmful':>18} {'benign':>18} {'gap':>19} {'echo h/b':>11} {'lost':>6} {'repl':>6}"
    print(header)
    for stage in STAGE_ORDER:
        s = stages[stage]
        print(
            f"{stage:6} {'PLAINTEXT':11} "
            f"{s['plain_harmful_refusal']:>18.2f} {s['plain_benign_refusal']:>18.2f} "
            f"{s['plain_gap']:>+19.2f}"
        )
        for family in REPORTED:
            e = s["families"].get(family)
            if e is None:
                continue
            hci = f"{e['harmful_refusal']:.2f} [{e['harmful_refusal_ci'][0]:.2f},{e['harmful_refusal_ci'][1]:.2f}]"
            bci = f"{e['benign_refusal']:.2f} [{e['benign_refusal_ci'][0]:.2f},{e['benign_refusal_ci'][1]:.2f}]"
            gci = f"{e['gap']:+.2f} [{e['gap_ci'][0]:+.2f},{e['gap_ci'][1]:+.2f}]"
            rep_s = f"{e['replicate_range']:.2f}" if "replicate_range" in e else "  --"
            clears = e["echo_screen_clears"]
            mark = "  " if clears else ("!!" if clears is False else "??")
            print(
                f"{'':6} {family:11} {hci:>18} {bci:>18} {gci:>19} "
                f"{e['harmful_echo_rate']:.2f}/{e['benign_echo_rate']:.2f} "
                f"{e['gap_lost']:>6.2f} {rep_s:>6} {mark}"
            )
    print(f"\nendpoint contrast, {STAGE_ORDER[0]} minus {STAGE_ORDER[-1]}, paired over items:")
    for family, e in endpoints.items():
        verdict = "EXCLUDES 0" if e["excludes_zero"] else "includes 0"
        print(
            f"  {family:11} d(gap) {e['delta_gap_sft_minus_rlvr']:+.3f} "
            f"[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]  {verdict}   "
            f"harmful-arm McNemar p={e['harmful_arm_mcnemar_p']:.3f} "
            f"({e['harmful_arm_discordant_sft_only']}/{e['harmful_arm_discordant_rlvr_only']} discordant)"
        )
    failing = [
        f"{stage}/{family}"
        for stage in STAGE_ORDER
        for family in REPORTED
        if stages[stage]["families"].get(family, {}).get("echo_screen_clears") is False
    ]
    print(
        f"\necho screen (|gap - clean gap| <= the gap's own 95% half-width): "
        f"{len(failing)} of {len(STAGE_ORDER) * len(REPORTED)} reported cells FAIL"
    )
    for stage in STAGE_ORDER:
        for family in REPORTED:
            e = stages[stage]["families"].get(family)
            if e is None:
                continue
            verdict = "clears" if e["echo_screen_clears"] else "FAILS"
            print(
                f"  {stage:5} {family:11} displacement {e['echo_screen_displacement']:.3f} "
                f"vs bar {e['echo_screen_bar']:.3f}  {verdict}"
            )
    print("\necho sensitivity, gap under three treatments:")
    for stage in STAGE_ORDER:
        for family in REPORTED:
            e = stages[stage]["families"].get(family)
            if e is None:
                continue
            s = e["echo_sensitivity"]
            drop = f"{s['echo_dropped']:+.2f}" if s["echo_dropped"] is not None else " n/a"
            print(
                f"  {stage:5} {family:11} dropped {drop}  "
                f"as-refusal {s['echo_as_refusal']:+.2f}  "
                f"as-non-refusal {s['echo_as_non_refusal']:+.2f}  "
                f"(clean n {e['n_harmful_clean']}/{e['n_benign_clean']})"
            )

    out = args.out or (RUNS_DIR.parent / "analysis" / "pipeline_table_20260822.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stages": stages,
                "endpoint_contrast": endpoints,
                "bootstrap_draws": draws,
                "bootstrap_seed": conf.controls.bootstrap_seed,
                "alpha": alpha,
            },
            indent=1,
            default=float,
        )
    )
    print(f"\nwrote {out.relative_to(RUNS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
