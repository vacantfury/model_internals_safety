#!/usr/bin/env python3
"""Draw AS-5's headline figure: the two arms invert under encoding.

**What it shows.** Four independently post-trained models, refusal rate on the
harmful arm (left) and the benign arm (right), plaintext vs encoded. The harmful
arm fans IN — a spread of 0.57 collapses to 0.08, inside what sampling noise
alone produces — while the benign arm fans OUT, 0.15 to 0.69. That inversion is
the paper's claim, and it is the one thing a table states less well than a
picture: fan-in and fan-out are shapes.

**Every number is read from `outputs/`, none is typed here.** The repo rule is
that a figure whose value cannot be regenerated is a number with no home
(`paper/as-5/README.md`), so this script locates the runs, pulls the rates out of
their `readings` blocks, and recomputes the noise null rather than trusting a
figure caption written by hand. `--print-table` emits the LaTeX table body from
the same reads, which is how the drafted table is checked against disk instead of
against memory.

**The noise null is recomputed here, not imported.** Cross-model spreads are
max-minus-min over four estimates, which is a max statistic and therefore
n-dependent — the failure this repo already paid for once in the control floor
(`instrument_layer.md` §2.4). So every spread is read against a bootstrap of the
spread four IDENTICAL models at the observed mean would produce at the same n.
Seeded, so the figure is reproducible.

Keyless, no GPU, no model load: seconds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from internals_safety.config import load_measurements_config
from internals_safety.paths import PROJECT_ROOT, RUNS_DIR

# The four models of the cross-family result, in the order the paper reports
# them (descending plaintext harm gap). Keys are the run-dir model names.
MODELS: list[tuple[str, str]] = [
    ("llama3_1_8b_instruct", "Llama-3.1-8B"),
    ("qwen2_5_7b_instruct", "Qwen2.5-7B"),
    ("tulu3_8b", "Tülu-3-8B"),
    ("mistral_7b_instruct", "Mistral-7B-v0.3"),
]

# The runs that carry a plaintext arm AND an encoded arm in the same job. The
# pairing is the point: a plaintext baseline from a different run would reopen
# the cross-run comparison the in-job design exists to close.
RUN_PREFIX = "plain-baseline-"



@dataclass(frozen=True)
class ModelReading:
    """One model's four cells, plus the encoder they were measured under."""

    key: str
    label: str
    family: str
    n: int
    plain_harmful: float
    plain_benign: float
    enc_harmful: float
    enc_benign: float
    run_name: str

    @property
    def plain_gap(self) -> float:
        return self.plain_harmful - self.plain_benign

    @property
    def enc_gap(self) -> float:
        return self.enc_harmful - self.enc_benign


def _find_run(model_key: str) -> Path:
    """The newest `plain-baseline-*` run dir for one model.

    Newest by directory name, which carries an ISO timestamp — deliberately not
    by mtime, since an rsync down from the cluster rewrites mtimes in transfer
    order and would silently pick whichever file landed last.
    """
    model_dir = RUNS_DIR / "phase0" / model_key
    if not model_dir.is_dir():
        raise SystemExit(f"no run dir for {model_key} under {model_dir}")
    candidates = sorted(p for p in model_dir.glob(f"{RUN_PREFIX}*") if (p / "results.json").exists())
    if not candidates:
        raise SystemExit(
            f"no completed {RUN_PREFIX}* run for {model_key}. This figure needs the "
            "plaintext baseline runs; down-sync them from the cluster first."
        )
    return candidates[-1]


def read_model(model_key: str, label: str) -> ModelReading:
    run_dir = _find_run(model_key)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    plain = encoded = None
    for reading in results["readings"]:
        if reading["instrument"] == "behavior_plain":
            plain = reading["detail"]
        elif reading["instrument"] == "behavior":
            encoded = reading["detail"]
    if plain is None or encoded is None:
        raise SystemExit(
            f"{run_dir.name} lacks a behavior_plain or behavior reading. Every "
            "encoded rate needs its denominator; a run without both cannot enter "
            "this figure."
        )

    return ModelReading(
        key=model_key,
        label=label,
        family=encoded["family"],
        n=int(encoded["n"]),
        plain_harmful=float(plain["plain_harmful_refusal_rate"]),
        plain_benign=float(plain["plain_benign_refusal_rate"]),
        enc_harmful=float(encoded["refusal_rate"]),
        enc_benign=float(encoded["benign_arm_refusal_rate"]),
        run_name=results["run_name"],
    )


def noise_null(
    rates: list[float], n: int, rng: np.random.Generator, *, draws: int
) -> tuple[float, float, float]:
    """Median and 95% interval of the spread IDENTICAL models would show.

    The observed statistic is max-minus-min over len(rates) estimates. Under the
    null every model shares the observed mean rate, so each estimate is
    Binomial(n, p)/n and the spread is pure sampling noise. Returning the
    interval rather than a point matters: the plaintext benign spread sits just
    above the median and just below the ceiling, and the paper calls it marginal
    on exactly that basis.

    `draws` is KEYWORD-ONLY with no default, following this repo's rule that a
    settled parameter is threaded by making its omission a `TypeError` rather
    than by remembering to pass it.
    """
    p = float(np.mean(rates))
    sampled = rng.binomial(n, p, size=(draws, len(rates))) / n
    spreads = sampled.max(axis=1) - sampled.min(axis=1)
    return (
        float(np.median(spreads)),
        float(np.percentile(spreads, 2.5)),
        float(np.percentile(spreads, 97.5)),
    )


def paired_noise_null(
    per_item: "np.ndarray", rng: np.random.Generator, *, draws: int
) -> tuple[float, float, float]:
    """The same statistic, with item difficulty held in common (TODO 84 #3).

    `noise_null` above draws each model's rate as an INDEPENDENT Binomial(n, p).
    Every model in this paper answers the SAME prompts, so item difficulty is
    shared and independent draws overstate how far identical models drift apart.
    Here each simulated model draws per item from that item's own difficulty,
    estimated as the fraction of the observed models that refused it.

    **It cost a claim.** On harmful homoglyph the observed spread is 0.08: the
    independent null puts it comfortably inside (median 0.05, p=0.12) and the
    paired null puts it outside (median 0.04, 95th 0.07, p=0.03). The paper said
    "inside the null, and we cannot distinguish them"; it now says compression to
    the edge of resolution, because the paired null is the correct one.

    **And it is conservative.** Difficulty is estimated from the same models
    being tested, so wherever they genuinely disagree an item lands near 0.5 and
    contributes maximal Bernoulli variance --- inflating the null. The verdict
    above survives that inflation.

    `per_item` is models x items of booleans. Requires per-prompt verdicts, so it
    is unavailable for any condition whose cells were never persisted; callers
    fall back to `noise_null` and must SAY which they used rather than letting
    two different nulls share one name.
    """
    difficulty = per_item.mean(axis=0)
    n_models, n_items = per_item.shape
    sampled = (rng.random((draws, n_models, n_items)) < difficulty).mean(axis=2)
    spreads = sampled.max(axis=1) - sampled.min(axis=1)
    return (
        float(np.median(spreads)),
        float(np.percentile(spreads, 2.5)),
        float(np.percentile(spreads, 97.5)),
    )


def build_figure(readings: list[ModelReading], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display on the cluster or in CI
    import matplotlib.pyplot as plt

    controls = load_measurements_config().controls
    rng = np.random.default_rng(controls.noise_null_seed)
    draws = controls.noise_null_draws
    n = readings[0].n
    family = readings[0].family

    # Greyscale-safe: distinct markers and dash patterns carry the identity, and
    # colour only reinforces it. AAAI proceedings are read on paper as often as
    # on screen, and four colours at 45% column width are four grey lines there.
    styles = [
        ("o", "-", "#1b3a6b"),
        ("s", "--", "#8c3a12"),
        ("^", "-.", "#1c5c34"),
        ("D", ":", "#5c2b6b"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    arms = [
        ("harmful requests", lambda r: (r.plain_harmful, r.enc_harmful)),
        ("benign requests", lambda r: (r.plain_benign, r.enc_benign)),
    ]

    for ax, (arm_label, getter) in zip(axes, arms):
        values = [getter(r) for r in readings]
        plain_rates = [v[0] for v in values]
        enc_rates = [v[1] for v in values]

        # The spread band is the figure's actual subject: it narrows on the left
        # panel and widens on the right. Drawn under the lines so it reads as
        # ground rather than as a fifth series.
        for x, rates in ((0, plain_rates), (1, enc_rates)):
            ax.fill_between(
                [x - 0.13, x + 0.13],
                min(rates),
                max(rates),
                color="0.85",
                zorder=0,
                linewidth=0,
            )

        for reading, (marker, dash, colour) in zip(readings, styles):
            y0, y1 = getter(reading)
            ax.plot(
                [0, 1],
                [y0, y1],
                marker=marker,
                linestyle=dash,
                color=colour,
                linewidth=1.4,
                markersize=5,
                label=reading.label,
                zorder=3,
            )

        # Annotations ride at a FIXED height rather than above each band's max.
        # Anchoring them to the data put the benign panel's plaintext label at
        # 0.16, on top of the lines it was describing.
        label_transform = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for x, rates in ((0, plain_rates), (1, enc_rates)):
            spread = max(rates) - min(rates)
            null_median, _, null_hi = noise_null(rates, n, rng, draws=draws)
            inside = spread <= null_hi
            ax.text(
                x,
                0.97,
                f"spread {spread:.2f}\nnull {null_median:.2f}",
                transform=label_transform,
                ha="center",
                va="top",
                fontsize=6.5,
                color="#8c1515" if inside else "0.25",
                fontweight="bold" if inside else "normal",
            )

        ax.set_title(arm_label, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["plaintext", family], fontsize=8)
        ax.set_xlim(-0.32, 1.32)
        ax.set_ylim(-0.05, 1.28)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.92", linewidth=0.6, zorder=-1)

    axes[0].set_ylabel("refusal rate", fontsize=9)
    # The legend lives in the HARMFUL panel's floor, the one region empty in both
    # panels: harmful refusal never drops below 0.38, and the benign panel's
    # lower right is where its steepest lines land.
    axes[0].legend(fontsize=7, loc="lower center", ncol=2, frameon=False, handlelength=2.2,
                   columnspacing=1.0, borderpad=0.2)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def print_table(readings: list[ModelReading]) -> None:
    """Emit the LaTeX table body, so the draft is checked against disk.

    A table typed into a .tex and a figure drawn from `outputs/` are two homes
    for one number. This makes the .tex the derived one.
    """
    controls = load_measurements_config().controls
    rng = np.random.default_rng(controls.noise_null_seed)
    draws = controls.noise_null_draws
    n = readings[0].n
    print("% generated by scripts/figure_arm_inversion.py --print-table")
    for r in readings:
        print(
            f"{r.label:16s} & {r.plain_harmful:.2f} & {r.plain_benign:.2f} & "
            f"${r.plain_gap:+.2f}$ & {r.enc_harmful:.2f} & {r.enc_benign:.2f} & "
            f"${r.enc_gap:+.2f}$ \\\\"
        )
    cells = {
        "plain harmful": [r.plain_harmful for r in readings],
        "plain benign": [r.plain_benign for r in readings],
        "enc harmful": [r.enc_harmful for r in readings],
        "enc benign": [r.enc_benign for r in readings],
    }
    print("\n% spread vs noise null (median [2.5%, 97.5%])")
    for name, rates in cells.items():
        spread = max(rates) - min(rates)
        median, low, high = noise_null(rates, n, rng, draws=draws)
        verdict = "INSIDE null" if spread <= high else "outside null"
        print(f"%   {name:14s} spread {spread:.2f}   null {median:.2f} [{low:.2f}, {high:.2f}]   {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "paper/as-5/aaai_2027_ai_alignment/aaai_aia_latex/figs/arm_inversion.pdf",
        help="output path; `paper/` is gitignored, so a fresh clone regenerates it",
    )
    parser.add_argument("--print-table", action="store_true", help="emit the LaTeX table body and exit")
    args = parser.parse_args(argv)

    readings = [read_model(key, label) for key, label in MODELS]

    families = {r.family for r in readings}
    if len(families) != 1:
        raise SystemExit(f"models were measured under different encoders: {families}")
    sizes = {r.n for r in readings}
    if len(sizes) != 1:
        raise SystemExit(f"models were measured at different n: {sizes} — the null is per-n")

    for r in readings:
        print(f"{r.label:16s} {r.run_name}")

    if args.print_table:
        print()
        print_table(readings)
        return 0

    build_figure(readings, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
