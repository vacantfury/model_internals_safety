"""Re-tune `conf/cost.yaml`'s throughput from runs that have actually finished.

**The tuning path the knob was created with, finally walked.** `conf/cost.yaml`
says of `decode_tokens_per_s`: *"the first real run measures it"*. The first real
run did — and the value it wrote, `[40, 71]`, was measured on the phase-0 pilot,
BEFORE `a0855fa` pinned probe fits single-threaded (31x faster on an 8-CPU
allocation). So the configured range absorbs a defect that has since been fixed,
and every estimate since has been pessimistic by that factor.

## What this calibrates, and in whose units

Not a kernel benchmark. The estimator's model is

    wall_clock = load_seconds + prefill_tokens/prefill_rate + decode_tokens/decode_rate

where `decode_tokens` is the BUDGET (`max_new_tokens` x generations), not tokens
actually emitted. So the only rate worth fitting is the one that makes THAT
equation reproduce a real job, which is what this computes: the census the cost
model would have built for a preset, divided by the wall clock the run took.

Two honest caveats, stated rather than buried:

* **Prefill is folded in.** The implied rate is `decode_tokens / elapsed`, so it
  absorbs prefill, probe fitting and judge latency. Prefill is 0.01-0.05 h of a
  4-7 h estimate (<1%), so the double-count is smaller than the spread of the
  measurements; separating them would need per-phase timing the run record does
  not carry.
* **The denominator excludes model load.** `throughput.elapsed_seconds` in a run
  record is the sum of per-family elapsed times, so `load_seconds` stays the
  separate term the estimator already treats it as.

Keyless for the tokenizer of an ungated model; gated ones need `./run`. No GPU,
no weights, no spend.

    ./run python scripts/tune_cost_model.py --preset ladder_plain_sft --preset causal_sweep
    ./run python scripts/tune_cost_model.py --all
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from internals_safety.config import (
    load_corpus_config,
    load_judge_config,
    load_measurements_config,
    load_model_config,
    load_preset,
)
from internals_safety.cost import census_phase0, load_cost_config
from internals_safety.data import prompt_set
from internals_safety.encodings.registry import load_ladder
from internals_safety.judges.harmbench import HarmBenchJudge
from internals_safety.judges.refusal import RefusalJudge
from internals_safety.paths import RUNS_DIR

# Only presets the phase-0 census describes can be calibrated against: an
# entrypoint that generates nothing has no decode budget to divide. Same guard
# `cost_model.py` carries, and for the same reason — a confident number for a
# different run shape is worse than no number.
PHASE0_SHAPED = frozenset({"phase0_regime_map"})


def measured_seconds(target: str, run_name: str) -> tuple[float, str] | None:
    """Wall clock of the completed run matching `run_name`, from its own record.

    Prefers the run record's `throughput.elapsed_seconds` — the family-sum the
    rate was already reported against — and falls back to summing the family
    rows, so a record written before the throughput block still calibrates.
    """
    root = RUNS_DIR / "phase0" / target
    if not root.exists():
        return None
    matches = sorted(d for d in root.iterdir() if d.name.startswith(run_name))
    for directory in reversed(matches):          # newest by timestamped name
        record = directory / "results.json"
        if not record.exists():
            continue
        payload = json.loads(record.read_text())
        elapsed = (payload.get("throughput") or {}).get("elapsed_seconds")
        if elapsed is None:
            families = (payload.get("metrics") or {}).get("families") or []
            rows = [f.get("elapsed_seconds") for f in families if f.get("elapsed_seconds")]
            elapsed = sum(rows) if rows else None
        if elapsed:
            return float(elapsed), directory.name
    return None


def decode_budget(preset, corpus, measurements, judge_template_chars: int) -> int:
    """The decode-token budget `cost_model.py` would build for this preset."""
    model_config = load_model_config(preset.target)
    ladder = load_ladder()
    families = preset.families if isinstance(preset.families, list) else list(ladder)
    n_prompts = preset.n_prompts or corpus.n_prompts
    harmful = prompt_set(corpus.harmful_set, limit=n_prompts)
    harmless = prompt_set(corpus.harmless_set, limit=n_prompts)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)

    def count_tokens(text: str) -> int:
        messages = []
        if model_config.system_prompt:
            messages.append({"role": "system", "content": model_config.system_prompt})
        messages.append({"role": "user", "content": text})
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    census = census_phase0(
        count_tokens, harmful, harmless, ladder, families, measurements, judge_template_chars
    )
    return census.decode_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", action="append", default=[])
    parser.add_argument("--all", action="store_true",
                        help="every phase-0-shaped preset with a completed local run")
    # The estimator has NO per-run fixed-cost term, so runs of different SHAPE
    # imply different rates: a 1-rung plaintext baseline amortises probe fitting
    # and judge setup over eight times less work than a ladder sweep and looks
    # ~1.8x "faster" per budgeted token. Calibrating across shapes would produce
    # a 5x range that is useless to a gate. These select the shape the estimator
    # is actually ASKED about, and everything excluded is printed, never dropped.
    parser.add_argument("--min-rungs", type=int, default=3)
    parser.add_argument("--min-prompts", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus = load_corpus_config()
    measurements = load_measurements_config()
    cost_config = load_cost_config()
    template_chars = max(len(HarmBenchJudge.prompt_template), len(RefusalJudge.prompt_template))

    names = args.preset
    if args.all:
        names = sorted(p.stem for p in (Path("conf") / "experiment").glob("*.yaml"))
    if not names:
        raise SystemExit("--preset or --all is required")

    rows = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        try:
            preset = load_preset(name)
        except Exception as error:                      # unreadable preset is data, not a crash
            skipped.append((name, f"unreadable: {error}"))
            continue
        if preset.entrypoint not in PHASE0_SHAPED:
            skipped.append((name, f"entrypoint {preset.entrypoint} — no decode budget"))
            continue
        if not preset.target or not preset.run_name:
            skipped.append((name, "no single target/run_name"))
            continue
        found = measured_seconds(preset.target, preset.run_name)
        if found is None:
            skipped.append((name, "no completed local run"))
            continue
        elapsed, directory = found
        budget = decode_budget(preset, corpus, measurements, template_chars)
        n_rungs = len(preset.families) if isinstance(preset.families, list) else len(load_ladder())
        n_prompts = preset.n_prompts or corpus.n_prompts
        rows.append({
            "preset": name, "target": preset.target, "run": directory,
            "n_rungs": n_rungs, "n_prompts": n_prompts,
            "decode_tokens": budget, "elapsed_seconds": elapsed,
            "implied_decode_tokens_per_s": budget / elapsed,
            "in_shape": n_rungs >= args.min_rungs and n_prompts >= args.min_prompts,
        })

    if skipped:
        print("skipped:")
        for name, why in skipped:
            print(f"  {name:<32} {why}")
        print()

    if not rows:
        print("!! nothing calibratable — no preset had both a phase-0 census and a local run")
        return 1

    rows.sort(key=lambda r: r["implied_decode_tokens_per_s"])
    print("%-28s %-20s %5s %5s %11s %8s %8s %6s" % (
        "preset", "target", "rung", "n", "decode_tok", "sec", "tok/s", "shape"))
    for row in rows:
        print("%-28s %-20s %5d %5d %11d %8.0f %8.1f %6s" % (
            row["preset"], row["target"][:20], row["n_rungs"], row["n_prompts"],
            row["decode_tokens"], row["elapsed_seconds"],
            row["implied_decode_tokens_per_s"], "in" if row["in_shape"] else "OUT"))

    out_of_shape = [r for r in rows if not r["in_shape"]]
    if out_of_shape:
        print(f"\nEXCLUDED as a different shape (< {args.min_rungs} rungs or < "
              f"{args.min_prompts} prompts), rates {min(r['implied_decode_tokens_per_s'] for r in out_of_shape):.1f}"
              f"-{max(r['implied_decode_tokens_per_s'] for r in out_of_shape):.1f}:")
        print("  " + ", ".join(r["preset"] for r in out_of_shape))
        print("  They are not slower or faster hardware — the estimator has no per-run")
        print("  fixed-cost term, so a small run amortises probe fitting and judge setup")
        print("  over less work. Excluding them is the LESS conservative choice where they")
        print("  sit low, so it is stated rather than silent.")

    rows = [r for r in rows if r["in_shape"]]
    if not rows:
        print("\n!! every calibratable run was out of shape — widen --min-rungs/--min-prompts")
        return 1
    rates = [r["implied_decode_tokens_per_s"] for r in rows]
    low, high = min(rates), max(rates)
    configured = cost_config.hardware["h200_144gb"].decode_tokens_per_s
    print(f"\nn = {len(rates)}   min {low:.1f}   median {statistics.median(rates):.1f}   max {high:.1f}")
    print(f"configured (h200_144gb): {list(configured)}")
    # The gate must not UNDER-predict time, so the low end is the one that has to
    # be defensible: it is the slowest real run, not a percentile. Rounding is
    # outward for the same reason.
    proposed = [int(low), int(high) + 1]
    print(f"proposed:                {proposed}   (slowest and fastest completed run, rounded outward)")
    if configured[1] < proposed[0]:
        factor = proposed[0] / configured[1]
        print(f"\n!! THE CONFIGURED RANGE DOES NOT OVERLAP THE MEASURED ONE. Every estimate")
        print(f"   made with it over-predicts time by at least {factor:.1f}x.")

    report = {"hardware": "h200_144gb", "configured": list(configured),
              "proposed": proposed, "n": len(rates),
              "shape_filter": {"min_rungs": args.min_rungs, "min_prompts": args.min_prompts},
              "excluded": [r["preset"] for r in out_of_shape], "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
