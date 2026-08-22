"""Cost model — what the experiment-run approval gate is owed, before any run.

Family rule (owner 2026-07-22): report GPU count and type, money, and wall-clock
and get an explicit go BEFORE launching. This script produces those three
numbers for phase 0 from measured token counts, and projects phases 1-3 from the
design in `s1_idea_check.md` §7.

It loads a tokenizer (a few MB) but never the model weights, so it needs no GPU,
no API key and no cluster — it is meant to be run on a laptop while deciding
whether to queue anything.

    uv run python scripts/cost_model.py --model qwen2_5_7b_instruct
    uv run python scripts/cost_model.py --model qwen2_5_7b_instruct --hardware v100_32gb
    uv run python scripts/cost_model.py --model qwen2_5_7b_instruct --all-phases

## What is measured and what is assumed

Measured, by tokenising the real corpus under every rung: prompt lengths, and
therefore the prefill half of the estimate. This matters because the ladder
inflates prompt length by 1.2x to 17x, so no single assumed inflation factor
would be within an order of magnitude across the rungs.

Assumed, from `conf/cost.yaml`: hardware throughput, which cannot be known
before running on the actual node. Every such number is a range and every
output is a range. The first real run records its observed rates into the run
record, which is what replaces these guesses for phases 1-3.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from internals_safety.config import (
    load_judge_config,
    load_measurements_config,
    load_model_config,
    load_corpus_config,
    load_preset,
)
from internals_safety.cost import (
    census_phase0,
    estimate,
    format_estimate,
    format_range,
    load_cost_config,
)
from internals_safety.pipeline import load_contrast_sets
from internals_safety.encodings.registry import load_ladder
from internals_safety.judges.harmbench import HarmBenchJudge
from internals_safety.judges.refusal import RefusalJudge

# Default when the caller does not name one. NOT the highest-availability GPU
# on the target cluster — that is the V100, which this project's torch cannot
# use at all (Volta/sm_70, dropped by the cu130 build; see conf/cost.yaml). The
# smallest card the job can actually run on is the honest default, because an
# estimate against hardware the job would refuse to start on is worse than no
# estimate.
DEFAULT_HARDWARE = "h200_144gb"  # the only measured profile (conf/cost.yaml) and what ops/phase0.sbatch requests

# Which entrypoints the phase-0 census actually describes. `census_phase0`
# prices a ladder sweep with generation and judge calls; an entrypoint doing
# none of those gets a confident number for work it will never do.
#
# **This set exists because the wrong number was silent.** `--preset
# sae_pregate_instruct` reported 5.8-10.3 GPU-hours, a $2.08-6.26 judge bill and
# "EXCEEDS the 8h partition limit" for a job the same preset declares as three
# 1-hour tasks that run no ladder and call no judge: the preset sets no
# `families`, so the census fell back to all 19 rungs. A crash would have been
# kinder — the approval gate's worst failure is a plausible number for a
# different run.
#
# `as6_guard_probe` is deliberately NOT here even though it sweeps the ladder:
# it reads a verdict from the logits at one position, so it neither generates
# nor judges, and the decode half of the census would be pure invention.
_PHASE0_SHAPED = frozenset({"phase0_regime_map"})

# Where each other entrypoint's forward-pass census actually lives. DELEGATED,
# not reimplemented: `sae_pregate --dry-run` already prints its pass count and
# its zero judge spend, and a second copy here would be the same two-sources-of-
# truth defect one level up. `relicense_probes` has no dry-run because it has
# nothing to census — no model, no generation, no API.
# `--sae-layer` is passed explicitly even though the pass count does not depend
# on it: the script's dry-run echoes the layer back, and defaulting it would
# print a census header naming a layer the preset does not run — the same
# describes-a-different-run defect this whole dispatch exists to close.
_COST_ELSEWHERE: dict[str, str | None] = {
    "sae_pregate": "./run python scripts/sae_pregate.py --model {target} "
    "--n-prompts {n_prompts} --sae-layer {first_layer} --dry-run",
    "as6_guard_probe": "./run python scripts/as6_guard_probe.py --guard {target} --dry-run",
    # Reads guard verdicts over CACHED restatements, so it generates nothing and
    # calls no judge; its dry-run prints the pass count and a $0.00 line.
    "decoder_guard_pipeline": "./run python scripts/decoder_guard_pipeline.py "
    "--guard {target} --base-model {decoder} --families {families} "
    "--n-prompts {n_prompts} --dry-run",
    "relicense_probes": None,
    # I7. Generates AND judges, unlike the three above, so it is the reason
    # `_JUDGES` below exists — the "$0.00, calls no judge" line was unconditional
    # and would have printed a false zero into an approval request. Its census
    # is its own dry-run, which prints generations and judge calls per task.
    "encoding_ablation": "./run python scripts/encoding_ablation.py --model {target} "
    "--families {families} --n-prompts {n_prompts} --dry-run",
}

# Entrypoints on the delegated route that DO spend judge money. The gate line
# below is driven by this rather than asserting zero for everything that is not
# phase-0 shaped: a false $0.00 in an approval request is worse than no number.
_JUDGES = frozenset({"encoding_ablation"})

# `PresetConfig.tasks` needs an outputs root to build paths with, but this
# script only ever counts the rows. Any root gives the same count; the real one
# is the launcher's business, so naming a fake one keeps the two from drifting.
_TASK_COUNT_ROOT = "/outputs"


def report_declared_cost(name: str, preset, outputs_dir: str) -> int:
    """The approval-gate triple for a preset the phase-0 census does not describe.

    Every number printed here is one the preset itself declares or that this
    repo knows structurally — task count, the resource ask, and the fact that a
    job with no judge spends nothing. Nothing is estimated, because an estimate
    of the wrong shape is what this branch exists to prevent.
    """
    resources = preset.resources
    n_tasks = len(preset.tasks(outputs_dir))
    target = preset.target or ", ".join(preset.targets)

    print(f"costing preset {name!r}\n")
    print(f"entrypoint            {preset.entrypoint}")
    print(f"target                {target}")
    print(f"array tasks           {n_tasks}")
    print()
    print("THE APPROVAL-GATE TRIPLE (declared, not estimated)")
    if resources.is_gpu_job:
        print(f"  hardware            {n_tasks} x {resources.gres} on partition '{resources.partition}'")
    else:
        print(f"  hardware            none — {resources.cpus} CPU / {resources.mem} on partition '{resources.partition}'")
    print(f"  wall-clock          {resources.time} per task (a CEILING the scheduler enforces)")
    if preset.entrypoint in _JUDGES:
        print("  judge API spend     NOT ZERO — this entrypoint judges every generation;")
        print("                      the per-task call count is in its dry-run below")
    else:
        print(f"  judge API spend     $0.00 — this entrypoint calls no judge")
    print()

    census = _COST_ELSEWHERE[preset.entrypoint]
    if census is None:
        print("The phase-0 census does not describe this run and there is nothing to")
        print("census: it loads no model, generates no tokens and calls no API. The")
        print("ceiling above IS the wall-clock ask.")
    else:
        command = census.format(
            target=target,
            n_prompts=preset.n_prompts or 100,
            first_layer=preset.sae_layers[0] if preset.sae_layers else 15,
            families=" ".join(preset.families) if isinstance(preset.families, list) else "all",
            decoder=preset.decoder or "UNSET",
        )
        print("The phase-0 census does not describe this run. For the forward-pass")
        print(f"count, ask the entrypoint that owns it (PER TASK — there are {n_tasks}):")
        print(f"\n    {command}\n")
    return 0


def judge_prices(model: str) -> tuple[float, float]:
    """Dollars per million input / output tokens, from llm_utils' price table.

    Imported lazily so `--help` and the hermetic tests never pull in the
    provider layer.
    """
    from llm_utils import LLMModel

    spec = LLMModel.from_string(model)
    return spec.input_price, spec.output_price


def main(argv: Sequence[str] | None = None) -> int:
    corpus = load_corpus_config()
    cost_config = load_cost_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None, help=f"conf/models/<name>.yaml; pilot set: {corpus.models}")
    parser.add_argument(
        "--preset",
        default=None,
        help="cost exactly what conf/experiment/<name>.yaml declares, filling in --model, "
        "--families and --n-prompts from it. Without this the estimate prices the FULL "
        "ladder at the pilot's corpus size, which is not what a scoped preset submits — "
        "an approval gate reading a number for a different run is worse than no number.",
    )
    parser.add_argument(
        "--hardware",
        default=DEFAULT_HARDWARE,
        choices=sorted(cost_config.hardware),
        help=f"GPU type to cost against (default: {DEFAULT_HARDWARE})",
    )
    parser.add_argument("--n-prompts", type=int, default=corpus.n_prompts)
    parser.add_argument(
        "--corpus",
        default=None,
        choices=sorted(corpus.pairs),
        help=f"contrast pair from conf/corpus.yaml (default: {corpus.default_pair})",
    )
    parser.add_argument("--families", nargs="+", default=None, help="default: every configured rung")
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="also project phases 1-3 from conf/cost.yaml (planning figures, not measurements)",
    )
    parser.add_argument("--per-family", action="store_true", help="show the token inflation per rung")
    args = parser.parse_args(argv)

    if args.preset:
        preset = load_preset(args.preset)
        for flag, value in (("--model", args.model), ("--families", args.families)):
            if value is not None:
                raise SystemExit(f"{flag} and --preset both set; the preset IS the declaration")
        # Dispatch on the entrypoint the preset already declares, BEFORE filling
        # in the phase-0 arguments. Reaching the census with a preset it does
        # not describe is how `sae_pregate_instruct` came to be priced at 19
        # rungs: `families` is unset because the entrypoint has no ladder, and
        # the phase-0 default for unset families is "every configured rung".
        if preset.entrypoint not in _PHASE0_SHAPED:
            return report_declared_cost(args.preset, preset, _TASK_COUNT_ROOT)
        args.model = preset.target
        if isinstance(preset.families, list):
            args.families = preset.families
        if preset.n_prompts is not None:
            args.n_prompts = preset.n_prompts
        # ⚠️ The preset's CORPUS, not the default one. The census tokenises real
        # prompts, so a held-out pair with different prompt lengths gives a
        # different estimate — and this script IS the approval-gate artifact, so
        # costing the wrong corpus would be costing a run nobody is proposing.
        args.corpus = preset.corpus
        print(f"costing preset {args.preset!r}\n")
    if not args.model:
        raise SystemExit("--model or --preset is required")

    model_config = load_model_config(args.model)
    measurements = load_measurements_config()
    judge_config = load_judge_config()
    ladder = load_ladder()
    families = list(ladder) if args.families is None else args.families
    unknown = [family for family in families if family not in ladder]
    if unknown:
        raise SystemExit(f"unknown encoding families {unknown}; have {sorted(ladder)}")

    pair_name, pair = corpus.pair(args.corpus)
    # Through `load_contrast_sets`, not two bare `prompt_set` calls: the matched
    # subset is derived there, and a census taken over the raw files would count
    # prompts the run will never send.
    harmful, harmless = load_contrast_sets(
        pair.harmful_set, pair.harmless_set, args.n_prompts, matching=pair.matching
    )
    print(f"corpus pair {pair_name!r}: {len(harmful)} harmful + {len(harmless)} benign\n")

    print(f"tokenising {model_config.hf_id} ...", flush=True)
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

    # The judge bill is charged against the longer of the two templates, since
    # every response is sent to both and the difference is not worth a second
    # census pass.
    template_chars = max(
        len(HarmBenchJudge.prompt_template), len(RefusalJudge.prompt_template)
    )
    census = census_phase0(
        count_tokens, harmful, harmless, ladder, families, measurements, template_chars
    )

    hardware = cost_config.hardware[args.hardware]
    input_price, output_price = judge_prices(judge_config.model)
    phase0 = estimate(
        census,
        hardware,
        cost_config.scheduler,
        cost_config.judge,
        input_price,
        output_price,
    )

    context_limit = getattr(tokenizer, "model_max_length", 0) or 0
    print()
    print(f"corpus                {len(harmful)} harmful + {len(harmless)} harmless, {len(families)} rungs")
    print(f"prefill tokens        {census.prefill_tokens:,}  (measured)")
    print(f"decode tokens         {census.decode_tokens:,}  (max budget)")
    print(f"judge calls           {census.judge_calls:,} at {judge_config.model} "
          f"(${input_price}/M in, ${output_price}/M out)")
    print(f"longest prompt        {census.max_prompt_tokens:,} tokens", end="")
    if 0 < context_limit < 10**6:
        fits = "fits" if census.max_prompt_tokens < context_limit else "EXCEEDS"
        print(f"  ({fits} the {context_limit:,}-token context)")
    else:
        print()

    if args.per_family:
        print()
        plain = census.prefill_tokens and min(census.per_family_mean_prompt_tokens.values())
        print(f"{'rung':<20}{'mean prompt tokens':>20}{'vs shortest rung':>18}")
        for family, mean in sorted(census.per_family_mean_prompt_tokens.items(), key=lambda kv: kv[1]):
            print(f"{family:<20}{mean:>20.0f}{mean / plain:>17.2f}x")

    print()
    print(format_estimate(phase0, f"PHASE 0 — one model, {len(families)} rungs"))

    n_models = len(corpus.models) or 1
    print()
    print(
        f"PHASE 0 — full pilot ({n_models} models)\n"
        f"  GPU-hours           {format_range((phase0.gpu_hours[0] * n_models, phase0.gpu_hours[1] * n_models))}\n"
        f"  judge API spend     ${format_range((phase0.judge_usd[0] * n_models, phase0.judge_usd[1] * n_models), places=2)}\n"
        f"  jobs                {n_models} x 1 GPU on partition "
        f"'{cost_config.scheduler.partition}' (limit {cost_config.scheduler.max_concurrent_jobs} concurrent)"
    )

    if args.all_phases:
        print()
        print("PROJECTED — planning figures from conf/cost.yaml, not measurements.")
        print("Each is re-derived from real numbers once the phase before it has run.")
        for name, scale in cost_config.phases.items():
            scaled = census.scaled(
                scale.prefill_multiple, scale.decode_multiple, scale.judge_multiple
            )
            projected = estimate(
                scaled,
                hardware,
                cost_config.scheduler,
                cost_config.judge,
                input_price,
                output_price,
            )
            hours = (
                projected.gpu_hours[0] * n_models
                + scale.finetunes * scale.gpu_hours_per_finetune[0],
                projected.gpu_hours[1] * n_models
                + scale.finetunes * scale.gpu_hours_per_finetune[1],
            )
            print()
            print(f"{name} — {scale.description}")
            print(f"  GPU-hours           {format_range(hours)}", end="")
            if scale.finetunes:
                print(f"   (incl. {scale.finetunes} fine-tunes)")
            else:
                print()
            print(
                "  judge API spend     $"
                f"{format_range((projected.judge_usd[0] * n_models, projected.judge_usd[1] * n_models), places=2)}"
            )

    print()
    print(
        "These are the numbers the approval gate needs (family rule, owner "
        "2026-07-22). Nothing launches from this script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
