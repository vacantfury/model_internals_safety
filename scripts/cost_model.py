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
    load_pilot_config,
)
from internals_safety.cost import (
    census_phase0,
    estimate,
    format_estimate,
    format_range,
    load_cost_config,
)
from internals_safety.data import prompt_set
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


def judge_prices(model: str) -> tuple[float, float]:
    """Dollars per million input / output tokens, from llm_utils' price table.

    Imported lazily so `--help` and the hermetic tests never pull in the
    provider layer.
    """
    from llm_utils import LLMModel

    spec = LLMModel.from_string(model)
    return spec.input_price, spec.output_price


def main(argv: Sequence[str] | None = None) -> int:
    pilot = load_pilot_config()
    cost_config = load_cost_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help=f"conf/models/<name>.yaml; pilot set: {pilot.models}")
    parser.add_argument(
        "--hardware",
        default=DEFAULT_HARDWARE,
        choices=sorted(cost_config.hardware),
        help=f"GPU type to cost against (default: {DEFAULT_HARDWARE})",
    )
    parser.add_argument("--n-prompts", type=int, default=pilot.n_prompts)
    parser.add_argument("--families", nargs="+", default=None, help="default: every configured rung")
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="also project phases 1-3 from conf/cost.yaml (planning figures, not measurements)",
    )
    parser.add_argument("--per-family", action="store_true", help="show the token inflation per rung")
    args = parser.parse_args(argv)

    model_config = load_model_config(args.model)
    measurements = load_measurements_config()
    judge_config = load_judge_config()
    ladder = load_ladder()
    families = list(ladder) if args.families is None else args.families
    unknown = [family for family in families if family not in ladder]
    if unknown:
        raise SystemExit(f"unknown encoding families {unknown}; have {sorted(ladder)}")

    harmful = prompt_set(pilot.harmful_set, limit=args.n_prompts)
    harmless = prompt_set(pilot.harmless_set, limit=args.n_prompts)

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

    n_models = len(pilot.models) or 1
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
