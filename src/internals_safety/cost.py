"""The cost model behind the experiment-run approval gate.

Family rule (owner 2026-07-22): before launching ANY run, report GPU count and
type, money, and wall-clock, and get an explicit go. This module turns a token
census into those three numbers; `scripts/cost_model.py` is its entrypoint and
`scripts/phase0_regime_map.py --dry-run` prints it inline.

## Why a token census rather than a pass count

`Plan` in the pilot counts forward passes and generations, which is the right
unit for "what work happens" but the wrong unit for "how long does it take".
A transformer's wall-clock is driven by tokens, in two regimes with throughputs
that differ by an order of magnitude:

* **prefill** — the prompt, processed in one compute-bound pass;
* **decode** — new tokens, generated one step at a time, memory-bandwidth-bound.

Encodings make the distinction load-bearing here rather than pedantic: the
ladder inflates prompt length by 1.2x (reverse_words) to 17x (binary), so a
per-family estimate built on pass counts alone is wrong by an order of magnitude
across the rungs it is summing over. `census_phase0` therefore tokenises the
real corpus under every rung with the real tokenizer.

## Ranges, not point estimates

Hardware throughput cannot be known before running on the actual node, so every
hardware number in `conf/cost.yaml` is a range and every result here is a range.
The gate is served better by "1-4 hours, and the partition wall is 8" than by a
confident single number that is wrong. `Estimate.fits_wall_clock` reports
against the scheduler's wall rather than leaving the reader to compare.

## Prices come from llm_utils, not from our config

`llm_utils.LLMModel` already carries the per-model price table the whole family
bills against. Copying prices into `conf/cost.yaml` would create a second home
for a number that changes upstream, so the judge bill reads them from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from internals_safety.config import MeasurementsConfig, StrictModel, load_yaml
from internals_safety.paths import CONF_DIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from internals_safety.data import Prompt
    from internals_safety.encodings.base import Encoder

# A (low, high) pair from config. Kept as a plain tuple rather than a class
# because it is arithmetic, not a domain object.
Range = tuple[float, float]

SECONDS_PER_HOUR = 3600.0  # constant: seconds per hour


class HardwareProfile(StrictModel):
    """One GPU type's throughput assumptions."""

    label: str
    vram_gb: float
    prefill_tokens_per_s: Range
    decode_tokens_per_s: Range
    load_seconds: float


class SchedulerLimits(StrictModel):
    """The queue's hard walls — see `conf/cost.yaml` for provenance."""

    partition: str
    max_wall_clock_hours: float
    gpus_per_job: int
    max_concurrent_jobs: int
    gpu_usd_per_hour: float = 0.0


class JudgeCostConfig(StrictModel):
    chars_per_token: int = 4
    output_tokens_per_verdict: Range = (150.0, 700.0)


class PhaseScale(StrictModel):
    """A later phase expressed as multiples of the phase-0 census.

    The rough half of "a rough cost model": phase 0 is costed from measured
    tokens, phases 1-3 from the design plus a multiplier, re-derived from real
    numbers once the phase before them has run.
    """

    description: str
    prefill_multiple: float = 0.0
    decode_multiple: float = 0.0
    judge_multiple: float = 0.0
    finetunes: int = 0
    gpu_hours_per_finetune: Range = (0.0, 0.0)


class CostConfig(StrictModel):
    hardware: dict[str, HardwareProfile]
    scheduler: SchedulerLimits
    judge: JudgeCostConfig = JudgeCostConfig()
    phases: dict[str, PhaseScale] = {}


def load_cost_config(conf_dir: Path = CONF_DIR) -> CostConfig:
    return CostConfig(**load_yaml(conf_dir / "cost.yaml"))


@dataclass(frozen=True)
class TokenCensus:
    """Exactly how many tokens a run moves, per regime.

    Every field is a measured count when it comes from `census_phase0`, not an
    assumption — that is the point of the class.
    """

    prefill_tokens: int
    decode_tokens: int
    judge_calls: int
    judge_input_chars: int
    # Longest single rendered prompt, for the truncation check below.
    max_prompt_tokens: int = 0  # plumbing: accumulator start; the real value is measured
    # Per-rung mean attack-prompt length, for the report's inflation column.
    per_family_mean_prompt_tokens: dict[str, float] = field(default_factory=dict)

    def scaled(self, prefill: float, decode: float, judge: float) -> "TokenCensus":
        """This census times per-regime multiples — how later phases are costed."""
        return TokenCensus(
            prefill_tokens=int(self.prefill_tokens * prefill),
            decode_tokens=int(self.decode_tokens * decode),
            judge_calls=int(self.judge_calls * judge),
            judge_input_chars=int(self.judge_input_chars * judge),
            max_prompt_tokens=self.max_prompt_tokens,
        )


@dataclass(frozen=True)
class Estimate:
    """What the approval gate is owed, as ranges."""

    hardware: str
    gpus: int
    gpu_hours: Range
    prefill_hours: Range
    decode_hours: Range
    judge_usd: Range
    gpu_usd: Range
    max_wall_clock_hours: float

    @property
    def total_usd(self) -> Range:
        return (self.judge_usd[0] + self.gpu_usd[0], self.judge_usd[1] + self.gpu_usd[1])

    @property
    def fits_wall_clock(self) -> bool:
        """Whether even the pessimistic end fits the partition's wall.

        Checked rather than reported because the failure is not slowness: a job
        that exceeds the wall is killed part-way through the sweep, losing the
        families it had not reached yet.
        """
        return self.gpu_hours[1] <= self.max_wall_clock_hours


def _bounds(value: Range) -> Range:
    low, high = float(value[0]), float(value[1])
    return (min(low, high), max(low, high))


def estimate(
    census: TokenCensus,
    hardware: HardwareProfile,
    scheduler: SchedulerLimits,
    judge_config: JudgeCostConfig,
    judge_input_price: float,
    judge_output_price: float,
    # config(gpus): cost.scheduler.gpus_per_job
    gpus: int = 1,
) -> Estimate:
    """Turn a census into GPU-hours, dollars and a wall-clock verdict.

    `judge_*_price` are dollars per million tokens, passed in rather than looked
    up so this function stays free of provider imports (the hermetic test suite
    never touches llm_utils).

    Slow throughput gives the pessimistic time, so the ranges cross: the low end
    of the hours range uses the HIGH end of tokens/second.
    """
    prefill_low, prefill_high = _bounds(hardware.prefill_tokens_per_s)
    decode_low, decode_high = _bounds(hardware.decode_tokens_per_s)

    prefill_hours = (
        census.prefill_tokens / prefill_high / SECONDS_PER_HOUR,
        census.prefill_tokens / prefill_low / SECONDS_PER_HOUR,
    )
    decode_hours = (
        census.decode_tokens / decode_high / SECONDS_PER_HOUR,
        census.decode_tokens / decode_low / SECONDS_PER_HOUR,
    )
    load_hours = hardware.load_seconds / SECONDS_PER_HOUR
    gpu_hours = (
        prefill_hours[0] + decode_hours[0] + load_hours,
        prefill_hours[1] + decode_hours[1] + load_hours,
    )

    input_tokens = census.judge_input_chars / judge_config.chars_per_token
    out_low, out_high = _bounds(judge_config.output_tokens_per_verdict)
    judge_usd = (
        (input_tokens * judge_input_price + census.judge_calls * out_low * judge_output_price)
        / 1_000_000,
        (input_tokens * judge_input_price + census.judge_calls * out_high * judge_output_price)
        / 1_000_000,
    )
    gpu_usd = (
        gpu_hours[0] * gpus * scheduler.gpu_usd_per_hour,
        gpu_hours[1] * gpus * scheduler.gpu_usd_per_hour,
    )

    return Estimate(
        hardware=hardware.label,
        gpus=gpus,
        gpu_hours=gpu_hours,
        prefill_hours=prefill_hours,
        decode_hours=decode_hours,
        judge_usd=judge_usd,
        gpu_usd=gpu_usd,
        max_wall_clock_hours=scheduler.max_wall_clock_hours,
    )


def census_phase0(
    count_tokens: "Callable[[str], int]",
    harmful: "Sequence[Prompt]",
    harmless: "Sequence[Prompt]",
    ladder: "dict[str, Encoder]",
    families: "Sequence[str]",
    measurements: "MeasurementsConfig",
    judge_template_chars: int,
) -> TokenCensus:
    """Measure exactly what a phase-0 run over `families` moves.

    `count_tokens` renders and tokenises one user message — passed in so this
    stays importable (and testable) without transformers or a weights download;
    `scripts/cost_model.py` supplies the model's real tokenizer.

    The four token streams, matching `phase0_regime_map.run_family`:

    1. plain harmful + plain harmless captures, once per model (the plain
       condition does not depend on the rung, which is what the cache buys);
    2. encoded harmful + encoded harmless captures, per family;
    3. restate prompts, prefilled before measurement #1 generates;
    4. attack prompts, prefilled again before measurement #4 generates.

    Stream 4 repeats stream 2's prompts because capture and generation are
    separate forward passes over the same text — real work, so it is counted
    twice on purpose rather than deduplicated into a prettier number.

    Judge input is charged at the *maximum* response budget: responses do not
    exist at census time, and the gate wants the ceiling.
    """
    prefill = sum(count_tokens(prompt.text) for prompt in harmful)
    prefill += sum(count_tokens(prompt.text) for prompt in harmless)

    per_family_mean: dict[str, float] = {}
    max_prompt_tokens = 0
    decode = 0
    judge_calls = 0
    for family in families:
        encoder = ladder[family]
        attack_harmful = [encoder.encode(prompt.text).attack_prompt for prompt in harmful]
        attack_harmless = [encoder.encode(prompt.text).attack_prompt for prompt in harmless]
        restate = [encoder.encode(prompt.text).restate_prompt for prompt in harmful]

        harmful_tokens = [count_tokens(text) for text in attack_harmful]
        harmless_tokens = [count_tokens(text) for text in attack_harmless]
        restate_tokens = [count_tokens(text) for text in restate]

        # Capture (both classes) + the attack prefill before generation.
        prefill += sum(harmful_tokens) + sum(harmless_tokens) + sum(harmful_tokens)
        prefill += sum(restate_tokens)
        decode += len(harmful) * (
            measurements.ability.max_new_tokens + measurements.behavior.max_new_tokens
        )
        judge_calls += 2 * len(harmful)

        combined = harmful_tokens + harmless_tokens
        per_family_mean[family] = sum(combined) / len(combined) if combined else 0.0
        max_prompt_tokens = max(max_prompt_tokens, max(combined + restate_tokens, default=0))

    response_chars = measurements.behavior.max_new_tokens * 4
    behaviour_chars = sum(len(prompt.text) for prompt in harmful) * len(families)
    judge_input_chars = judge_calls * (judge_template_chars + response_chars)
    judge_input_chars += 2 * behaviour_chars

    return TokenCensus(
        prefill_tokens=prefill,
        decode_tokens=decode,
        judge_calls=judge_calls,
        judge_input_chars=judge_input_chars,
        max_prompt_tokens=max_prompt_tokens,
        per_family_mean_prompt_tokens=per_family_mean,
    )


# plumbing(places): decimal places in a printed range — presentation only
def format_range(value: Range, unit: str = "", places: int = 1) -> str:
    low, high = value
    suffix = f" {unit}" if unit else ""
    return f"{low:.{places}f}-{high:.{places}f}{suffix}"


def format_estimate(estimate_: Estimate, name: str) -> str:
    lines = [
        f"{name}",
        f"  hardware            {estimate_.gpus} x {estimate_.hardware}",
        f"  GPU-hours           {format_range(estimate_.gpu_hours)}"
        f"   (prefill {format_range(estimate_.prefill_hours, places=2)}"
        f" + decode {format_range(estimate_.decode_hours, places=2)})",
        f"  judge API spend     ${format_range(estimate_.judge_usd, places=2)}",
    ]
    if estimate_.gpu_usd[1] > 0:
        lines.append(f"  GPU spend           ${format_range(estimate_.gpu_usd, places=2)}")
    verdict = "fits" if estimate_.fits_wall_clock else "EXCEEDS"
    lines.append(
        f"  wall-clock wall     {verdict} the {estimate_.max_wall_clock_hours:g}h partition limit"
    )
    return "\n".join(lines)
