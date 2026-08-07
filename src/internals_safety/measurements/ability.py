"""Measurement #1 — decode-and-restate ability.

*Can* the model recover the plaintext when recovering it is the task? This is
the (C) axis: a rung the model cannot decode under direct instruction is a
capability limit, and no safety intervention can be expected to repair it.

**What this measurement is not.** It is not evidence that the model decoded
anything during an attack. It runs a different prompt in a different forward
pass, with decoding explicitly requested. Reading a regime off this number alone
was the design hole in the v1 framing; measurement #2 (deployment, the in-situ
content probe on the attack forward pass) is what closes it, and the (D) regime
is precisely the gap between the two.

Secondary use: this is also the ladder's difficulty axis. Because it is a
measured quantity, "obfuscation depth" in RQ2 becomes an empirical ordering
rather than a hand-sorted list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from internals_safety.config import AbilityConfig
from internals_safety.measurements.ability_control import AbilityControl, zero_count_margin
from internals_safety.measurements.contract import Claim, Kind, Reading
from internals_safety.encodings.base import EncodedPrompt
from internals_safety.encodings.recovery import RecoveryScore, score_recovery
from internals_safety.models.generate import generate
from internals_safety.models.loader import LoadedModel


@dataclass(frozen=True)
class AbilityRecord:
    family: str
    plaintext: str
    ciphertext: str
    response: str
    score: RecoveryScore


def measure_ability(
    loaded: LoadedModel,
    encoded: list[EncodedPrompt],
    config: AbilityConfig | None = None,
) -> list[AbilityRecord]:
    """Ask for the plaintext directly and score what comes back."""
    settings = config or AbilityConfig()
    responses = generate(
        loaded,
        [item.restate_prompt for item in encoded],
        max_new_tokens=settings.max_new_tokens,
        batch_size=settings.batch_size,
    )
    return [
        AbilityRecord(
            family=item.family,
            plaintext=item.plaintext,
            ciphertext=item.ciphertext,
            response=response,
            score=score_recovery(item.plaintext, response, item.ciphertext),
        )
        for item, response in zip(encoded, responses)
    ]


@dataclass(frozen=True)
class FamilyAbility:
    family: str
    n: int
    recovery_rate: float
    mean_similarity: float
    echo_rate: float

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"{self.family:<20} n={self.n:<4} recovered={self.recovery_rate:.2f} "
            f"similarity={self.mean_similarity:.2f} echoed={self.echo_rate:.2f}"
        )


def summarize_by_family(
    records: list[AbilityRecord], config: AbilityConfig | None = None
) -> list[FamilyAbility]:
    """Per-rung ability. Sorted hardest-first — this ordering IS the difficulty
    axis, and it is why the ladder does not need to be sorted by intuition."""
    settings = config or AbilityConfig()
    grouped: dict[str, list[AbilityRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)

    summaries = [
        FamilyAbility(
            family=family,
            n=len(group),
            recovery_rate=sum(
                record.score.is_recovered(
                    settings.similarity_threshold,
                    settings.content_overlap_threshold,
                    settings.order_blind_overlap_threshold,
                )
                for record in group
            )
            / len(group),
            mean_similarity=sum(record.score.similarity for record in group) / len(group),
            echo_rate=sum(record.score.echoed_ciphertext for record in group) / len(group),
        )
        for family, group in grouped.items()
    ]
    return sorted(summaries, key=lambda summary: (summary.recovery_rate, summary.mean_similarity))


# P1 — checked across the roster by `assert_distinct_questions`.
QUESTION = "can the model recover the plaintext when it is explicitly asked to decode"
KIND: Kind = "correlational"


def claim_direction(recovery_rate: float, max_ability: float) -> Claim:
    """Is this condition's reading asserting a decode, or asserting an absence?

    **One home for the concept, deliberately.** `max_ability` is
    `controls.control_ability_max` — the same cut that decides whether a rung may
    serve as an ability-0 negative control for the other instruments. A rung that
    reads no decoding IS the rung making a null claim, so a second, separately
    tuned cut for "counts as a null" would be the `DEFAULT_LENGTH_BINS` failure
    again: two copies of one idea, free to drift, with a claim silently meaning
    different things in different places.

    NaN reads as a positive claim, i.e. the strict route. An unreadable rate is
    not evidence of absence.
    """
    if recovery_rate != recovery_rate:  # NaN
        return "positive"
    return "null" if recovery_rate <= max_ability else "positive"


def reading(
    summary: FamilyAbility,
    *,
    control: AbilityControl | None = None,
    control_reading: float | None = None,
    control_margin: float | None = None,
    length_null_margin: float | None = None,
    claim: Claim = "positive",
    sensitivity: float | None = None,
    sensitivity_floor: float | None = None,
    detail: dict | None = None,
) -> Reading:
    """Measurement #1's condition-level verdict.

    **The negative control this measurement lacked is now built** —
    `ability_control.measure_ability_control`, offline over cached text. Pass it
    as `control` and every contract axis is filled from it: the free derangement
    supplies P2, the length-matched derangement supplies P3, and `identity_rate`
    supplies the sensitivity arm. The explicit `control_reading` /
    `control_margin` / `length_null_margin` / `sensitivity` arguments remain for
    callers that compute their own; `control` wins where both are given.

    **The ability-0 case is now reportable, and it was not before (TODO 42).** An
    ability-0 rung's value equals its negative control by construction, so P2 can
    never license it — which used to mean the rungs that CALIBRATE three other
    instruments were themselves unreportable. Pass `claim="null"` (see
    `claim_direction`) and the contract routes to the sensitivity arm instead:
    the evidence becomes "this scorer demonstrably fires when it should", not
    "this scorer stayed quiet".

    The control's bar is still the derived rule-of-three bound for this
    condition's n, not a chosen constant — see `ability_control.zero_count_margin`.
    """
    if control is not None:
        control_reading = control.mismatched_rate
        control_margin = zero_count_margin(control.n)
        length_null_margin = control.length_margin
        sensitivity = control.identity_rate
        detail = {
            "control_identity_rate": control.identity_rate,
            "control_scorer_is_functional": control.scorer_is_functional,
            "control_n_length_matched": control.n_length_matched,
            "control_length_mismatched_rate": control.length_mismatched_rate,
            **(detail or {}),
        }
    return Reading(
        instrument="ability",
        kind=KIND,
        value=summary.recovery_rate,
        claim=claim,
        sensitivity=sensitivity,
        sensitivity_floor=sensitivity_floor,
        operating_point=(
            "fraction of prompts recovered under the three-route rule (exact/contains "
            "short-circuit, similarity >= cut with a content-overlap veto, order-blind "
            "overlap branch), cuts in conf/measurements.yaml"
        ),
        # Direct behavioural read: there is no null to fail, only an empty
        # condition to be unable to read.
        licensed=None if summary.n == 0 else True,
        control_reading=control_reading,
        control_margin=control_margin,
        length_null_margin=length_null_margin,
        # No layer or position is selected — nothing is searched over.
        selection_inside_null=True,
        detail={
            "family": summary.family,
            "n": summary.n,
            "mean_similarity": summary.mean_similarity,
            "echo_rate": summary.echo_rate,
            **(detail or {}),
        },
    )
