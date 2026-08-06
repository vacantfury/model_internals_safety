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
from internals_safety.measurements.contract import Kind, Reading
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


def reading(
    summary: FamilyAbility,
    *,
    control_reading: float | None = None,
    control_margin: float | None = None,
    length_null_margin: float | None = None,
    detail: dict | None = None,
) -> Reading:
    """Measurement #1's condition-level verdict.

    **⚠️ This measurement has NO negative control, and the contract is how that
    became visible rather than a matter of remembering.** Every other instrument
    on the roster has one — deployment reads the format-decorrelation 2x2,
    decode_lens scores against a deranged plaintext, trajectory and entropy read
    against the ability-0 floor. Measurement #1 scores a response against its own
    plaintext and nothing else, so a scorer firing on incidental overlap (shared
    stopwords, echoed fragments of the instruction, boilerplate) is
    indistinguishable from a decode.

    The control that would close it is cheap and offline: score each response
    against a MISMATCHED plaintext from the same condition, exactly as
    `decode_lens.derange` does, and require the real-plaintext score to beat it.
    Filed rather than built here, because building it inside a wiring change
    would hide the finding.

    Until then `control_reading` defaults to None and every ability reading is
    non-reportable, naming "no negative control was run (P2)" as the reason.
    That is the correct state, not a bug: the numbers stay usable as run output
    and are barred from a paper table until the control exists.
    """
    return Reading(
        instrument="ability",
        kind=KIND,
        value=summary.recovery_rate,
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
