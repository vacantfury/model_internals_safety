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
