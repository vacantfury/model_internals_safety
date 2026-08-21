"""Screen a reading against a floor derived in a DIFFERENT run — safely.

**The problem this exists to solve, and why it is not a cross-run import.**
`instrument_layer.md` §2.4 settled that the control-floor statistic is
n-dependent, so a floor derived in one run may not be carried to a number
measured in another. That rule is correct and this module does not weaken it.

What it adds is the case §2.4 does not cover: **the same measurement present in
two runs.** `deployment` is deterministic given (model, family, corpus, cached
activations, probe config) — it fits on the plain contrast and transfers to the
encoded arm with no sampling anywhere. So a run that reads the identical value
from the identical cached tensors is not "another run" for screening purposes;
it is the same reading, in a run that also happened to carry control rungs.

AS-5's internals leg is exactly that case. Its number comes from the
single-family `scaffold-control` runs, where `control_floor.usable` is false
because a run with one rung has no can't-decode rung to build a floor from. The
identical reading also sits in runs carrying 2-6 controls.

**The identity check is the whole safety property, so it is not optional and it
is not a boolean a caller passes in.** `WitnessScreen` takes two `Provenance`
records and computes the match itself; a mismatch yields `None`, never a
verdict. The alternative shape — a `same_measurement=True` flag — is the one
that has failed in this repo four times in a week, and the fix each time was to
make the omission inexpressible rather than to document the requirement.

**What a witness does NOT license.** It screens the reading against a floor; it
says nothing about the floor's own grade. A `bound` floor stays a bound (§2.4:
the max statistic moves with n), and callers must report `floor.kind` beside
the verdict or they are quoting a bound as a distribution — the exact error
§2.2's table caused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from internals_safety.measurements.contract import Screen
from internals_safety.measurements.control_floor import ControlFloor

SCREEN_NAME = "control_floor_via_witness"

# Every field that must agree before two runs are called the same measurement.
# Listed once, here, so `matches` and the mismatch report cannot drift apart.
IDENTITY_FIELDS: tuple[str, ...] = (
    "model",
    "family",
    "harmful_digest",
    "harmless_digest",
    "n_prompts",
    "plain_harmful_activations",
    "plain_harmless_activations",
    "encoded_harmful_activations",
    "encoded_harmless_activations",
    "layer",
    "position",
    "transfer_auroc",
)


@dataclass(frozen=True)
class Provenance:
    """Everything that must agree for two runs to hold the same reading.

    Corpus digests AND activation paths AND the selected cell AND the value —
    deliberately redundant. Any one of them could in principle match by
    accident; all twelve cannot, and the cost of checking is a dict comparison.
    """

    model: str
    family: str
    harmful_digest: str | None
    harmless_digest: str | None
    n_prompts: int | None
    plain_harmful_activations: str | None
    plain_harmless_activations: str | None
    encoded_harmful_activations: str | None
    encoded_harmless_activations: str | None
    layer: int | None
    position: str | None
    transfer_auroc: float | None
    run: str = ""  # identity-irrelevant: the two runs differ by name, that is the point

    def mismatches(self, other: "Provenance") -> tuple[str, ...]:
        """Which identity fields disagree. Empty means the same measurement.

        A `None` on either side is a MISMATCH, not a wildcard: an absent
        activation path or an unrecorded digest is missing evidence, and
        missing evidence must never read as agreement.
        """
        bad: list[str] = []
        for name in IDENTITY_FIELDS:
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if mine is None or theirs is None or mine != theirs:
                bad.append(name)
        return tuple(bad)


@dataclass(frozen=True)
class WitnessScreen:
    """A reading, a witness run's floor, and whether the pairing is legitimate.

    `verdict` is tri-state and the `None` branch carries two very different
    causes, which is why `reason` is not optional: the witness may be the wrong
    measurement (never screen it), or the floor may be unusable (nothing to
    screen against). Collapsing them would repeat §2.7's `caesar3` error, where
    "the instrument could not resolve this" and "this did not decode" were read
    as the same answer.
    """

    reading: Provenance
    witness: Provenance
    floor: ControlFloor
    mismatched_fields: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mismatched_fields", self.reading.mismatches(self.witness)
        )

    @property
    def is_same_measurement(self) -> bool:
        return not self.mismatched_fields

    @property
    def verdict(self) -> bool | None:
        """Does the reading clear the witness run's floor? `None` = cannot say."""
        if not self.is_same_measurement:
            return None
        if self.reading.transfer_auroc is None:
            return None
        return self.floor.clears(self.reading.transfer_auroc, family=self.reading.family)

    @property
    def margin(self) -> float | None:
        if not self.is_same_measurement or self.floor.value is None:
            return None
        if self.reading.transfer_auroc is None:
            return None
        return self.reading.transfer_auroc - self.floor.value

    @property
    def reason(self) -> str:
        if not self.is_same_measurement:
            return (
                "NOT the same measurement — refusing to screen; disagrees on: "
                + ", ".join(self.mismatched_fields)
            )
        if self.floor.value is None:
            return f"witness floor unusable (kind={self.floor.kind}, n={self.floor.n})"
        if self.reading.family in self.floor.controls:
            return "the reading's own family is a control in the witness run"
        return (
            f"screened against a {self.floor.kind} floor from {self.witness.run} "
            f"(n={self.floor.n} controls)"
        )

    def screen(self) -> Screen:
        """Bounded ABOVE: a real reading must sit above the surface-feature floor."""
        observed = (
            float("nan")
            if (self.reading.transfer_auroc is None or not self.is_same_measurement)
            else self.reading.transfer_auroc
        )
        return Screen(
            name=SCREEN_NAME,
            observed=observed,
            floor=float("nan") if self.floor.value is None else self.floor.value,
            direction="above",
            defeats=(
                "a probe licensed by permutation alone, reading surface features "
                "at the level its own can't-decode rungs read them"
            ),
        )
