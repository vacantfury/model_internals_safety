"""The instrument contract — what every instrument must declare, in types.

**Why this exists.** Build plan §2 states seven principles every instrument must
satisfy (P1 distinct question · P2 silent on negative controls · P3 clears the
length null · P4 beats a black-box baseline · P5 correlational and causal never
mixed · P6 non-circular operating point · P7 selection inside the null). Until
now each instrument declared those in prose, in its module docstring, where
nothing could check them and nothing could compare across instruments.

That is not a style problem. The repo's three worst instrument defects were all
*undeclared* properties: a probe asserting `False` where it meant "unmeasured"
(§1.5), an operating point with a 50% false-positive rate by construction that
`assign_regime` never saw (§2.1), and a probe riding character length that the
licensing test structurally could not detect (§5). Each is a field on `Reading`
below, because a field is a thing a reviewer and a script can both read.

**What this is not.** Not a framework, not a runner, not a registry with
lifecycle. It is a Protocol and a result type. Instruments stay plain functions;
the wrappers here only assert what they already compute.

**The load-bearing method is `reportable`.** One place answers "may this number
appear in a paper", and it fails CLOSED on every axis: unmeasured, unlicensed,
loud on controls, or inside the length null all return False. A number that
cannot say why it is trustworthy is not trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

Kind = Literal["correlational", "causal"]


@dataclass(frozen=True)
class Reading:
    """One instrument's verdict on one condition, with its evidence attached."""

    instrument: str
    # P5. Correlational and causal claims are labelled separately and never
    # merged into a single "it works" number. Stored, not inferred, because the
    # distinction is about what the measurement DID, not what it looks like.
    kind: Kind
    # The quantity the instrument reports. Deliberately unconstrained — an AUROC,
    # a probability margin, a bypass fraction. `operating_point` says how to read
    # it.
    value: float

    # P6. How `value` was turned into a decision, in words a reviewer can check —
    # e.g. "transfer AUROC at the permutation-licensed cell, read at the 95th
    # percentile of the same-condition negative class". Required, because the
    # band run's hard incoherence was traced to an operating point stated only
    # in a config comment that never reached the label.
    operating_point: str

    # Tri-state, and this is the point (`instrument_layer.md` §1.5). None means
    # THIS INSTRUMENT COULD NOT READ THIS CONDITION. It never means "absent".
    # A False here asserts a measured negative; conflating the two is the defect
    # that made the cipher band's uniform (R) an artefact of a silent False.
    licensed: bool | None

    # P2. What the instrument reads on a condition where the true answer is
    # known to be nothing — the can't-decode rungs, whose ability is 0.00 on
    # both models. None means no control was run, which `reportable` treats as
    # disqualifying rather than as a pass.
    control_reading: float | None = None
    # The margin the control reading must be beaten by. Kept beside the reading
    # so a stale threshold cannot silently re-license an old number.
    control_margin: float | None = None

    # P3. How far `value` clears the length null. None means the null was not
    # computed — again disqualifying, not passing. Character length separates
    # the harmful corpus from the benign one at AUROC 0.654 and every encoder
    # preserves it, so an uncontrolled number is uninterpretable.
    length_null_margin: float | None = None

    # P7. Whether layer/position selection was inside the null rather than an
    # uncorrected search over the grid.
    selection_inside_null: bool = False

    # Free-form provenance a reader may need: layer, position, n, p-value.
    detail: dict = field(default_factory=dict)

    @property
    def clears_controls(self) -> bool:
        """P2 — read loudly where the answer is known to be nothing? Fails closed."""
        if self.control_reading is None or self.control_margin is None:
            return False
        return (self.value - self.control_reading) >= self.control_margin

    @property
    def clears_length_null(self) -> bool:
        """P3 — distinguishable from what raw length alone would produce?"""
        if self.length_null_margin is None:
            return False
        return self.length_null_margin > 0.0

    @property
    def reportable(self) -> bool:
        """May this number appear in a paper?

        Fails closed on every axis. Note `licensed is True` rather than a truth
        test: `None` is unmeasured and must not pass by looking falsy-or-not.
        """
        return (
            self.licensed is True
            and self.clears_controls
            and self.clears_length_null
            and self.selection_inside_null
        )

    def why_not_reportable(self) -> list[str]:
        """The failing axes, named. Empty iff `reportable`.

        Exists so a run record says WHY a cell was withheld. "Not reportable"
        with no reason is how an instrument defect hides as a null result.
        """
        reasons = []
        if self.licensed is None:
            reasons.append("unmeasured: this instrument could not read this condition")
        elif self.licensed is False:
            reasons.append("not licensed: no signal above the instrument's null")
        if self.control_reading is None or self.control_margin is None:
            reasons.append("no negative control was run (P2)")
        elif not self.clears_controls:
            reasons.append(
                f"reads {self.value:.3f} against {self.control_reading:.3f} on the "
                f"negative control, margin {self.control_margin:.3f} (P2)"
            )
        if self.length_null_margin is None:
            reasons.append("no length null was computed (P3)")
        elif not self.clears_length_null:
            reasons.append(f"inside the length null by {-self.length_null_margin:.3f} (P3)")
        if not self.selection_inside_null:
            reasons.append("layer/position selection was not inside the null (P7)")
        return reasons


@runtime_checkable
class Instrument(Protocol):
    """What every instrument declares. P1 is enforced across a set, not here."""

    @property
    def name(self) -> str: ...

    @property
    def question(self) -> str:
        """The question this instrument answers, and no other instrument does."""

    @property
    def kind(self) -> Kind: ...


def assert_distinct_questions(instruments: Sequence[Instrument]) -> None:
    """P1 — no two instruments may answer the same question.

    Enforced across the SET because that is where the principle lives: the
    roster's value is that failure modes are independent, and two instruments
    with the same question are one instrument counted twice. The build plan's
    argument for I3 is exactly this — it earns its slot by having an independent
    failure mode, not by being good.
    """
    seen: dict[str, str] = {}
    for instrument in instruments:
        question = instrument.question.strip().lower()
        if question in seen:
            raise ValueError(
                f"P1 violated: {instrument.name!r} and {seen[question]!r} answer the "
                f"same question ({instrument.question!r}) — they are one instrument "
                "counted twice, not two independent ones"
            )
        seen[question] = instrument.name


def reportable_only(readings: Sequence[Reading]) -> list[Reading]:
    """The subset a paper may quote. The complement is a finding, not waste."""
    return [reading for reading in readings if reading.reportable]


def withheld_summary(readings: Sequence[Reading]) -> dict[str, list[str]]:
    """Every non-reportable reading with its reasons, keyed by instrument.

    A run record carries this beside the results: a table of what was measured
    is only honest next to a table of what was measured and thrown away.
    """
    return {
        reading.instrument: reading.why_not_reportable()
        for reading in readings
        if not reading.reportable
    }
