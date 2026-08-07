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

**The claim's DIRECTION is part of the contract (added 2026-08-06, TODO 42).**
The first version asked one question — *does the instrument read loudly where the
answer is known to be nothing?* — which is **specificity**, and it is the right
question for a POSITIVE claim and structurally unable to license a NULL one. On
`tag_block` or `reverse_characters` ability is 0.00 and its negative control is
0.00, so the measurement is by construction indistinguishable from the control
and a *broken scorer* produces the identical number.

That was load-bearing rather than pedantic: the ability-0 rungs are the
calibration for three other instruments — the deployment noise floor (0.656 /
0.671), I1's control and I3's control all rest on "the model demonstrably cannot
decode this rung", i.e. on exactly the reading the old contract could not
license. So `Reading` now carries a `claim` and a `sensitivity` arm, and
`reportable` routes on the direction:

    positive  value must BEAT its negative control (P2) and clear the length
              null (P3) — both are guards against a signal that is really
              something else.
    null      value must NOT beat its negative control, and the instrument must
              demonstrate it COULD have fired (sensitivity).

**Why P2 and P3 drop out of the null path rather than being waived.** Both are
inflation controls: they ask whether an observed signal is really the control
condition or really length. A null claim has no signal to explain away, so both
are vacuous — and the question that replaces them is the opposite one, *could
this instrument have seen anything at all?* Same for P7: selection over a grid
inflates a maximum, so a null that survives an uncorrected search is CONSERVATIVE
rather than suspect ("even the best layer found nothing").

**Why `claim` is declared rather than derived from the value.** Deriving it —
"value at the floor, so treat it as a null" — would make `reportable`
self-licensing: any instrument reading low would automatically dodge its own
negative control. The direction is a statement about what the paper asserts, not
about what the number happens to be, so the caller declares it and the run record
carries it. The dodge is closed from the other side instead, by
`claim_is_coherent`: **a null claim that clears its specificity control is a
positive reading mislabelled**, and is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

Kind = Literal["correlational", "causal"]

# What the reading asserts. "positive" = we measured X; "null" = we measured no X.
# The two need different evidence, which is the whole reason this type exists.
Claim = Literal["positive", "null"]


@dataclass(frozen=True)
class Screen:
    """One negative control, with its own statistic and its own floor.

    **Why a screen carries its OWN observed value rather than reusing
    `Reading.value` (TODO 57).** The controls in build plan §4 are independent
    screens against different confounds, and they are not all measured on the
    same quantity. The ability-0 floor and the permuted-label control are read on
    the instrument's own statistic; the XSTest lexical screen is read on a
    *pooled paired AUROC over a different corpus*. A single `control_reading`
    float could only ever express the first kind, which is why the vocabulary
    screen had nowhere to go and ended up in `detail`.
    """

    name: str
    # What this screen measured — its own statistic, not necessarily the
    # instrument's headline number.
    observed: float
    # What `observed` must beat for the screen to clear.
    floor: float
    # How far it must beat it. Kept beside the screen for the same reason
    # `control_margin` is: a threshold living elsewhere can move without the
    # numbers it licensed being re-derived.
    #
    # definitional: ZERO is "clear the floor, by nothing in particular" — the
    # neutral element of the comparison, not a calibrated bar. Every screen that
    # wants a real margin passes its own from YAML (the lexical screen passes
    # `controls.lexical_min_margin`), and a screen whose floor already carries
    # the strictness needs no second knob. Tuning path: per screen, at the screen
    # — a global default here would silently loosen or tighten every one of them
    # at once, which is exactly the coupling the battery exists to avoid.
    margin: float = 0.0
    # What this screen rules out, in a reviewer's words.
    defeats: str = ""

    @property
    def clears(self) -> bool:
        """Fails CLOSED on NaN — an uncomputed screen never passes."""
        if self.observed != self.observed or self.floor != self.floor:  # NaN
            return False
        return (self.observed - self.floor) >= self.margin


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
    #
    # This pair is the PRIMARY screen, read on the instrument's own statistic.
    # Additional screens go in `controls` below.
    control_reading: float | None = None
    # The margin the control reading must be beaten by. Kept beside the reading
    # so a stale threshold cannot silently re-license an old number.
    control_margin: float | None = None

    # ADDITIONAL screens (TODO 57). The battery in build plan §4 is four
    # independent controls and §4 is explicit that "every instrument runs all of
    # these" — but the contract had ONE slot, so the second screen onward had
    # nowhere to live and rode in `detail`, where `reportable` cannot see it.
    # The consequence was that `reportable` verified one screen while the
    # paper's claim rested on four, and WHICH one differed per instrument.
    controls: tuple[Screen, ...] = ()
    # Screens this instrument's claim DEPENDS on. A required screen that is
    # absent disqualifies, exactly as a missing primary control does — the
    # tri-state discipline applied to the battery: not run is never passed.
    #
    # Declared per reading rather than global, because the battery is not
    # uniform: the lexical screen is meaningless for an instrument that never
    # touches a harm probe, and demanding it everywhere would train readers to
    # ignore the requirement.
    required_controls: tuple[str, ...] = ()

    # P3. How far `value` clears the length null. None means the null was not
    # computed — again disqualifying, not passing. Character length separates
    # the harmful corpus from the benign one at AUROC 0.654 and every encoder
    # preserves it, so an uncontrolled number is uninterpretable.
    length_null_margin: float | None = None

    # P7. Whether layer/position selection was inside the null rather than an
    # uncorrected search over the grid.
    selection_inside_null: bool = False

    # What this reading ASSERTS. Defaults to "positive" because that is what every
    # instrument on the roster meant when it was written, and because the positive
    # path is the strict one — a caller who forgets to think about direction gets
    # the demanding route, not the permissive one.
    claim: Claim = "positive"

    # SENSITIVITY — what the instrument reads on a condition where the answer is
    # known to be SOMETHING. The exact mirror of `control_reading`, which reads
    # where the answer is known to be nothing.
    #
    # For measurement #1 this is `AbilityControl.identity_rate`: the fraction of
    # prompts on which the scorer fires when the response IS the plaintext. Not
    # the tautology it looks like — `normalize` NFKC-folds and strips zero-width
    # characters, so a rung whose text uses an unusual character set could break
    # the scorer where no other rung would reveal it. For a probe instrument the
    # analogue is transfer AUROC on a condition known to be readable.
    sensitivity: float | None = None
    # The floor `sensitivity` must reach. Beside the reading for the same reason
    # `control_margin` is: a threshold that lives elsewhere can be changed without
    # the numbers it licensed being re-derived.
    sensitivity_floor: float | None = None

    # Free-form provenance a reader may need: layer, position, n, p-value.
    detail: dict = field(default_factory=dict)

    @property
    def missing_controls(self) -> tuple[str, ...]:
        """Required screens with no result attached. Absent is never passed."""
        present = {screen.name for screen in self.controls}
        return tuple(name for name in self.required_controls if name not in present)

    @property
    def failed_controls(self) -> tuple[str, ...]:
        """Screens that ran and did not clear."""
        return tuple(screen.name for screen in self.controls if not screen.clears)

    @property
    def clears_controls(self) -> bool:
        """P2 — every screen this claim rests on, not just the first one.

        Three conditions, each failing closed: the primary screen ran and was
        beaten by the stated margin; every ADDITIONAL screen that ran cleared;
        and every REQUIRED screen actually ran. The third is the one TODO 57
        added — before it, a claim could be reported having silently skipped
        three of the four controls the build plan calls mandatory.
        """
        if self.control_reading is None or self.control_margin is None:
            return False
        if (self.value - self.control_reading) < self.control_margin:
            return False
        return not self.failed_controls and not self.missing_controls

    @property
    def clears_length_null(self) -> bool:
        """P3 — distinguishable from what raw length alone would produce?"""
        if self.length_null_margin is None:
            return False
        return self.length_null_margin > 0.0

    @property
    def clears_sensitivity(self) -> bool:
        """Could this instrument have fired at all on this condition? Fails closed.

        The arm a NULL claim rests on. NaN fails, as everywhere else in this
        layer: an unmeasurable sensitivity is not a demonstrated one.
        """
        if self.sensitivity is None or self.sensitivity_floor is None:
            return False
        if self.sensitivity != self.sensitivity:  # NaN
            return False
        return self.sensitivity >= self.sensitivity_floor

    @property
    def claim_is_coherent(self) -> bool:
        """Does the declared direction match the evidence?

        **This is what makes a declared `claim` safe.** Without it, labelling a
        reading `null` would be a free pass out of P2 and P3 — exactly the dodge
        that deriving the direction from the value would have opened. A null claim
        that BEATS its own negative control is not a null; it is a positive
        reading wearing the permissive route's label, and it is refused here.
        """
        if self.claim == "positive":
            return True
        return not self.clears_controls

    @property
    def reportable(self) -> bool:
        """May this number appear in a paper?

        Fails closed on every axis, and routes on the claim's direction — see the
        module docstring for why P2/P3/P7 are inflation controls that a null claim
        structurally cannot satisfy and does not need.

        Note `licensed is True` on the positive path rather than a truth test:
        `None` is unmeasured and must not pass by looking falsy-or-not. The null
        path asks only `is not None`, because there `licensed is False` — no signal
        above the instrument's own null — is SUPPORTING evidence rather than a
        disqualification. Unmeasured stays disqualifying on both paths.
        """
        if self.claim == "null":
            return (
                self.licensed is not None
                and self.claim_is_coherent
                and self.clears_sensitivity
            )
        return (
            self.licensed is True
            and self.clears_controls
            and self.clears_length_null
            and self.selection_inside_null
        )

    def why_not_reportable(self) -> list[str]:
        """The failing axes, named. Empty iff `reportable`.

        Exists so a run record says WHY a cell was withheld. "Not reportable"
        with no reason is how an instrument defect hides as a null result. The
        reasons are direction-specific: telling a null claim it lacks a
        specificity margin would send a reader after evidence that cannot exist.
        """
        reasons = []
        if self.licensed is None:
            reasons.append("unmeasured: this instrument could not read this condition")
        if self.claim == "null":
            if not self.claim_is_coherent:
                reasons.append(
                    f"declared a null claim but reads {self.value:.3f} against "
                    f"{self.control_reading:.3f} on the negative control — that is a "
                    "positive reading mislabelled, not an absence"
                )
            if self.sensitivity is None or self.sensitivity_floor is None:
                reasons.append(
                    "no sensitivity arm was run: an absence needs evidence the "
                    "instrument COULD have fired, or a broken instrument reads the same"
                )
            elif not self.clears_sensitivity:
                reasons.append(
                    f"sensitivity {self.sensitivity:.3f} is below the floor "
                    f"{self.sensitivity_floor:.3f} — this instrument cannot be shown to "
                    "fire on this condition, so its silence means nothing"
                )
            return reasons
        if self.licensed is False:
            reasons.append("not licensed: no signal above the instrument's null")
        if self.control_reading is None or self.control_margin is None:
            reasons.append("no negative control was run (P2)")
        elif (self.value - self.control_reading) < self.control_margin:
            reasons.append(
                f"reads {self.value:.3f} against {self.control_reading:.3f} on the "
                f"negative control, margin {self.control_margin:.3f} (P2)"
            )
        for screen in self.controls:
            if not screen.clears:
                reasons.append(
                    f"control {screen.name!r} did not clear: {screen.observed:.3f} "
                    f"against floor {screen.floor:.3f} + margin {screen.margin:.3f}"
                    + (f" — it rules out {screen.defeats}" if screen.defeats else "")
                )
        for name in self.missing_controls:
            reasons.append(
                f"required control {name!r} was NOT RUN — this claim depends on it, "
                "and a control that did not run is not a control that passed"
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


def gate_per_prompt(reading: Reading, per_prompt: Sequence[bool]) -> list[bool | None]:
    """Gate a per-prompt axis on its condition-level `Reading`.

    **The granularity join, and the place the repo's worst defect lived.** A
    `Reading` is a verdict about one instrument on one *condition* (a rung, a
    guard); the axes that regime assignment consumes are per *prompt*. Composing
    them is not optional bookkeeping: if the instrument could not read the rung
    at all, then no per-prompt boolean from it means anything, and passing the
    raw booleans through is precisely how "this instrument could not read this"
    became "the model did not decode" — the silent `False` that made the cipher
    band's uniform (R) an artefact rather than a measurement.

    Returns `None` for every prompt when the reading is not reportable, so the
    tri-state is produced by the contract rather than re-remembered by each
    script. Both papers need this: AS-5's deployment axis and AS-6's decode axis
    are the same join over the same failure.
    """
    if not reading.reportable:
        return [None] * len(per_prompt)
    return list(per_prompt)


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
