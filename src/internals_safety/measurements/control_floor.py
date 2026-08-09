"""The surface-feature noise floor, derived from the run's own can't-decode rungs.

**One home for a derivation that was living in two scripts.** `recalibrate_
deployment.py` had a `control_floor` using max; `control_floor.py` grew a second
one reporting max and mean+kSD side by side. A floor computed two ways is a floor
that can disagree with itself, and this is the number three instruments are
judged against (deployment, I1's control, I3's).

## What a control rung is

A rung whose ability is ~0 is a **free negative control**: whatever the probe
reads there is by construction NOT decoded content, so its transfer AUROC is what
surface features alone buy on this corpus, this model and this instrument.

## Why the statistic changed (SETTLED 2026-08-07, owner go)

`max` was adopted when the control set had TWO rungs, where it is the binding
constraint and nothing better is estimable. It is an extreme-value statistic, so
it is **monotone non-decreasing in the size of the control set**: measured on the
same instrument and the same models, the floor moved 0.656 -> 0.674 (Llama) and
0.671 -> 0.683 (Qwen) purely by going from 2 controls to 11/10. A rung's
measurability then depends on how many rungs the model happened to be unable to
decode in that run, which is not a property of the rung being screened, and it
makes floors from different runs incomparable.

`mean + k*SD` describes the control DISTRIBUTION, so it converges as evidence
accumulates instead of ratcheting.

## Why k = 2.0 is not the knob the answer rests on

Derived, not chosen, and deliberately from a requirement that says nothing about
which rungs we want to pass: **no control rung may clear its own floor.** On the
2026-08-07 pilot ladder that alone requires k >= 1.531 (Qwen binds; Llama needs
only 0.846). The genuine rungs sit at k = 6.2 (Llama `reverse_words`) through
19.5 (Qwen `zero_width`).

    k must be >= 1.53   (no control may pass)
    k may  be <= 6.17   (both genuine rungs must still pass)

**Every conclusion is invariant across that whole range**, so k is a knob with no
reachable setting that changes an answer. `hex`, the rung the gate turned on,
sits at k = 1.383 — below the floor the no-control-may-pass requirement imposes
before 2.0 is chosen at all. Tuning path: re-derive both bounds whenever the
ladder or corpus changes; if they ever cross, the screen has no valid setting and
must say so rather than pick one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

FloorKind = Literal["distribution", "bound", "none"]


@dataclass(frozen=True)
class AbilitySource:
    """The ability rates that SELECT the control set, and whose they are.

    **Both model names are required, and that is the whole point (2026-08-08).**
    §2.6 of `instrument_layer.md` establishes that a guard has no ability
    measurement of its own — Llama Guard 3's template hard-wires the
    classification task, so there is no in-distribution channel through which it
    can be asked to restate a payload. The non-circular route is to inherit its
    base model's ability (Llama Guard 3 8B <- Llama-3.1-8B, WildGuard <-
    Mistral-7B-v0.3), and §2.6 also states the honest label for such a floor:
    *inherited*, with the assumption stated.

    An optional `inherited=True` flag would have been the obvious way to carry
    that, and it is exactly the shape this repo has been burned by four times in
    one week — a rule that is adopted, tested, documented and approved, and then
    governs nothing because omitting it is expressible. So the two model names
    are REQUIRED fields and `is_inherited` is derived: a floor cannot be built
    without saying what its ability was measured on and what it screens, and no
    caller can report it as a measured floor by forgetting a keyword.
    """

    rates: Mapping[str, float]
    measured_on: str
    screens: str

    @property
    def is_inherited(self) -> bool:
        """True when ability came from a different model than the floor screens.

        An inherited floor is an ESTIMATE: fine-tuning can move decoding
        capability, so the base model's 0.00 on `caesar3` is evidence about the
        fine-tune, not a measurement of it. What it is not is *circular*, which
        is what §3.4 requires and what selecting controls on the screened
        statistic can never be.
        """
        return self.measured_on != self.screens


@dataclass(frozen=True)
class ControlFloor:
    """A floor, the evidence behind it, and how much to trust it.

    `kind` is not decoration. A `bound` floor (too few controls for a spread) is
    weaker than a `distribution` floor and callers must be able to say so in a
    run record — the band run's 0.656 was a bound reported as though it were a
    property of the instrument, which is how it came to be copied.

    `ability_measured_on` / `ability_screens` travel with the value for the same
    reason, one axis over: a floor whose provenance is optional gets quoted
    without its assumption.
    """

    value: float | None
    kind: FloorKind
    n: int
    controls: tuple[str, ...]
    ability_measured_on: str
    ability_screens: str
    mean: float | None = None
    stdev: float | None = None
    observed_max: float | None = None

    @property
    def is_usable(self) -> bool:
        return self.value is not None

    @property
    def is_inherited(self) -> bool:
        return self.ability_measured_on != self.ability_screens

    def clears(self, auroc: float, family: str | None = None) -> bool | None:
        """Does `auroc` clear the floor? `None` when it cannot be judged.

        Tri-state, for the same reason every other axis here is: with no floor
        there is nothing to judge a reading against, and returning False would
        say "this rung reads nothing" when the truth is "this run could not
        screen it". A control rung never clears its own floor.
        """
        if self.value is None:
            return None
        if family is not None and family in self.controls:
            return False
        return auroc > self.value


def derive(
    transfer_auroc: Mapping[str, float],
    *,
    ability: AbilitySource,
    max_ability: float,
    sigma: float,
    min_controls: int,
) -> ControlFloor:
    """Derive the floor from the rungs its ability source can call controls.

    A rung with NO ability measurement is excluded from the control set rather
    than defaulted into or out of it — an unmeasured rung must never be allowed
    to set the floor that everything else is judged against.

    `ability` is keyword-only and has no default. It was a positional
    `ability_rate: Mapping` until 2026-08-08; the change is deliberate and is
    the same fix `strata` got on `measure_deployment` — a stale positional call
    is now a `TypeError` at import-adjacent time rather than a floor silently
    reported without its provenance.
    """
    rates = ability.rates
    controls = tuple(
        sorted(f for f in transfer_auroc if f in rates and rates[f] <= max_ability)
    )
    values = [transfer_auroc[f] for f in controls]
    n = len(values)
    where = {"ability_measured_on": ability.measured_on, "ability_screens": ability.screens}
    if n == 0:
        return ControlFloor(None, "none", 0, controls, **where)

    observed_max = max(values)
    mean = statistics.mean(values)
    if n < max(min_controls, 2):
        # Fail SOFT to the older, weaker statistic rather than refusing outright:
        # a bound is still a bound, and the band run's two-control screen was
        # informative. But it is labelled, so a reader can see it is not a
        # distributional estimate and must not be carried to another run.
        return ControlFloor(observed_max, "bound", n, controls, **where,
                            mean=mean, observed_max=observed_max)

    stdev = statistics.stdev(values)
    return ControlFloor(mean + sigma * stdev, "distribution", n, controls, **where,
                        mean=mean, stdev=stdev, observed_max=observed_max)


def sigma_bounds(
    transfer_auroc: Mapping[str, float],
    *,
    ability: AbilitySource,
    max_ability: float,
    genuine: Sequence[str],
) -> tuple[float | None, float | None]:
    """The window of `sigma` in which no answer changes: (must exceed, must not).

    Lower bound = the largest control's own k, so no control passes. Upper bound
    = the smallest k among `genuine`, so every rung believed real still passes.
    Returns `None` for a bound that cannot be computed. **If lower >= upper the
    screen has no valid setting** and the caller must report that rather than
    pick one — the whole point of deriving k instead of choosing it.

    ⚠️ **The window is a property of the object of study, not of the method**
    (2026-08-08). On the AS-5 ladder it is [1.53, 6.17] and 2.0 sits inside it.
    On Llama Guard 3's own control distribution the lower bound is **2.21** —
    `caesar3` reads 0.7168 against a control mean of 0.644 — so the configured
    2.0 is *not a valid setting there*. Re-derive per guard; never import a
    sigma the way §2.2's floors were once imported.
    """
    ability_rate = ability.rates
    controls = [f for f in transfer_auroc if f in ability_rate and ability_rate[f] <= max_ability]
    if len(controls) < 2:
        return None, None
    values = [transfer_auroc[f] for f in controls]
    mean, stdev = statistics.mean(values), statistics.stdev(values)
    if stdev == 0:
        return None, None
    lower = (max(values) - mean) / stdev
    present = [transfer_auroc[f] for f in genuine if f in transfer_auroc]
    upper = min((v - mean) / stdev for v in present) if present else None
    return lower, upper
