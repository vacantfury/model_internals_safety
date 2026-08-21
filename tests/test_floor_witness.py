"""The witness screen refuses to screen anything but the same measurement.

The safety property here is entirely in the identity check: a witness run
supplies a floor for the internals leg ONLY because its reading and the leg's
are the same measurement. If that check can be skipped, weakened, or defaulted
past, the module becomes a cross-run floor import — which `instrument_layer.md`
§2.4 forbids — wearing a name that says it is not one.

So these tests attack the check rather than exercising it: every identity field
is mutated in turn and the screen must refuse, and `None` must read as missing
evidence rather than as agreement.
"""

from __future__ import annotations

import dataclasses

import pytest

from internals_safety.measurements.control_floor import ControlFloor
from internals_safety.measurements.floor_witness import (
    IDENTITY_FIELDS,
    Provenance,
    WitnessScreen,
)

# Modelled on the real records this screen runs against
# (`outputs/runs/phase0/llama3_1_8b_instruct/lens-floor_*` and
# `scaffold-control-llama_*`), not on a convenient minimum: every field carries
# a realistic value, because a fixture whose fields are None everywhere would
# pass a check that treats None as a wildcard — the exact defect being tested.
LEG = Provenance(
    model="llama3_1_8b_instruct",
    family="homoglyph",
    harmful_digest="b42d6be2c97a2125",
    harmless_digest="fd34b465c7880052",
    n_prompts=100,
    plain_harmful_activations="/scratch/acts/plain-harmful-05ff5a5b.pt",
    plain_harmless_activations="/scratch/acts/plain-harmless-fd34b465.pt",
    encoded_harmful_activations="/scratch/acts/encoded-harmful-homoglyph-ad28cdad.pt",
    encoded_harmless_activations="/scratch/acts/encoded-harmless-homoglyph-0a5b39d5.pt",
    layer=21,
    position="instruction_final",
    transfer_auroc=0.9808,
    run="scaffold-control-llama_20260809T102157Z_9033595",
)
WITNESS = dataclasses.replace(LEG, run="lens-floor_20260808T072638Z_9010530")

FLOOR = ControlFloor(
    value=0.6765217627681936,
    kind="distribution",
    n=6,
    controls=("base64", "caesar7", "morse", "reverse_characters", "rot13", "tag_block"),
    ability_measured_on="llama3_1_8b_instruct",
    ability_screens="llama3_1_8b_instruct",
    mean=0.6529,
    stdev=0.0118,
    observed_max=0.6681,
)


class TestTheIdentityCheckIsTheSafetyProperty:
    def test_the_same_measurement_screens_and_clears(self):
        screen = WitnessScreen(reading=LEG, witness=WITNESS, floor=FLOOR)
        assert screen.is_same_measurement
        assert screen.verdict is True
        assert screen.margin == pytest.approx(0.9808 - 0.6765217627681936)
        assert "distribution" in screen.reason

    @pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
    def test_mutating_ANY_identity_field_refuses_to_screen(self, field_name):
        """One changed field is enough. Mutation, not enumeration by hand."""
        current = getattr(WITNESS, field_name)
        altered = current + 1 if isinstance(current, (int, float)) else f"{current}-x"
        witness = dataclasses.replace(WITNESS, **{field_name: altered})

        screen = WitnessScreen(reading=LEG, witness=witness, floor=FLOOR)

        assert screen.is_same_measurement is False
        assert screen.mismatched_fields == (field_name,)
        assert screen.verdict is None, (
            f"a witness differing on {field_name} is a DIFFERENT measurement and "
            "screening it would be the cross-run floor import §2.4 forbids"
        )
        assert screen.margin is None
        assert "NOT the same measurement" in screen.reason

    @pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
    def test_None_is_missing_evidence_NOT_a_wildcard(self, field_name):
        """A None on either side must never read as agreement."""
        blanked_witness = dataclasses.replace(WITNESS, **{field_name: None})
        assert WitnessScreen(
            reading=LEG, witness=blanked_witness, floor=FLOOR
        ).verdict is None

        blanked_leg = dataclasses.replace(LEG, **{field_name: None})
        assert WitnessScreen(
            reading=blanked_leg, witness=WITNESS, floor=FLOOR
        ).verdict is None

    def test_the_run_NAME_is_deliberately_not_an_identity_field(self):
        """Two runs differing only by name is the whole point of a witness."""
        assert "run" not in IDENTITY_FIELDS
        assert LEG.run != WITNESS.run
        assert WitnessScreen(reading=LEG, witness=WITNESS, floor=FLOOR).verdict is True


class TestTheNoneBranchesStayDistinguishable:
    def test_an_unusable_floor_is_unjudged_not_failed(self):
        empty = ControlFloor(
            value=None, kind="none", n=0, controls=(),
            ability_measured_on="m", ability_screens="m",
        )
        screen = WitnessScreen(reading=LEG, witness=WITNESS, floor=empty)

        assert screen.is_same_measurement, "identity is fine; the FLOOR is missing"
        assert screen.verdict is None
        assert screen.margin is None
        assert "unusable" in screen.reason
        assert "NOT the same measurement" not in screen.reason, (
            "the two None causes must stay distinguishable — collapsing them "
            "repeats §2.7's caesar3 error"
        )

    def test_a_reading_whose_own_family_is_a_control_never_clears(self):
        floor = dataclasses.replace(FLOOR, controls=FLOOR.controls + ("homoglyph",))
        screen = WitnessScreen(reading=LEG, witness=WITNESS, floor=floor)

        assert screen.verdict is False, (
            "a control rung must not clear the floor it helped define, however "
            "high it reads — this is the caesar3 case"
        )
        assert "control in the witness run" in screen.reason


class TestTheScreenTypeReportsHonestly:
    def test_a_refused_pairing_reports_nan_rather_than_a_number(self):
        witness = dataclasses.replace(WITNESS, layer=99)
        screen = WitnessScreen(reading=LEG, witness=witness, floor=FLOOR).screen()
        assert screen.observed != screen.observed  # NaN

    def test_a_valid_pairing_reports_the_reading_against_the_floor(self):
        screen = WitnessScreen(reading=LEG, witness=WITNESS, floor=FLOOR).screen()
        assert screen.observed == pytest.approx(0.9808)
        assert screen.floor == pytest.approx(0.6765217627681936)
        assert screen.direction == "above"
