"""The pre-gate has two arms asking different questions, and conflating them inverted it.

**The measured inversion this fixes.** `min_variance_explained` stood at 0.75 as
an absolute bar applied to both arms. Job 9009915 measured the CEILING arm —
Llama-3.1-8B-Base on plain text, the model the dictionary was actually fitted on
— at variance explained 0.698-0.723. So the guessed bar was FAILING the run whose
entire job is to SET it, and `licensed=False` was reporting a property of our
guess rather than of the dictionary.

* **Ceiling arm** (`ceiling is None`): does our loader work? Judged on
  reconstructing at all — positive variance, above its own matched control, KL
  term clearing its bar. Nothing exists above this arm to compare it against.
* **Target arm** (`ceiling` supplied): does the dictionary transfer? Judged as a
  FRACTION of what the ceiling arm reached.

The KL term keeps an absolute bar because it is already relative by construction
— a fraction of the layer's own downstream contribution. Variance explained is
not, which is precisely why an absolute bar on it means different things per
model and per layer.
"""

from __future__ import annotations

import pytest

from internals_safety.config import SAEConfig
from internals_safety.measurements.sae_reconstruction import (
    Ceiling,
    ReconstructionQuality,
    reading,
)

CONFIG = SAEConfig(trained_on="t", min_kl_recovered=0.80, min_transfer_ratio=0.80)

# The real Base numbers, job 9009915 layer 18. Used verbatim so the test fails if
# anyone reintroduces a bar above the measured ceiling.
BASE_VE, BASE_KL_SAE, BASE_KL_ABLATED = 0.6979, 0.081, 1.0
# The ceiling arm's own KL-recovered, measured. The KL bar is a fraction of THIS
# from 2026-08-08, not an absolute 0.80 -- see reading()'s docstring.
BASE_KL_RECOVERED = 0.9189
CEILING = Ceiling(variance_explained=BASE_VE, kl_recovered=BASE_KL_RECOVERED)


def quality(variance: float, *, control: float | None = 0.0, kl_sae: float = 0.081):
    return ReconstructionQuality(
        layer=18, n_prompts=100, mse=1.0, variance_explained=variance,
        l0=199.3, kl_sae=kl_sae, kl_ablated=1.0,
        control_variance_explained=control,
    )


class TestTheCeilingArm:
    def test_the_measured_base_ceiling_LICENSES(self):
        """**The regression this file exists for.** 0.698 against the retired
        0.75 bar was the inversion; the ceiling arm must now pass on it."""
        record = reading(quality(BASE_VE), CONFIG)
        assert record.licensed is True
        assert record.detail["arm"] == "ceiling"

    def test_a_bar_above_the_ceiling_cannot_be_reintroduced(self):
        """Any absolute variance bar the ceiling arm is judged against would have
        to exceed 0 — pinning the floor at exactly 0 is what keeps a future
        'reasonable-looking' 0.75 from silently returning."""
        assert reading(quality(BASE_VE), CONFIG).detail["variance_floor_applied"] == 0.0

    def test_negative_variance_REFUSES_however_good_the_kl(self):
        """The -915 case. A dictionary worse than predicting the mean is not
        reconstructing, whatever the downstream term says."""
        assert reading(quality(-915.1, kl_sae=0.0), CONFIG).licensed is False

    def test_zero_variance_refuses_because_the_mean_is_free(self):
        assert reading(quality(0.0), CONFIG).licensed is False

    def test_a_poor_kl_term_still_refuses(self):
        """The KL term is the gate the foundational paper says should decide; a
        relaxed variance floor must not smuggle a bad one through."""
        assert reading(quality(BASE_VE, kl_sae=0.9), CONFIG).licensed is False


class TestTheTargetArm:
    def test_it_is_judged_as_a_fraction_of_the_ceiling(self):
        record = reading(quality(0.60), CONFIG, ceiling=CEILING)
        assert record.detail["arm"] == "target"
        assert record.detail["ceiling_variance_explained"] == BASE_VE
        assert record.detail["variance_floor_applied"] == pytest.approx(BASE_VE * 0.80)
        assert record.licensed is True

    def test_the_KL_bar_is_ALSO_a_fraction_of_the_ceiling(self):
        """**The 2026-08-08 fix.** An absolute 0.80 failed job 9010205's L18 by
        0.0017 while L20/L22 passed -- against their own ceilings all three
        retain 87-89%, so the split was an artefact of the bar's units."""
        record = reading(quality(0.60), CONFIG, ceiling=CEILING)
        assert record.detail["ceiling_kl_recovered"] == BASE_KL_RECOVERED
        assert record.detail["kl_floor_applied"] == pytest.approx(BASE_KL_RECOVERED * 0.80)

    def test_the_MEASURED_L18_transfer_now_licenses(self):
        """0.7983 against an absolute 0.80 was the knife-edge; against 0.80 of
        the 0.9189 ceiling it clears with room. Real numbers so the regression
        fails if anyone restores the absolute bar."""
        record = reading(quality(0.7094, kl_sae=0.0), CONFIG, ceiling=CEILING)
        assert record.detail["kl_floor_applied"] < 0.7983
        assert record.licensed is True

    def test_a_KL_far_below_the_ceilings_fraction_still_REFUSES(self):
        """Relative is not lax -- mutating the floor away must break something."""
        record = reading(quality(0.60, kl_sae=0.60), CONFIG, ceiling=CEILING)
        assert record.licensed is False

    def test_below_the_transfer_floor_refuses(self):
        assert reading(quality(0.50), CONFIG, ceiling=CEILING).licensed is False

    def test_the_floor_MOVES_with_the_ceiling(self):
        """A fixed bar would not. This is the whole difference between the two
        designs, so it is asserted rather than assumed."""
        low = reading(quality(0.30), CONFIG, ceiling=Ceiling(0.35, BASE_KL_RECOVERED))
        high = reading(quality(0.30), CONFIG, ceiling=Ceiling(0.90, BASE_KL_RECOVERED))
        assert low.licensed is True and high.licensed is False


class TestNeitherArmPassesWithoutAControl:
    """A missing control is never a passed one: without it, "the dictionary
    reconstructs" is a statement about linear algebra rather than about training."""

    @pytest.mark.parametrize("ceiling", [None, CEILING])
    def test_an_absent_control_refuses(self, ceiling):
        assert reading(quality(BASE_VE, control=None), CONFIG, ceiling=ceiling).licensed is False

    @pytest.mark.parametrize("ceiling", [None, CEILING])
    def test_losing_to_the_control_refuses(self, ceiling):
        assert reading(
            quality(BASE_VE, control=BASE_VE + 0.01), CONFIG, ceiling=ceiling
        ).licensed is False


class TestTheReadingSaysWhichArmItIs:
    def test_a_record_can_always_tell_whether_it_SET_a_bar_or_was_measured_against_one(self):
        """Without this a later reader cannot interpret `licensed` at all — the
        same reading means different things in the two arms."""
        assert reading(quality(BASE_VE), CONFIG).detail["ceiling_variance_explained"] is None
        assert "CEILING arm" in reading(quality(BASE_VE), CONFIG).operating_point
        assert "TARGET arm" in reading(quality(0.6), CONFIG, ceiling=CEILING).operating_point
