"""The control floor's two statistics, and the n-dependence that motivated both.

Pins `instrument_layer.md` §2.4. The finding under test is not a bug fix — it is
that the floor `recalibrate_deployment.control_floor` uses is an extreme-value
statistic, so the same instrument on the same model yields a different bar
depending on how many rungs the model happened to be unable to decode. A rung's
measurability must not depend on that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from control_floor import floors  # noqa: E402


class TestTheFloorGrowsWithTheControlSet:
    def test_max_can_only_ratchet_upward_as_controls_are_added(self):
        """The defect, stated as the property that causes it.

        `max` is monotone non-decreasing in the control set: every rung added can
        only raise the bar, never lower it. So a run that happens to contain more
        can't-decode rungs applies a strictly harsher screen — for a reason that
        has nothing to do with the rung being screened.
        """
        small = [0.64, 0.66]
        for extra in ([0.63], [0.63, 0.645], [0.63, 0.645, 0.68]):
            assert floors(small + extra)["max"] >= floors(small)["max"]

    def test_mean_plus_2sd_can_move_DOWN_which_is_the_qualitative_difference(self):
        """A distributional floor responds to the distribution, not the extreme.

        Adding rungs that tighten the spread LOWERS mean+2SD. `max` cannot do
        that under any addition, which is exactly why it does not converge to a
        property of the instrument as evidence accumulates.
        """
        wide = [0.62, 0.70]
        tightened = wide + [0.66, 0.66, 0.66, 0.66, 0.66, 0.66]
        assert floors(tightened)["mean_plus_2sd"] < floors(wide)["mean_plus_2sd"]
        assert floors(tightened)["max"] == floors(wide)["max"]

    def test_the_real_pilot_numbers_reproduce_the_reported_floors(self):
        """The 2026-08-07 Llama control set, to three decimals."""
        llama = [0.6736, 0.6680, 0.6645, 0.6585, 0.6571,
                 0.6562, 0.6533, 0.6475, 0.6393, 0.6376, 0.5559]
        stats = floors(llama)
        assert stats["n"] == 11
        assert round(stats["max"], 3) == 0.674
        assert round(stats["mean_plus_2sd"], 3) == 0.711
        # hex, the rung the gate turned on, falls between the two floors — which
        # is the whole reason both are reported and neither is used alone.
        assert stats["max"] < 0.6908 < stats["mean_plus_2sd"]


class TestItFailsClosedOnAnUnusableControlSet:
    def test_no_controls_yields_no_floor_rather_than_a_default(self):
        """`None`, never 0.0. A floor of zero would pass every rung — the silent
        default this repo has now been bitten by on three separate axes."""
        stats = floors([])
        assert stats["n"] == 0
        assert stats["max"] is None and stats["mean_plus_2sd"] is None

    def test_a_single_control_gives_a_max_but_no_distributional_floor(self):
        """One control has no spread. Returning mean+0 would turn the strict
        floor into 'anything above the one control passes' — strictly weaker
        than the max rule while looking more principled."""
        stats = floors([0.66])
        assert stats["max"] == 0.66
        assert stats["mean_plus_2sd"] is None
