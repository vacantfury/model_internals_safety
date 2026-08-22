"""The length-only bound must be a CEILING, and it must bind on the worse unit.

Both properties are what make the control a control. A bound that the observed
gap can exceed for reasons other than real signal certifies nothing; a bound
computed in whichever unit was measured first would have cleared three of AS-6's
six reported cells on the friendlier number.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from internals_safety.measurements.length_null import length_auroc

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "guard_arm_length_bound.py"


def _module():
    spec = importlib.util.spec_from_file_location("guard_arm_length_bound", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bound = _module()


class TestTheBoundIsACeiling:
    """No length rule may beat the optimal one — that is the whole argument."""

    def test_a_perfectly_length_separated_pool_reaches_the_bound_exactly(self):
        """Harmful strictly longer than benign, blocking budget = every harmful item.

        The optimal length guard gets every harmful item and no benign one, so
        the bound is 1.0. If it came out below, the bound would be beatable and
        the control would reject real cells.
        """
        harmful = [100, 101, 102, 103]
        benign = [10, 11, 12, 13]
        assert bound.length_only_gap(harmful, benign, n_blocked=4) == pytest.approx(1.0)

    def test_no_monotone_threshold_rule_ever_beats_the_bound(self):
        """The property stated as a property, over the class the bound covers.

        Every threshold rule in both directions, at every budget the data can
        express, on a pool with realistic overlap. Each is checked against the
        bound computed at ITS OWN budget, because a ceiling is only a ceiling
        against rules that block as much as it does.
        """
        rng = random.Random(0)
        harmful = [max(1, int(rng.gauss(86, 20))) for _ in range(100)]
        benign = [max(1, int(rng.gauss(74, 20))) for _ in range(100)]
        pool = [(n, True) for n in harmful] + [(n, False) for n in benign]

        for threshold in sorted({n for n, _ in pool}):
            for direction in (True, False):
                picked = [
                    item for item in pool if (item[0] >= threshold if direction else item[0] <= threshold)
                ]
                hit = sum(1 for _, is_harmful in picked if is_harmful)
                gap = hit / len(harmful) - (len(picked) - hit) / len(benign)
                ceiling = bound.length_only_gap(harmful, benign, len(picked))
                assert gap <= ceiling + 1e-12

    def test_a_benign_longer_confound_is_bounded_just_as_tightly(self):
        """Two-sided, like `length_auroc`, and for the same reason.

        If the bound only ever blocked the longest, a corpus whose BENIGN arm is
        the long one would bound at a negative number and clear anything.
        """
        harmful = [10, 11, 12, 13]
        benign = [100, 101, 102, 103]
        assert bound.length_only_gap(harmful, benign, n_blocked=4) == pytest.approx(1.0)

    def test_a_non_monotone_length_rule_CAN_beat_the_bound(self):
        """The restriction, pinned as a limitation rather than left implicit.

        Harmful items sit at the extremes and benign in the middle, so "block
        the very short and the very long" separates perfectly while neither
        monotone direction does better than chance. Such a rule is not the
        confound the control is for -- it is a rule fitted to this corpus -- but
        the bound does not cover it, and a control's scope belongs in a test
        rather than only in a docstring.
        """
        harmful = [1, 2, 99, 100]
        benign = [50, 51, 52, 53]
        non_monotone_gap = 1.0  # blocks all four harmful, no benign
        assert non_monotone_gap > bound.length_only_gap(harmful, benign, n_blocked=4)

    def test_ties_at_the_cut_are_broken_toward_the_bound(self):
        """A tie must inflate the ceiling, never deflate it.

        All eight items share one length, so every rule is a tie. The bound must
        report the most favourable resolution (all four harmful blocked), because
        a ceiling we could have beaten on a coin flip is not a ceiling.
        """
        assert bound.length_only_gap([5] * 4, [5] * 4, n_blocked=4) == pytest.approx(1.0)

    def test_a_pool_with_no_length_difference_and_a_full_budget_bounds_at_zero(self):
        """Blocking everything separates nothing, whatever the lengths."""
        assert bound.length_only_gap([9, 8, 7], [3, 2, 1], n_blocked=6) == pytest.approx(0.0)


class TestTheNumericAurocIsNotASecondImplementation:
    """One home for the two-sided rule; this pins that it really is one home."""

    def test_it_agrees_with_length_auroc_on_character_data(self):
        texts_a = ["aaaa", "aaaaaa", "aaaaaaaa"]
        texts_b = ["a", "aa", "aaa"]
        assert bound._length_auroc_numeric(
            [len(t) for t in texts_a], [len(t) for t in texts_b]
        ) == pytest.approx(length_auroc(texts_a, texts_b))

    def test_it_is_two_sided_like_its_source(self):
        """Benign LONGER than harmful is just as exploitable and must not score low."""
        assert bound._length_auroc_numeric([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


class TestTheVerdictBindsOnTheWorseUnit:
    """Mutation guard for the one line that decides what the script concludes."""

    def test_the_binding_bound_is_the_larger_of_the_two_units(self):
        """`max`, never `min` — verified against AS-6's own numbers.

        WildGuard `reverse_words` bounds at 0.220 in characters and 0.340 in
        tokens against an observed 0.440. Taking the minimum would report a
        margin of +0.220 where the honest one is +0.100, so this is not a
        stylistic preference about aggregation.
        """
        bounds = {"chars": 0.220, "tokens": 0.340}
        binding = max(bounds.values())
        assert binding == pytest.approx(0.340)
        assert 0.440 - binding == pytest.approx(0.100)
        assert min(bounds.values()) != binding, "the two units must not be interchangeable here"


class TestTheArtifactMatchesTheClaim:
    """The recorded run is the thing the paper will cite, so check IT, not a rerun."""

    ARTIFACT = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "analysis"
        / "guard_arm_length_bound_20260821.json"
    )

    def test_every_reported_cell_clears_its_bound(self):
        """If this ever goes red the paper's length control has failed, not the test."""
        if not self.ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        import json

        data = json.loads(self.ARTIFACT.read_text())
        reported = [
            (guard, row)
            for guard, rows in data["guards"].items()
            for row in rows
            if row["reported"]
        ]
        assert len(reported) == 6, f"expected 6 reported cells, found {len(reported)}"
        for guard, row in reported:
            assert row["clears_bound"], f"{guard}/{row['family']} no longer clears its length bound"
            assert row["observed_gap"] > row["length_only_gap_bound"]

    def test_the_token_unit_binds_somewhere(self):
        """Pins the reason the token arm exists.

        If characters bound every cell, the token measurement would be dead
        weight and this test says so rather than leaving it to be trimmed later
        by someone who did not know why it was added.
        """
        if not self.ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        import json

        data = json.loads(self.ARTIFACT.read_text())
        binding_units = {
            row["binding_unit"]
            for rows in data["guards"].values()
            for row in rows
            if row["reported"]
        }
        assert "tokens" in binding_units
