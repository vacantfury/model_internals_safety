"""The paired ordering test, and the guard that stops it running on the wrong read.

Two properties carry the result. The reconstruction must reproduce Table 2 before
anything is compared, or the comparison is about a different operating point than
the paper's. And the degeneracy check must fire, because the finding is not "the
cells can now be ordered" but "they can be, and the ordering restates the block
rate".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "paired_cell_ordering.py"
ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "analysis"
    / "paired_cell_ordering_20260821.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("paired_cell_ordering", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ordering = _module()


class TestTheReadIsPinnedToThePublishedTable:
    def test_table_two_is_declared_and_not_derived(self):
        """The check must compare against the PAPER, not against a recomputation.

        A self-check that recomputes both sides agrees with itself whatever the
        threshold is, which is precisely the failure it exists to catch.
        """
        assert ordering.TABLE_TWO["llama_guard_3_8b"] == {
            "homoglyph": 7,
            "zero_width": 17,
            "reverse_words": 8,
        }
        assert ordering.TABLE_TWO["wildguard"] == {
            "homoglyph": 23,
            "zero_width": 23,
            "reverse_words": 9,
        }

    def test_the_reported_operating_point_is_not_the_source_run_s(self):
        """`scores-b10` is at percentile 50, which is retired; the paper reports 75."""
        assert ordering.READING_PERCENTILE == 75.0
        record = json.loads(
            (ordering.RUNS / "llama_guard_3_8b" / ordering.SOURCE / "results.json").read_text()
        )
        produced_at = record["config"]["measurements"]["probes"]["reading_percentile"]
        assert produced_at != ordering.READING_PERCENTILE, (
            "if the source run were already at the reported point the reconstruction "
            "would be trivial and this script's threshold logic untested by its own data"
        )

    def test_the_reconstruction_reproduces_every_published_count(self):
        if not (ordering.RUNS / "llama_guard_3_8b" / ordering.SOURCE / "cells.jsonl").exists():
            pytest.skip("run records absent (fresh clone: outputs/ is gitignored)")
        for guard, expected in ordering.TABLE_TWO.items():
            cells, _ = ordering.per_prompt_cells(guard)
            for family, count in expected.items():
                assert sum(cells[family].values()) == count, f"{guard}/{family}"


class TestTheDegeneracyOfTheConjunction:
    """Why the abstention survives a test that overturned its stated reason."""

    def test_the_decode_term_is_near_one_on_the_surface_conditions(self):
        if not ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        data = json.loads(ARTIFACT.read_text())
        for guard, block in data["guards"].items():
            for family in ("homoglyph", "zero_width"):
                rate = block["marginals"][family]["decode_rate"]
                assert rate >= 0.90, f"{guard}/{family} decode rate {rate}"

    def test_the_count_recovers_most_of_the_unblocked_prompts_there(self):
        """The concrete form of the degeneracy, as the number the paper prints.

        This test caught the paper claiming "within one or two prompts", which
        holds on three of the four surface cells and fails on WildGuard
        `zero_width` at 23 of 29. The published range is 79 to 100 per cent, and
        it is pinned here so a looser sentence cannot come back.
        """
        if not ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        data = json.loads(ARTIFACT.read_text())
        shares = []
        for guard, block in data["guards"].items():
            for family in ("homoglyph", "zero_width"):
                unblocked = block["marginals"][family]["unblocked"]
                shares.append(ordering.TABLE_TWO[guard][family] / unblocked)
        assert min(shares) == pytest.approx(0.79, abs=0.005), f"shares {shares}"
        assert max(shares) == pytest.approx(1.00, abs=0.005), f"shares {shares}"

        # The control must NOT be in that band; if it were, the degeneracy would
        # be a property of the statistic rather than of these four cells.
        controls = [
            ordering.TABLE_TWO[guard]["reverse_words"]
            / block["marginals"]["reverse_words"]["unblocked"]
            for guard, block in data["guards"].items()
        ]
        assert max(controls) < min(shares)

    def test_reverse_words_is_NOT_degenerate_and_is_the_contrast(self):
        """The control's decode term is far from one, which is what makes it a contrast."""
        if not ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        data = json.loads(ARTIFACT.read_text())
        for guard, block in data["guards"].items():
            assert block["marginals"]["reverse_words"]["decode_rate"] < 0.90, guard


class TestTheOrderingResult:
    def test_three_pairs_separate_and_none_did_unpaired(self):
        if not ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        data = json.loads(ARTIFACT.read_text())
        pairs = [row for block in data["guards"].values() for row in block["pairs"]]
        assert len(pairs) == 6
        separating = [row for row in pairs if row["separates_after_adjustment"]]
        assert len(separating) == 3
        assert all(row["wilson_intervals_overlap"] for row in pairs), (
            "the paper's caption says the independent intervals overlap on every pair; "
            "if one separates, the caption is wrong rather than this test"
        )

    def test_identical_cells_do_not_separate(self):
        """WildGuard reads 23 on both surface conditions; a test calling that a
        difference would be broken in the most visible possible way."""
        if not ARTIFACT.exists():
            pytest.skip("analysis artifact absent (fresh clone: outputs/ is gitignored)")
        data = json.loads(ARTIFACT.read_text())
        row = next(
            r
            for r in data["guards"]["wildguard"]["pairs"]
            if {r["left"], r["right"]} == {"homoglyph", "zero_width"}
        )
        assert row["left_count"] == row["right_count"] == 23
        assert row["mcnemar_p"] == pytest.approx(1.0)
        assert not row["separates_after_adjustment"]
