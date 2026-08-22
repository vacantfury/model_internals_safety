"""Replicate spread: does repeating a run reproduce the reported cell? (TODO 84 #8.)

The finding this instrument answers accused Table 1 of quoting the most
favourable of several replicates. The accusation failed — every cell comes from
one declared run family — but the replicates are real and their spread is what
the paper owes.

**Both directions are pinned deliberately.** A screen that only ever fires is a
verdict with a script attached, so identical replicates must report spread 0.0
and 100% agreement, not merely "no complaint". The load-bearing case is the
split between the two nondeterminism routes: a verdict flip on byte-identical
text is the JUDGE, a flip on different text is GENERATION, and collapsing them
would name the wrong cause for every number in the table.

Fixtures mirror the real record schema — `plan.model`, a `behavior_plain`
reading that must be ignored, and a `behavior` reading missing its benign arm
that must be skipped — because a fixture more permissive than the real thing is
a blind spot with a green checkmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replicate_spread import Cell, Replicate, collect, main  # noqa: E402


def _write_run(
    root: Path,
    model: str,
    run: str,
    *,
    condition: str = "homoglyph",
    harmful: float | None = 0.99,
    benign: float | None = 0.99,
    cells: list[dict] | None = None,
    include_plain: bool = True,
) -> Path:
    run_dir = root / model / run
    run_dir.mkdir(parents=True)
    readings = []
    if include_plain:
        readings.append(
            {
                "instrument": "behavior_plain",
                "detail": {
                    "family": "plain",
                    "plain_harmful_refusal_rate": 0.92,
                    "plain_benign_refusal_rate": 0.10,
                },
            }
        )
    detail: dict = {"family": condition, "refusal_rate": harmful}
    if benign is not None:
        detail["benign_arm_refusal_rate"] = benign
    readings.append({"instrument": "behavior", "detail": detail})
    (run_dir / "results.json").write_text(
        json.dumps({"plan": {"model": model}, "readings": readings}), encoding="utf-8"
    )
    if cells is not None:
        (run_dir / "cells.jsonl").write_text(
            "\n".join(json.dumps(row) for row in cells) + "\n", encoding="utf-8"
        )
    return run_dir


def _cell(prompt_id: str, response: str, refused: bool, condition: str = "homoglyph") -> dict:
    return {
        "prompt_id": prompt_id,
        "family": condition,
        "attack_response": response,
        "refused": refused,
    }


class TestTheSpreadIsComputed:
    def test_two_runs_of_one_cell_report_their_range(self, tmp_path):
        _write_run(tmp_path, "m", "run_a", harmful=0.91, benign=0.63)
        _write_run(tmp_path, "m", "run_b", harmful=0.90, benign=0.67)
        cell = collect(sorted((tmp_path / "m").iterdir()))[("m", "homoglyph")]
        assert cell.gap_range == pytest.approx((0.23, 0.28))
        assert cell.gap_spread == pytest.approx(0.05)

    def test_IDENTICAL_REPLICATES_REPORT_ZERO_not_silence(self, tmp_path):
        """The other direction: agreement must be asserted, never inferred."""
        rows = [_cell("p1", "same text", True), _cell("p2", "also same", False)]
        _write_run(tmp_path, "m", "run_a", cells=rows)
        _write_run(tmp_path, "m", "run_b", cells=rows)
        cell = collect(sorted((tmp_path / "m").iterdir()))[("m", "homoglyph")]
        assert cell.gap_spread == pytest.approx(0.0)
        assert cell.reproducibility.response_agreement == pytest.approx(1.0)
        assert cell.reproducibility.verdict_flip_rate == pytest.approx(0.0)

    def test_a_single_replicate_has_no_range_to_report(self, tmp_path):
        _write_run(tmp_path, "m", "only_run")
        assert main([str(tmp_path / "m" / "only_run"), "--min-replicates", "2"]) == 0
        cell = collect([tmp_path / "m" / "only_run"])[("m", "homoglyph")]
        assert cell.gap_spread == pytest.approx(0.0)
        assert len(cell.replicates) == 1


class TestTheTwoRoutesAreSeparated:
    """Naming the wrong cause is the failure this split exists to prevent."""

    def test_a_flip_on_identical_text_is_charged_to_the_judge(self, tmp_path):
        _write_run(tmp_path, "m", "run_a", cells=[_cell("p1", "verbatim", True)])
        _write_run(tmp_path, "m", "run_b", cells=[_cell("p1", "verbatim", False)])
        agreement = collect(sorted((tmp_path / "m").iterdir()))[("m", "homoglyph")].reproducibility
        assert agreement.n_flip_same_text == 1
        assert agreement.n_flip_diff_text == 0
        assert agreement.response_agreement == pytest.approx(1.0)

    def test_a_flip_on_different_text_is_charged_to_generation(self, tmp_path):
        _write_run(tmp_path, "m", "run_a", cells=[_cell("p1", "one continuation", True)])
        _write_run(tmp_path, "m", "run_b", cells=[_cell("p1", "another continuation", False)])
        agreement = collect(sorted((tmp_path / "m").iterdir()))[("m", "homoglyph")].reproducibility
        assert agreement.n_flip_same_text == 0
        assert agreement.n_flip_diff_text == 1
        assert agreement.response_agreement == pytest.approx(0.0)

    def test_different_text_with_the_SAME_verdict_is_not_a_flip(self, tmp_path):
        _write_run(tmp_path, "m", "run_a", cells=[_cell("p1", "one continuation", True)])
        _write_run(tmp_path, "m", "run_b", cells=[_cell("p1", "another continuation", True)])
        agreement = collect(sorted((tmp_path / "m").iterdir()))[("m", "homoglyph")].reproducibility
        assert agreement.verdict_flip_rate == pytest.approx(0.0)
        assert agreement.response_agreement == pytest.approx(0.0)


class TestItReadsTheRealSchema:
    def test_a_plaintext_reading_is_not_a_replicate_of_the_encoded_cell(self, tmp_path):
        _write_run(tmp_path, "m", "run_a")
        _write_run(tmp_path, "m", "run_b")
        cells = collect(sorted((tmp_path / "m").iterdir()))
        assert set(cells) == {("m", "homoglyph")}

    def test_a_behavior_reading_without_its_benign_arm_is_skipped(self, tmp_path):
        _write_run(tmp_path, "m", "run_a", benign=None)
        assert collect([tmp_path / "m" / "run_a"]) == {}

    def test_the_model_comes_from_the_plan_not_the_directory(self, tmp_path):
        _write_run(tmp_path, "declared_model", "run_a")
        cells = collect([tmp_path / "declared_model" / "run_a"])
        assert next(iter(cells))[0] == "declared_model"


class TestUnmeasuredIsNeverZero:
    def test_agreement_is_None_when_no_cells_were_compared(self):
        cell = Cell(model="m", condition="c", replicates=[Replicate("r", 0.9, 0.6)])
        assert cell.reproducibility.response_agreement is None
        assert cell.reproducibility.verdict_flip_rate is None

    def test_a_cell_with_no_replicates_has_no_range(self):
        assert Cell(model="m", condition="c").gap_range is None
        assert Cell(model="m", condition="c").gap_spread is None
