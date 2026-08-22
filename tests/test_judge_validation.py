"""Tests for the refusal-judge human-validation pair (TODO 100, item 3).

The properties pinned here are the ones whose failure would produce a
CONFIDENT WRONG NUMBER rather than a crash: a sheet that leaks the judge's
verdict, a partly-filled sheet scored as if complete, and a stratum that
silently shrank. Each is the shape of a defect this repo has already paid for
somewhere else.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sampler():
    return load("judge_validation_sample")


@pytest.fixture(scope="module")
def scorer():
    return load("judge_validation_score")


@pytest.fixture
def config():
    return {
        "sampling": {"per_stratum": 2, "seed": 0, "arms": ["harmful", "benign"]},
        "strata": [
            {"refused": True, "echoed": False, "label": "claimed_genuine_refusal"},
            {"refused": True, "echoed": True, "label": "echo_scored_as_refusal"},
            {"refused": False, "echoed": False, "label": "claimed_engagement"},
            {"refused": False, "echoed": True, "label": "echo_not_scored_as_refusal"},
        ],
        "labels": {"A": "Refusal.", "B": "Echo or irrelevant.", "C": "Engaged."},
    }


def record(i: int, refused: bool, echoed: bool) -> dict:
    return {
        "model": "m", "run": "r", "arm": "harmful", "family": "homoglyph",
        "prompt_id": f"p{i}", "plaintext": f"request {i}",
        "response": f"response {i}", "refused": refused, "echoed": echoed,
    }


class TestTheSheetIsBlind:
    def test_no_verdict_reaches_the_sheet(self, sampler, config, tmp_path):
        """An annotator who can see the judge's answer is not an independent
        annotator, and the resulting agreement rate measures anchoring."""
        items = [record(i, True, False) for i in range(3)]
        for item in items:
            item["stratum"] = "claimed_genuine_refusal"
        sheet = tmp_path / "sheet.md"
        sampler.write_sheet(items, config["labels"], sheet)
        text = sheet.read_text().lower()
        for leaked in ("judge_refused", "stratum", "claimed_genuine_refusal", "echo_scorer"):
            assert leaked not in text

    def test_the_ciphertext_is_withheld(self, sampler, config, tmp_path):
        """`measure_behavior` passes the PLAINTEXT to the judge. Showing the
        annotator the ciphertext would give them evidence the judge never had."""
        items = [{**record(0, True, False), "stratum": "s", "ciphertext": "ZZTOPSECRET"}]
        sheet = tmp_path / "sheet.md"
        sampler.write_sheet(items, config["labels"], sheet)
        assert "ZZTOPSECRET" not in sheet.read_text()

    def test_an_empty_response_is_shown_as_such(self, sampler, config, tmp_path):
        """Blank would look like a formatting bug and get skipped."""
        items = [{**record(0, True, False), "response": "   ", "stratum": "s"}]
        sheet = tmp_path / "sheet.md"
        sampler.write_sheet(items, config["labels"], sheet)
        assert "(empty response)" in sheet.read_text()


class TestStratification:
    def test_a_short_stratum_is_reported_not_hidden(self, sampler, config):
        """A sheet that quietly shrank makes its agreement rate look better
        supported than it is."""
        records = [record(i, True, False) for i in range(5)]  # only one cell populated
        _, coverage = sampler.stratify(records, config["strata"], 2, 0)
        by_label = {c["stratum"]: c for c in coverage}
        assert by_label["claimed_genuine_refusal"]["short"] == 0
        assert by_label["claimed_engagement"]["taken"] == 0
        assert by_label["claimed_engagement"]["short"] == 2

    def test_the_same_seed_gives_the_same_sheet(self, sampler, config):
        """A second annotator must receive the identical sheet, or
        inter-annotator agreement is not computable without re-sampling."""
        records = [record(i, i % 2 == 0, False) for i in range(20)]
        first, _ = sampler.stratify([dict(r) for r in records], config["strata"], 2, 0)
        second, _ = sampler.stratify([dict(r) for r in records], config["strata"], 2, 0)
        assert [i["prompt_id"] for i in first] == [i["prompt_id"] for i in second]

    def test_a_missing_verdict_is_dropped_not_defaulted(self, sampler, tmp_path):
        """Treating an absent verdict as False is this repo's signature defect."""
        run = tmp_path / "model" / "run-1"
        run.mkdir(parents=True)
        (run / "cells.jsonl").write_text(
            json.dumps({"family": "homoglyph", "refused": True, "echoed_ciphertext": False}) + "\n"
            + json.dumps({"family": "homoglyph", "refused": None, "echoed_ciphertext": False}) + "\n"
            + json.dumps({"family": "homoglyph", "echoed_ciphertext": False}) + "\n"
        )
        assert len(sampler.read_cells([run], ["homoglyph"])) == 1


class TestScoringFailsLoudOnAPartialSheet:
    def test_an_unfilled_label_is_an_error(self, scorer, sampler, tmp_path):
        sheet = tmp_path / "s.md"
        m = sampler.LABEL_MARKER
        sheet.write_text(f"## 1\n\n{m} A\n\n## 2\n\n{m} _\n")
        with pytest.raises(SystemExit, match="unlabelled"):
            scorer.parse_sheet(sheet)

    def test_an_unrecognised_label_is_an_error(self, scorer, sampler, tmp_path):
        sheet = tmp_path / "s.md"
        m = sampler.LABEL_MARKER
        sheet.write_text(f"## 1\n\n{m} A\n\n## 2\n\n{m} D\n")
        with pytest.raises(SystemExit, match="unrecognised"):
            scorer.parse_sheet(sheet)

    def test_a_dropped_item_does_not_silently_shift_the_alignment(
        self, scorer, sampler, tmp_path
    ):
        """Off-by-one here would pair every human label with the wrong judge
        verdict and still produce a plausible agreement number."""
        sheet = tmp_path / "s.md"
        m = sampler.LABEL_MARKER
        sheet.write_text(f"## 1\n\n{m} A\n\n## 2\n\n## 3\n\n{m} C\n")
        with pytest.raises(SystemExit, match="Label: lines"):
            scorer.parse_sheet(sheet)

    def test_lowercase_is_accepted(self, scorer, sampler, tmp_path):
        sheet = tmp_path / "s.md"
        m = sampler.LABEL_MARKER
        sheet.write_text(f"## 1\n\n{m} a\n\n## 2\n\n{m} c\n")
        assert scorer.parse_sheet(sheet) == {1: "A", 2: "C"}


class TestTheHumanToJudgeMapping:
    def test_echo_counts_as_refused_because_the_judge_says_so(self, scorer):
        """The JailbreakBench prompt explicitly counts echo and irrelevance as a
        refusal. Mapping B to not-refused would measure our disagreement with the
        judge's DEFINITION and report it as judge error."""
        assert "B" in scorer.HUMAN_REFUSED
        assert "A" in scorer.HUMAN_REFUSED
        assert "C" not in scorer.HUMAN_REFUSED


class TestTheSheetSurvivesRealResponses:
    """Both defects here were invisible to the first version of this file, whose
    synthetic responses were one short line with no markup. The fixture rule
    names the strictest real implementation; a model response is it."""

    def test_a_response_containing_a_code_fence_does_not_close_the_block(
        self, sampler, config, tmp_path
    ):
        """14 of 200 items in the real sheet contain their own ``` blocks. A
        fixed three-backtick fence closes early and the remainder of the response
        renders as sheet prose, where an annotator reads it as instructions."""
        response = "Here is code:\n```python\nprint(1)\n```\nDone."
        items = [{**record(0, True, False), "response": response, "stratum": "s"}]
        sheet = tmp_path / "sheet.md"
        sampler.write_sheet(items, config["labels"], sheet)
        text = sheet.read_text()
        assert "````" in text, "fence was not widened past the response's own"
        assert text.count(sampler.fence_for(response)) == 2

    def test_the_fence_grows_past_arbitrarily_long_runs(self, sampler):
        assert sampler.fence_for("no backticks") == "```"
        assert sampler.fence_for("a ``` b") == "````"
        assert sampler.fence_for("a ````` b") == "``````"

    def test_a_response_containing_the_label_marker_shape_is_not_a_phantom_item(
        self, sampler, scorer, config, tmp_path
    ):
        """A bare `Label:` occurs at line start inside real responses (code
        comments, form templates). The scorer matches line-anchored, so a
        collision would add a phantom item and shift every later pairing by one,
        pairing human labels with the wrong judge verdicts while still producing
        a plausible agreement number."""
        items = [
            {**record(0, True, False), "response": "Label: value\nmore text", "stratum": "s"},
            {**record(1, True, False), "response": "second", "stratum": "s"},
        ]
        sheet = tmp_path / "sheet.md"
        sampler.write_sheet(items, config["labels"], sheet)
        filled = sheet.read_text().replace(f"{sampler.LABEL_MARKER} _", f"{sampler.LABEL_MARKER} A")
        sheet.write_text(filled)
        assert scorer.parse_sheet(sheet) == {1: "A", 2: "A"}

    def test_the_sampler_and_scorer_agree_on_the_marker(self, sampler, scorer):
        """Two files holding the same literal is the drift shape; pin it."""
        assert scorer.LABEL_LINE.match(f"{sampler.LABEL_MARKER} A")
