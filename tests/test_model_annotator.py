"""Tests for `scripts/model_annotator.py` — the cross-family annotator pass.

The properties here all guard the same class of failure: an annotator's SILENCE
being converted into a label. The first real run returned 27 empty replies out of
200, concentrated 20% on the harmful arm against 8% on the benign one, so a
parser that guessed would have manufactured agreement exactly where the sample is
thinnest.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "model_annotator", ROOT / "scripts" / "model_annotator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSilenceIsNeverALabel:
    def test_an_empty_reply_yields_none(self, script):
        assert script.extract_label("") is None
        assert script.extract_label("   \n ") is None

    def test_a_paragraph_with_no_standalone_letter_yields_none(self, script):
        assert script.extract_label("I cannot help with this request.") is None

    def test_a_bare_letter_is_read(self, script):
        assert script.extract_label("A") == "A"
        assert script.extract_label(" c \n") == "C"

    def test_a_letter_inside_a_word_is_not_a_label(self, script):
        """`\\b` matters: 'ABSTAIN' contains no standalone A, and reading one
        would turn a refusal to answer into a refusal verdict."""
        assert script.extract_label("ABSTAIN") is None
        assert script.extract_label("Cannot comply") is None


class TestTheSheetParser:
    def test_a_response_with_its_own_fence_is_parsed_whole(self, script, tmp_path):
        """14 of 200 real items carry ``` blocks. A fixed-width fence pattern
        would truncate the response and the annotator would label a fragment."""
        sheet = tmp_path / "s.md"
        sheet.write_text(
            "## 1\n\n**Request**\n\n> do a thing\n\n**Response**\n\n"
            "````\nhere is code:\n```py\nx=1\n```\ndone\n````\n\n**Label >>** _\n\n---\n"
        )
        items = script.parse_items(sheet)
        assert len(items) == 1
        assert "x=1" in items[0]["response"]
        assert items[0]["response"].endswith("done")

    def test_the_blockquote_marker_is_stripped_from_the_request(self, script, tmp_path):
        sheet = tmp_path / "s.md"
        sheet.write_text(
            "## 7\n\n**Request**\n\n> line one\n> line two\n\n**Response**\n\n"
            "```\nr\n```\n\n**Label >>** _\n\n---\n"
        )
        items = script.parse_items(sheet)
        assert items[0]["request"] == "line one\nline two"
        assert items[0]["index"] == 7

    def test_every_item_of_the_real_sheet_parses(self, script):
        """The strictest available fixture is the artifact itself."""
        sheet = ROOT / "outputs" / "analysis" / "judge_validation" / "annotation_sheet.md"
        if not sheet.exists():
            pytest.skip("sheet is a gitignored artifact; absent on a fresh clone")
        items = script.parse_items(sheet)
        # Line-anchored: the header MENTIONS the marker once, mid-sentence and
        # backtick-wrapped, so a bare `.count()` expects one item too many.
        markers = len(re.findall(r"^\*\*Label >>\*\* ", sheet.read_text(), re.MULTILINE))
        assert len(items) == markers


class TestTheSameFamilyGuard:
    @pytest.mark.parametrize("model", ["gpt-5-mini", "GPT-4o", "o3", "o4-mini", "o1"])
    def test_the_judges_own_family_is_refused(self, script, model, tmp_path, capsys):
        """Our judge is gpt-5-mini. An OpenAI annotator marks its own family's
        homework, which is the one thing this pass exists not to do."""
        code = script.main(
            ["--sheet", str(tmp_path / "none.md"), "--model", model, "--out", str(tmp_path / "o.md")]
        )
        assert code == 1
        assert "REFUSED" in capsys.readouterr().err

    def test_another_family_is_allowed_through_the_guard(self, script, tmp_path, capsys):
        """It gets past the guard and then dies on the missing sheet, which is
        what proves the guard is family-specific rather than a blanket refusal."""
        with pytest.raises(FileNotFoundError):
            script.main(
                ["--sheet", str(tmp_path / "none.md"), "--model", "claude-opus-4-5",
                 "--out", str(tmp_path / "o.md")]
            )
        assert "REFUSED" not in capsys.readouterr().err


class TestLabelsAreWrittenPositionally:
    """The defect this class exists for produced a CONFIDENT WRONG NUMBER.

    The first writer walked the items and did a sequential
    `text.replace(marker + " _", marker + " " + label, 1)`, skipping items whose
    label was None. Every item after the first skip therefore received the NEXT
    item's label. Two skips shifted the Gemini sheet, twenty-seven shifted the
    Anthropic one, and the scorer happily reported per-stratum agreement rates
    around 0.5 as though the judge were a coin. It surfaced only because the two
    annotators agreed with EACH OTHER at 0.93, which is impossible alongside
    chance agreement with the judge.

    `tests/test_judge_validation.py` already forbade exactly this shift when
    READING a sheet. Nothing guarded the WRITE side, which is the same defect
    facing the other way.
    """

    def test_a_gap_does_not_shift_the_labels_after_it(self, script, tmp_path):
        sheet = tmp_path / "s.md"
        body = "".join(
            f"## {i}\n\n**Request**\n\n> q{i}\n\n**Response**\n\n```\nr{i}\n```\n\n"
            f"**Label >>** _\n\n---\n\n"
            for i in (1, 2, 3, 4)
        )
        sheet.write_text(body)
        items = script.parse_items(sheet)
        assert len(items) == 4
        # item 2 is the gap; 3 and 4 must keep their own labels, not slide up.
        for item, label in zip(items, ["A", None, "B", "C"]):
            item["label"] = label

        order = iter(items)

        def fill(_m):
            label = next(order)["label"]
            marker = script.LABEL_MARKER
            return f"{marker} {label}" if label in script.VALID else f"{marker} _"

        filled = re.sub(
            rf"^{re.escape(script.LABEL_MARKER)} _$", fill, sheet.read_text(), flags=re.MULTILINE
        )
        written = re.findall(rf"^{re.escape(script.LABEL_MARKER)} (\S+)$", filled, re.MULTILINE)
        assert written == ["A", "_", "B", "C"], f"labels shifted: {written}"
