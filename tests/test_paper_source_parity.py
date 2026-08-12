"""A paper's LaTeX copies must not drift apart.

`paper/<paper-id>/` holds one directory per output kit, and each kit carries its
own `paper.tex`. The prose is the same paper; only the preamble differs (author
block, class file). So the body is DUAL-TRUTH by construction, and this file is
the guard that keeps it single-truth in practice.

**Why it exists.** On 2026-08-12 a results table lost a row to a control that
had withdrawn the cell, and the fix was applied to one copy. Four prose blocks
were then ported to the second copy by hand and one — the abstract — was
missed, so the two copies disagreed about a withdrawn number for two days. A
session had flagged the hazard in writing at the end of the same turn that
created it ("worth settling which is canonical before the next edit, or this
recurs") and it recurred inside that very edit. A note that predicts a defect
is not a guard against it; this is, and it takes milliseconds.

**Three properties that are deliberate.**

*It skips when `paper/` is absent, and that is the only legitimate skip.*
`paper/` is gitignored — this repo is public and a paper directory accumulates
reviewer text, venue style files under their own licences, and build artifacts
— so a fresh clone has no paper sources at all. A skip there is correct; a skip
anywhere else means the discovery below broke.

*The preamble is excluded on purpose.* Comparison starts at `\\begin{abstract}`,
because the author block legitimately differs between kits. Everything from the
abstract to `\\end{document}` is the paper and must match.

*Comments are stripped and whitespace collapsed BEFORE splitting.* The copies
are hard-wrapped differently, so a line-level diff reports the wrapping rather
than the content. `tests/test_public_repo_hygiene.py` paid for this lesson once
already — its first version split on newlines and its strictness was therefore a
function of line width.

**No kit directory is named here.** Pairs are discovered by globbing, so adding
a third kit to a paper puts it under this guard without editing this file — and
no venue token has to appear in a committed file to make that work.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent / "paper"

BODY_START = r"\begin{abstract}"
BODY_END = r"\end{document}"


def _body_sentences(path: Path) -> list[str]:
    """The paper's prose as a sentence list, wrapping-independent."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*%.*$", "", text)  # whole-line LaTeX comments
    start = text.find(BODY_START)
    end = text.find(BODY_END)
    if start == -1 or end == -1:
        raise AssertionError(
            f"{path} has no {BODY_START!r}..{BODY_END!r} body — the comparison "
            "window is gone, so this guard would silently compare nothing."
        )
    body = re.sub(r"\s+", " ", text[start:end])
    return [s.strip() for s in re.split(r"(?<=\.)\s+", body) if s.strip()]


def _paper_ids() -> list[str]:
    if not PAPER_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in PAPER_ROOT.iterdir()
        if d.is_dir() and len(list(d.rglob("paper.tex"))) > 1
    )


@pytest.mark.parametrize("paper_id", _paper_ids())
def test_every_kit_of_a_paper_carries_the_same_body(paper_id: str) -> None:
    """Two kits, one paper: the body must be sentence-identical across kits."""
    sources = sorted((PAPER_ROOT / paper_id).rglob("paper.tex"))
    reference = sources[0]
    expected = _body_sentences(reference)

    for other in sources[1:]:
        actual = _body_sentences(other)
        matcher = difflib.SequenceMatcher(None, expected, actual)
        divergences = [op for op in matcher.get_opcodes() if op[0] != "equal"]
        if not divergences:
            continue

        report = [
            f"{paper_id}: {reference.parent.name} and {other.parent.name} "
            f"disagree in {len(divergences)} place(s). "
            "One copy was edited and the other was not.",
        ]
        for tag, i1, i2, j1, j2 in divergences[:5]:
            report.append(f"  [{tag}]")
            for sentence in expected[i1:i2][:3]:
                report.append(f"    {reference.parent.name}: {sentence[:200]}")
            for sentence in actual[j1:j2][:3]:
                report.append(f"    {other.parent.name}: {sentence[:200]}")
        raise AssertionError("\n".join(report))


def test_the_guard_found_something_to_guard() -> None:
    """A discovery guard that discovers nothing is a green checkmark over air.

    This is the one place the absent-`paper/` skip is allowed, and it is stated
    rather than implied: if the tree IS present and multi-kit papers exist, the
    parametrised test above must have been given work to do.
    """
    if not PAPER_ROOT.is_dir():
        pytest.skip("`paper/` is gitignored and absent — nothing to compare")

    multi_kit = _paper_ids()
    if not multi_kit:
        pytest.skip("no paper in `paper/` has more than one kit yet")

    for paper_id in multi_kit:
        sources = list((PAPER_ROOT / paper_id).rglob("paper.tex"))
        assert len(sources) > 1, f"{paper_id} was selected but has one source"
