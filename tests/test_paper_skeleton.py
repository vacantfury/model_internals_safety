"""Every paper kit carries the section skeleton a desk check requires.

Founded 2026-08-21, paid for by an automated desk check that FAILED AS-6 on
minimum quality: the draft ran Introduction -> Scope -> Related work -> Method ->
Results -> Limitations -> bibliography and stopped, with no Discussion and no
Conclusion. Length and topic passed; the science was never reached.

⚠️ The internal reviewer-lens self-review run the same day did not catch it. It
graded each section and never asked whether the SET of sections was complete,
because it took "structurally complete" from this repo's own status line, which
enumerates those same six. That is the lit-search lesson one level up: a failed
lookup is not evidence of absence, and neither is an index that was never built
to answer the question being asked of it. So the check is MECHANICAL and lives
here rather than in a reviewer's judgment.

Two properties, both deliberate:

* **Kits are DISCOVERED by glob**, like the parity guard, so a new kit or a new
  paper is covered without editing this file and no venue token has to appear in
  a committed file.
* **The concluding section is matched by INTENT, not by one title.** AS-5 titles
  it ``Conclusion`` and AS-6 ``Discussion and conclusion``; both discharge the
  requirement, and a guard that pinned one spelling would fail the wrong paper.

``paper/`` is gitignored, so a fresh clone has no kits at all. That is the one
legitimate skip and it is stated rather than implied.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent / "paper"

SECTION_PATTERN = re.compile(r"(?m)^\\section\{([^}]*)\}")

#: A section discharges the "does it conclude" requirement if its title contains
#: any of these. Kept broad on purpose: the desk check asks for a Discussion OR a
#: Conclusion, and this guard should never be the thing that dictates a title.
CONCLUDING = ("conclusion", "concluding", "discussion")

#: Likewise for the opening. A paper with no introduction is a different defect,
#: but it is the same check and costs nothing to include.
OPENING = ("introduction",)


def _kits() -> list[Path]:
    if not PAPER_ROOT.is_dir():
        return []
    return sorted(PAPER_ROOT.rglob("paper.tex"))


def _titles(path: Path) -> list[str]:
    body = path.read_text(encoding="utf-8")
    # Comments first: a \section inside a commented-out block is not a section,
    # and this repo leaves long comment blocks in its drafts.
    body = re.sub(r"(?m)(?<!\\)((?:\\\\)*)%.*$", r"\1", body)
    return [title.strip().lower() for title in SECTION_PATTERN.findall(body)]


@pytest.mark.parametrize("kit", _kits(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_kit_has_a_concluding_section(kit: Path) -> None:
    titles = _titles(kit)
    assert titles, f"{kit} declares no sections at all"
    assert any(any(word in title for word in CONCLUDING) for title in titles), (
        f"{kit.relative_to(PAPER_ROOT)} has no Discussion or Conclusion section.\n"
        f"Sections found: {titles}\n"
        "A draft that stops at Limitations reads as truncated and is a desk-reject "
        "on minimum quality before the science is read."
    )


@pytest.mark.parametrize("kit", _kits(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_kit_has_an_introduction(kit: Path) -> None:
    titles = _titles(kit)
    assert any(any(word in title for word in OPENING) for title in titles), (
        f"{kit.relative_to(PAPER_ROOT)} has no Introduction. Sections: {titles}"
    )


def test_the_guard_found_kits_to_guard() -> None:
    """Without this, an empty glob makes the whole file vacuously green.

    ``paper/`` is gitignored, so on a fresh clone there is genuinely nothing to
    check and the skip is correct. What must never happen silently is the glob
    breaking while kits exist.
    """
    if not PAPER_ROOT.is_dir():
        pytest.skip("paper/ is gitignored and absent from this checkout")
    assert _kits(), f"{PAPER_ROOT} exists but no paper.tex was discovered"
