"""A paper must agree with itself about its own numbered list.

Pins `internals_safety.paper_claims`. Every case is MUTATED in both directions:
a clean document passes and each defect class fails. A screen that only ever
fires is a verdict with a script attached, and one that never fires is a green
checkmark over a live failure — this file exists to rule out both.

**Fixture strictness.** The synthetic documents below are hard-wrapped mid-
sentence and carry LaTeX comments, because the real kits are. A fixture written
as one long line would make the guard look robust while its strictness was
really a function of line width, which is the defect `test_public_repo_hygiene.py`
already paid for once.

**The one legitimate skip is `paper/` being absent**, stated rather than implied:
the tree is gitignored (this repo is public and a paper directory accumulates
reviewer text and venue style files), so a fresh clone has no kits to check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from internals_safety.paper_claims import (
    numbered_items,
    ordinal_references,
    problems,
    referenced_numbers,
)

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "paper"

# Hard-wrapped and commented, like the real thing.
CLEAN = r"""
\begin{abstract}
We report three instrument defects, each with the control
that caught it. Two of the three inflate apparent
safety, which is the direction a broken evaluation fails in.
\end{abstract}

% a comment mentioning nine defects, which must be ignored
\section{The instrument}

\paragraph{(1) No denominator.} Refusal was computed on
encoded corpora only.

\paragraph{(2) No benign arm.} Benchmarks send only harmful
content, so $0.99$ has nothing to be read against.

\paragraph{(3) The binary judge.} It fires at $0.70$ on
plaintext benign prompts.

\paragraph{The direction.} Defects (1) and (2) all inflate
apparent safety. Defect (3) is the exception.
"""


def mutate(text: str, before: str, after: str) -> str:
    assert text.count(before) == 1, f"fixture drift: {before!r}"
    return text.replace(before, after)


class TestACleanDocumentPasses:
    def test_no_problems(self):
        assert problems(CLEAN) == []

    def test_the_list_is_found_at_all(self):
        """Guards against passing by parsing nothing — the failure mode of every
        regex checker, and the reason this assertion is not redundant."""
        assert [i.number for i in numbered_items(CLEAN)] == [1, 2, 3]

    def test_an_empty_document_is_not_judged(self):
        """No numbered list means no claims about one. Silence, not a failure."""
        assert problems(r"\section{Nothing here}") == []


class TestTheDefectsThatActuallyHappened:
    """Each of these is a real 2026-08-22 defect in AS-5, reduced to the fixture."""

    def test_a_stated_total_disagreeing_with_the_list(self):
        """The abstract said nine while the section carried ten."""
        broken = mutate(CLEAN, "three instrument defects", "nine instrument defects")
        assert any("__total__" in p for p in problems(broken))

    def test_a_partition_count_disagreeing_with_its_enumeration(self):
        """'the eight on the behaviour axis' against a six-item enumeration."""
        broken = mutate(
            CLEAN,
            "Two of the three inflate apparent\nsafety",
            "the three on the behaviour axis inflate apparent\nsafety",
        )
        found = problems(broken)
        assert any("inflate-apparent-safety" in p for p in found), found

    def test_an_ordinal_position_reference(self):
        """§6 called an unnumbered defect 'the ninth' while (9) was another one."""
        broken = mutate(CLEAN, "Defect (3) is the exception.", "The third defect is the exception.")
        assert any("ORDINAL" in p for p in problems(broken))

    def test_a_reference_to_an_item_the_list_does_not_contain(self):
        broken = mutate(CLEAN, "Defect (3) is the exception.", "Defect (7) is the exception.")
        assert any("does not contain" in p for p in problems(broken))

    def test_a_gap_in_the_numbering(self):
        broken = mutate(CLEAN, r"\paragraph{(3) The binary judge.}", r"\paragraph{(4) The binary judge.}")
        assert any("contiguous" in p for p in problems(broken))


class TestTheParserDoesNotFireOnItsOwnCorrectCase:
    """An enumeration IS its own cardinality, so it must not also report as an
    unbindable count. Without this the partition sentence flags itself, and a
    guard that fails on correct input is a guard that gets deleted."""

    def test_an_enumeration_is_not_reported_as_unparseable(self):
        assert not any("could not be bound" in p for p in problems(CLEAN))

    def test_a_single_id_reference_is_not_read_as_a_partition(self):
        """'Defect (3) is the exception' names one item; it is a pointer, not a
        claim that the predicate holds of exactly one."""
        assert referenced_numbers(CLEAN) == {1, 2, 3}

    def test_ordinals_outside_the_list_vocabulary_are_left_alone(self):
        """'the second we found' is prose, not a position claim about the list."""
        text = CLEAN + "\nThis is the second one we found by turning an instrument on a result."
        assert ordinal_references(text) == []


@pytest.mark.skipif(not PAPER_DIR.exists(), reason="paper/ is gitignored; a fresh clone has no kits")
class TestTheRealKits:
    @pytest.mark.parametrize("kit", sorted(PAPER_DIR.glob("as-*/**/paper.tex")), ids=str)
    def test_every_kit_agrees_with_itself(self, kit: Path):
        """Kits are DISCOVERED by glob, so a third venue is covered without
        editing this file and no venue token has to appear in a committed one."""
        found = problems(kit.read_text())
        assert not found, f"{kit.relative_to(REPO)}:\n  " + "\n  ".join(found)


# --------------------------------------------------------------------------
# The artefact-backed half's ledger. The RECOMPUTATION cannot run here
# (`outputs/` is gitignored), but the ledger's own integrity can, and a ledger
# naming a recipe nobody implemented is the failure that would make
# `scripts/claim_sets.py` silently check fewer claims than it prints.
# --------------------------------------------------------------------------

sys.path.insert(0, str(REPO / "scripts"))
import claim_sets  # noqa: E402


class TestTheArtefactBackedLedger:
    @staticmethod
    def ledger() -> list[dict]:
        return yaml.safe_load(claim_sets.LEDGER.read_text())["claims"]

    def test_every_recipe_named_is_implemented(self):
        """The closed vocabulary. An unknown recipe must be a hard failure and
        never a skip, which is the typed residual failing toward attention."""
        named = {claim["recipe"] for claim in self.ledger()}
        assert not named - set(claim_sets.RECIPES), sorted(named - set(claim_sets.RECIPES))

    def test_every_implemented_recipe_is_used(self):
        """The vacuity guard, borrowed from `test_entrypoint_call_sites`: a
        recipe matching no ledger entry checks nothing and reads as coverage."""
        used = {claim["recipe"] for claim in self.ledger()}
        assert not set(claim_sets.RECIPES) - used, sorted(set(claim_sets.RECIPES) - used)

    def test_every_locate_pattern_has_exactly_one_capturing_group(self):
        """The captured group IS the paper's assertion. Zero groups or two means
        the comparison silently reads the wrong token."""
        for claim in self.ledger():
            assert re.compile(claim["locate"]).groups == 1, claim["id"]

    def test_the_ledger_holds_no_second_copy_of_any_value(self):
        """One truth per claim, and it is the paper. A ledger carrying its own
        expected number would go stale in the direction nobody checks."""
        for claim in self.ledger():
            assert "expect" not in claim, claim["id"]
            assert "value" not in claim, claim["id"]

    @pytest.mark.skipif(not PAPER_DIR.exists(), reason="paper/ is gitignored; a fresh clone has no kits")
    def test_every_locate_pattern_finds_exactly_one_sentence_in_every_kit(self):
        """A pattern matching nothing is a claim that stopped being checked when
        someone reworded the sentence, and it would pass as silently as a green
        recomputation."""
        for claim in self.ledger():
            kits = sorted((PAPER_DIR / claim["paper"]).glob("**/paper.tex"))
            if not kits:
                pytest.skip(f"no kits under paper/{claim['paper']}/")
            for kit in kits:
                flat = re.sub(r"\s+", " ", kit.read_text())
                hits = re.findall(claim["locate"], flat)
                assert len(hits) == 1, f"{claim['id']} in {kit.relative_to(REPO)}: {len(hits)} hits"
