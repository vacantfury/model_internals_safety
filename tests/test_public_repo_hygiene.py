"""This repo is PUBLIC and name-linked. It must never say where a paper goes.

**The rule, checked against the primary source 2026-08-08.** The venue's own
instructions — quoted in full in the science organ's venue records, deliberately
NOT here — permit preprints and public online material under double-blind review
on two conditions: the anonymous PDF must carry no citation or pointer to the
non-anonymous material, and the non-anonymous material must not reference the
fact that the work was sent anywhere. Breaking either is grounds for summary
rejection.

**This file names no venue outside its own fixtures, and that is deliberate.** A
public file explaining where the work goes is the violation one level up — the
first drafts of this docstring, of the `CLAUDE.md` bullet and of the
`proposal.md` replacement all failed the check they were describing. State the
rule; never the destination.

A public GitHub repo under the author's name IS non-anonymous online material.
So the first condition is the paper's problem and the second is THIS REPO'S, and
only the second can be enforced here.

**Why a test.** `text_docs/as5/proposal.md` carried a venue name, its special
track and two deadline dates for eleven days, inside this paper's own
venue-strategy section. Nothing caught it: the repo has public-grade discipline
for secrets and PII, and venue linkage is a different failure with the same
shape — invisible to the author, obvious to a reviewer, and expensive exactly
once.

**What it deliberately does NOT forbid: naming a venue as a CITATION.** "Patchscopes
(ICML 2024)" and "Tulu 3 is COLM 2025" are evidence-tier claims the
method-provenance law requires. The signal is not the venue token, it is a venue
token sitting next to submission machinery — a deadline, a track, a CFP. Citation
rows never contain those.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Venues this line of work could plausibly be sent to. Grow it when a venue
# enters the plan — the cost of a false positive is one allowlist entry, and the
# cost of a false negative is a summary rejection.
VENUES = (
    r"AAAI|IJCAI|ICLR|NeurIPS|ICML|COLM|AISTATS|"
    r"\bACL\b|EMNLP|NAACL|EACL|AACL|ARR|"
    r"USENIX|IEEE S&P|Oakland|\bCCS\b|NDSS|SaTML"
)
# Submission machinery. A citation never carries one of these; a plan always does.
SUBMISSION = (
    r"deadline|submit|submitted|submission site|camera.ready|abstract due|"
    r"\bCFP\b|call for papers|special track|rebuttal|author response|desk reject"
)
# Same sentence, either order. Sentence-scoped rather than line-scoped because
# this repo's markdown puts a whole paragraph on one line, so a line window would
# pair a citation in one sentence with a deadline three sentences later.
#
# ⚠️ **Paragraphs are UNWRAPPED first, and the first version of this file did not
# do that.** Treating `\n` as a sentence terminator makes the guard depend on how
# a file happens to be wrapped: `CLAUDE.md` puts a paragraph on one line and was
# flagged, while the hard-wrapped `proposal.md` passed with the SAME violating
# sentence, because the venue token and "submitted" sat on different physical
# lines. A checker whose strictness varies with line width is not a checker.
# ---------------------------------------------------------------------------
# A SECOND, SEPARATE RULE, sharing this file only because it needs the same
# sentence machinery: no committed file states the page count of a draft.
#
# Standing owner rule: pages are considered right before submission and at no
# other time (global law, research-process time blindness -- "page limits ...
# weighed ONLY when the owner raises them in the current arc"). Reporting a page
# count invites length to steer content, which is the whole reason the rule
# exists.
#
# ⚠️ This is an ENFORCEMENT-lane fix, not a third statement of the rule. The law
# already said it; it was violated in two separate paper namespaces on the same
# day (AS-5's §4m and the session board, AS-6's build note). The improvement
# loop's escalation rule is explicit that a law violated a second time gets a
# deterministic assist rather than more prose. Making the omission inexpressible
# is the fix; remembering it is not.
PAGE_COUNT = r"\b\d+\s*pages?\b|\bpage (?:limit|count|budget)s?\b"
# What makes a page count a DRAFT's page count. Bare "pages" in a citation field
# (`pages = {37830--37838}`) carries no digits-then-"pages" form and is unaffected.
DRAFT = (
    r"\bkits?\b|\bpapers?\b|\bdrafts?\b|\bmanuscripts?\b|"
    r"\bbuilds?\b|\bbuilt\b|compiles?|rebuilds?|\.tex\b|arxiv|camera.ready"
)

_SENTENCE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


def sentences(text: str) -> list[str]:
    """Sentences, with soft line wrapping removed but paragraphs kept apart."""
    for paragraph in re.split(r"\n\s*\n", text):
        unwrapped = " ".join(paragraph.split())
        for match in _SENTENCE.finditer(unwrapped):
            sentence = match.group().strip()
            if sentence:
                yield sentence


def tracked_text_files() -> list[Path]:
    """Committed text files only — `git ls-files` is the definition of public.

    Not a filesystem walk: `paper/`, `outputs/`, `TODO.md` and `NOW.md` are
    gitignored precisely so they may hold this material, and a walk would flag
    the files the gitignore exists to protect.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    suffixes = {".md", ".txt", ".rst", ".yaml", ".yml", ".py", ".tex", ".bib"}
    return [ROOT / name for name in listed if name and Path(name).suffix in suffixes]


def offending_sentences(text: str) -> list[str]:
    return [
        sentence[:200]
        for sentence in sentences(text)
        if re.search(VENUES, sentence) and re.search(SUBMISSION, sentence, re.IGNORECASE)
    ]


def test_the_scanner_sees_a_real_corpus():
    """A checker over an empty set is a green tick, not evidence.

    This repo's own recorded defect: absence measured against a narrow index,
    reported as absence in general.
    """
    files = tracked_text_files()
    assert len(files) > 50, f"only {len(files)} tracked text files — the scan is not running"


# The guard's own fixtures ARE violations by construction, so it cannot scan
# itself. Narrow and named rather than a pattern: exactly one file, and its
# fixtures use a stand-in venue this work is not aimed at, so the exemption
# hides no real linkage.
SELF = Path(__file__).name


@pytest.mark.parametrize("path", tracked_text_files(), ids=lambda p: p.name)
def test_no_committed_file_links_this_work_to_a_venue_deadline(path: Path):
    if path.name == SELF:
        pytest.skip("the guard's fixtures are violations by construction")
    hits = offending_sentences(path.read_text(encoding="utf-8", errors="replace"))
    assert not hits, (
        f"{path.relative_to(ROOT)} names a venue next to submission machinery, in a "
        f"PUBLIC repo. Under double-blind review the non-anonymous online material "
        f"must not reference the fact that the work was sent anywhere, and breaking "
        f"that is grounds for summary rejection. Move it to the venue records "
        f"(private) and leave a pointer. Offending sentence(s): {hits}"
    )


class TestTheGuardWouldHaveCaughtTheRealOne:
    """Mutation, on the sentence that was actually committed for eleven days."""

    def test_the_withdrawn_proposal_line_is_rejected(self):
        """The shape of the real one, with a stand-in venue.

        The venue is swapped for one this line of work is not aimed at, so the
        fixture exercises the guard without this file becoming the linkage it
        exists to prevent. Structure — venue, special track, two dates, CFP —
        is byte-for-byte the sentence that sat in `proposal.md`.
        """
        withdrawn = (
            "Nearest-window fact (from the canonical timeline, captured 2026-08-02): "
            "IJCAI-27 Some Special Track — abstract 2026-08-14, full "
            "paper 2026-08-21, AoE; CFP text/submission site still pending as of 07-29."
        )
        assert offending_sentences(withdrawn)

    @pytest.mark.parametrize("citation", [
        "Patchscopes (arXiv 2401.06102, ICML 2024, 233c) is the peer-reviewed framework.",
        "Tulu-3 is COLM 2025 — the OpenReview PDF carries the header.",
        "| 2607.08883 | AAAI Symposium Series | 0-4 |",  # a real committed row
        "arXiv 2507.11878 (NeurIPS 2025) studies exactly our two capture positions.",
    ])
    def test_a_plain_citation_is_NOT_rejected(self, citation):
        """The guard must not tax the method-provenance law it sits beside."""
        assert not offending_sentences(citation)

    def test_a_deadline_with_no_venue_is_not_rejected(self):
        """Only the LINKAGE is forbidden; this repo may still discuss its own work."""
        assert not offending_sentences("The phase-1 build has no deadline attached.")

    def test_HARD_WRAPPING_DOES_NOT_HIDE_A_VIOLATION(self):
        """The defect this guard had on its first run, pinned.

        Splitting on `\\n` made strictness a function of line width: the same
        sentence passed in a hard-wrapped file and failed in an unwrapped one.
        Both forms below are one sentence and both must be rejected.
        """
        unwrapped = (
            "Checked against the IJCAI-27 call for papers: the material "
            "must not reference the fact that the work was submitted."
        )
        wrapped = (
            "Checked against the IJCAI-27 call for papers: the material\n"
            "must not reference the fact that the work was submitted."
        )
        assert offending_sentences(unwrapped)
        assert offending_sentences(wrapped), "hard wrapping still hides the linkage"

    def test_a_paragraph_break_still_separates(self):
        """Unwrapping must not fuse two paragraphs into one window."""
        text = "Patchscopes is ICML 2024.\n\nThe build has no deadline attached."
        assert not offending_sentences(text)


def page_count_sentences(text: str) -> list[str]:
    """Sentences stating a draft's page count."""
    return [
        sentence[:200]
        for sentence in sentences(text)
        if re.search(PAGE_COUNT, sentence, re.IGNORECASE)
        and re.search(DRAFT, sentence, re.IGNORECASE)
    ]


@pytest.mark.parametrize("path", tracked_text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_committed_file_states_a_draft_page_count(path: Path):
    if path.name == SELF:
        pytest.skip("the guard's fixtures are violations by construction")
    offenders = page_count_sentences(path.read_text(encoding="utf-8", errors="ignore"))
    assert not offenders, (
        f"{path.relative_to(ROOT)} states a draft's page count:\n  "
        + "\n  ".join(offenders)
        + "\n\nPages are considered right before submission and at no other time"
        "\n(global law: research-process time blindness). Report what the build"
        "\nactually certifies instead: 0 errors, no undefined references, no"
        "\noverfull boxes."
    )


class TestThePageCountGuardIsCalibrated:
    """A guard that fires on nothing, or on everything, is not a guard."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "Both kits build clean: 0 errors, 0 overfull boxes, 6 pages.",
            "arXiv kit rebuilds at 7 pages, zero LaTeX warnings.",
            "The draft is over the page limit.",
            "The paper compiles to 8 pages.",
        ],
    )
    def test_a_draft_page_count_is_rejected(self, sentence):
        assert page_count_sentences(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            # A citation's page RANGE is not a page count and must survive.
            "Prakash et al., Proceedings of AAAI, pages 37830-37838.",
            # Prose about a PDF page of someone else's paper is not our draft.
            "Their Appendix D spans several pages of detailed results.",
            # The word "pages" with no draft in the sentence.
            "The corpus is 200 pages of scraped text.",
            # A build report that says what a build actually certifies.
            "Both kits build clean: 0 errors, no undefined references.",
        ],
    )
    def test_a_legitimate_use_is_NOT_rejected(self, sentence):
        assert not page_count_sentences(sentence)

    def test_HARD_WRAPPING_DOES_NOT_HIDE_A_VIOLATION(self):
        """Same lesson the venue guard paid for: unwrap before splitting."""
        wrapped = "The arXiv kit rebuilds at\n7 pages, zero warnings."
        assert page_count_sentences(wrapped)
