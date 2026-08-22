"""Integrity of the counted claims a paper makes about its own numbered lists.

**The failure this exists for.** A screen adopted after a table exists will pass
its own tests, appear in the Method, be called required-for-validity in
Limitations, and never have touched the numbers. Nothing catches it, because no
artefact records which screens a given number went through. That is TODO 97, and
it arrived twice in one day from two directions: the echo screen had never been
run on AS-5's pipeline table, and AS-6's ``across all 38 guard-condition pairs''
ranged over a set the control floor had shrunk three subsections earlier.

Both were found the same way, and NOT by tracing provenance forward: by
recomputing the claim's set and comparing it to the set the document actually
contains. That is the cheap half of the guard and it is what this module does.

**The split, stated so the next session does not mistake one for the other.**

*Mechanical (here).* A claim whose set lives inside the document — ``ten
instrument defects'', ``six of the ten inflate apparent safety'', ``defect~(10)''
— can be checked against the document with no run records, no artefacts and no
ledger. This is a comparison of literal facts, so it cannot be green while the
failure is live.

*Artefact-backed (`scripts/claim_sets.py`).* A claim whose set lives in a run
record — ``the 27 model-by-rung cells we screened'', ``rejects four of the nine
encoded cells'' — needs the artefact. That half cannot run in the suite, because
``outputs/'' is gitignored.

Keeping them apart matters: a checker that silently skipped the artefact half
while reporting the mechanical half green would be the ``--dry-run returns
before the guard'' shape, which has cost this repo two queue cycles.

**Why the residual fails rather than warns.** A sentence naming a tracked
predicate together with a cardinality the parser cannot bind is exactly the case
the guard exists for, so it is a failure and not a note. The scope is deliberately
narrow — only sentences about a tracked predicate — so the noise is bounded and
the test survives ordinary prose edits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Spelled-out cardinals a paper actually uses about a defect list, plus digits.
#: Kept small on purpose: a wider table buys nothing and matches more prose.
WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15,
}
_NUM = r"(?:\d+|" + "|".join(WORD_NUMBERS) + r")"

#: Predicates whose cardinality is tracked across the whole document.
#:
#: definitional: this is a closed vocabulary of the claims that partition a
#: numbered list, not a tunable. A predicate enters when a paper starts making a
#: counted claim with it. The residual member is the parse failure below, which
#: fails toward attention rather than defaulting into a convenient class.
TRACKED_PREDICATES: tuple[tuple[str, str], ...] = (
    (r"inflates?\s+apparent\s+safety", "inflate-apparent-safety"),
    (r"on\s+the\s+behaviour\s+axis", "behaviour-axis"),
    (r"on\s+the\s+probe\s+axis", "probe-axis"),
)

#: Ordinal words that name a position in a numbered list.
#:
#: A paper referring to ``the ninth defect'' is making a POSITION claim, and a
#: position claim goes stale the moment the list grows while the sentence stays
#: put. That is not hypothetical here: this file was written because §6 called an
#: unnumbered defect ``the ninth'' while ``\paragraph{(9)}'' was already a
#: different one, and nothing noticed. The list has a stable id syntax, so the
#: rule is that ordinals are never used for it.
ORDINALS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12,
}

#: The noun a total-count claim attaches to.
LIST_NOUN = r"instrument\s+defects"


@dataclass(frozen=True)
class Item:
    """One numbered paragraph, e.g. ``\\paragraph{(4) The refusal judge ...}``."""

    number: int
    title: str


@dataclass(frozen=True)
class Bound:
    """A cardinality bound to a predicate key, and where it was asserted."""

    predicate: str
    cardinality: int
    source: str
    sentence: str


def _strip(tex: str) -> str:
    """Comments out, whitespace collapsed. Paragraph breaks survive as periods.

    Collapsing BEFORE any sentence split is the rule
    `test_public_repo_hygiene.py` paid for: a guard that splits on newlines has a
    strictness that is a function of line width, so the same violating sentence
    passes hard-wrapped and fails unwrapped.
    """
    no_comments = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", no_comments)


def sentences(tex: str) -> list[str]:
    flat = _strip(tex)
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\$])", flat) if s.strip()]


def numbered_items(tex: str) -> list[Item]:
    """Numbered paragraphs, in document order."""
    return [
        Item(int(n), t.strip())
        for n, t in re.findall(r"\\paragraph\{\((\d+)\)\s*([^}]*)\}", _strip(tex))
    ]


def referenced_numbers(tex: str) -> set[int]:
    """Every id the prose points at: ``defect~(10)``, ``Defects (1), (2) and (4)``."""
    flat = _strip(tex)
    found: set[int] = set()
    for match in re.finditer(
        r"\b(?:defects?|items?)~?\s*((?:\(\d+\)[,\s]*(?:and\s+)?)+)", flat, re.I
    ):
        found.update(int(d) for d in re.findall(r"\((\d+)\)", match.group(1)))
    return found


def _cardinal(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else WORD_NUMBERS[token]


def enumerated_bounds(tex: str) -> list[Bound]:
    """``Defects (1), (2) and (10) all inflate apparent safety`` -> 3."""
    out: list[Bound] = []
    for sentence in sentences(tex):
        for match in re.finditer(
            r"\b(?:defects?|items?)~?\s*((?:\(\d+\)[,\s]*(?:and\s+)?)+)", sentence, re.I
        ):
            ids = {int(d) for d in re.findall(r"\((\d+)\)", match.group(1))}
            if len(ids) < 2:  # a pointer to one item is a reference, not a partition
                continue
            tail = sentence[match.end():]
            for pattern, key in TRACKED_PREDICATES:
                if re.search(pattern, tail, re.I):
                    out.append(Bound(key, len(ids), "enumeration", sentence))
    return out


def _enumerates(sentence: str) -> bool:
    """Does this sentence partition the list by naming ids, e.g. ``(1), (2) and (4)``?"""
    return any(
        len(re.findall(r"\((\d+)\)", m.group(1))) >= 2
        for m in re.finditer(
            r"\b(?:defects?|items?)~?\s*((?:\(\d+\)[,\s]*(?:and\s+)?)+)", sentence, re.I
        )
    )


def counted_bounds(tex: str) -> tuple[list[Bound], list[str]]:
    """Cardinalities asserted in words, plus sentences that could not be parsed.

    Two shapes, and the ratio shape is the useful one because it states the
    TOTAL as well: ``six of the ten inflate apparent safety`` pins both the
    subject count and the list length in one phrase.
    """
    out: list[Bound] = []
    unparsed: list[str] = []
    for sentence in sentences(tex):
        keys = [k for p, k in TRACKED_PREDICATES if re.search(p, sentence, re.I)]
        if not keys:
            continue
        ratio = re.search(rf"\b({_NUM})\s+of\s+(?:the\s+)?({_NUM})\b", sentence, re.I)
        bare = re.search(rf"\bthe\s+({_NUM})\s+(?=on\s+the)", sentence, re.I)
        if ratio:
            subject = _cardinal(ratio.group(1))
            out.extend(Bound(k, subject, "ratio", sentence) for k in keys)
            out.append(Bound("__total__", _cardinal(ratio.group(2)), "ratio", sentence))
        elif bare:
            out.extend(Bound(k, _cardinal(bare.group(1)), "bare", sentence) for k in keys)
        elif _enumerates(sentence):
            # Already bound by `enumerated_bounds` — the ids ARE the cardinality.
            # Without this the partition sentence reports itself as unparseable,
            # and a guard that fires on its own correct case is a guard that gets
            # deleted.
            continue
        elif re.search(rf"\b{_NUM}\b", sentence, re.I):
            unparsed.append(sentence)
    return out, unparsed


def total_bounds(tex: str) -> list[Bound]:
    """``ten instrument defects`` -> the list length."""
    return [
        Bound("__total__", _cardinal(n), "list-noun", sentence)
        for sentence in sentences(tex)
        for n in re.findall(rf"\b({_NUM})\s+{LIST_NOUN}", sentence, re.I)
    ]


def ordinal_references(tex: str) -> list[tuple[int, str]]:
    """``the ninth defect`` -> (9, sentence). Position claims, which go stale."""
    ordinals = "|".join(ORDINALS)
    return [
        (ORDINALS[m.group(1).lower()], sentence)
        for sentence in sentences(tex)
        for m in re.finditer(rf"\bthe\s+({ordinals})\s+(?:defect|item)\b", sentence, re.I)
    ]


def problems(tex: str) -> list[str]:
    """Every integrity failure, in a reviewer's words. Empty iff the paper agrees
    with itself about its own numbered list."""
    items = numbered_items(tex)
    found: list[str] = []
    if not items:
        return found

    numbers = [i.number for i in items]
    expected = list(range(1, len(items) + 1))
    if numbers != expected:
        found.append(
            f"numbered items are {numbers}, not a contiguous 1..{len(items)} — "
            "two items claiming one number is how a paper ends up with two ninth defects"
        )

    dangling = referenced_numbers(tex) - set(numbers)
    if dangling:
        found.append(
            f"prose references item(s) {sorted(dangling)}, which the numbered "
            "list does not contain"
        )

    found.extend(
        f"item referred to by ORDINAL position ('the {word} defect') in "
        f"{sentence[:110]!r} — use the stable id, defect~({position}); an ordinal "
        "is a position claim that goes stale when the list grows, and a paper can "
        "end up with two ninth defects without either sentence changing"
        for position, sentence in ordinal_references(tex)
        for word in [next(w for w, n in ORDINALS.items() if n == position)]
    )

    counted, unparsed = counted_bounds(tex)
    bounds = enumerated_bounds(tex) + counted + total_bounds(tex)
    bounds.append(Bound("__total__", len(items), "the list itself", "(counted paragraphs)"))

    by_predicate: dict[str, list[Bound]] = {}
    for bound in bounds:
        by_predicate.setdefault(bound.predicate, []).append(bound)
    for predicate, group in sorted(by_predicate.items()):
        sizes = {b.cardinality for b in group}
        if len(sizes) > 1:
            detail = "; ".join(
                f"{b.cardinality} ({b.source}: {b.sentence[:90]!r})" for b in group
            )
            found.append(f"{predicate!r} is claimed at {sorted(sizes)} — {detail}")

    found.extend(
        f"a cardinality about a tracked predicate that could not be bound: "
        f"{sentence[:140]!r}"
        for sentence in unparsed
    )
    return found
