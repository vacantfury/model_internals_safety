"""Build the sampling frame for AS-5's arm survey from the science bib master.

The survey asks whether the encoded-attack literature ever sends BENIGN content
through its own transformation. That is a claim of ABSENCE, and this repo has
already made the absence mistake twice (the 2026-08-06 coverage sweep measured
absence against a narrow index; the model-slate sweep read an aggregator and
concluded a paper was unpublished). So the screen here is deliberately
OVER-INCLUSIVE and decides nothing: it produces a candidate frame that a human
adjudicates against each paper's own text.

Keyless, no network, no model. Reads the science organ's master bib as a
read-only reference (declared oikos edge: reference-read).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# The two populations the survey's 2x2 is built from. Screens are recall-first:
# a term here proposes a candidate, it never classifies one.
TRANSFORM_TERMS = (
    "encod", "cipher", "base64", "base32", "rot13", "caesar", "morse", "atbash",
    "obfusc", "ascii art", "artprompt", "leetspeak", "homoglyph", "unicode",
    "zero-width", "zero width", "fullwidth", "invisible character",
    "bad character", "imperceptible", "low-resource language", "multilingual",
    "translat", "flipattack", "flip attack", "scrambl", "token smuggl",
    "character-level", "steganograph", "puzzle", "word game", "wordgame",
    "mathematical encoding", "novel cipher", "cipher character",
)
OVERREFUSAL_TERMS = (
    "over-refus", "over refus", "overrefus", "exaggerated safety", "xstest",
    "or-bench", "false refusal", "pseudo-harmful", "pseudoharmful",
    "benign", "helpfulness", "oktest", "compliance rate", "utility",
)
# A paper only occupies the empty cell if it measures REFUSAL on transformed
# benign content. These terms flag entries whose abstract already hints at it,
# so adjudication starts with the most likely counterexamples.
CROSS_TERMS = (
    "false positive", "over-refus", "over refus", "benign", "utility",
    "harmless", "helpfulness", "clean",
)


@dataclass(frozen=True)
class Record:
    key: str
    title: str
    year: str
    venue: str
    abstract: str
    pool_transform: bool
    pool_overrefusal: bool
    cross_hint: bool


def _field(body: str, *names: str) -> str:
    """Pull one bibtex field, tolerating both {..} and ".." delimiters."""
    for name in names:
        m = re.search(rf"\b{name}\s*=\s*", body, re.IGNORECASE)
        if not m:
            continue
        i = m.end()
        if i >= len(body):
            continue
        if body[i] == "{":
            depth, j = 0, i
            while j < len(body):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return _clean(body[i + 1 : j])
                j += 1
        elif body[i] == '"':
            j = body.index('"', i + 1)
            return _clean(body[i + 1 : j])
    return ""


def _clean(text: str) -> str:
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse(bib: str) -> list[Record]:
    records: list[Record] = []
    # Entries start at a column-0 @type{key,  -- split there rather than on "@",
    # which also appears inside abstracts and email addresses.
    for m in re.finditer(r"^@(\w+)\s*\{\s*([^,]+),", bib, re.MULTILINE):
        start = m.start()
        nxt = bib.find("\n@", m.end())
        body = bib[start : nxt if nxt != -1 else len(bib)]
        title = _field(body, "title")
        if not title:
            continue
        abstract = _field(body, "abstract", "abstractNote")
        venue = _field(body, "booktitle", "journal") or _field(body, "publisher")
        hay = f"{title} {abstract}".lower()
        records.append(
            Record(
                key=m.group(2).strip(),
                title=title,
                year=_field(body, "year"),
                venue=venue,
                abstract=abstract,
                pool_transform=any(t in hay for t in TRANSFORM_TERMS),
                pool_overrefusal=any(t in hay for t in OVERREFUSAL_TERMS),
                cross_hint=any(t in hay for t in CROSS_TERMS),
            )
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", type=Path, required=True, help="master .bib to screen")
    ap.add_argument("--out", type=Path, help="write the frame as JSON")
    args = ap.parse_args()

    records = parse(args.bib.read_text(encoding="utf-8"))
    transform = [r for r in records if r.pool_transform]
    overrefusal = [r for r in records if r.pool_overrefusal]
    both = [r for r in records if r.pool_transform and r.pool_overrefusal]

    print(f"parsed            {len(records)} entries from {args.bib}")
    print(f"pool A transform  {len(transform)}")
    print(f"pool B overrefusal{len(overrefusal)}")
    print(f"screened into BOTH{len(both)}   <- adjudicate these first")
    print()
    for r in sorted(both, key=lambda r: r.year):
        flag = "!" if r.cross_hint else " "
        print(f"  {flag} {r.year}  {r.key:<44.44} {r.title[:82]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "source_bib": str(args.bib),
                    "n_parsed": len(records),
                    "pool_transform": [asdict(r) for r in transform],
                    "pool_overrefusal": [asdict(r) for r in overrefusal],
                    "screened_both": [asdict(r) for r in both],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
