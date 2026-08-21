"""Surface the evidence needed to adjudicate one arm-survey question per paper.

The question: does an input-transformation attack paper send BENIGN content
through its own transformation and report a REFUSAL / false-positive rate on
that arm?

This script decides nothing. It extracts the paper's own text and prints every
sentence that could bear on the question, so the verdict is recorded as a QUOTE
from the primary source rather than as a recollection. That discipline is not
optional here: the claim the survey supports is a claim of ABSENCE, and a
keyword screen that finds nothing is a failed lookup, not evidence.

Keyless, no network, no model. Requires `pdftotext` (poppler).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# definitional: the longest run of text still treated as ONE claim. Above this a
# "sentence" is a reference block or a flattened table, which carries no argument
# and drowns the output. Tuning path: raise it until no adjudicated verdict in
# arm_survey.md changes; it is a --flag so a disputed row can be re-checked
# without editing code.
MAX_CLAIM_CHARS = 600

# Sentences mentioning a benign corpus AT ALL. Recall-first: the adjudicator
# reads them and decides whether the benign content went through the transform.
BENIGN = re.compile(
    r"\b(benign|harmless|safe prompts?|non-?harmful|innocuous|over-?refus|"
    r"false positive|false-positive|FPR|exaggerated safety|XSTest|OR-Bench|"
    r"AlpacaEval|MT-Bench|utility|helpfulness|clean (?:prompts?|queries|inputs?))\b",
    re.IGNORECASE,
)
# Sentences that name a transformation. A paper occupies the empty cell only if
# a single claim joins both, so co-occurrence within a sentence is the signal.
TRANSFORM = re.compile(
    r"\b(encod\w*|cipher\w*|base ?64|base ?32|rot-?13|caesar|morse|atbash|"
    r"obfuscat\w*|ASCII art|art ?prompt|leet\w*|homoglyph\w*|unicode|"
    r"zero-?width|full-?width|translat\w*|low-?resource|scrambl\w*|"
    r"transform\w*|our attack|our method|the attack prompt)\b",
    re.IGNORECASE,
)


@dataclass
class Paper:
    path: Path
    hits_both: list[str] = field(default_factory=list)
    hits_benign_only: list[str] = field(default_factory=list)
    # plumbing: counter seeds. Nothing reported depends on their starting value.
    n_sentences: int = 0
    n_chars: int = 0  # plumbing: counter seed


def extract(pdf: Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-q", str(pdf), "-"], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {pdf}")
    # Un-wrap hard line breaks so sentences survive the two-column layout, then
    # split on terminal punctuation. A line-split search would make strictness a
    # function of column width -- the defect test_public_repo_hygiene.py paid
    # for once already.
    text = re.sub(r"-\n(?=[a-z])", "", out.stdout)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]


def adjudicate(pdf: Path, max_claim_chars: int = MAX_CLAIM_CHARS) -> Paper:
    text = extract(pdf)
    paper = Paper(path=pdf, n_chars=len(text))
    for s in sentences(text):
        if len(s) > max_claim_chars:
            continue
        if not BENIGN.search(s):
            continue
        paper.n_sentences += 1
        (paper.hits_both if TRANSFORM.search(s) else paper.hits_benign_only).append(s)
    return paper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--show-benign-only", action="store_true",
                    help="also print benign sentences with no transform term")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-claim-chars", type=int, default=MAX_CLAIM_CHARS,
                    help="longest run of text still treated as one claim")
    args = ap.parse_args()

    results = []
    for pdf in args.pdfs:
        if not pdf.exists():
            print(f"MISSING  {pdf}", file=sys.stderr)
            continue
        paper = adjudicate(pdf, args.max_claim_chars)
        results.append(paper)
        print("=" * 100)
        print(f"{pdf.name}")
        print(f"  {paper.n_chars:,} chars | {paper.n_sentences} benign-mentioning "
              f"sentences | {len(paper.hits_both)} ALSO name a transform")
        for s in paper.hits_both:
            print(f"   >> {s}")
        if args.show_benign_only:
            for s in paper.hits_benign_only:
                print(f"    . {s}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                [
                    {
                        "paper": p.path.name,
                        "n_chars": p.n_chars,
                        "n_benign_sentences": p.n_sentences,
                        "hits_both": p.hits_both,
                        "hits_benign_only": p.hits_benign_only,
                    }
                    for p in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
