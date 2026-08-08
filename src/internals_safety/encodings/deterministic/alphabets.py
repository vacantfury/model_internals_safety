"""Letter-alphabet substitutions — the rungs that un-confound `fullwidth`.

## The confound this module exists to break

Two independent screens landed on 2026-08-07 and **no sound rung passes both**
(`text_docs/shared/instrument_layer.md` §3.6, §3.7):

| rung | refusal tracks harm? | (S) free of echo? |
|---|---|---|
| `zero_width` | ✗ 0.93 harmful vs 0.90 benign | ✗ 74% echoed |
| `homoglyph`  | ✗ 0.99 vs 0.99                | ✓ 9% echoed |
| `fullwidth`  | **✓ 0.90 vs 0.59**            | ✗ 71% echoed |

`fullwidth` is the only rung whose refusal demonstrably tracks harm rather than
surface form, which is what promoted it to phase-1 substrate. But it differs
from the other two on **two axes at once**, and the band run cannot say which
one earned the 31-point gap:

* **completeness of substitution** — `fullwidth` maps *every* printable ASCII
  into a disjoint range; `homoglyph` maps 21 characters and leaves the rest
  Latin; `zero_width` maps none and only interleaves.
* **corpus naturalness** — fullwidth Latin is pervasive in CJK-adjacent
  pretraining text, so it is a *script the model has read as ordinary text*.
  Zero-width characters and mixed-script confusables appear in pretraining
  almost exclusively in spam, phishing and homograph-attack contexts, so a
  learned surface-cue refusal for those two would swamp harm-sensitivity.

Those two explanations predict opposite things about phase 1. If completeness
drives it, any complete substitution is substrate and the effect is cheap to
reproduce. If naturalness drives it, substrate is a property of *which* script,
and a rung has to be chosen rather than constructed.

## How these rungs separate them

Unicode supplies several **complete, exactly-invertible A-Z/a-z alphabets** at
graded corpus frequency. Holding completeness fixed at "complete over the
letters" and varying only the alphabet turns the confound into a series:

| rung | block | what a reader sees |
|---|---|---|
| `fullwidth_letters` | U+FF21 / U+FF41 | ordinary English, wide |
| `math_bold` | U+1D400 / U+1D41A | ordinary English, bold |
| `math_sans` | U+1D5A0 / U+1D5BA | ordinary English, sans-serif |
| `math_monospace` | U+1D670 / U+1D68A | ordinary English, typewriter |
| `circled` | U+24B6 / U+24D0 | ordinary English, each letter ringed |
| `math_fraktur` | U+1D504 / U+1D51E | ordinary English, blackletter |

`fullwidth_letters` is the **bridge point**: it is `fullwidth` restricted to
letters, so the pair isolates the punctuation-and-space half of `fullwidth`'s
substitution as its own control. Every other rung here is letters-only too, so
digits, punctuation and spacing are held byte-identical across the whole band
and the alphabet is the only thing that moves.

## The ordering variable is MEASURED, not asserted

"Corpus naturalness" is not observable and this module does not claim it. What
is observable, keylessly and instantly, is **tokenizer fertility** — tokens per
character under the target model's own tokenizer. A script a tokenizer has seen
often earns efficient merges; a rare one falls back to bytes. `scripts/
alphabet_fertility.py` measures it per rung per model and that measurement, not
the table above, orders the band.

Stated honestly up front: fertility and corpus exposure are **not separable by
this design**. Fertility is a proxy for exposure and may equally be the causal
variable itself — a rung that costs four tokens per character may fail to
transmit harm for reasons that have nothing to do with how familiar it looks.
Either result is informative and the two readings are named in the write-up
rather than one being assumed.

## Why the tables are generated rather than typed

Every alphabet here is built by walking a codepoint range and **verifying each
glyph NFKC-folds back to its ASCII letter**, raising at import if any letter is
missing or wrong. Hand-typed codepoint tables fail silently — a mistyped offset
encodes to a neighbouring glyph and the round-trip still succeeds, because the
inverse table carries the same mistake. Fraktur is the case that motivates it:
five uppercase positions (C H I R Z) are reserved in the Mathematical
Alphanumeric Symbols block because the characters already exist in Letterlike
Symbols, so a naive contiguous range silently produces reserved codepoints.
"""

from __future__ import annotations

import unicodedata

from internals_safety.encodings.base import Encoder

ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"

# Fraktur capitals reserved in the Mathematical Alphanumeric Symbols block; the
# characters live in Letterlike Symbols instead. Generated ranges MUST override
# these five or they emit reserved codepoints.
# constant: Unicode Letterlike Symbols substitutes for reserved Fraktur capitals
_FRAKTUR_CAPITAL_OVERRIDES = {
    "C": "ℭ", "H": "ℌ", "I": "ℑ", "R": "ℜ", "Z": "ℨ",
}


def _build_alphabet(
    upper_start: int, lower_start: int, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Generate a letter map from two range starts and VERIFY it by NFKC.

    The verification is the point. A wrong start offset produces a perfectly
    self-consistent encoder — encode and decode carry the same error, so the
    round-trip test passes while the model is shown a different script than the
    one the rung claims. NFKC folding is an independent check: it is the
    Unicode-declared compatibility mapping, so it disagrees with a mistyped
    offset even though our own inverse table cannot.
    """
    overrides = overrides or {}
    table: dict[str, str] = {}
    for letters, start in ((ASCII_UPPER, upper_start), (ASCII_LOWER, lower_start)):
        for index, letter in enumerate(letters):
            glyph = overrides.get(letter) or chr(start + index)
            folded = unicodedata.normalize("NFKC", glyph)
            if folded != letter:
                raise ValueError(
                    f"alphabet is wrong at {letter!r}: U+{ord(glyph):04X} "
                    f"({unicodedata.name(glyph, '?')}) folds to {folded!r}, not {letter!r}"
                )
            table[letter] = glyph
    # No injectivity check here, deliberately: the fold above already PROVES it.
    # Distinct letters fold from distinct glyphs, so two letters sharing a glyph
    # would require that glyph to fold to both — which the loop rejects first.
    # A separate guard could never fire, and unreachable guards read as
    # protection that is not there. `test_alphabet_band` asserts injectivity as
    # a property of every built alphabet instead.
    return table


LETTER_ALPHABETS: dict[str, dict[str, str]] = {
    "fullwidth_letters": _build_alphabet(0xFF21, 0xFF41),
    "math_bold": _build_alphabet(0x1D400, 0x1D41A),
    "math_sans": _build_alphabet(0x1D5A0, 0x1D5BA),
    "math_monospace": _build_alphabet(0x1D670, 0x1D68A),
    "circled": _build_alphabet(0x24B6, 0x24D0),
    "math_fraktur": _build_alphabet(0x1D504, 0x1D51E, _FRAKTUR_CAPITAL_OVERRIDES),
}


class AlphabetMapEncoder(Encoder):
    """One letter-for-letter substitution over a complete A-Z/a-z alphabet.

    Letters only, by deliberate design: digits, punctuation and whitespace pass
    through untouched so that the entire band differs in the *alphabet* and in
    nothing else. `fullwidth` (which also maps punctuation and the space) is the
    off-axis point that makes that choice measurable rather than assumed.
    """

    def __init__(
        self, family: str, attack_template: str, restate_template: str, alphabet: str
    ) -> None:
        if alphabet not in LETTER_ALPHABETS:
            raise ValueError(
                f"{family}: unknown alphabet {alphabet!r}; have {sorted(LETTER_ALPHABETS)}"
            )
        super().__init__(family, attack_template, restate_template, alphabet=alphabet)
        self.alphabet = alphabet
        self._forward = LETTER_ALPHABETS[alphabet]
        self._inverse = {glyph: letter for letter, glyph in self._forward.items()}

    def canonicalize(self, plaintext: str) -> str:
        # A plaintext already containing this alphabet's glyphs could not be
        # told from encoded content, so it is not recoverable — same reasoning
        # as HomoglyphEncoder.
        return "".join(char for char in plaintext if char not in self._inverse)

    def encode_text(self, plaintext: str) -> str:
        return "".join(self._forward.get(char, char) for char in plaintext)

    def decode_text(self, ciphertext: str) -> str:
        return "".join(self._inverse.get(char, char) for char in ciphertext)
