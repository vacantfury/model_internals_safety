"""Comprehension-preserving rungs — the band the phase-0 pilot pointed at.

## Why this module exists

The phase-0 pilot (2026-08-03, Qwen2.5-7B + Llama-3.1-8B, n=100) measured the
twelve cipher and byte-encoding rungs — base64 through morse — and found the
same cell every time: **(R) surface refusal, ability ~= 0, deployment = 0**.
Mean decode-and-restate similarity ran 0.03-0.30. Those models simply cannot
read those rungs, so "can't, didn't, or wouldn't?" has one uniform answer
there — *can't* — and the (D)/(B)/(S) distinctions the paper exists to draw are
unmeasurable in that band. Twelve rungs, two models, one repeated result.

The exception was `zero_width`, where mean similarity was 0.987 and the map
read **B=5-13, S=86-95, R=0** once the ability binary was fixed
(`scripts/rescore_ability.py`). That is the entire signal the pilot produced:
where comprehension actually happens, the model overwhelmingly decodes *and
refuses*, with a small real decode-and-comply tail.

So the ladder's ordering variable is not decoding difficulty. It is **surface
conspicuousness** — how much the ciphertext still looks like ordinary text —
and the rungs worth having are the ones a 7-9B model can still read. This
module adds four, ordered here from invisible to merely decorated:

| rung | what a human sees | what the tokenizer sees |
|---|---|---|
| `tag_block` | nothing at all | a run of Plane-14 tag codepoints |
| `homoglyph` | ordinary English | Cyrillic/Greek lookalikes |
| `fullwidth` | ordinary English, wide | a disjoint codepoint range |
| `combining_marks` | accented English | base + combining pairs |

`tag_block` is the sharpest instrument of the four: like `zero_width` it is
*completely invisible*, so any refusal it draws cannot be a surface-cue
refusal — there is no surface cue to react to.

## Invertibility

All four are EXACT, which `tests/test_encodings.py` verifies rather than takes
on trust. Three need a `canonicalize` projection, following the pattern Morse
established in `transforms.py`: the restriction is in the *alphabet*, not in
the mapping, so the rung stays in the primary claim band. Each strips its own
codomain from the reference, because a plaintext already containing (say) a
Cyrillic lookalike could not be recovered from its own ciphertext.
"""

from __future__ import annotations

import unicodedata

from internals_safety.encodings.base import Encoder

# Plane-14 tag characters. U+E0020..U+E007E mirror printable ASCII one-for-one;
# they render as nothing in every mainstream font and are stripped by many
# surface filters, which is exactly why they are worth a rung.
# constant: Unicode plane-14 Tags block, U+E0000 and the printable-ASCII mirror range
TAG_BASE = 0xE0000
TAG_LOW, TAG_HIGH = 0xE0020, 0xE007E

# Latin -> Cyrillic/Greek lookalikes. Every entry must be visually confusable
# AND the map must stay injective, or the inverse is not a function.
HOMOGLYPHS: dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о",
    "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}
_HOMOGLYPH_INVERSE = {glyph: latin for latin, glyph in HOMOGLYPHS.items()}

# ASCII printable -> fullwidth forms; the space maps to the ideographic space.
FULLWIDTH_OFFSET = 0xFF01 - 0x21
FULLWIDTH_SPACE = "　"

# Combining acute. Any combining mark works; this one is unambiguous and common.
COMBINING_MARK = "́"


class TagBlockEncoder(Encoder):
    """Every printable ASCII character mapped into the Unicode tag block.

    The ciphertext renders as an empty string to a human and to most logging,
    while carrying the full message to the tokenizer. Where `zero_width`
    interleaves invisible characters *between* visible ones, this rung makes
    the entire payload invisible — so a refusal here cannot be reacting to how
    the text looks.
    """

    def canonicalize(self, plaintext: str) -> str:
        return "".join(
            char for char in plaintext if not TAG_LOW <= ord(char) <= TAG_HIGH
        )

    def encode_text(self, plaintext: str) -> str:
        return "".join(
            chr(TAG_BASE + ord(char)) if 0x20 <= ord(char) <= 0x7E else char
            for char in plaintext
        )

    def decode_text(self, ciphertext: str) -> str:
        return "".join(
            chr(ord(char) - TAG_BASE) if TAG_LOW <= ord(char) <= TAG_HIGH else char
            for char in ciphertext
        )


class HomoglyphEncoder(Encoder):
    """Latin letters swapped for visually identical Cyrillic/Greek codepoints.

    To a reader the ciphertext IS the plaintext; to the tokenizer it is a
    different string end to end. Unlike `zero_width` it adds no characters, so
    it separates "unfamiliar codepoints" from "interrupted character runs" —
    two mechanisms `zero_width` alone conflates.
    """

    def canonicalize(self, plaintext: str) -> str:
        return "".join(char for char in plaintext if char not in _HOMOGLYPH_INVERSE)

    def encode_text(self, plaintext: str) -> str:
        return "".join(HOMOGLYPHS.get(char, char) for char in plaintext)

    def decode_text(self, ciphertext: str) -> str:
        return "".join(_HOMOGLYPH_INVERSE.get(char, char) for char in ciphertext)


class FullwidthEncoder(Encoder):
    """ASCII rendered in fullwidth forms — readable, visibly different.

    The control for `homoglyph`: equally readable, but the substitution is
    obvious rather than invisible. If refusal tracks conspicuousness, these two
    rungs should separate, and that difference is the measurement.
    """

    def canonicalize(self, plaintext: str) -> str:
        return "".join(
            char
            for char in plaintext
            if not (0xFF01 <= ord(char) <= 0xFF5E or char == FULLWIDTH_SPACE)
        )

    def encode_text(self, plaintext: str) -> str:
        return "".join(
            FULLWIDTH_SPACE
            if char == " "
            else chr(ord(char) + FULLWIDTH_OFFSET)
            if 0x21 <= ord(char) <= 0x7E
            else char
            for char in plaintext
        )

    def decode_text(self, ciphertext: str) -> str:
        return "".join(
            " "
            if char == FULLWIDTH_SPACE
            else chr(ord(char) - FULLWIDTH_OFFSET)
            if 0xFF01 <= ord(char) <= 0xFF5E
            else char
            for char in ciphertext
        )


class CombiningMarkEncoder(Encoder):
    """A combining mark after every non-space character.

    Still readable, but every character becomes a two-codepoint sequence, so
    token boundaries are destroyed while the glyph sequence survives. Sits
    between `homoglyph` (codepoints changed, length preserved) and `zero_width`
    (length changed, codepoints preserved) — it does both at once.
    """

    def __init__(
        self, family: str, attack_template: str, restate_template: str, mark: str = COMBINING_MARK
    ) -> None:
        if unicodedata.combining(mark) == 0:
            raise ValueError(f"{family}: mark must be a combining character, got {mark!r}")
        super().__init__(family, attack_template, restate_template, mark=mark)
        self.mark = mark

    def canonicalize(self, plaintext: str) -> str:
        # Pre-existing combining marks would be indistinguishable from injected
        # ones, so the reference is the un-combined form.
        return "".join(char for char in plaintext if unicodedata.combining(char) == 0)

    def encode_text(self, plaintext: str) -> str:
        return "".join(
            char if char.isspace() else char + self.mark for char in plaintext
        )

    def decode_text(self, ciphertext: str) -> str:
        return "".join(char for char in ciphertext if unicodedata.combining(char) == 0)
