"""Orthographic and symbolic transforms — exactly invertible rungs.

None of these exist in the sibling repo. They are here because the primary
claim band is the exactly-invertible one, and these are the cheapest way to add
rungs that differ in *kind* from the cipher and byte-encoding families: they
leave the alphabet intact and rearrange or interleave instead. A ladder whose
rungs all share one mechanism cannot separate "decode difficulty" from "this
particular mechanism".

Morse is the CipherChat `morse` family, which the sibling left unported.
"""

from __future__ import annotations

from internals_safety.encodings.base import Encoder

ZERO_WIDTH_CHARS = "​‌‍﻿"

# Standard ITU table plus the punctuation CipherChat's expert carries.
MORSE_TABLE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.", "!": "-.-.--",
    "/": "-..-.", "(": "-.--.", ")": "-.--.-", "&": ".-...", ":": "---...",
    ";": "-.-.-.", "=": "-...-", "+": ".-.-.", "-": "-....-", "_": "..--.-",
    '"': ".-..-.", "$": "...-..-", "@": ".--.-.",
}
_MORSE_INVERSE = {code: char for char, code in MORSE_TABLE.items()}
_WORD_SEPARATOR = " / "


class ReverseCharactersEncoder(Encoder):
    """Whole string reversed. Self-inverse."""

    def encode_text(self, plaintext: str) -> str:
        return plaintext[::-1]

    def decode_text(self, ciphertext: str) -> str:
        return ciphertext[::-1]


class ReverseWordsEncoder(Encoder):
    """Word order reversed, spacing preserved. Self-inverse.

    Splits on the literal space rather than on whitespace runs, so repeated
    spaces survive the round trip — `"a  b"` -> `"b  a"` -> `"a  b"`.
    """

    def encode_text(self, plaintext: str) -> str:
        return " ".join(plaintext.split(" ")[::-1])

    def decode_text(self, ciphertext: str) -> str:
        return self.encode_text(ciphertext)


class ZeroWidthEncoder(Encoder):
    """A zero-width space between every pair of characters.

    The interesting rung of the set: to a human reader and to most surface
    filters the ciphertext *is* the plaintext, but the tokenizer sees something
    with no relationship to the original tokens. It isolates tokenisation from
    semantics about as cleanly as a rung can.
    """

    def __init__(
        self, family: str, attack_template: str, restate_template: str, filler: str = "​"
    ):
        if filler not in ZERO_WIDTH_CHARS:
            raise ValueError(f"{family}: filler must be one of {ZERO_WIDTH_CHARS!r}")
        super().__init__(family, attack_template, restate_template, filler=filler)
        self.filler = filler

    def canonicalize(self, plaintext: str) -> str:
        # A plaintext that already contains zero-width characters could not be
        # recovered from the ciphertext, so they are removed from the reference.
        return "".join(char for char in plaintext if char not in ZERO_WIDTH_CHARS)

    def encode_text(self, plaintext: str) -> str:
        return self.filler.join(plaintext)

    def decode_text(self, ciphertext: str) -> str:
        return "".join(char for char in ciphertext if char not in ZERO_WIDTH_CHARS)


class MorseEncoder(Encoder):
    """ITU Morse; letters separated by a space, words by ` / `.

    Morse carries no case and no character outside its table, so this rung
    canonicalizes its reference to upper case over the supported alphabet (see
    `Encoder.canonicalize`). The mapping itself stays exact — the restriction is
    in the alphabet, not the inverse — so this stays in the primary claim band.
    """

    def canonicalize(self, plaintext: str) -> str:
        upper = " ".join(plaintext.upper().split())
        kept = "".join(char for char in upper if char == " " or char in MORSE_TABLE)
        return " ".join(kept.split())

    def encode_text(self, plaintext: str) -> str:
        words = [
            " ".join(MORSE_TABLE[char] for char in word if char in MORSE_TABLE)
            for word in plaintext.split(" ")
        ]
        return _WORD_SEPARATOR.join(words)

    def decode_text(self, ciphertext: str) -> str:
        words = []
        # Split on the bare `/` rather than on ` / ` so that a model's
        # restatement with sloppy spacing still decodes. An encoded literal
        # slash is `-..-.`, never `/`, so this stays unambiguous.
        for word in ciphertext.split("/"):
            codes = word.split()
            words.append("".join(_MORSE_INVERSE[code] for code in codes if code in _MORSE_INVERSE))
        return " ".join(word for word in words if word)
