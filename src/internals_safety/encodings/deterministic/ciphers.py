"""Letter-substitution ciphers — exactly invertible rungs.

Caesar's body is a port of `llm_guardrail_security`'s
`prompt_transformations/text/encoders/non_llm_cipher_encoder.py` (commit
ba3036e), whose own provenance is CipherChat — "GPT-4 Is Too Smart To Be Safe:
Stealthy Chat with LLMs via Cipher" (Yuan et al., ICLR 2024;
github.com/RobustNLP/CipherChat), `encode_experts.py::CaesarExpert.encode`:
only `[a-zA-Z]` are shifted, case-preserving, everything else passes through.

Atbash is the rung the sibling deliberately did NOT port. CipherChat's
`AtbashExpert.encode` reflects with a single constant `N - ord(s)` where
`N = ord('z') + ord('a')` (219), which is correct for lowercase only; uppercase
input maps into the non-ASCII control range. Since English sentences start
capitalized, a faithful port would corrupt essentially every prompt. It is
ported here **with the bug fixed** — upper and lower case reflected within their
own ranges — and the divergence from the published implementation is stated
here so a reader can tell which of our rungs are faithful ports and which are
corrected variants (method-provenance rule).
"""

from __future__ import annotations

import string

from internals_safety.encodings.base import Encoder

_ALPHABET_SIZE = 26


def _shift_letters(text: str, shift: int) -> str:
    out = []
    for char in text:
        if "a" <= char <= "z":
            out.append(chr(ord("a") + (ord(char) - ord("a") + shift) % _ALPHABET_SIZE))
        elif "A" <= char <= "Z":
            out.append(chr(ord("A") + (ord(char) - ord("A") + shift) % _ALPHABET_SIZE))
        else:
            out.append(char)
    return "".join(out)


class CaesarEncoder(Encoder):
    """Mod-26 shift. `shift=13` is ROT13; `shift=3` is CipherChat's default.

    ROT-n for n != 13 is the cheap way to get several rungs that are
    *identical in kind* but differ in how familiar the mapping is — useful for
    RQ2, where the ladder wants to be ordered by measured decode difficulty
    rather than by intuition.
    """

    def __init__(self, family: str, attack_template: str, restate_template: str, shift: int = 3):
        super().__init__(family, attack_template, restate_template, shift=int(shift))
        self.shift = int(shift) % _ALPHABET_SIZE

    def encode_text(self, plaintext: str) -> str:
        return _shift_letters(plaintext, self.shift)

    def decode_text(self, ciphertext: str) -> str:
        return _shift_letters(ciphertext, -self.shift)


class AtbashEncoder(Encoder):
    """a<->z reflection, case-preserving. Self-inverse. See module docstring for
    the divergence from CipherChat's published implementation."""

    def encode_text(self, plaintext: str) -> str:
        out = []
        for char in plaintext:
            if "a" <= char <= "z":
                out.append(chr(ord("z") - (ord(char) - ord("a"))))
            elif "A" <= char <= "Z":
                out.append(chr(ord("Z") - (ord(char) - ord("A"))))
            else:
                out.append(char)
        return "".join(out)

    def decode_text(self, ciphertext: str) -> str:
        return self.encode_text(ciphertext)


class VigenereEncoder(Encoder):
    """Keyed poly-alphabetic shift.

    The key advances only on letters, so non-letter characters pass through
    without consuming key material — that is what keeps the inverse exact
    regardless of punctuation. Not a CipherChat cipher; included because it is
    the natural "same mechanism, higher decode difficulty" companion to Caesar,
    which is exactly the kind of pair RQ2's difficulty axis needs.
    """

    def __init__(
        self, family: str, attack_template: str, restate_template: str, key: str = "safety"
    ):
        key_letters = [char for char in key.lower() if char in string.ascii_lowercase]
        if not key_letters:
            raise ValueError(f"{family}: vigenere key must contain at least one letter")
        super().__init__(family, attack_template, restate_template, key=key)
        self.key_shifts = [ord(char) - ord("a") for char in key_letters]

    def _apply(self, text: str, sign: int) -> str:
        out = []
        position = 0
        for char in text:
            if char.isalpha() and char.isascii():
                shift = sign * self.key_shifts[position % len(self.key_shifts)]
                out.append(_shift_letters(char, shift))
                position += 1
            else:
                out.append(char)
        return "".join(out)

    def encode_text(self, plaintext: str) -> str:
        return self._apply(plaintext, 1)

    def decode_text(self, ciphertext: str) -> str:
        return self._apply(ciphertext, -1)
