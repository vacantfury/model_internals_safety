"""Byte- and codepoint-level re-encodings — exactly invertible rungs.

Base64's body comes from the sibling's `non_llm_cipher_encoder.py` (commit
ba3036e), where it is noted as *not* a CipherChat cipher — CipherChat has no
base64 variant; it is the well-known encoding baseline. The rest are the
standard encodings the sibling never implemented, added here because the primary
claim band is the exactly-invertible one, so every additional deterministic rung
is worth more than any semantic one (`text_docs/project_structure.md` §3).

`ascii_decimal` and `unicode_escape` are the CipherChat `ascii` / `unicode`
families, ported. `gbk` is deliberately not ported: it is a CJK byte encoding
and reduces to ASCII passthrough for English prompts, so it would be a rung that
does nothing to an English corpus.
"""

from __future__ import annotations

import base64
import binascii
import re

from internals_safety.encodings.base import Encoder


class Base64Encoder(Encoder):
    def encode_text(self, plaintext: str) -> str:
        return base64.b64encode(plaintext.encode("utf-8")).decode("ascii")

    def decode_text(self, ciphertext: str) -> str:
        return base64.b64decode(ciphertext.strip().encode("ascii")).decode("utf-8")


class Base32Encoder(Encoder):
    def encode_text(self, plaintext: str) -> str:
        return base64.b32encode(plaintext.encode("utf-8")).decode("ascii")

    def decode_text(self, ciphertext: str) -> str:
        return base64.b32decode(ciphertext.strip().encode("ascii")).decode("utf-8")


class HexEncoder(Encoder):
    """UTF-8 bytes as hex. `separator` is a real knob, not cosmetics: spacing
    changes tokenisation, and tokenisation is plausibly part of what makes a
    rung hard — so it stays in config where it can be varied."""

    def __init__(
        self, family: str, attack_template: str, restate_template: str, separator: str = " "
    ):
        super().__init__(family, attack_template, restate_template, separator=separator)
        self.separator = separator

    def encode_text(self, plaintext: str) -> str:
        return self.separator.join(f"{byte:02x}" for byte in plaintext.encode("utf-8"))

    def decode_text(self, ciphertext: str) -> str:
        compact = re.sub(r"\s+", "", ciphertext) if self.separator.strip() == "" else ciphertext.replace(self.separator, "")
        return bytes.fromhex(compact).decode("utf-8")


class BinaryEncoder(Encoder):
    """UTF-8 bytes as 8-bit binary, space separated."""

    def encode_text(self, plaintext: str) -> str:
        return " ".join(f"{byte:08b}" for byte in plaintext.encode("utf-8"))

    def decode_text(self, ciphertext: str) -> str:
        chunks = ciphertext.split()
        return bytes(int(chunk, 2) for chunk in chunks).decode("utf-8")


class AsciiDecimalEncoder(Encoder):
    """Each character as its decimal codepoint, space separated (CipherChat's
    `ascii`). Codepoints rather than bytes, so non-ASCII survives."""

    def encode_text(self, plaintext: str) -> str:
        return " ".join(str(ord(char)) for char in plaintext)

    def decode_text(self, ciphertext: str) -> str:
        return "".join(chr(int(chunk)) for chunk in ciphertext.split())


class UnicodeEscapeEncoder(Encoder):
    r"""Every character as a `\uXXXX` (or `\UXXXXXXXX`) escape — CipherChat's
    `unicode`/`utf` family. Escapes *everything*, not just non-ASCII, so the
    ciphertext contains no readable plaintext for a surface filter to catch."""

    _PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")

    def encode_text(self, plaintext: str) -> str:
        parts = []
        for char in plaintext:
            code = ord(char)
            parts.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        return "".join(parts)

    def decode_text(self, ciphertext: str) -> str:
        def replace(match: re.Match) -> str:
            return chr(int(match.group(1) or match.group(2), 16))

        return self._PATTERN.sub(replace, ciphertext)


def is_decodable(encoder: Encoder, ciphertext: str) -> bool:
    """Whether `ciphertext` is well-formed for this encoder.

    Used by the recovery scorers: a model that emits a *malformed* ciphertext
    echo is a different failure from one that emits a wrong plaintext, and the
    regime map should not confuse them.
    """
    try:
        encoder.decode_text(ciphertext)
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return False
    return True
