"""Encoder contract — the paper's independent variable.

Re-authored rather than copied from `llm_guardrail_security`. That repo's
encoder contract is "given a harmful prompt, produce an attack string"; the four
measurements need three things it does not carry (`text_docs/project_structure.md` §2):

1. **the ground-truth plaintext** — measurements #1 (ability) and #2
   (deployment) score *recovery*, so they need a reference;
2. **an invertibility class** — which claim band a rung belongs to;
3. **two prompt forms** — see below.

## Why two prompt forms

The attack condition must not ask the model to decode. That is the entire point
of regime (D): a decoding ability that is demonstrably present goes *unspent* in
the forward pass where it would have mattered. So every family renders:

* `attack_prompt` — the encoded request as an attacker would send it. This is
  the condition every regime is defined over.
* `restate_prompt` — "decode this and write out the plaintext". This is
  measurement #1 and it is a *different* forward pass. Conflating the two is the
  design hole that the v2 four-regime reframe exists to close: decode-and-restate
  accuracy measures ability when decoding *is* the task, which is not the same
  question as whether the ability is used when it is not requested.

## Why the invertibility class is verified, not declared

The owner settled (2026-08-02) that the primary (C)/(D)/(B) claims rest on the
exactly-invertible band, because a reviewer attacking "you cannot tell *didn't
decode* from *decoded loosely*" must be met with string-match evidence rather
than a judge score. A class that an encoder author merely *asserts* would put
that guarantee on the honour system. So `Invertibility.EXACT` is a claim the
test suite checks: `tests/test_encodings.py` round-trips every EXACT encoder
over a corpus and fails if `decode(encode(x)) != x`. An encoder that cannot pass
is SEMANTIC by definition, whatever its author intended.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Invertibility(str, Enum):
    """Which claim band a rung can carry."""

    # A unique plaintext exists and we can compute it. Recovery is scored by
    # string match. Primary claim band.
    EXACT = "exact"
    # No unique inverse (paraphrase, translation, formalisation). Recovery is
    # scored by judge or embedding similarity, which puts judge noise inside the
    # measurement — carried as the labelled extended-ladder check only.
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class EncodedPrompt:
    """One rung applied to one prompt, with everything the measurements need."""

    plaintext: str
    ciphertext: str
    family: str
    invertibility: Invertibility
    attack_prompt: str
    restate_prompt: str
    params: dict = field(default_factory=dict)


class Encoder(ABC):
    """Base class for every rung of the ladder.

    Subclasses implement `encode_text` (and `decode_text` when EXACT); prompt
    rendering is shared so that the attack/restate distinction cannot drift
    between families.
    """

    invertibility: Invertibility = Invertibility.EXACT

    def __init__(
        self,
        family: str,
        attack_template: str,
        restate_template: str,
        **params: object,
    ) -> None:
        for template, label in ((attack_template, "attack"), (restate_template, "restate")):
            if "{ciphertext}" not in template:
                raise ValueError(f"{family}: {label}_template must contain {{ciphertext}}")
        self.family = family
        self.attack_template = attack_template
        self.restate_template = restate_template
        self.params = dict(params)

    @abstractmethod
    def encode_text(self, plaintext: str) -> str:
        """Plaintext -> ciphertext. Must be deterministic."""

    def decode_text(self, ciphertext: str) -> str:
        """Ciphertext -> plaintext. Required for EXACT encoders."""
        raise NotImplementedError(f"{self.family} is {self.invertibility.value}; no exact inverse")

    def canonicalize(self, plaintext: str) -> str:
        """Project a plaintext into the form this rung can round-trip exactly.

        Some rungs are exactly invertible only over a restricted alphabet —
        Morse carries no case, so it round-trips uppercase text and nothing
        else. Rather than quietly demote such a rung to SEMANTIC (it is fully
        deterministic; the loss is in the *alphabet*, not the mapping), each
        rung declares the projection it needs and the EncodedPrompt records the
        canonical plaintext as the reference the scorers compare against.

        Identity for most rungs. Projections are deliberately **per rung, not
        composed across the ladder**: composing them would impose Morse's
        uppercase-and-filter projection on every other rung to satisfy one,
        and case is exactly what several cipher rungs act on. Each cell's
        recovery is scored against its own reference, so per-rung references are
        sound; what would not be sound is a reference that disagrees with the
        ciphertext it was built from.

        The cost is that rungs no longer see byte-identical prompts.
        `registry.canonicalization_report` surfaces which rungs alter the corpus
        and by how much, so the S3 design can decide whether to restrict the
        corpus instead — a corpus decision, which is where it belongs.
        """
        return plaintext

    def encode(self, plaintext: str) -> EncodedPrompt:
        canonical = self.canonicalize(plaintext)
        ciphertext = self.encode_text(canonical)
        return EncodedPrompt(
            plaintext=canonical,
            ciphertext=ciphertext,
            family=self.family,
            invertibility=self.invertibility,
            attack_prompt=self.attack_template.format(ciphertext=ciphertext),
            restate_prompt=self.restate_template.format(ciphertext=ciphertext),
            params=dict(self.params),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(family={self.family!r}, params={self.params!r})"
