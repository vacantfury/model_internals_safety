"""Family name -> encoder, built from `conf/encodings.yaml`.

Adding a rung is a config entry plus (if it is a new mechanism) an encoder
class. Nothing in the measurement or analysis code names a family directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from internals_safety.config import load_yaml
from internals_safety.encodings.base import EncodedPrompt, Encoder, Invertibility
from internals_safety.encodings.deterministic.ciphers import (
    AtbashEncoder,
    CaesarEncoder,
    VigenereEncoder,
)
from internals_safety.encodings.deterministic.encodings import (
    AsciiDecimalEncoder,
    Base32Encoder,
    Base64Encoder,
    BinaryEncoder,
    HexEncoder,
    UnicodeEscapeEncoder,
)
from internals_safety.encodings.deterministic.transforms import (
    MorseEncoder,
    ReverseCharactersEncoder,
    ReverseWordsEncoder,
    ZeroWidthEncoder,
)
from internals_safety.encodings.deterministic.surface import (
    CombiningMarkEncoder,
    FullwidthEncoder,
    HomoglyphEncoder,
    TagBlockEncoder,
)
from internals_safety.paths import CONF_DIR

ENCODER_CLASSES: dict[str, type[Encoder]] = {
    cls.__name__: cls
    for cls in (
        Base64Encoder,
        Base32Encoder,
        HexEncoder,
        BinaryEncoder,
        AsciiDecimalEncoder,
        UnicodeEscapeEncoder,
        CaesarEncoder,
        AtbashEncoder,
        VigenereEncoder,
        MorseEncoder,
        ReverseCharactersEncoder,
        ReverseWordsEncoder,
        ZeroWidthEncoder,
        TagBlockEncoder,
        HomoglyphEncoder,
        FullwidthEncoder,
        CombiningMarkEncoder,
    )
}


def _config_path(conf_dir: Path) -> Path:
    return conf_dir / "encodings.yaml"


def load_ladder(conf_dir: Path = CONF_DIR) -> dict[str, Encoder]:
    """Build every configured rung. Insertion order follows the config file.

    The config's order is *not* the difficulty order. Difficulty is measured
    (decode-and-restate accuracy, measurement #1) precisely so the ladder is not
    ordered by intuition — see `text_docs/project_structure.md` §3.
    """
    config = load_yaml(_config_path(conf_dir))
    defaults = config.get("defaults", {})
    ladder: dict[str, Encoder] = {}

    for family, spec in config["families"].items():
        class_name = spec["encoder"]
        if class_name not in ENCODER_CLASSES:
            raise ValueError(f"{family}: unknown encoder {class_name!r}; have {sorted(ENCODER_CLASSES)}")
        encoding_name = spec["encoding_name"]
        templates = {
            key: spec.get(key, defaults[key]).replace("{encoding_name}", encoding_name)
            for key in ("attack_template", "restate_template")
        }
        ladder[family] = ENCODER_CLASSES[class_name](
            family=family, **templates, **spec.get("params", {})
        )
    return ladder


def get_encoder(family: str, conf_dir: Path = CONF_DIR) -> Encoder:
    ladder = load_ladder(conf_dir)
    if family not in ladder:
        raise KeyError(f"no encoding family {family!r}; have {sorted(ladder)}")
    return ladder[family]


def encode_corpus(
    ladder: dict[str, Encoder], prompts: list[str]
) -> list[EncodedPrompt]:
    """Cross every prompt with every rung — the phase-0 sweep's input."""
    return [encoder.encode(prompt) for encoder in ladder.values() for prompt in prompts]


@dataclass(frozen=True)
class CanonicalizationReport:
    """How much each rung had to project its reference away from the raw prompt.

    Read this before treating cross-rung differences as findings: a rung that
    rewrote 12% of its corpus is not measuring the same string as one that
    rewrote none. `Encoder.canonicalize` explains why the projections are
    per-rung rather than composed.
    """

    family: str
    n_prompts: int
    n_altered: int
    examples: list[tuple[str, str]]

    @property
    def fraction_altered(self) -> float:
        return self.n_altered / self.n_prompts if self.n_prompts else 0.0


def canonicalization_report(
    ladder: dict[str, Encoder], prompts: list[str], n_examples: int = 3
) -> list[CanonicalizationReport]:
    reports = []
    for family, encoder in ladder.items():
        altered = [
            (prompt, encoder.canonicalize(prompt))
            for prompt in prompts
            if encoder.canonicalize(prompt) != prompt
        ]
        reports.append(
            CanonicalizationReport(
                family=family,
                n_prompts=len(prompts),
                n_altered=len(altered),
                examples=altered[:n_examples],
            )
        )
    return reports


def exact_families(ladder: dict[str, Encoder]) -> list[str]:
    """The primary claim band."""
    return [
        family
        for family, encoder in ladder.items()
        if encoder.invertibility is Invertibility.EXACT
    ]
