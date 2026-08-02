"""Config loading.

Every tunable lives in `conf/*.yaml`, never as a literal in code (global law).
This module is the only place that reads YAML; everything else takes a typed
config object.

Deliberately plain: `yaml.safe_load` + pydantic validation. No configuration
framework is assumed — whether this project ever needs one is an open decision
deferred until the phase-0 pilot shows the real run shapes
(`text_docs/project_structure.md` §7.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from internals_safety.paths import CONF_DIR

DType = Literal["auto", "bfloat16", "float16", "float32"]
Device = Literal["auto", "cuda", "mps", "cpu"]
Site = Literal["resid_pre", "resid_post"]

# Position names understood by the capture layer. Both are resolved per prompt
# against its own tokenization (see models.loader.resolve_position).
#   last              — final token of the rendered prompt (start of generation)
#   instruction_final — last token of the user message content, before any
#                       end-of-turn / assistant-header template tokens. This is
#                       the readout site for measurement #3 (recognition).
PositionName = Literal["last", "instruction_final"]


class StrictModel(BaseModel):
    """Base: unknown YAML keys are an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CaptureConfig(StrictModel):
    site: Site = "resid_pre"
    # "all" = every layer; otherwise an explicit list of layer indices.
    layers: Literal["all"] | list[int] = "all"
    positions: list[PositionName] = Field(default_factory=lambda: ["instruction_final", "last"])


class ModelConfig(StrictModel):
    """One entry of `conf/models/*.yaml`."""

    name: str
    hf_id: str
    dtype: DType = "auto"
    device: Device = "auto"
    trust_remote_code: bool = False
    # Applied as the chat system message when present; null = no system message.
    system_prompt: str | None = None
    # Forward-pass batch size for activation capture.
    capture_batch_size: int = 8
    capture: CaptureConfig = CaptureConfig()


class AbilityConfig(StrictModel):
    """Measurement #1 — decode-and-restate."""

    # Long enough for the plaintext plus the "the decoded text is:" preamble
    # models tend to add. Too short would score a truncated correct answer as a
    # decode failure, which is exactly the confusion the (C)/(D) split exists to
    # avoid — so this is a knob, and its tuning path is the phase-0 pilot's
    # response-length distribution.
    max_new_tokens: int = 256
    batch_size: int = 8


class ProbeConfig(StrictModel):
    """Measurements #2 and #3 — the linear-probe layer."""

    # The first stochastic knob in the project (train/test splits). The
    # run-record schema requires it to be recorded per run — see
    # `.claude/skills/reproducible-run-logging`.
    seed: int = 0
    test_fraction: float = 0.3
    # Inverse L2 regularisation for the logistic probe. Tuning path: the phase-0
    # pilot sweeps it against held-out AUROC on the plain-text condition, where
    # the answer is known to be near-ceiling — a value that cannot separate
    # plain harmful from plain harmless is misconfigured, not informative.
    regularization_c: float = 1.0
    max_iter: int = 2000
    # A probe is only counted as reading a signal above this AUROC. 0.5 is
    # chance; the margin is set from the control-task distribution measured in
    # the pilot, not chosen by taste.
    auroc_threshold: float = 0.70


class MeasurementsConfig(StrictModel):
    ability: AbilityConfig = AbilityConfig()
    probes: ProbeConfig = ProbeConfig()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_model_config(name: str, conf_dir: Path = CONF_DIR) -> ModelConfig:
    """Load `conf/models/<name>.yaml`.

    The file's `name` field must match its filename — the filename is how runs
    refer to a model, so a mismatch would make results ambiguous.
    """
    path = conf_dir / "models" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (conf_dir / "models").glob("*.yaml"))
        raise FileNotFoundError(f"no model config {path}; available: {available}")
    config = ModelConfig(**load_yaml(path))
    if config.name != name:
        raise ValueError(f"{path}: name field is {config.name!r} but filename says {name!r}")
    return config


def load_measurements_config(conf_dir: Path = CONF_DIR) -> MeasurementsConfig:
    return MeasurementsConfig(**load_yaml(conf_dir / "measurements.yaml"))
