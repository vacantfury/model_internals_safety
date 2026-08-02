"""Config tests — the YAML files under conf/ are part of the contract."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from internals_safety.config import ModelConfig, load_model_config
from internals_safety.paths import CONF_DIR


def _model_config_names() -> list[str]:
    return sorted(path.stem for path in (CONF_DIR / "models").glob("*.yaml"))


@pytest.mark.parametrize("name", _model_config_names())
def test_shipped_model_configs_load(name):
    config = load_model_config(name)
    assert config.name == name
    assert config.hf_id
    assert config.capture.positions


def test_missing_config_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="available"):
        load_model_config("no_such_model")


def test_filename_and_name_field_must_agree(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "alpha.yaml").write_text(
        yaml.safe_dump({"name": "beta", "hf_id": "org/beta"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="filename"):
        load_model_config("alpha", conf_dir=tmp_path)


def test_unknown_keys_are_rejected():
    """A typo'd knob must fail loudly rather than silently keep the default."""
    with pytest.raises(ValidationError):
        ModelConfig(name="x", hf_id="org/x", capture_bath_size=8)
