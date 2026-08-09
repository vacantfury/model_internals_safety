"""Config tests — the YAML files under conf/ are part of the contract."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from internals_safety.config import ModelConfig, load_measurements_config, load_model_config
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


def test_the_operating_point_is_the_tuned_one():
    """THE one place `reading_percentile`'s value is pinned. Tuned, not chosen.

    Every other test asserts the invariant `harmless_rate == 1 - pct/100` and
    derives the percentile, so this is the single line that must be edited
    deliberately when the knob moves — and the single line that records why it
    sits where it does.

    Basis (`instrument_layer.md` §2.8, artifacts
    `outputs/analysis/operating_point_*_20260808.json`): swept over both guards'
    re-emitted phase-1 runs, 75 halves the benign false-positive rate 0.50 ->
    0.25 while costing 0-1 cells on every floor-surviving sound rung of BOTH
    guards. 90 was rejected on the stabilise-don't-optimise criterion — free on
    Llama Guard but cutting WildGuard `homoglyph` 23 -> 13.

    The YAML and the dataclass default must AGREE, because a run that loads the
    shipped config and a unit test that constructs `ProbeConfig()` would
    otherwise sit at two different operating points and neither would say so.
    """
    from internals_safety.config import ProbeConfig

    assert load_measurements_config().probes.reading_percentile == 75.0
    assert ProbeConfig().reading_percentile == 75.0
    # The retired value, rejected by mutation: 50 is the median, so the benign
    # control reads positive half the time by construction (§2.1).
    assert load_measurements_config().probes.reading_percentile != 50.0


def test_the_swept_grid_contains_the_operating_point():
    """A knob outside its own sweep grid is a value no evidence ever covered."""
    probes = load_measurements_config().probes
    assert probes.reading_percentile in probes.reading_percentile_sweep
