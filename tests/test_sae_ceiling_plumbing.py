"""The ceiling must reach the target arm, and a WRONG ceiling must be impossible.

The target arm is judged as a fraction of the ceiling arm's variance explained,
so the ceiling is an input to a verdict. Every way of getting it wrong here is
silent — a ceiling from the wrong layer shifts the floor by a few percent and
never errors; a target reading used as a ceiling compounds two transfer ratios
into a confident number describing nothing; an absent ceiling defaulting to 0.0
would license anything that reconstructs at all.

So `ceiling_from` raises on each, and these tests pin that it does.
"""

from __future__ import annotations

import json

import pytest

from internals_safety.config import PresetConfig, load_preset

from internals_safety.measurements.sae_reconstruction import ceiling_from

RESOURCES = {"partition": "gpu", "cpus": 8, "mem": "64G", "time": "01:00:00"}


def record(tmp_path, *, arm="ceiling", layer=18, variance=0.6979, n=1):
    reading = {
        "instrument": "sae_reconstruction",
        "detail": {"arm": arm, "layer": layer, "variance_explained": variance},
    }
    path = tmp_path / f"{arm}-{layer}-{n}.json"
    path.write_text(json.dumps({"readings": [reading] * n}))
    return str(path)


class TestTheCeilingIsReadCorrectly:
    def test_it_returns_the_ceiling_arms_variance(self, tmp_path):
        assert ceiling_from(record(tmp_path), layer=18) == pytest.approx(0.6979)


class TestEveryWrongCeilingRAISES:
    def test_a_target_reading_cannot_serve_as_a_ceiling(self, tmp_path):
        """Compounding two transfer ratios produces a number describing nothing."""
        with pytest.raises(ValueError, match="not a ceiling"):
            ceiling_from(record(tmp_path, arm="target"), layer=18)

    def test_a_layer_mismatch_raises(self, tmp_path):
        """**The silent one.** Base measures 0.698/0.708/0.723 at 18/20/22, so a
        ceiling from the wrong layer shifts the floor a few percent and never
        errors — it just quietly licenses or refuses the wrong thing."""
        with pytest.raises(ValueError, match="per layer"):
            ceiling_from(record(tmp_path, layer=20), layer=18)

    def test_an_ambiguous_record_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ambiguous"):
            ceiling_from(record(tmp_path, n=2), layer=18)

    def test_a_record_with_no_variance_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"readings": [
            {"instrument": "sae_reconstruction",
             "detail": {"arm": "ceiling", "layer": 18, "variance_explained": None}}
        ]}))
        with pytest.raises(ValueError, match="no variance_explained"):
            ceiling_from(str(path), layer=18)

    def test_a_missing_file_raises_rather_than_defaulting_to_zero(self, tmp_path):
        """A ceiling of 0.0 would make the transfer floor 0.0, licensing anything
        that reconstructs at all — the gate would silently become a no-op."""
        with pytest.raises(OSError):
            ceiling_from(str(tmp_path / "absent.json"), layer=18)


class TestThePresetBuildsAPerLAYERCeilingPath:
    def _preset(self, **overrides) -> PresetConfig:
        base = dict(
            entrypoint="sae_pregate", description="d", gates="g",
            target="llama3_1_8b_instruct", sae_layers=[17, 19, 21],
            source_runs={"llama3_1_8b_base": "pregate-base-plain"},
            resources=RESOURCES,
        )
        return PresetConfig(**{**base, **overrides})

    def test_each_task_points_at_its_OWN_layer(self):
        tasks = self._preset().tasks("/out")
        for layer, argv in zip((17, 19, 21), tasks):
            path = argv[argv.index("--ceiling-from") + 1]
            assert path.endswith(f"pregate-base-plain-L{layer}/results.json"), path

    def test_no_source_runs_means_no_flag_and_the_run_IS_the_ceiling(self):
        argv = self._preset(source_runs={}).tasks("/out")[0]
        assert "--ceiling-from" not in argv

    def test_two_source_models_raise_rather_than_picking_one(self):
        preset = self._preset(source_runs={"a": "x", "b": "y"})
        with pytest.raises(ValueError, match="ONE source_runs entry"):
            preset.tasks("/out")


class TestTheShippedPresetsAreConsistent:
    def test_both_instruct_arms_share_one_ceiling_and_one_layer_set(self):
        templated = load_preset("sae_pregate_instruct")
        plain = load_preset("sae_pregate_instruct_plain")
        assert templated.source_runs == plain.source_runs
        assert templated.sae_layers == plain.sae_layers == load_preset(
            "sae_pregate_base_plain"
        ).sae_layers
        assert templated.render_chat is True and plain.render_chat is False

    def test_the_ceiling_preset_names_no_ceiling_of_its_own(self):
        """The Base arm SETS the bar; giving it one would be circular."""
        assert load_preset("sae_pregate_base_plain").source_runs == {}

    def test_the_ceiling_source_names_the_PLAIN_base_run(self):
        """Pointing at `pregate-base` (templated, VE -915) would produce a
        negative floor that licenses anything."""
        assert load_preset("sae_pregate_instruct").source_runs == {
            "llama3_1_8b_base": "pregate-base-plain"
        }
