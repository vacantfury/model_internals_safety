"""The cluster run presets, and the closure property that makes them safe.

A preset declares WHICH RUN. It may never declare HOW an instrument reads —
`conf/measurements.yaml` owns every tunable together with its tuning path, and
`tests/test_config_discipline.py` enforces that pairing. If a preset could carry
a knob, a run would ship a number nobody registered, which is the magic-number
problem re-entering through the launcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from internals_safety.config import (
    PresetConfig,
    ResourceConfig,
    list_presets,
    load_measurements_config,
    load_model_config,
    load_preset,
)
from internals_safety.paths import CONF_DIR

PRESETS = list_presets()


def leaf_keys(node, into: set[str]) -> set[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            into.add(key)
            leaf_keys(value, into)
    elif isinstance(node, list):
        for item in node:
            leaf_keys(item, into)
    return into


class TestTheSchemaIsClosed:
    """The property the whole design rests on."""

    def test_no_preset_field_shares_a_name_with_a_measurement_knob(self):
        """The closure invariant, asserted against the real knob file.

        Name collision is the mechanism by which a preset would start looking
        like a legitimate place to put a knob. Catching it here means the
        temptation is a build failure rather than a code-review question.
        """
        knobs = leaf_keys(yaml.safe_load((CONF_DIR / "measurements.yaml").read_text()), set())
        fields = set(PresetConfig.model_fields) | set(ResourceConfig.model_fields)
        collisions = fields & knobs
        assert not collisions, (
            f"preset field(s) {sorted(collisions)} share a name with a measurements.yaml knob. "
            "A preset declares which run, never how an instrument reads."
        )

    def test_an_unknown_key_is_an_error_not_a_no_op(self):
        with pytest.raises(Exception):
            PresetConfig(
                entrypoint="phase0_regime_map",
                description="d",
                gates="g",
                target="m",
                resources={"partition": "gpu", "time": "01:00:00"},
                reading_percentile=99.0,  # a knob, smuggled in
            )

    def test_a_field_the_entrypoint_ignores_is_refused(self):
        """A silently-dropped field would make the approved artifact a lie."""
        with pytest.raises(ValueError, match="ignores"):
            PresetConfig(
                entrypoint="sae_pregate",
                description="d",
                gates="g",
                target="m",
                instruments=["decode_lens"],  # sae_pregate has no --instruments
                resources={"partition": "gpu", "time": "01:00:00"},
            )


class TestTheGateFieldIsRequired:
    """Owner defect report 2026-08-06, made structural.

    "A run must be a GATE, not a measurement, until the instrument roster is
    complete." A rule in prose is a rule enforced by memory; a required field is
    enforced by the loader.
    """

    def test_a_preset_without_gates_cannot_be_constructed(self):
        with pytest.raises(Exception):
            PresetConfig(
                entrypoint="phase0_regime_map",
                description="d",
                target="m",
                resources={"partition": "gpu", "time": "01:00:00"},
            )

    def test_whitespace_does_not_satisfy_it(self):
        with pytest.raises(ValueError, match="gates"):
            PresetConfig(
                entrypoint="phase0_regime_map",
                description="d",
                gates="   ",
                target="m",
                resources={"partition": "gpu", "time": "01:00:00"},
            )

    @pytest.mark.parametrize("name", PRESETS)
    def test_every_shipped_preset_answers_the_gate_question(self, name):
        """Not just non-empty — long enough to be an actual answer.

        The failure mode this guards is `gates: "yes"`, which satisfies a
        required field while restoring exactly the situation the field exists
        to prevent.
        """
        assert len(load_preset(name).gates.split()) >= 15


class TestTheWallClockCeiling:
    def test_over_eight_hours_is_refused(self):
        """The pilot was KILLED at the 8h wall having written nothing recoverable."""
        with pytest.raises(ValueError, match="8h wall"):
            ResourceConfig(partition="gpu", time="12:00:00")

    def test_a_malformed_time_is_refused(self):
        with pytest.raises(ValueError, match="HH:MM:SS"):
            ResourceConfig(partition="gpu", time="2h")

    @pytest.mark.parametrize("name", PRESETS)
    def test_every_shipped_preset_fits_the_queue(self, name):
        assert load_preset(name).resources.time <= "08:00:00"


class TestEveryShippedPresetIsValid:
    @pytest.mark.parametrize("name", PRESETS)
    def test_it_loads(self, name):
        assert load_preset(name).description.strip()

    @pytest.mark.parametrize("name", PRESETS)
    def test_its_targets_are_real_model_configs(self, name):
        """A typo'd model name must fail here, not after the queue wait."""
        preset = load_preset(name)
        for target in filter(None, [preset.target, *preset.targets]):
            if preset.entrypoint == "as6_guard_probe":
                continue  # guards live in conf/guards/, checked by their own loader
            load_model_config(target)

    @pytest.mark.parametrize("name", PRESETS)
    def test_its_instruments_are_ones_that_exist(self, name):
        """Guards the same class of typo one level over from --families."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from phase0_regime_map import OPTIONAL_INSTRUMENTS

        for instrument in load_preset(name).instruments:
            assert instrument in OPTIONAL_INSTRUMENTS, instrument

    @pytest.mark.parametrize("name", PRESETS)
    def test_it_builds_at_least_one_command(self, name):
        tasks = load_preset(name).tasks("/scratch/x/out")
        assert tasks and all(task[0].startswith("scripts/") for task in tasks)


class TestCommandConstruction:
    """Built in Python, because the thing it replaced was array-index arithmetic
    over two bash arrays in a file that was never in git."""

    def test_relicense_fans_out_over_model_times_family(self):
        tasks = load_preset("relicense_all").tasks("/s/out")
        preset = load_preset("relicense_all")
        assert len(tasks) == len(preset.targets) * len(preset.families)

    def test_relicense_names_a_source_run_for_every_target(self):
        preset = load_preset("relicense_all")
        for target in preset.targets:
            assert target in preset.source_runs

    def test_a_missing_source_run_is_an_error_not_a_guessed_path(self):
        preset = PresetConfig(
            entrypoint="relicense_probes",
            description="d",
            gates="w " * 20,
            targets=["a", "b"],
            families=["hex"],
            source_runs={"a": "run1"},
            resources={"partition": "short", "time": "01:00:00"},
        )
        with pytest.raises(ValueError, match="no source_runs"):
            preset.tasks("/s/out")

    def test_sae_layers_become_separate_tasks(self):
        """One --sae-layer per invocation; argparse would keep only the last.

        A repeated flag would have run one layer while the approved artifact
        said three — the run looks complete and covers a third of what it claimed.
        """
        tasks = load_preset("sae_pregate_base").tasks("/s/out")
        assert len(tasks) == 3
        layers = [task[task.index("--sae-layer") + 1] for task in tasks]
        assert layers == ["17", "19", "21"]
        assert len({tuple(task) for task in tasks}) == 3

    def test_every_task_writes_to_the_outputs_dir_it_was_given(self):
        """Cluster scratch, never the repo tree — the activations are GB-scale."""
        for name in PRESETS:
            for task in load_preset(name).tasks("/scratch/me/out"):
                assert any("/scratch/me/out" in part for part in task)

    def test_the_guard_entrypoint_uses_guard_not_model(self):
        preset = PresetConfig(
            entrypoint="as6_guard_probe",
            description="d",
            gates="w " * 20,
            target="wildguard",
            resources={"partition": "gpu", "time": "02:00:00"},
        )
        assert "--guard" in preset.tasks("/s")[0]
        assert "--model" not in preset.tasks("/s")[0]


class TestTheSAEPreGateOrdering:
    """Base before Instruct is a correctness ordering, not a preference."""

    def test_both_arms_read_the_same_layers(self):
        """Otherwise the transfer gap is confounded by which layer was compared."""
        base = load_preset("sae_pregate_base")
        instruct = load_preset("sae_pregate_instruct")
        assert base.sae_layers == instruct.sae_layers

    def test_both_arms_use_the_same_corpus_size(self):
        base = load_preset("sae_pregate_base")
        instruct = load_preset("sae_pregate_instruct")
        assert base.n_prompts == instruct.n_prompts

    def test_the_base_arm_borrows_the_instruct_template(self):
        """The model is the only variable the pre-gate is allowed to change."""
        base = load_model_config("llama3_1_8b_base")
        assert base.chat_template_from == "llama3_1_8b_instruct"

    def test_the_two_arms_are_the_same_depth(self):
        """A layer-for-layer dictionary comparison needs matching block counts."""
        assert load_model_config("llama3_1_8b_base").n_layers == (
            load_model_config("llama3_1_8b_instruct").n_layers
        )


class TestAPresetCannotApproveAnUNREPORTABLERun:
    """The contract's `required_controls`, enforced where the run is declared.

    Found by running `--dry-run` on the generated argv, 2026-08-07: two presets
    read the deployment probe and neither declared `lexical`. Deployment's
    `REQUIRED_CONTROLS` names the XSTest vocabulary screen, so `Reading.reportable`
    would have been False for every deployment number in both runs — an approved,
    costed, hours-long GPU job whose central quantity could not appear in a paper.

    The contract already fails closed, so nothing false would have been
    published. But failing closed AFTER the GPU time is spent is the expensive
    place to fail, and the preset is the cheap one.
    """

    # Entrypoints that run the four measurements, deployment among them.
    MEASUREMENT_ENTRYPOINTS = {"phase0_regime_map", "as6_guard_probe"}

    @pytest.mark.parametrize("name", PRESETS)
    def test_a_measurement_run_declares_deployments_required_control(self, name):
        preset = load_preset(name)
        if preset.entrypoint not in self.MEASUREMENT_ENTRYPOINTS:
            return
        if preset.n_prompts is not None and preset.n_prompts < 32:
            return  # a smoke run reports nothing and is not held to it

        from internals_safety.measurements.deployment import REQUIRED_CONTROLS

        assert "lexical_vocabulary" in REQUIRED_CONTROLS  # the rule this mirrors
        assert "lexical" in preset.instruments, (
            f"{name} runs the deployment probe without the XSTest lexical control, so "
            "every deployment reading it produces is non-reportable by the contract. "
            "Declare `lexical` in instruments, or the run buys a number it cannot use."
        )


class TestNPromptsFitsTheCorpus:
    """`--n-prompts` is PER CLASS, and JBB ships exactly 100 of each.

    Caught twice in one day, 2026-08-07: once as a script default of 200, and
    then again in two presets that copied "200 prompts" out of the band-run
    write-up — where 200 is the TOTAL across both classes. A number that reads
    as right and is off by a factor of two against a fixed corpus is exactly
    what a test is for; the failure otherwise surfaces after the queue wait.
    """

    @pytest.mark.parametrize("name", PRESETS)
    def test_it_does_not_ask_for_more_prompts_than_exist(self, name):
        from internals_safety.config import load_corpus_config
        from internals_safety.paths import DATA_DIR

        preset = load_preset(name)
        if preset.n_prompts is None:
            return
        pilot = load_corpus_config()
        for prompt_set_name in (pilot.harmful_set, pilot.harmless_set):
            path = DATA_DIR / prompt_set_name
            if not path.exists():
                pytest.skip(f"{path} absent; data/ is gitignored and re-copied per clone")
            available = sum(1 for line in path.read_text().splitlines() if line.strip())
            assert preset.n_prompts <= available, (
                f"{name} asks for {preset.n_prompts} prompts per class; {prompt_set_name} "
                f"holds {available}. --n-prompts is PER CLASS, not the total across both."
            )


class TestNoKnobDriftIntoPresets:
    def test_presets_do_not_mention_any_measurement_knob_by_name(self):
        """Belt and braces over the field-name check: catches a knob smuggled in
        as a nested mapping under an allowed field."""
        knobs = leaf_keys(yaml.safe_load((CONF_DIR / "measurements.yaml").read_text()), set())
        # Names too generic to be evidence of anything.
        generic = {"name", "layers", "positions", "site", "families", "n_prompts", "model", "models"}
        for name in PRESETS:
            raw = yaml.safe_load((CONF_DIR / "experiment" / f"{name}.yaml").read_text())
            present = leaf_keys(raw, set()) & (knobs - generic)
            assert not present, f"{name}.yaml carries knob-shaped key(s) {sorted(present)}"
