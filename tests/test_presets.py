"""The cluster run presets, and the closure property that makes them safe.

A preset declares WHICH RUN. It may never declare HOW an instrument reads —
`conf/measurements.yaml` owns every tunable together with its tuning path, and
`tests/test_config_discipline.py` enforces that pairing. If a preset could carry
a knob, a run would ship a number nobody registered, which is the magic-number
problem re-entering through the launcher.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import get_args

import pytest
import yaml

from internals_safety.config import (
    GUARD_TARGET_ENTRYPOINTS,
    Entrypoint,
    PresetConfig,
    ResourceConfig,
    list_presets,
    load_measurements_config,
    load_model_config,
    load_preset,
)
from internals_safety.paths import CONF_DIR, PROJECT_ROOT

PRESETS = list_presets()


def _script(name: str):
    """Import a scripts/*.py module by path — they are entrypoints, not package
    members, so there is no import path to them."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                resources={"cluster": "nurc", "partition": "gpu", "time": "01:00:00"},
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
                resources={"cluster": "nurc", "partition": "gpu", "time": "01:00:00"},
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
                resources={"cluster": "nurc", "partition": "gpu", "time": "01:00:00"},
            )

    def test_whitespace_does_not_satisfy_it(self):
        with pytest.raises(ValueError, match="gates"):
            PresetConfig(
                entrypoint="phase0_regime_map",
                description="d",
                gates="   ",
                target="m",
                resources={"cluster": "nurc", "partition": "gpu", "time": "01:00:00"},
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
            ResourceConfig(cluster="nurc", partition="gpu", time="12:00:00")

    def test_a_malformed_time_is_refused(self):
        with pytest.raises(ValueError, match="HH:MM:SS"):
            ResourceConfig(cluster="nurc", partition="gpu", time="2h")

    @pytest.mark.parametrize("name", PRESETS)
    def test_every_shipped_preset_fits_its_own_clusters_queue(self, name):
        """⚠️ Its OWN cluster's, not a literal `08:00:00` (2026-08-22).

        This read `<= "08:00:00"` for every preset, which was right while every
        preset ran on NURC and silently wrong the moment one did not: xc has no
        wall at all, and a lexicographic string comparison against a NURC
        constant would have refused a legitimate 09:00:00 xc job in a test whose
        name claims to be about the queue.
        """
        from internals_safety.config import load_cluster_config

        preset = load_preset(name)
        cap = load_cluster_config(preset.resources.cluster).max_walltime_hours
        if cap is None:
            return  # no wall to fit
        assert int(preset.resources.time.split(":")[0]) <= cap


class TestEveryShippedPresetIsValid:
    @pytest.mark.parametrize("name", PRESETS)
    def test_it_loads(self, name):
        assert load_preset(name).description.strip()

    @pytest.mark.parametrize("name", PRESETS)
    def test_its_targets_are_real_model_configs(self, name):
        """A typo'd model name must fail here, not after the queue wait."""
        preset = load_preset(name)
        for target in filter(None, [preset.target, *preset.targets]):
            if preset.entrypoint in GUARD_TARGET_ENTRYPOINTS:
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
            resources={"cluster": "nurc", "partition": "short", "time": "01:00:00"},
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
            resources={"cluster": "nurc", "partition": "gpu", "time": "02:00:00"},
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

        from internals_safety.config import load_corpus_config
        from internals_safety.measurements.deployment import REQUIRED_CONTROLS
        from internals_safety.measurements.lexical_decorrelation import (
            LEXICAL_CONTROL_SETS,
        )

        assert "lexical_vocabulary" in REQUIRED_CONTROLS  # the rule this mirrors

        # ⚠️ ONE EXEMPTION, DERIVED RATHER THAN LISTED (2026-08-22). Once a run
        # could name its own contrast pair, a pair could BE this control's
        # corpus, and then declaring `lexical` would screen the probe on the
        # corpus it was fitted on — a control that cannot fail, passing more
        # comfortably than before. `phase0_regime_map` refuses that combination
        # outright, so the two rules cannot both be satisfied.
        #
        # The exemption is narrow and it is not a loophole: it does NOT say the
        # deployment numbers are fine, it says the run knowingly does not buy
        # them. The contract still fails closed and marks every deployment
        # reading non-reportable, which is asserted below rather than assumed —
        # an exemption tied to nothing would be a way to approve exactly the run
        # this class exists to reject.
        _, pair = load_corpus_config().pair(preset.corpus)
        if LEXICAL_CONTROL_SETS & {pair.harmful_set, pair.harmless_set}:
            assert "lexical" not in preset.instruments, (
                f"{name} pairs on {sorted(LEXICAL_CONTROL_SETS)}, which IS the lexical "
                "control's corpus; the entrypoint refuses this combination"
            )
            assert "lexical_vocabulary" in REQUIRED_CONTROLS, (
                "the exemption is only safe while the contract still withholds a "
                "deployment reading whose lexical control was never computed"
            )
            return

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
        # ⚠️ The PRESET's pair, not the default one. Checking a held-out preset
        # against the default corpus is the same reach-for-the-default defect the
        # registry exists to make unwritable, and it would fail a valid preset
        # rather than pass an invalid one — which is the direction that gets a
        # correct guard deleted.
        corpus = load_corpus_config()
        pair_name, pair = corpus.pair(preset.corpus)
        for prompt_set_name in (pair.harmful_set, pair.harmless_set):
            path = DATA_DIR / prompt_set_name
            if not path.exists():
                pytest.skip(f"{path} absent; data/ is gitignored and re-copied per clone")
            available = sum(1 for line in path.read_text().splitlines() if line.strip())
            if pair.matching == "contrast_type":
                # The matched subset is smaller than either file, so the file
                # count is the wrong denominator — derive the real one.
                from internals_safety.pipeline import load_contrast_sets

                harmful, _ = load_contrast_sets(
                    pair.harmful_set, pair.harmless_set, 10**6, matching=pair.matching
                )
                available = min(available, len(harmful))
            assert preset.n_prompts <= available, (
                f"{name} asks for {preset.n_prompts} prompts per class; pair {pair_name!r} "
                f"({prompt_set_name}) holds {available}. --n-prompts is PER CLASS, not the "
                "total across both."
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


class TestEveryShippedPresetCanBeCosted:
    """The approval gate is a THREE-command loop, and step 2 must not break.

    `submit.py <preset>` -> `cost_model.py --preset <preset>` -> the owner's go
    -> `--submit`. Every preset shipped here has passed the other invariants in
    this file for weeks while three of six could not be costed at all:

    - `sae_pregate_base` CRASHED — the Base checkpoint ships no chat template,
      which the repo already knew (`chat_template_from` in its model config) and
      the cost script did not consult.
    - `relicense_all` refused with "--model or --preset is required" while
      --preset was set, because it declares `targets:` (plural).
    - `sae_pregate_instruct` was the dangerous one: it SUCCEEDED, reporting
      5.8-10.3 GPU-hours, a $2.08-6.26 judge bill and "EXCEEDS the 8h partition
      limit" for three 1-hour tasks that run no ladder and call no judge. It
      sets no `families`, and the phase-0 default for unset families is every
      configured rung.

    A number of the wrong shape is worse than a crash, so the invariant is
    asserted structurally rather than left to whoever next reads the output.
    """

    cost_model = _script("cost_model")
    _PHASE0_SHAPED = cost_model._PHASE0_SHAPED
    _COST_ELSEWHERE = cost_model._COST_ELSEWHERE
    _JUDGES = cost_model._JUDGES
    report_declared_cost = staticmethod(cost_model.report_declared_cost)

    @pytest.mark.parametrize("name", PRESETS)
    def test_its_entrypoint_has_a_cost_route(self, name):
        """Either the phase-0 census describes it, or something else does."""
        entrypoint = load_preset(name).entrypoint
        assert entrypoint in self._PHASE0_SHAPED or entrypoint in self._COST_ELSEWHERE, (
            f"{name} runs {entrypoint!r}, which no cost route covers; the approval "
            "gate cannot be satisfied for it"
        )

    def test_the_two_routes_do_not_overlap(self):
        """An entrypoint on both routes would be costed twice, differently."""
        assert not (self._PHASE0_SHAPED & set(self._COST_ELSEWHERE))

    def test_every_entrypoint_is_routed(self):
        """Adding a fifth entrypoint must not silently inherit the phase-0 shape."""
        routed = set(self._PHASE0_SHAPED) | set(self._COST_ELSEWHERE)
        assert routed == set(get_args(Entrypoint)), (
            f"unrouted entrypoint(s) {set(get_args(Entrypoint)) - routed}"
        )

    @pytest.mark.parametrize("name", PRESETS)
    def test_a_non_phase0_preset_reports_without_a_tokenizer(self, name, capsys):
        """The branch must not reach the census — no network, no weights.

        `report_declared_cost` is the whole non-phase-0 path, so running it here
        proves the crash and the misleading refusal are both gone.
        """
        preset = load_preset(name)
        if preset.entrypoint in self._PHASE0_SHAPED:
            pytest.skip("phase-0 shaped; its census needs a tokenizer")
        assert self.report_declared_cost(name, preset, "/outputs") == 0
        out = capsys.readouterr().out
        # ⚠️ The judge line is per-entrypoint, not a constant. It asserted
        # `$0.00` unconditionally until 2026-08-09, when `encoding_ablation`
        # became the first delegated-route entrypoint that DOES judge — and an
        # approval request carrying a false zero is worse than one carrying no
        # number at all. What must hold is that the line is TRUE, not that it is
        # always free.
        if preset.entrypoint in self._JUDGES:
            assert "judge API spend     NOT ZERO" in out
            assert "$0.00" not in out
        else:
            assert "judge API spend     $0.00" in out
        assert preset.entrypoint in out

    def test_the_phase0_census_is_not_claimed_for_the_guard_entrypoint(self):
        """AS-6 sweeps the ladder but reads a verdict from the logits.

        No generation and no judge, so the decode half of the census would be
        invention. It is the one entrypoint whose shape makes it look like it
        belongs on the phase-0 route.
        """
        assert "as6_guard_probe" not in self._PHASE0_SHAPED


class TestEveryPresetsCommandLineParses:
    """Bind each preset's RENDERED argv to the target script's real flags.

    **This exists because a declaration went unreconciled and it cost a job.**
    `_CONSUMES` claimed `as6_guard_probe` consumes `instruments`. It does not —
    the script has no such flag. So `guard_benign_arm_wildguard` carried
    `instruments: [lexical]`, the schema accepted it, `command()` rendered
    `--instruments lexical`, and job 9010529 died in 22 seconds on
    `unrecognized arguments` after a queue wait.

    Neither existing guard could see it. `tests/test_presets.py` validated the
    preset against the schema, and the schema was the thing that was wrong;
    `tests/test_entrypoint_call_sites.py` binds calls to library FUNCTIONS, and
    a command-line flag is not a function. And a `--dry-run` does not cover it
    either, because a dry run is invoked by hand with hand-typed flags — the
    exact gap the preset system was built to close, reopened one level up.

    The check is on the RENDERED command line rather than on `_CONSUMES`
    directly, so it needs no field-to-flag mapping to drift out of date: whatever
    `command()` emits must be something the script can parse.
    """

    def _flags_defined_in(self, path: Path) -> set[str]:
        """Every `--flag` string passed to an add_argument call in this file."""
        flags: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            attr = getattr(node.func, "attr", None)
            if attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--"):
                        flags.add(arg.value)
        return flags

    def test_every_flag_the_launcher_emits_exists_in_its_script(self):
        repo = Path(__file__).resolve().parents[1]
        # Shared flags live in the pipeline helper every entrypoint calls.
        shared = self._flags_defined_in(repo / "src" / "internals_safety" / "pipeline.py")

        checked = 0
        for path in sorted((repo / "conf" / "experiment").glob("*.yaml")):
            preset = load_preset(path.stem)
            for row in preset.tasks(Path("/tmp/outputs")):
                script = repo / row[0]
                assert script.exists(), f"{path.stem}: {row[0]} does not exist"
                available = self._flags_defined_in(script) | shared
                emitted = {token for token in row if token.startswith("--")}
                missing = emitted - available
                assert not missing, (
                    f"preset {path.stem!r} renders {sorted(missing)}, which "
                    f"{row[0]} cannot parse — the job would die at argparse "
                    f"after a queue wait (job 9010529)"
                )
                checked += 1
        assert checked, "no preset command lines were checked"
