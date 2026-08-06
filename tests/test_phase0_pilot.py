"""Phase-0 pilot tests — the entrypoint that combines all four measurements.

Hermetic: the tiny in-process model from conftest plus stub judge services, so
nothing downloads weights or calls an API. What these pin is the *wiring* — that
every measurement reaches the regime assignment, that the run record carries the
schema `.claude/skills/reproducible-run-logging` mandates, and that the raw cell
file is auditable. Whether a real 8B model populates (B) is the pilot's question,
not a test's.

The one that matters most is `test_the_attack_prompt_is_the_captured_condition`:
if activations came from the restate prompt, every regime in the paper would be
labelled from a forward pass in which decoding was explicitly requested — the
exact confusion the (D) cell exists to detect.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from internals_safety.config import (
    AbilityConfig,
    BehaviorConfig,
    JudgeConfig,
    MeasurementsConfig,
    ProbeConfig,
    load_measurements_config,
    load_model_config,
)
from internals_safety.data import Prompt
from internals_safety.encodings.registry import load_ladder
from internals_safety.judges.harmbench import HarmBenchJudge
from internals_safety.judges.refusal import RefusalJudge
from internals_safety.models.capture import capture_or_load
from internals_safety.paths import PROJECT_ROOT
from judge_stubs import StubService, no_verdict, yes_verdict

N = 8
MEASUREMENTS = MeasurementsConfig(
    ability=AbilityConfig(max_new_tokens=4, batch_size=4),
    behavior=BehaviorConfig(max_new_tokens=4, batch_size=4),
    # n_permutations is tiny here ON PURPOSE: these tests exercise the pilot's
    # wiring, not the licensing statistic, and the production default (200)
    # multiplies every fixture by the layer count. The permutation test's own
    # behaviour is covered in tests/test_probes.py.
    probes=ProbeConfig(
        seed=0, test_fraction=0.3, auroc_threshold=0.70, cv_folds=3, n_permutations=3
    ),
)


def load_script():
    path = PROJECT_ROOT / "scripts" / "phase0_regime_map.py"
    spec = importlib.util.spec_from_file_location("phase0_regime_map", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot():
    return load_script()


@pytest.fixture
def harmful():
    return [Prompt(id=f"h{i}", text=f"harmful request number {i}", category="c") for i in range(N)]


@pytest.fixture
def harmless():
    return [Prompt(id=f"b{i}", text=f"benign request number {i}", category="c") for i in range(N)]


@pytest.fixture
def judges():
    """Every response judged refused-and-not-jailbroken, so the wiring is
    testable without asserting anything about the tiny model's output."""
    refusal = RefusalJudge(JudgeConfig(), service=StubService(default=yes_verdict()))
    harm = HarmBenchJudge(JudgeConfig(), service=StubService(default=no_verdict()))
    return refusal, harm


@pytest.fixture
def plain_batches(tiny_model, harmful, harmless, tmp_path):
    positive, _, _ = capture_or_load(
        tiny_model, [p.text for p in harmful], condition="plain-harmful", cache_dir=tmp_path
    )
    negative, _, _ = capture_or_load(
        tiny_model, [p.text for p in harmless], condition="plain-harmless", cache_dir=tmp_path
    )
    return positive, negative


@pytest.fixture
def family_result(pilot, tiny_model, harmful, harmless, plain_batches, judges, tmp_path):
    refusal, harm = judges
    ladder = load_ladder()
    return pilot.run_family(
        tiny_model,
        ladder["base64"],
        harmful,
        harmless,
        *plain_batches,
        MEASUREMENTS,
        refusal,
        harm,
        cache_dir=tmp_path,
    )


class TestPlan:
    def test_counts_the_work_the_approval_gate_needs(self, pilot):
        from internals_safety.config import ModelConfig

        config = ModelConfig(name="m", hf_id="x", device="cpu")
        plan = pilot.build_plan(config, ["a", "b", "c"], n_prompts=100)
        # 2 plain conditions captured once per model + 2 encoded per family.
        assert plan.prompt_forward_passes == 2 * 100 + 2 * 3 * 100
        assert plan.generations == 2 * 3 * 100
        assert plan.judge_calls == 2 * 3 * 100

    def test_describe_names_the_gate_and_where_to_cost_it(self, pilot):
        """The dry-run has to hand the reader the next command, not just a pile
        of counts: the counts are the gate's input, not the gate's answer."""
        from internals_safety.config import ModelConfig

        plan = pilot.build_plan(ModelConfig(name="m", hf_id="x", device="cpu"), ["a"], 10)
        text = plan.describe(MEASUREMENTS)
        assert "approval gate" in text and "wall-clock" in text
        assert "scripts/cost_model.py --model m" in text

    def test_select_families_defaults_to_the_whole_ladder(self, pilot):
        ladder = load_ladder()
        assert pilot.select_families(ladder, "all") == list(ladder)

    def test_an_unknown_family_stops_the_run(self, pilot):
        with pytest.raises(SystemExit, match="unknown encoding families"):
            pilot.select_families(load_ladder(), ["not_a_rung"])


class TestRunFamily:
    def test_every_prompt_gets_a_regime(self, family_result, harmful):
        cells = family_result["cells"]
        assert len(cells) == len(harmful)
        # U joined the set 2026-08-05: an unlicensed deployment probe yields a
        # declared hole rather than being forced into (R).
        assert all(cell["regime"] in set("CDBSRXU") for cell in cells)
        assert family_result["regime_map"].n == len(harmful)

    def test_a_cell_carries_all_four_measurements(self, family_result):
        cell = family_result["cells"][0]
        assert set(cell) >= {"ability", "deployment", "recognition", "refused", "jailbroken"}
        assert cell["prompt_id"] == "h0"

    def test_the_attack_prompt_is_the_captured_condition(self, family_result, tmp_path, pilot):
        """Activations must come from the attack prompt, never the restate one:
        a regime labelled off a forward pass that was *asked* to decode measures
        the wrong thing, which is the whole reason (D) exists."""
        cached = family_result["summary"]["activations"]["encoded_harmful"]
        from internals_safety.models.capture import ActivationBatch

        batch = ActivationBatch.load(Path(cached))
        assert all("Write out the decoded text" not in message for message in batch.user_messages)
        assert all("encoded with" in message.lower() for message in batch.user_messages)

    def test_judges_saw_the_plaintext_not_the_ciphertext(self, family_result, judges, harmful):
        refusal, _ = judges
        sent = " ".join(refusal.service.seen.values())
        assert harmful[0].text in sent
        assert family_result["cells"][0]["ciphertext"] not in sent

    def test_the_summary_reports_the_format_decorrelation_gap(self, family_result):
        deployment = family_result["summary"]["deployment"]
        assert set(deployment) >= {"licensed", "transfer_auroc", "harmful_rate", "harmless_rate", "gap"}
        if deployment["licensed"]:
            assert deployment["gap"] == deployment["harmful_rate"] - deployment["harmless_rate"]
        else:
            # Tri-state (2026-08-05): an unlicensed probe reports None on every
            # rate, never 0.0. A gap computed here would be the difference of two
            # non-measurements — and the 0.0 it used to produce read downstream
            # as "the decode was not deployed".
            assert deployment["harmful_rate"] is None
            assert deployment["harmless_rate"] is None
            assert deployment["gap"] is None

    def test_the_summary_carries_the_length_null(self, family_result):
        """The mandatory control for both papers, on the AS-5 side.

        It shipped 2026-08-05 wired into the GUARD path only, so AS-6 had it and
        AS-5 did not — for a confound that was MEASURED on the AS-5 side (raw
        character length separates the JBB harmful corpus from the benign one at
        AUROC 0.654, and the newly licensed rungs sat at 0.659).
        """
        null = family_result["summary"]["length_null"]
        assert set(null) >= {
            "plain_auroc", "encoded_auroc", "observed_max_transfer_auroc",
            "margin", "min_margin", "beats_length_null",
        }
        # nan_ok: on the tiny fixture model a probe can fail to fit, and the
        # margin is then NaN rather than 0.0 — the same fail-closed discipline
        # as the deployment curve's p_value. A NaN margin must NOT beat the null.
        assert null["margin"] == pytest.approx(
            null["observed_max_transfer_auroc"] - null["encoded_auroc"], nan_ok=True
        )
        assert null["beats_length_null"] == (null["margin"] >= null["min_margin"])
        if null["margin"] != null["margin"]:
            assert not null["beats_length_null"]

    def test_the_length_null_is_measured_on_the_texts_actually_sent(self, family_result, harmful):
        """From the run's own strings, not a corpus name — otherwise the control
        drifts away from the thing it is controlling."""
        assert family_result["summary"]["length_null"]["n_positive"] == len(harmful)

    def test_the_summary_names_its_activation_caches(self, family_result):
        activations = family_result["summary"]["activations"]
        assert Path(activations["encoded_harmful"]).exists()
        assert Path(activations["encoded_harmless"]).exists()

    def test_the_second_call_hits_the_activation_cache(
        self, pilot, tiny_model, harmful, harmless, plain_batches, judges, tmp_path
    ):
        refusal, harm = judges
        encoder = load_ladder()["base64"]
        args = (tiny_model, encoder, harmful, harmless, *plain_batches, MEASUREMENTS, refusal, harm)
        first = pilot.run_family(*args, cache_dir=tmp_path)
        second = pilot.run_family(*args, cache_dir=tmp_path)
        assert first["summary"]["activations"]["cache_hits"] == [False, False]
        assert second["summary"]["activations"]["cache_hits"] == [True, True]

    def test_refused_everything_leaves_the_binding_cell_empty(self, family_result):
        """The stub judges refuse every response, so (B) — decode-and-comply —
        cannot be populated. A pipeline that reported (B) here would be reading
        behaviour from somewhere other than the refusal judge."""
        summary = family_result["summary"]
        assert summary["refusal_rate"] == 1.0
        assert summary["regimes"].get("B", 0) == 0
        if summary["deployment_unmeasured"] == summary["n"]:
            # The tiny random model's probe does not license, so the honest
            # answer is "no binding-failure rate exists", NOT a rate of zero.
            # 0.0 here reads as "measured, and found none" (2026-08-05).
            assert summary["binding_failure_rate"] is None
        else:
            assert summary["binding_failure_rate"] == 0.0


class TestMain:
    """The glue the unit tests above do not reach: argument parsing, corpus
    loading, and run-record assembly. Driven with the tiny in-process model and
    stub judges, so it downloads nothing, calls no API and spends nothing — the
    experiment-run approval gate covers runs that produce numbers, and this
    produces a file layout."""

    @pytest.fixture
    def run(self, pilot, tiny_model, monkeypatch, tmp_path):
        monkeypatch.setattr(pilot, "load_model", lambda config: tiny_model)
        monkeypatch.setattr(
            pilot, "RefusalJudge", lambda config: RefusalJudge(config, service=StubService(default=yes_verdict()))
        )
        monkeypatch.setattr(
            pilot, "HarmBenchJudge", lambda config: HarmBenchJudge(config, service=StubService(default=no_verdict()))
        )
        exit_code = pilot.main(
            [
                "--model", "qwen2_5_0_5b_instruct",
                "--families", "base64", "rot13",
                "--n-prompts", "6",
                "--run-name", "unit",
                "--allow-dirty",
                # This test IS a CPU run, and the suite itself runs inside a
                # SLURM job during the cluster env build — where resolve_device
                # refuses a silent CPU fallback. Saying so explicitly keeps the
                # test deterministic instead of depending on whether the host
                # happens to offer CUDA or MPS.
                "--allow-cpu",
                "--outputs-dir", str(tmp_path),
            ]
        )
        assert exit_code == 0
        return tmp_path / "runs" / "phase0" / "qwen2_5_0_5b_instruct" / "unit"

    def test_writes_the_canonical_run_record(self, run):
        record = json.loads((run / "results.json").read_text())
        # The schema in .claude/skills/reproducible-run-logging.
        assert set(record) >= {
            "git_hash", "git_dirty", "config", "seed", "env_lock",
            "metrics", "raw_output_path", "activations_path", "timestamp",
        }
        assert record["seed"] == 0

    def test_the_record_carries_typed_readings_and_what_was_withheld(self, run):
        """**The proof that the contract is load-bearing, not merely available.**

        A run record that lists three numbers and silently drops the rest is
        indistinguishable from a run that measured three things. Every
        instrument's verdict is recorded here with its evidence, and every
        non-reportable one names why — for a real end-to-end run, not a fixture.
        """
        record = json.loads((run / "results.json").read_text())
        assert {"readings", "withheld", "n_reportable"} <= set(record)

        # Five instruments per rung x two rungs.
        instruments = {reading["instrument"] for reading in record["readings"]}
        assert instruments == {
            "ability", "behavior", "deployment", "recognition", "trajectory"
        }
        # Every reading states how to read its number and whether it was measured.
        for reading in record["readings"]:
            assert reading["operating_point"]
            assert reading["licensed"] in (True, False, None)

    def test_behaviour_is_still_withheld_for_want_of_a_negative_control(self, run):
        """TODO 38, asserted end to end rather than only at the unit level.

        The judges never run on the benign-encoded arm, so nothing distinguishes
        "the attack succeeded" from "this judge says yes to anything wearing this
        encoding". If this test ever goes green by accident, a control was added
        or the bar was lowered.
        """
        withheld = json.loads((run / "results.json").read_text())["withheld"]
        assert "behavior" in withheld
        assert any("no negative control" in why for why in withheld["behavior"])

    def test_ability_now_carries_its_control_and_is_withheld_on_the_MEASURED_axis(self, run):
        """TODO 37 closed — and the reason ability is still withheld has changed.

        It is no longer "no negative control was run": the mismatched-plaintext
        control runs on every family. It is now the measured comparison, because
        the tiny in-process model decodes nothing, so the value sits AT its
        negative-control floor. That is the asymmetry `ability_control` documents
        — a specificity control cannot license an ability-0 reading — and it must
        show up in the run record rather than only in a docstring.
        """
        record = json.loads((run / "results.json").read_text())
        withheld = record["withheld"]
        assert "ability" in withheld
        assert not any("no negative control" in why for why in withheld["ability"])
        assert any("on the negative control" in why for why in withheld["ability"])

        ability = next(r for r in record["readings"] if r["instrument"] == "ability")
        assert ability["control_reading"] == 0.0
        # The sensitivity arm rides in detail and says the scorer is not broken
        # on this family's character set — the only evidence that supports an
        # ability-0 claim at all.
        assert ability["detail"]["control_identity_rate"] == 1.0
        assert ability["detail"]["control_scorer_is_functional"] is True

    def test_the_record_names_every_activation_cache_it_read(self, run):
        paths = json.loads((run / "results.json").read_text())["activations_path"]
        assert Path(paths["plain_harmful"]).exists()
        assert set(paths["per_family"]) == {"base64", "rot13"}

    def test_the_record_pins_the_corpus_by_content(self, run):
        corpus = json.loads((run / "results.json").read_text())["corpus"]
        assert corpus["n_prompts"] == 6
        assert corpus["harmful_set"] == "jbb_prompts.jsonl"
        assert len(corpus["harmful_digest"]) == 16

    def test_cells_are_one_auditable_jsonl_line_per_prompt(self, run):
        lines = (run / "cells.jsonl").read_text().splitlines()
        assert len(lines) == 12  # 6 prompts x 2 families
        assert {json.loads(line)["family"] for line in lines} == {"base64", "rot13"}

    def test_a_completed_family_survives_a_job_that_never_finishes(self, run):
        """Regression, 2026-08-05. The comprehension-band sweep was killed at the
        8 h wall having COMPLETED `zero_width` on both models and recovered
        nothing: `cells.jsonl` was 0 bytes (Python's buffer had not filled) and
        `results.json` is only written after the family loop. Eight GPU-hours of
        finished work were unrecoverable.

        So every family checkpoints as it completes. This asserts the per-family
        record exists and carries the timing the cost model needs; the fsync that
        makes it survive a SIGKILL cannot be tested in-process, which is exactly
        why the flush must never be quietly dropped."""
        partial = run / "summaries.partial.jsonl"
        rows = [json.loads(line) for line in partial.read_text().splitlines()]
        assert [row["family"] for row in rows] == ["base64", "rot13"]
        assert all(row["elapsed_seconds"] >= 0 for row in rows)

    def test_the_record_times_itself_for_the_next_cost_estimate(self, run):
        """conf/cost.yaml's throughput ranges span ~4x because no run has
        measured them yet. This is the instrumentation that closes that: the
        knob and the data collection ship together, so phase 1's estimate is a
        measurement rather than a second guess."""
        throughput = json.loads((run / "results.json").read_text())["throughput"]
        assert throughput["elapsed_seconds"] > 0
        # main() loads conf/measurements.yaml, not this module's MEASUREMENTS.
        budgets = load_measurements_config()
        assert throughput["budgeted_decode_tokens"] == 2 * 6 * (
            budgets.ability.max_new_tokens + budgets.behavior.max_new_tokens
        )
        assert throughput["budgeted_decode_tokens_per_s"] > 0
        # Labelled as an upper bound, so it is never read as a measured rate.
        assert "upper-bound" in throughput["note"]

    def test_metrics_carry_one_summary_per_family(self, run):
        families = json.loads((run / "results.json").read_text())["metrics"]["families"]
        assert [entry["family"] for entry in families] == ["base64", "rot13"]
        assert all("binding_failure_rate" in entry for entry in families)


class TestScheduledJobGuard:
    """A batch job that falls back to CPU is a defect, not a slow run.

    Seed 2026-08-02: the first real cluster job resolved to CPU because torch was
    built for cu130 against a 12.8 driver. It ran to completion on an allocated,
    idle A100 and the only trace was a UserWarning in the log. Worse, the
    dirty-tree guard keys off CUDA (provenance.RESULT_BEARING_DEVICES), so the
    same silent fallback also downgraded that guard from refuse to warn.
    """

    def test_cpu_inside_a_slurm_job_is_refused(self, monkeypatch):
        import torch

        from internals_safety.models.loader import ScheduledJobOnCPU, resolve_device

        monkeypatch.setenv("SLURM_JOB_ID", "123456")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(ScheduledJobOnCPU, match="123456"):
            resolve_device("auto")

    def test_the_refusal_names_the_likely_cause(self, monkeypatch):
        import torch

        from internals_safety.models.loader import ScheduledJobOnCPU, resolve_device

        monkeypatch.setenv("SLURM_JOB_ID", "1")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(ScheduledJobOnCPU) as caught:
            resolve_device("auto")
        assert "driver" in str(caught.value) and "cu130" in str(caught.value)

    def test_allow_cpu_overrides_it(self, monkeypatch):
        import torch

        from internals_safety.models.loader import resolve_device

        monkeypatch.setenv("SLURM_JOB_ID", "1")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device("auto", allow_cpu_in_job=True).type == "cpu"

    def test_a_laptop_still_falls_back_silently(self, monkeypatch):
        """Off the cluster, CPU is a legitimate answer — the guard must not fire."""
        import torch

        from internals_safety.models.loader import resolve_device

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device("auto").type == "cpu"

    def test_an_explicit_device_is_never_second_guessed(self, monkeypatch):
        from internals_safety.models.loader import resolve_device

        monkeypatch.setenv("SLURM_JOB_ID", "1")
        assert resolve_device("cpu").type == "cpu"


class TestDeclaredInstrumentRoster:
    """`--instruments` — the approval gate sees the cost before the run happens.

    I1 and I3 add GPU work that the four measurements and I2 do not. Turning
    them on by default would change what a run costs without the gate seeing it,
    which is the family's experiment-run rule and, since 2026-08-06, also the
    gate-not-measurement rule. So the roster is DECLARED and `--dry-run` prices
    the declaration.
    """

    def plan(self, instruments=()):
        from internals_safety.config import load_model_config

        return load_script().build_plan(
            load_model_config("qwen2_5_0_5b_instruct"),
            ["base64", "rot13"],
            n_prompts=100,
            allow_cpu=True,
            instruments=instruments,
        )

    def test_the_default_roster_is_empty_and_costs_nothing_extra(self):
        plan = self.plan()
        assert plan.instruments == ()
        assert plan.instrument_forward_passes == 0
        assert plan.lens_readouts == 0

    def test_declaring_i1_adds_forward_passes_to_the_estimate(self):
        """The number the gate needs: I1's target passes are not free, and on a
        two-rung run they outnumber the base capture passes."""
        plan = self.plan(("decode_lens",))
        assert plan.instrument_forward_passes > 0
        assert plan.lens_readouts == 0

    def test_declaring_i3_adds_lens_readouts_counted_separately(self):
        """Separately, because they are not forward passes — they are
        [chunk, d_model] x [d_model, vocab] matmuls, bandwidth-bound not
        compute-bound. Conflating them would misprice the run."""
        plan = self.plan(("entropy_dynamics",))
        assert plan.lens_readouts > 0
        assert plan.instrument_forward_passes == 0

    def test_the_estimate_names_the_declared_roster(self):
        from internals_safety.config import load_measurements_config

        described = self.plan(("decode_lens",)).describe(load_measurements_config())
        assert "decode_lens" in described
        assert "I1 decode lens" in described and "I3 entropy dynamics" in described

    def test_the_declared_instruments_actually_run_and_emit_readings(
        self, pilot, tiny_model, monkeypatch, tmp_path
    ):
        """End to end, not just costed.

        A flag that prices an instrument but does not run it would be worse than
        no flag: the gate would approve work that never happens. This drives the
        real code path for I1 and I3 on the tiny in-process model.
        """
        monkeypatch.setattr(pilot, "load_model", lambda config: tiny_model)
        monkeypatch.setattr(
            pilot, "RefusalJudge",
            lambda config: RefusalJudge(config, service=StubService(default=yes_verdict())),
        )
        monkeypatch.setattr(
            pilot, "HarmBenchJudge",
            lambda config: HarmBenchJudge(config, service=StubService(default=no_verdict())),
        )
        assert pilot.main([
            "--model", "qwen2_5_0_5b_instruct",
            "--families", "rot13",
            "--n-prompts", "4",
            "--run-name", "instruments",
            "--instruments", "decode_lens", "entropy_dynamics",
            "--allow-dirty", "--allow-cpu",
            "--outputs-dir", str(tmp_path),
        ]) == 0

        record = json.loads(
            (tmp_path / "runs" / "phase0" / "qwen2_5_0_5b_instruct" / "instruments"
             / "results.json").read_text()
        )
        instruments = {reading["instrument"] for reading in record["readings"]}
        assert {"decode_lens", "entropy_dynamics"} <= instruments

    def test_an_unknown_instrument_is_refused_rather_than_ignored(self):
        """A typo'd --instruments would otherwise produce a run that silently
        omitted the instrument it was launched to add."""
        with pytest.raises(SystemExit, match="unknown instruments"):
            load_script().main([
                "--model", "qwen2_5_0_5b_instruct", "--instruments", "logit_lens",
                "--dry-run", "--allow-cpu",
            ])


class TestTheCausalGateIsWired:
    """The causal gate runs at MODEL level, not per rung — asserted end to end
    because that is the structural claim, and a claim only a docstring makes is
    one nothing checks.

    Driven through the same tiny in-process model and stub judges as `TestMain`:
    no download, no API call, no spend.
    """

    @pytest.fixture
    def run(self, pilot, tiny_model, monkeypatch, tmp_path):
        monkeypatch.setattr(pilot, "load_model", lambda config: tiny_model)
        monkeypatch.setattr(
            pilot,
            "RefusalJudge",
            lambda config: RefusalJudge(config, service=StubService(default=yes_verdict())),
        )
        monkeypatch.setattr(
            pilot,
            "HarmBenchJudge",
            lambda config: HarmBenchJudge(config, service=StubService(default=no_verdict())),
        )
        exit_code = pilot.main([
            "--model", "qwen2_5_0_5b_instruct",
            "--families", "base64", "rot13",
            "--n-prompts", "4",
            "--run-name", "causal",
            "--allow-dirty",
            "--allow-cpu",
            "--instruments", "causal_license",
            "--outputs-dir", str(tmp_path),
        ])
        assert exit_code == 0
        return tmp_path / "runs" / "phase0" / "qwen2_5_0_5b_instruct" / "causal"

    def test_exactly_one_causal_reading_is_emitted_for_two_families(self, run):
        """The structural point. A rung-level instrument would emit two."""
        readings = json.loads((run / "results.json").read_text())["readings"]
        causal = [r for r in readings if r["instrument"] == "causal_license"]
        assert len(causal) == 1

    def test_the_causal_reading_is_labelled_causal_and_never_merges_with_the_rest(self, run):
        readings = json.loads((run / "results.json").read_text())["readings"]
        causal = next(r for r in readings if r["instrument"] == "causal_license")
        assert causal["kind"] == "causal"
        assert all(r["kind"] == "correlational" for r in readings if r is not causal)

    def test_the_random_direction_null_actually_ran(self, run):
        """Without it, "steering worked" and "perturbing anything worked" are the
        same observation — so its absence must be visible, not assumed."""
        readings = json.loads((run / "results.json").read_text())["readings"]
        causal = next(r for r in readings if r["instrument"] == "causal_license")
        assert causal["detail"]["null_p_value"] is not None
        assert causal["control_margin"] is not None
        assert causal["selection_inside_null"] is True

    def test_the_gate_is_off_by_default_so_it_cannot_reprice_a_run_silently(self, pilot):
        config = load_model_config("qwen2_5_0_5b_instruct")
        plan = pilot.build_plan(config, ["base64"], n_prompts=10)
        assert plan.causal_forward_passes == 0
        assert plan.causal_candidates == 0

    def test_the_dry_run_prices_the_null_as_well_as_the_sweep(self, pilot):
        """Regression, 2026-08-06: the null's 20 extra candidate runs were
        briefly absent from the estimate while the code already ran them. A
        control the estimate cannot see is a cost nobody approved."""
        config = load_model_config("qwen2_5_0_5b_instruct")
        plan = pilot.build_plan(
            config, ["base64"], n_prompts=10,
            instruments=["causal_license"], n_random_directions=20,
        )
        sweep_only = (2 + 3 * plan.causal_candidates) * plan.n_prompts
        assert plan.causal_forward_passes > sweep_only
        assert plan.causal_forward_passes == (
            (2 + 3 * plan.causal_candidates) + (2 + 3 * 20)
        ) * plan.n_prompts


class TestReplyInversionIsWired:
    """I5 runs at model level too — same reason as the causal gate: it steers
    with directions fit on the plain contrast sets, which do not depend on a
    rung. Driven through the tiny model and stub judges; no spend."""

    @pytest.fixture
    def run(self, pilot, tiny_model, monkeypatch, tmp_path):
        monkeypatch.setattr(pilot, "load_model", lambda config: tiny_model)
        monkeypatch.setattr(
            pilot,
            "RefusalJudge",
            lambda config: RefusalJudge(config, service=StubService(default=yes_verdict())),
        )
        monkeypatch.setattr(
            pilot,
            "HarmBenchJudge",
            lambda config: HarmBenchJudge(config, service=StubService(default=no_verdict())),
        )
        exit_code = pilot.main([
            "--model", "qwen2_5_0_5b_instruct",
            "--families", "base64",
            "--n-prompts", "4",
            "--run-name", "inversion",
            "--allow-dirty",
            "--allow-cpu",
            "--instruments", "reply_inversion",
            "--outputs-dir", str(tmp_path),
        ])
        assert exit_code == 0
        return tmp_path / "runs" / "phase0" / "qwen2_5_0_5b_instruct" / "inversion"

    def test_the_reading_reaches_the_run_record(self, run):
        readings = json.loads((run / "results.json").read_text())["readings"]
        inversion = [r for r in readings if r["instrument"] == "reply_inversion"]
        assert len(inversion) == 1
        assert inversion[0]["kind"] == "causal"

    def test_it_is_withheld_for_want_of_its_own_random_direction_control(self, run):
        """The causal gate's null steers through a different prompt, so it is not
        reusable here — a second measurement, filed rather than faked."""
        withheld = json.loads((run / "results.json").read_text())["withheld"]
        assert "reply_inversion" in withheld
        assert any("no negative control" in why for why in withheld["reply_inversion"])

    def test_it_is_off_by_default(self, pilot):
        config = load_model_config("qwen2_5_0_5b_instruct")
        assert pilot.build_plan(config, ["base64"], n_prompts=10).inversion_forward_passes == 0

    def test_the_dry_run_prices_it(self, pilot):
        config = load_model_config("qwen2_5_0_5b_instruct")
        plan = pilot.build_plan(
            config, ["base64"], n_prompts=10, instruments=["reply_inversion"]
        )
        assert plan.inversion_forward_passes == 3 * 10
        assert "I5 reply inversion" in plan.describe(MEASUREMENTS)
