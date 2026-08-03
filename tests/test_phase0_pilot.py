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
    probes=ProbeConfig(seed=0, test_fraction=0.3, auroc_threshold=0.70, cv_folds=3),
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
        assert all(cell["regime"] in set("CDBSRX") for cell in cells)
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
        assert deployment["gap"] == deployment["harmful_rate"] - deployment["harmless_rate"]

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
        assert family_result["summary"]["binding_failure_rate"] == 0.0
        assert family_result["summary"]["refusal_rate"] == 1.0


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
