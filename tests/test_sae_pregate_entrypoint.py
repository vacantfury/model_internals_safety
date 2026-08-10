"""`sae_pregate.py`'s own glue, run end to end on CPU with everything heavy stubbed.

**The test that should have existed before the first submit.** Three separate
defects in this one script reached the SLURM queue on 2026-08-07 — jobs
`8995805`, `9006556`, `9006846` — and were found one per queue cycle, each after
the job had loaded an 8B model and a 540 MB dictionary:

1. `guard_working_tree()` called against a signature requiring `device`.
2. A CUDA dictionary fed by a deliberately-CPU pipeline (`encode` raised).
3. The same mismatch one line later, after the seam fix returned a CUDA
   reconstruction into that CPU pipeline.

Every one is in the *glue* — the twenty lines between argument parsing and
`measure_reconstruction` — and every one was invisible to 1002 passing tests,
because the library is well covered and the glue was covered nowhere. `--dry-run`
could not help: it returns before any of it.

So this stubs the three heavy things (`load_model`, `download_llama_scope_sae`,
`prompt_set`) and runs `main()` for real. It needs no weights, no network and no
GPU, and it exercises the script's actual control flow: the guard call, device
resolution, `observed_l0`, the control's construction, the reconstruction, the
verdict, and the run record.

Where a second device exists the dictionary is placed on it, because that is the
cluster's shape — a CPU pipeline and an off-device dictionary — and it is the
only configuration in which defects 2 and 3 are distinguishable at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sae_pregate  # noqa: E402

from internals_safety.models.sae_loader import LlamaScopeSAE  # noqa: E402


def dictionary_device() -> str:
    """The cluster's shape where we can reproduce it, CPU where we cannot."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def stubbed(monkeypatch, tiny_model, tiny_config):
    """Everything that needs a download, replaced. The GLUE stays real."""
    d_model = tiny_model.model.config.hidden_size
    d_sae = 4 * d_model
    device = dictionary_device()

    # A real LlamaScopeSAE, not a mock: the script calls `observed_l0`,
    # `our_layer`, `hook_point`, `trained_on` and `nominal_top_k` on it, and a
    # mock would answer all of them without exercising one line of the class
    # whose device handling is what keeps breaking.
    generator = torch.Generator().manual_seed(0)
    sae = LlamaScopeSAE(
        encoder_weight=torch.randn(d_sae, d_model, generator=generator).to(device),
        encoder_bias=torch.zeros(d_sae).to(device),
        decoder_weight=torch.randn(d_model, d_sae, generator=generator).to(device),
        decoder_bias=torch.zeros(d_model).to(device),
        jump_relu_threshold=0.0,
        input_norm=float(d_model**0.5),
        output_norm=float(d_model**0.5),
        d_model=d_model,
        d_sae=d_sae,
        hook_point="blocks.1.hook_resid_post",
        trained_on="tiny-test-base",
        nominal_top_k=8,
    )

    monkeypatch.setattr(sae_pregate, "load_model_config", lambda name: tiny_config)
    monkeypatch.setattr(sae_pregate, "load_model", lambda config: tiny_model)
    monkeypatch.setattr(sae_pregate, "download_llama_scope_sae", lambda **kw: sae)
    # Real `Prompt` objects, not strings: the script does `p.text`, and a
    # stub that answers a different shape tests the stub.
    from internals_safety.data import Prompt

    monkeypatch.setattr(
        sae_pregate,
        "prompt_set",
        lambda *a, **k: [
            Prompt(id="t1", text="the cat sat", category="test"),
            Prompt(id="t2", text="a dog ran fast", category="test"),
        ],
    )
    return sae


def run(tmp_path, extra: list[str] | None = None) -> int:
    return sae_pregate.main(
        [
            "--model", "tiny_test_model",
            "--n-prompts", "2",
            "--sae-layer", "1",
            "--run-name", "pregate-test",
            "--outputs-dir", str(tmp_path),
            "--allow-dirty",   # the tree is routinely dirty under pytest
            "--allow-cpu",     # tiny model is CPU by construction
            *(extra or []),
        ]
    )


class TestTheGlueRuns:
    def test_main_completes_and_writes_a_run_record(self, stubbed, tmp_path):
        """The whole point: `main()` reaches the end and produces its artifact.

        Any of the three queue failures makes this raise instead.
        """
        assert run(tmp_path) == 0

        records = list(tmp_path.rglob("results.json"))
        assert len(records) == 1, f"expected one run record, found {records}"
        record = json.loads(records[0].read_text())
        assert record["model"] == "tiny_test_model"
        assert record["sae"]["trained_on"] == "tiny-test-base"
        assert record["n_prompts"] == 2

    def test_the_record_carries_the_observed_sparsity_not_the_nominal_one(
        self, stubbed, tmp_path
    ):
        """`nominal_top_k` is a training-schedule leftover in a jumprelu
        checkpoint; the control matches the OBSERVED L0. The script reads both
        and it must record the one it acted on."""
        run(tmp_path)
        record = json.loads(next(tmp_path.rglob("results.json")).read_text())
        assert "observed_l0" in record["sae"]
        assert record["sae"]["nominal_top_k"] == 8

    def test_the_dictionary_may_live_off_device(self, stubbed, tmp_path):
        """The cluster's configuration, stated as an assertion.

        On a machine with a second device this is the exact shape that killed
        9006556 and 9006846: a CPU capture pipeline and a dictionary somewhere
        else. On a CPU-only machine it degenerates to the covered case and still
        passes, which is why it is not the only guard.
        """
        assert stubbed.encoder_weight.device.type == dictionary_device()
        assert run(tmp_path) == 0


class TestTheGuardIsActuallyCalled:
    def test_a_dirty_tree_without_allow_dirty_reaches_the_guard(
        self, stubbed, tmp_path, monkeypatch
    ):
        """Defect 1 was a guard that could not run at all — a TypeError, not a
        refusal. Pin that the guard is invoked with arguments it can bind, by
        making it raise and checking the raise is what surfaces."""
        from internals_safety.provenance import DirtyWorkingTree

        def refuse(device, allow_dirty=False):
            raise DirtyWorkingTree(f"stub refusal (device={device})")

        monkeypatch.setattr(sae_pregate, "guard_working_tree", refuse)
        with pytest.raises(DirtyWorkingTree) as caught:
            run(tmp_path)
        # The device reached it: that is precisely the argument whose absence
        # was the first defect.
        assert "device=" in str(caught.value)


class TestDryRunProvesNothingAboutTheRealPath:
    def test_dry_run_returns_before_the_glue(self, stubbed, tmp_path, monkeypatch):
        """Documented as an assertion so nobody re-learns it from a queue wait.

        `--dry-run` is what the approval gate is built on, and it returns before
        the guard, the model load and the dictionary load — so it stays green
        while every one of the three defects is live. This test makes the whole
        glue explode and shows `--dry-run` still returns 0.
        """
        def explode(*a, **k):
            raise AssertionError("the dry-run path must not reach the glue")

        monkeypatch.setattr(sae_pregate, "guard_working_tree", explode)
        monkeypatch.setattr(sae_pregate, "load_model", explode)
        monkeypatch.setattr(sae_pregate, "download_llama_scope_sae", explode)

        assert run(tmp_path, ["--dry-run"]) == 0
        assert not list(tmp_path.rglob("results.json"))


class TestTheTargetArmIsTheBranchEveryPresetTakes:
    """⚠️ The class above ran `main()` end to end and still missed a defect that
    killed six H200 tasks — jobs `9049059`/`9049060`, 2026-08-10.

    **Because it never passed `--ceiling-from`.** Every committed preset that is
    not the ceiling arm passes it, so the branch the cluster actually executes
    was the one branch with no end-to-end coverage: `ceiling_from` returns a
    `Ceiling` dataclass (since `5bd8c32`, when the KL bar landed beside the
    variance bar) and the print beside it still formatted it as a float.

    **Why the static call-site guard could not help, again.**
    `test_entrypoint_call_sites.py` binds a caller's ARGUMENTS against a live
    signature. This is a RETURN type flowing into an f-string — the call binds
    perfectly. Same lesson as `plain_arm`: the durable fix is not another
    watcher, it is running the branch.

    The ceiling artifact is produced by running `main()` in ceiling mode rather
    than hand-written, per the fixture law: a hand-made ceiling file is a fixture
    more permissive than the real thing, and `ceiling_from` raises on exactly the
    fields such a fixture would be free to get wrong (`detail.arm`, the layer).
    """

    def ceiling_artifact(self, tmp_path) -> Path:
        """Run the CEILING arm for real and hand back its record."""
        assert run(tmp_path, ["--run-name", "pregate-ceiling"]) == 0
        records = [p for p in tmp_path.rglob("results.json") if "pregate-ceiling" in str(p)]
        assert len(records) == 1, f"expected one ceiling record, found {records}"
        return records[0]

    def test_the_target_arm_runs_end_to_end_against_a_real_ceiling(
        self, stubbed, tmp_path
    ):
        """The exact shape of the cluster command, and what six tasks died on."""
        ceiling = self.ceiling_artifact(tmp_path)

        assert run(tmp_path, [
            "--run-name", "pregate-target",
            "--ceiling-from", str(ceiling),
        ]) == 0

        record = json.loads(
            next(p for p in tmp_path.rglob("results.json") if "pregate-target" in str(p)
                 ).read_text()
        )
        assert record["n_prompts"] == 2

    def test_the_ceiling_is_printed_with_BOTH_of_its_terms(
        self, stubbed, tmp_path, capsys
    ):
        """`Ceiling` carries variance AND KL because both are relative bars, and
        shipping only one is the defect `5bd8c32` fixed one level down. A target
        run that reports only one of them is that defect resurfacing in the
        operator-facing output, where it is exactly as misleading."""
        ceiling = self.ceiling_artifact(tmp_path)
        capsys.readouterr()

        run(tmp_path, ["--run-name", "pregate-target", "--ceiling-from", str(ceiling)])

        printed = capsys.readouterr().out
        assert "ceiling variance" in printed
        assert "KL" in printed

    def test_a_target_record_is_not_accepted_as_a_ceiling(self, stubbed, tmp_path):
        """Chaining a target reading as a ceiling compounds one transfer ratio on
        another and yields a confident number describing nothing. `ceiling_from`
        guards it; this pins that the guard survives the entrypoint."""
        ceiling = self.ceiling_artifact(tmp_path)
        run(tmp_path, ["--run-name", "pregate-target", "--ceiling-from", str(ceiling)])
        target = next(p for p in tmp_path.rglob("results.json") if "pregate-target" in str(p))

        with pytest.raises(ValueError):
            run(tmp_path, ["--run-name", "pregate-chained", "--ceiling-from", str(target)])
