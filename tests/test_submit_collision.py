"""The duplicate-submission guard.

Two Claude sessions share one SLURM account. On 2026-08-07 one was seconds from
re-submitting a 30-task `relicense_all` array that the other already had
running, and the only thing that caught it was a human reading `squeue`. A guard
that depends on someone remembering to look is not a guard.

**These tests are written against a specific failure: the vacuous check.** The
peer session shipped a device-coercion test the same afternoon that passed on
the broken code, because its fake second device (`meta`) silently accepted the
operation it was supposed to reject. So every test here is checked in BOTH
directions — the guard must fire when it should and stay quiet when it should
not — and the stamp the guard reads is asserted to be one the submitter
actually writes. A check that reads a field nobody populates always returns
"no collision" and always passes.
"""

from __future__ import annotations

import importlib.util
import subprocess

import pytest

from internals_safety.config import list_presets
from internals_safety.paths import PROJECT_ROOT


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _script("submit")

# squeue -O "JobID:24,Comment:64,Name:64,StateCompact:12" — whitespace-separated.
STAMPED = "9006939                 preset=decode_lens_real   is_decode_lens_real   PD"
PRE_STAMP = "8995184                 (null)                    is_relicense_all      R"
UNRELATED = "9006940                 preset=causal_sweep       is_causal_sweep       R"
FOREIGN = "9995281                 (null)                    vllm_wildguard        R"


class _FakeCompleted:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def fake_squeue(monkeypatch, lines, returncode=0):
    """Route squeue to canned output; record any sbatch that gets through."""
    submitted: list[list[str]] = []

    def runner(command, **kwargs):
        if command[0] == "squeue":
            return _FakeCompleted(stdout="\n".join(lines), returncode=returncode,
                                  stderr="boom" if returncode else "")
        submitted.append(command)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(submit.subprocess, "run", runner)
    monkeypatch.setenv("USER", "tester")
    return submitted


class TestTheStampIsActuallyWritten:
    """A guard reading a field the submitter never sets is permanently vacuous."""

    @pytest.mark.parametrize("name", list_presets())
    def test_every_preset_gets_its_preset_name_in_the_comment(self, name):
        from internals_safety.config import load_preset

        flags = submit.sbatch_flags(load_preset(name), name, 1)
        assert f"--comment={submit.COMMENT_PREFIX}{name}" in flags

    def test_the_guard_reads_the_prefix_the_submitter_writes(self):
        """Pins the two halves together, so renaming one breaks the test."""
        flags = " ".join(submit.sbatch_flags(
            __import__("internals_safety.config", fromlist=["load_preset"]).load_preset("smoke"),
            "smoke", 1))
        assert submit.COMMENT_PREFIX in flags


class TestItFiresWhenItShould:
    def test_a_stamped_match_refuses_and_submits_nothing(self, monkeypatch, capsys):
        sent = fake_squeue(monkeypatch, [STAMPED, UNRELATED])
        assert submit.main(["decode_lens_real", "--submit"]) == 3
        assert sent == [], "REFUSED but sbatch still ran"
        assert "9006939" in capsys.readouterr().err, "refusal must name the offending job"

    def test_a_pre_stamp_job_still_matches_by_name(self, monkeypatch, capsys):
        """The population that exists right now: jobs queued before the stamp."""
        sent = fake_squeue(monkeypatch, [PRE_STAMP])
        assert submit.main(["relicense_all", "--submit"]) == 3
        assert sent == []
        assert "pre-stamp" in capsys.readouterr().err

    def test_it_matches_running_as_well_as_pending(self, monkeypatch):
        sent = fake_squeue(monkeypatch, [UNRELATED])  # state R
        assert submit.main(["causal_sweep", "--submit"]) == 3
        assert sent == []


class TestItStaysQuietWhenItShould:
    """The other direction — without these the guard could just always refuse."""

    def test_an_empty_queue_submits(self, monkeypatch):
        sent = fake_squeue(monkeypatch, [])
        assert submit.main(["decode_lens_real", "--submit"]) == 0
        assert len(sent) == 1 and sent[0][0] == "sbatch"

    def test_another_preset_is_not_a_collision(self, monkeypatch):
        """UNRELATED is causal_sweep; submitting decode_lens_real must go through."""
        sent = fake_squeue(monkeypatch, [UNRELATED, FOREIGN])
        assert submit.main(["decode_lens_real", "--submit"]) == 0
        assert len(sent) == 1

    def test_a_foreign_job_on_the_account_is_not_a_collision(self, monkeypatch):
        """The account has a third consumer — the guardrail sibling's vllm jobs."""
        sent = fake_squeue(monkeypatch, [FOREIGN])
        assert submit.main(["smoke", "--submit"]) == 0
        assert len(sent) == 1

    def test_printing_without_submit_never_calls_squeue(self, monkeypatch):
        """Describing a preset cannot collide with anything, so it must not pay."""
        calls: list[str] = []

        def runner(command, **kwargs):
            calls.append(command[0])
            return _FakeCompleted()

        monkeypatch.setattr(submit.subprocess, "run", runner)
        assert submit.main(["decode_lens_real"]) == 0
        assert calls == []


class TestForceIsLoud:
    def test_force_submits_but_names_what_it_duplicates(self, monkeypatch, capsys):
        sent = fake_squeue(monkeypatch, [STAMPED])
        assert submit.main(["decode_lens_real", "--submit", "--force"]) == 0
        assert len(sent) == 1
        out = capsys.readouterr().out
        assert "DUPLICATE" in out and "9006939" in out


class TestAFailedCheckIsNotACleanCheck:
    """Fail closed iff sbatch can actually reach a queue.

    The first version of this guard reasoned "no squeue means no sbatch, so
    warn and proceed". That is true on a laptop and silently wrong on the
    cluster: squeue failing because slurmctld is unreachable happens on a
    machine where sbatch still works, which is precisely when an unguarded
    submit makes a duplicate. The peer session caught it.

    The branch is mutation-tested on its own below — passing counts from
    mutating the guard as a whole would not have shown this path was covered.
    """

    def test_it_refuses_when_sbatch_exists_but_squeue_failed(self, monkeypatch, capsys):
        """The dangerous case: a real submit host with a blipping controller."""
        sent = fake_squeue(monkeypatch, [], returncode=1)
        monkeypatch.setattr(submit.shutil, "which", lambda name: "/usr/bin/sbatch")
        assert submit.main(["smoke", "--submit"]) == 4
        assert sent == [], "refused but sbatch still ran"
        assert "sbatch IS available" in capsys.readouterr().err

    def test_it_proceeds_when_sbatch_is_absent(self, monkeypatch, capsys):
        """The laptop: nothing can reach a queue, so the guard's opinion is moot."""
        sent = fake_squeue(monkeypatch, [], returncode=1)
        monkeypatch.setattr(submit.shutil, "which", lambda name: None)
        assert submit.main(["smoke", "--submit"]) == 0
        assert len(sent) == 1
        assert "collision check did not run" in capsys.readouterr().err

    def test_the_two_unavailable_causes_are_distinguished(self, monkeypatch):
        """'not installed' and 'ran and failed' are different facts."""
        def missing(command, **kwargs):
            raise FileNotFoundError("squeue")

        monkeypatch.setattr(submit.subprocess, "run", missing)
        with pytest.raises(submit.SqueueUnavailable) as absent:
            submit.active_jobs_for("smoke")
        assert absent.value.binary_missing is True

        monkeypatch.setattr(
            submit.subprocess, "run",
            lambda command, **kwargs: _FakeCompleted(returncode=1, stderr="slurmctld down"),
        )
        with pytest.raises(submit.SqueueUnavailable) as failed:
            submit.active_jobs_for("smoke")
        assert failed.value.binary_missing is False
        assert "slurmctld down" in str(failed.value)
