"""Run-record tests.

The dirty-tree guard is the one rule in `.claude/skills/reproducible-run-logging`
that carries weight: a git hash recorded from an uncommitted tree points at code
that is not what ran, and nothing downstream can detect that. So the guard is
tested in both directions — it must refuse on the machine that produces results,
and it must not block a laptop debugging run.
"""

from __future__ import annotations

import json

import pytest

from internals_safety import provenance
from internals_safety.provenance import (
    DirtyWorkingTree,
    capture_provenance,
    guard_working_tree,
    write_results,
)


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(provenance, "is_dirty", lambda: False)


@pytest.fixture
def dirty(monkeypatch):
    monkeypatch.setattr(provenance, "is_dirty", lambda: True)
    monkeypatch.setattr(provenance, "_git", lambda *args: "fake-diff" if args[0] == "diff" else "abc123")


class TestGuard:
    def test_a_clean_tree_passes_anywhere(self, clean):
        assert guard_working_tree("cuda").dirty is False
        assert guard_working_tree("cpu").dirty is False

    def test_a_dirty_tree_refuses_on_a_result_bearing_device(self, dirty):
        with pytest.raises(DirtyWorkingTree, match="does not describe the code that runs"):
            guard_working_tree("cuda")

    def test_allow_dirty_is_the_documented_override(self, dirty, capsys):
        assert guard_working_tree("cuda", allow_dirty=True).dirty is True
        assert "WARNING" in capsys.readouterr().err

    def test_a_laptop_run_warns_instead_of_blocking(self, dirty, capsys):
        """Blocking local CPU/MPS debugging buys nothing — the record stays
        honest either way — but it must never pass silently."""
        assert guard_working_tree("mps").dirty is True
        assert "WARNING" in capsys.readouterr().err


class TestRecord:
    def test_carries_the_whole_schema(self, clean):
        record = capture_provenance({"probes": {"seed": 0}}, seed=0, tree=guard_working_tree("cpu"))
        assert set(record) >= {
            "git_hash",
            "git_dirty",
            "config",
            "seed",
            "python",
            "platform",
            "env_lock",
            "timestamp",
        }
        assert record["git_dirty"] is False
        assert "git_diff" not in record

    def test_a_dirty_run_carries_the_diff_inline(self, dirty):
        """Inline, not a path: results get rsynced down from the cluster without
        the tree, so a path would leave the record unreadable where it lands."""
        record = capture_provenance({}, tree=guard_working_tree("mps", allow_dirty=True))
        assert record["git_dirty"] is True
        assert record["git_diff"] == "fake-diff"

    def test_env_lock_pins_the_committed_lockfile(self, clean):
        lock = capture_provenance({}, tree=guard_working_tree("cpu"))["env_lock"]
        assert lock["path"] == "uv.lock"
        assert len(lock["sha256_16"]) == 16

    def test_extra_fields_merge_in(self, clean):
        record = capture_provenance({}, extra={"phase": "phase0", "metrics": {"n": 1}}, tree=guard_working_tree("cpu"))
        assert record["phase"] == "phase0"
        assert record["metrics"] == {"n": 1}

    def test_write_results_creates_the_run_dir(self, clean, tmp_path):
        path = write_results(tmp_path / "runs" / "r1", capture_provenance({"a": 1}, tree=guard_working_tree("cpu")))
        assert path.name == "results.json"
        assert json.loads(path.read_text())["config"] == {"a": 1}


class TestTheHashIsTheStartHashNotTheWriteHash:
    """⚠️ The regression the whole `TreeState` change exists for.

    `capture_provenance` used to read git ITSELF, and every entrypoint calls it
    after the family loop — so the hash described the tree at WRITE time. On runs
    that last hours in a working directory two sessions share, those are
    different commits. Jobs `9010200`/`9010201` recorded `4563807`, a commit that
    did not exist when either started; `9010897`/`9011034`/`9010530` were hit the
    same day by a pull mid-run.

    The test moves the tree BETWEEN the guard and the write, which is exactly
    what happened, and asserts the record still describes the start.
    """

    def test_a_commit_landing_mid_run_does_not_change_the_record(self, monkeypatch):
        import internals_safety.provenance as provenance

        monkeypatch.setattr(provenance, "is_dirty", lambda: False)
        monkeypatch.setattr(provenance, "git_hash", lambda: "aaaaaaa_at_start")
        tree = provenance.guard_working_tree("cuda")

        # ... hours pass, the other session commits, someone pulls ...
        monkeypatch.setattr(provenance, "git_hash", lambda: "bbbbbbb_at_write")
        monkeypatch.setattr(provenance, "is_dirty", lambda: True)

        record = provenance.capture_provenance({}, tree=tree)
        assert record["git_hash"] == "aaaaaaa_at_start"
        assert record["git_dirty"] is False
        assert "git_diff" not in record

    def test_omitting_the_tree_is_a_TypeError_not_a_fresh_read(self):
        """Inexpressible, not documented — the `strata`/`device` fix shape."""
        import inspect

        from internals_safety.provenance import capture_provenance

        parameter = inspect.signature(capture_provenance).parameters["tree"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(TypeError):
            capture_provenance({})

    def test_a_dirty_start_carries_the_START_diff(self, monkeypatch):
        import internals_safety.provenance as provenance

        monkeypatch.setattr(provenance, "is_dirty", lambda: True)
        monkeypatch.setattr(provenance, "git_hash", lambda: "ccccccc")
        monkeypatch.setattr(provenance, "_git", lambda *a: "DIFF AT START")
        tree = provenance.guard_working_tree("mps")

        monkeypatch.setattr(provenance, "_git", lambda *a: "DIFF AT WRITE")
        record = provenance.capture_provenance({}, tree=tree)
        assert record["git_diff"] == "DIFF AT START"
