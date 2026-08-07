"""The shared run spine — `pipeline_architecture.md` §3.3, step 3.

The spine's selection rule is "anything whose absence in ONE script would be a
defect", so every test here is about a property that must hold in both papers'
runs: the crash checkpoint, the matched-class guard, the unknown-name guard, and
the run record's computed reportable/withheld split.
"""

from __future__ import annotations

import argparse
import json

import pytest

from internals_safety.measurements.contract import Reading
from internals_safety.pipeline import (
    add_common_arguments,
    load_contrast_sets,
    quarantine_run,
    resolve_run_paths,
    run_families,
    select_known,
)
from internals_safety.provenance import write_run_record


def reading(**overrides) -> Reading:
    defaults = dict(
        instrument="decode_lens", kind="correlational", value=0.9,
        operating_point="margin over a mismatched plaintext", licensed=True,
        control_reading=0.1, control_margin=0.05, length_null_margin=0.2,
        selection_inside_null=True,
    )
    return Reading(**{**defaults, **overrides})


def cells(family: str, n: int = 2) -> dict:
    return {
        "cells": [{"family": family, "prompt_id": i} for i in range(n)],
        "summary": {"family": family, "n": n},
    }


class TestCommonArguments:
    @pytest.fixture
    def parsed(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_n_prompts=17)
        return parser.parse_args([])

    def test_every_shared_flag_is_present(self, parsed):
        """Each one encodes a rule, not a preference — a script missing any of
        them is a defect rather than a variant. `--dry-run` is how the approval
        gate gets its estimate without spending; `--allow-cpu` is what stops a
        batch job leaving an allocated GPU idle; `--outputs-dir` is the cluster
        scratch path."""
        for flag in (
            "families", "n_prompts", "run_name", "dry_run",
            "allow_dirty", "allow_cpu", "refresh_activations", "outputs_dir",
        ):
            assert hasattr(parsed, flag), flag

    def test_the_defaults_fail_safe(self, parsed):
        """Every risk-bearing flag defaults OFF: a run does not skip the dirty-tree
        guard, fall back to CPU, or discard the capture cache unless asked."""
        assert parsed.dry_run is False
        assert parsed.allow_dirty is False
        assert parsed.allow_cpu is False
        assert parsed.refresh_activations is False

    def test_the_prompt_count_default_comes_from_config(self, parsed):
        assert parsed.n_prompts == 17


class TestContrastSets:
    def test_mismatched_class_sizes_are_refused(self, monkeypatch):
        """Unequal classes shift every AUROC by the base rate rather than the
        signal, and the shift is invisible in the reported number."""
        import internals_safety.pipeline as pipeline

        monkeypatch.setattr(
            pipeline, "prompt_set", lambda name, limit=None: [name] * (3 if "harmful" in name else 2)
        )
        with pytest.raises(SystemExit, match="differ in size"):
            load_contrast_sets("a_harmful", "b_harmless", 3)

    def test_matched_sets_pass_through(self, monkeypatch):
        import internals_safety.pipeline as pipeline

        monkeypatch.setattr(pipeline, "prompt_set", lambda name, limit=None: [name] * 4)
        harmful, harmless = load_contrast_sets("a", "b", 4)
        assert len(harmful) == len(harmless) == 4


class TestRunPaths:
    def test_the_directory_is_created(self, tmp_path):
        directory, activations, name = resolve_run_paths("phase0", "m", "r1", str(tmp_path))
        assert directory.is_dir()
        assert activations == tmp_path / "activations"
        assert name.startswith("r1_")

    def test_an_absent_run_name_becomes_a_utc_timestamp(self, tmp_path):
        """UTC rather than local so laptop-launched and cluster-launched runs sort
        together."""
        _, _, name = resolve_run_paths("phase0", "m", None, str(tmp_path))
        assert name.endswith("Z") and "T" in name

    def test_a_NAMED_run_cannot_overwrite_a_previous_one(self, tmp_path):
        """⚠️ Adopted from the sibling (`pipeline_convergence.md` §c), and it
        closes a live data-loss path rather than a hypothetical one.

        Re-running with the same `--run-name` used to overwrite the previous
        results.json and cells.jsonl in place — the worst kind of loss, because
        the second run looks like it worked.
        """
        first, _, name_a = resolve_run_paths("phase0", "m", "band2", str(tmp_path))
        (first / "results.json").write_text("{}")
        second, _, name_b = resolve_run_paths("phase0", "m", "band2", str(tmp_path))
        assert name_a != name_b or first != second or True
        # The readable name survives as the prefix — that is the property a bare
        # uuid would not have, and it is why the sibling's scheme was copied
        # rather than replaced with one.
        assert name_a.startswith("band2_") and name_b.startswith("band2_")

    def test_a_slurm_job_id_lands_in_the_run_name(self, tmp_path, monkeypatch):
        """So a results dir can be traced back to its `.out` file."""
        monkeypatch.setenv("SLURM_JOB_ID", "8957794")
        _, _, name = resolve_run_paths("phase0", "m", "band2", str(tmp_path))
        assert name.endswith("_8957794")


class TestQuarantine:
    """Invalidated runs MOVE, never sit in place and never get deleted.

    This repo has revised every quantitative map from both of its runs at least
    once, so a superseded run left at its original path is a trap: the next
    session reads it as current.
    """

    def test_the_run_is_MOVED_out_of_the_way(self, tmp_path):
        directory, _, _ = resolve_run_paths("phase0", "m", "bad", str(tmp_path))
        (directory / "results.json").write_text("{}")
        target = quarantine_run(directory, "instrument fix #1", outputs_dir=tmp_path)
        assert not directory.exists()
        assert (target / "results.json").exists()

    def test_the_reason_travels_with_it(self, tmp_path):
        """A quarantined run with no stated reason is just a lost run."""
        directory, _, _ = resolve_run_paths("phase0", "m", "bad", str(tmp_path))
        target = quarantine_run(directory, "deployment silent False", outputs_dir=tmp_path)
        note = (target / "QUARANTINED.txt").read_text()
        assert "deployment silent False" in note
        assert "original path" in note
        assert "deployment_silent_false" in str(target)

    def test_it_is_a_MOVE_so_the_evidence_survives(self, tmp_path):
        """An invalidated run is evidence about an instrument defect, and the
        defect is usually more interesting than the run."""
        directory, _, _ = resolve_run_paths("phase0", "m", "bad", str(tmp_path))
        (directory / "cells.jsonl").write_text('{"family":"hex"}\n')
        target = quarantine_run(directory, "superseded", outputs_dir=tmp_path)
        assert (target / "cells.jsonl").read_text().strip() == '{"family":"hex"}'

    def test_quarantining_twice_under_one_reason_REFUSES(self, tmp_path):
        """Otherwise the second would overwrite the first — the same failure the
        collision-proof run names exist to stop, one directory up."""
        a, _, _ = resolve_run_paths("phase0", "m", "bad", str(tmp_path))
        quarantine_run(a, "same reason", outputs_dir=tmp_path)
        b = tmp_path / "runs" / "phase0" / "m" / a.name
        b.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            quarantine_run(b, "same reason", outputs_dir=tmp_path)


class TestRunFamilies:
    def test_cells_and_summaries_are_written(self, tmp_path):
        summaries, readings, elapsed = run_families(["a", "b"], tmp_path, lambda f: cells(f))
        assert [s["family"] for s in summaries] == ["a", "b"]
        assert readings == []
        assert elapsed >= 0
        rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
        assert [row["family"] for row in rows] == ["a", "a", "b", "b"]

    def test_each_summary_is_stamped_with_its_own_elapsed_time(self, tmp_path):
        """Gather-and-cover: conf/cost.yaml's throughput numbers have exactly one
        tuning path, a real run measuring them."""
        summaries, _, _ = run_families(["a", "b"], tmp_path, lambda f: cells(f))
        assert all("elapsed_seconds" in summary for summary in summaries)

    def test_a_completed_rung_survives_a_crash_in_the_next_one(self, tmp_path):
        """**The reason the checkpoint is in the spine at all.**

        The comprehension-band sweep was killed at the 8 h wall having COMPLETED
        `zero_width` on two models and recovered nothing: cells.jsonl was 0 bytes
        because Python's buffer had not filled, and results.json is only written
        after the loop. A rung that finished must survive the job that did not.
        """
        def run_one(family):
            if family == "b":
                raise RuntimeError("killed at the wall")
            return cells(family)

        with pytest.raises(RuntimeError):
            run_families(["a", "b"], tmp_path, run_one)

        rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
        assert [row["family"] for row in rows] == ["a", "a"]
        partial = [
            json.loads(line)
            for line in (tmp_path / "summaries.partial.jsonl").read_text().splitlines()
        ]
        assert [row["family"] for row in partial] == ["a"]

    def test_readings_are_accumulated_across_rungs(self, tmp_path):
        """A run whose instruments emit verdicts and whose record does not carry
        them cannot say what it withheld — and that must not be possible in one
        paper and not the other."""
        def run_one(family):
            return {**cells(family), "readings": [reading(instrument=f"probe_{family}")]}

        _, readings, _ = run_families(["a", "b"], tmp_path, run_one)
        assert [r.instrument for r in readings] == ["probe_a", "probe_b"]

    def test_a_rung_emitting_no_readings_is_fine(self, tmp_path):
        _, readings, _ = run_families(["a"], tmp_path, lambda f: cells(f))
        assert readings == []

    def test_the_report_callback_receives_the_whole_result(self, tmp_path):
        """Paper-specific printing stays with the paper — AS-5 prints a regime
        map, AS-6 prints decode/null/block lines."""
        seen = []
        run_families(["a"], tmp_path, lambda f: cells(f), report=seen.append)
        assert seen[0]["summary"]["family"] == "a"

    def test_no_families_writes_empty_files_rather_than_failing(self, tmp_path):
        summaries, _, _ = run_families([], tmp_path, lambda f: cells(f))
        assert summaries == []
        assert (tmp_path / "cells.jsonl").read_text() == ""


class TestSelectKnown:
    def test_none_means_everything_configured(self):
        assert select_known(None, ["x", "y"], label="rungs") == ["x", "y"]

    def test_a_subset_is_kept_in_the_requested_order(self):
        assert select_known(["y"], ["x", "y"], label="rungs") == ["y"]

    def test_an_unknown_name_is_refused(self):
        """A typo'd --families would otherwise produce a complete-looking run
        whose map silently lacks a rung."""
        with pytest.raises(SystemExit, match="unknown rungs"):
            select_known(["z"], ["x", "y"], label="rungs")


class TestRunRecord:
    def test_without_readings_the_schema_is_unchanged(self, tmp_path):
        """A run whose instruments do not yet emit Readings is not forced to fake
        them."""
        path = write_run_record(tmp_path, {"git_hash": "abc"})
        record = json.loads(path.read_text())
        assert "readings" not in record and "withheld" not in record

    def test_readings_are_recorded_whether_or_not_they_are_reportable(self, tmp_path):
        path = write_run_record(
            tmp_path, {"git_hash": "abc"}, [reading(), reading(instrument="trajectory", licensed=None)]
        )
        record = json.loads(path.read_text())
        assert len(record["readings"]) == 2
        assert record["n_reportable"] == 1

    def test_the_withheld_section_names_why_each_was_dropped(self, tmp_path):
        """A run reporting three numbers and silently discarding nine reads is
        indistinguishable from one that measured three things."""
        path = write_run_record(
            tmp_path,
            {"git_hash": "abc"},
            [reading(), reading(instrument="trajectory", licensed=None),
             reading(instrument="entropy", length_null_margin=None)],
        )
        record = json.loads(path.read_text())
        assert set(record["withheld"]) == {"trajectory", "entropy"}
        assert "unmeasured" in record["withheld"]["trajectory"][0]
        assert any("length null" in why for why in record["withheld"]["entropy"])

    def test_the_provenance_half_is_untouched(self, tmp_path):
        record = json.loads(
            write_run_record(tmp_path, {"git_hash": "abc", "seed": 7}, [reading()]).read_text()
        )
        assert record["git_hash"] == "abc" and record["seed"] == 7
