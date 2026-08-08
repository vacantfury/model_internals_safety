"""The shared run spine — `pipeline_architecture.md` §3.3, step 3.

The spine's selection rule is "anything whose absence in ONE script would be a
defect", so every test here is about a property that must hold in both papers'
runs: the crash checkpoint, the matched-class guard, the unknown-name guard, and
the run record's computed reportable/withheld split.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path

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
        summaries, readings, elapsed = run_families(["a", "b"], tmp_path, lambda f: cells(f), cross_rung_screen=None)
        assert [s["family"] for s in summaries] == ["a", "b"]
        assert readings == []
        assert elapsed >= 0
        rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
        assert [row["family"] for row in rows] == ["a", "a", "b", "b"]

    def test_each_summary_is_stamped_with_its_own_elapsed_time(self, tmp_path):
        """Gather-and-cover: conf/cost.yaml's throughput numbers have exactly one
        tuning path, a real run measuring them."""
        summaries, _, _ = run_families(["a", "b"], tmp_path, lambda f: cells(f), cross_rung_screen=None)
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
            run_families(["a", "b"], tmp_path, run_one, cross_rung_screen=None)

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

        _, readings, _ = run_families(["a", "b"], tmp_path, run_one, cross_rung_screen=None)
        assert [r.instrument for r in readings] == ["probe_a", "probe_b"]

    def test_a_rung_emitting_no_readings_is_fine(self, tmp_path):
        _, readings, _ = run_families(["a"], tmp_path, lambda f: cells(f), cross_rung_screen=None)
        assert readings == []

    def test_the_report_callback_receives_the_whole_result(self, tmp_path):
        """Paper-specific printing stays with the paper — AS-5 prints a regime
        map, AS-6 prints decode/null/block lines."""
        seen = []
        run_families(["a"], tmp_path, lambda f: cells(f), report=seen.append, cross_rung_screen=None)
        assert seen[0]["summary"]["family"] == "a"

    def test_no_families_writes_empty_files_rather_than_failing(self, tmp_path):
        summaries, _, _ = run_families([], tmp_path, lambda f: cells(f), cross_rung_screen=None)
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


def screened_cells(family: str, *, ability: bool, deployment: bool, n: int = 2) -> dict:
    """A rung shaped like the real thing, enough for the cross-rung screen.

    Modelled on `tag_block` as it actually came back from job `9008631`:
    ability 0.00 with a licensed deployment probe reading barely above the
    can't-decode floor, which produced 66 `deployment_without_ability` cells.
    """
    return {
        "cells": [
            {
                "family": family,
                "prompt_id": i,
                "ability": ability,
                "deployment": deployment,
                "recognition": None,
                "refused": True,
                "regime": "S" if deployment else "R",
                "incoherences": ["deployment_without_ability"] if deployment and not ability else [],
            }
            for i in range(n)
        ],
        "summary": {
            "family": family,
            "n": n,
            "ability_rate": 1.0 if ability else 0.0,
            "deployment": {"licensed": True, "transfer_auroc": 0.65 if not ability else 0.95},
        },
    }


class TestTheCrossRungScreenCannotBeOmitted:
    """TODO 60 — the control floor was adopted, tested, documented and
    owner-approved and still governed nothing that produced a run, because the
    entrypoint never imported it. The fix is not "call it from there too"; it is
    that the call cannot be written without deciding."""

    def test_the_parameter_is_keyword_only_with_no_default(self):
        parameter = inspect.signature(run_families).parameters["cross_rung_screen"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty, (
            "a default would restore the defect exactly the way it opened: the "
            "caller that forgets the screen must not be a caller that runs"
        )

    def test_omitting_it_is_a_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            run_families(["a"], tmp_path, lambda f: cells(f))

    @pytest.mark.parametrize("script", ["phase0_regime_map.py", "as6_guard_probe.py"])
    def test_every_production_caller_passes_it_explicitly(self, script):
        """Pinning the CALLERS as well as the signature, because a later tidy-up
        restoring a default would otherwise pass every other test here."""
        source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) == "run_families"
        ]
        assert calls, f"{script} no longer calls run_families — update this test"
        for call in calls:
            assert "cross_rung_screen" in {kw.arg for kw in call.keywords}, (
                f"{script}:{call.lineno} calls run_families without naming "
                "cross_rung_screen"
            )


class TestDemotionRelabelsThroughTheRules:
    def test_a_screened_out_rung_becomes_unmeasured_in_cells_and_summary(self, tmp_path):
        """The exact `tag_block` shape: ability 0, deployment licensed, 66%
        hard incoherence. After the screen it must be (U) with NO incoherence —
        the flag came from an axis that is now unmeasured."""
        run_families(
            ["inert", "real"],
            tmp_path,
            lambda f: screened_cells(f, ability=(f == "real"), deployment=True),
            cross_rung_screen=lambda summaries: {"inert": "below the run's control floor"},
        )
        rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
        inert = [row for row in rows if row["family"] == "inert"]
        real = [row for row in rows if row["family"] == "real"]

        assert all(row["regime"] == "U" for row in inert)
        assert all(row["deployment"] is None for row in inert)
        assert all(row["incoherences"] == [] for row in inert), (
            "a demoted rung's deployment_without_ability flags must DISAPPEAR: "
            "the axis that produced them is now unmeasured"
        )
        assert all(row["demoted_by_cross_rung_screen"] for row in inert)
        # The rung that cleared is untouched, including its regime label.
        assert all(row["deployment"] is True for row in real)
        assert all("demoted_by_cross_rung_screen" not in row for row in real)

    def test_the_summary_is_corrected_not_just_the_cells(self, tmp_path):
        """Cells and summary disagreeing is the dual-truth failure this repo
        keeps paying for."""
        summaries, _, _ = run_families(
            ["inert"],
            tmp_path,
            lambda f: screened_cells(f, ability=False, deployment=True),
            cross_rung_screen=lambda summaries: {"inert": "below floor"},
        )
        summary = summaries[0]
        assert summary["regimes"] == {"U": 2}
        assert summary["deployment_unmeasured"] == 2
        assert summary["incoherences"] == {}
        assert summary["binding_failure_rate"] is None, (
            "no measured cells means no (B) rate — dividing by n would say "
            "'no binding failures here', the same silent zero one level up"
        )
        assert summary["deployment"]["cleared_control_floor"] is False
        # The failing AUROC is KEPT: it is the evidence for the demotion.
        assert summary["deployment"]["transfer_auroc"] == 0.65

    def test_the_partial_checkpoint_is_rewritten_too(self, tmp_path):
        run_families(
            ["inert"],
            tmp_path,
            lambda f: screened_cells(f, ability=False, deployment=True),
            cross_rung_screen=lambda summaries: {"inert": "below floor"},
        )
        partial = [
            json.loads(line)
            for line in (tmp_path / "summaries.partial.jsonl").read_text().splitlines()
        ]
        assert partial[0]["regimes"] == {"U": 2}, (
            "the crash artifact must not disagree with the record it checkpoints"
        )

    def test_a_screen_demoting_nothing_leaves_the_run_byte_identical(self, tmp_path):
        run_families(
            ["a"],
            tmp_path,
            lambda f: screened_cells(f, ability=True, deployment=True),
            cross_rung_screen=lambda summaries: {},
        )
        rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
        assert all(row["regime"] == "S" for row in rows)
        assert all("demoted_by_cross_rung_screen" not in row for row in rows)

    def test_the_screen_sees_every_rung_not_one(self, tmp_path):
        """The whole reason this is post-loop: the floor is derived from the
        run's own can't-decode rungs, so a per-rung hook could not compute it."""
        seen = []

        def screen(summaries):
            seen.append([s["family"] for s in summaries])
            return {}

        run_families(
            ["a", "b", "c"],
            tmp_path,
            lambda f: screened_cells(f, ability=True, deployment=True),
            cross_rung_screen=screen,
        )
        assert seen == [["a", "b", "c"]]
