"""The decoder-plus-guard baseline, exercised end to end on CPU with stubs.

Written before the run rather than after it, because this repo's record is that
a script whose `main()` no test invokes debugs itself on an H200 after a queue
wait, and that fixing only the first crash buys exactly one more cycle.

The properties that matter here are the ones a green run could still get wrong:
the benign arm cannot be skipped, the repair fraction cannot silently become
zero, and the restatement provenance has to survive into the record.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "decoder_guard_pipeline.py"


def _module():
    spec = importlib.util.spec_from_file_location("decoder_guard_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline = _module()


@dataclass(frozen=True)
class FakePrompt:
    id: str
    text: str


def _write_run(root: Path, model: str, name: str, families, ids, *, benign: bool, ability=True):
    run = root / "phase0" / model / name
    run.mkdir(parents=True, exist_ok=True)
    for arm, filename in (("harmful", "cells.jsonl"), ("benign", "benign_cells.jsonl")):
        if arm == "benign" and not benign:
            continue
        rows = [
            {
                "family": family,
                "prompt_id": pid,
                "restate_response": f"{name}:{arm}:{family}:{pid}",
                "ability": ability,
            }
            for family in families
            for pid in ids
        ]
        (run / filename).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


class TestTheBenignArmCannotBeSkipped:
    def test_missing_benign_restatements_exit_and_name_the_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        _write_run(tmp_path, "m", "r1", ["homoglyph"], [f"p{i}" for i in range(3)], benign=False)
        index, _ = pipeline._restatement_index("m", ["homoglyph"])
        with pytest.raises(SystemExit) as excinfo:
            pipeline._require_both_arms(index, ["homoglyph"], 3, "m")
        message = str(excinfo.value)
        assert "benign/homoglyph: 0 of 3" in message
        assert "comprehension_gap" in message, "must name what would produce the missing data"

    def test_a_short_benign_arm_is_also_refused(self, tmp_path, monkeypatch):
        """Present-but-incomplete is the likelier failure and the easier one to miss."""
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        _write_run(tmp_path, "m", "r1", ["homoglyph"], ["p0", "p1"], benign=True)
        index, _ = pipeline._restatement_index("m", ["homoglyph"])
        with pytest.raises(SystemExit, match="benign/homoglyph: 2 of 3"):
            pipeline._require_both_arms(index, ["homoglyph"], 3, "m")

    def test_both_arms_complete_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        _write_run(tmp_path, "m", "r1", ["homoglyph"], ["p0", "p1", "p2"], benign=True)
        index, _ = pipeline._restatement_index("m", ["homoglyph"])
        pipeline._require_both_arms(index, ["homoglyph"], 3, "m")

    def test_no_command_line_flag_relaxes_it(self):
        """Mutation guard on the design decision, not on the code path.

        The benign requirement is worth nothing if a later session adds
        `--harmful-only` to get a run out the door, so the absence of such a flag
        is asserted rather than left to the docstring.
        """
        import argparse

        parser = argparse.ArgumentParser()
        source = SCRIPT.read_text()
        for banned in ("--harmful-only", "--skip-benign", "--allow-missing", "--no-benign"):
            assert banned not in source, f"{banned} would defeat the fail-closed benign arm"
        assert "raise SystemExit" in source
        del parser


class TestTheRestatementIndex:
    def test_the_newest_run_wins_and_the_source_is_recorded(self, tmp_path, monkeypatch):
        """These runs span instrument fixes, so which one decoded a cell is a fact."""
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        _write_run(tmp_path, "m", "aaa-old", ["homoglyph"], ["p0"], benign=True)
        _write_run(tmp_path, "m", "zzz-new", ["homoglyph"], ["p0"], benign=True)
        index, source = pipeline._restatement_index("m", ["homoglyph"])
        assert index[("harmful", "homoglyph", "p0")]["restate_response"].startswith("zzz-new")
        assert source["harmful"]["homoglyph"] == "zzz-new"
        assert source["benign"]["homoglyph"] == "zzz-new"

    def test_an_unrequested_family_is_not_indexed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        _write_run(tmp_path, "m", "r1", ["homoglyph", "base64"], ["p0"], benign=True)
        index, _ = pipeline._restatement_index("m", ["homoglyph"])
        assert all(family == "homoglyph" for (_, family, _) in index)

    def test_a_zero_byte_benign_file_is_treated_as_absent(self, tmp_path, monkeypatch):
        """Every benign_cells.jsonl AS-6 has on disk is zero bytes, so this is the
        real-world case rather than a hypothetical one."""
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
        run = _write_run(tmp_path, "m", "r1", ["homoglyph"], ["p0"], benign=True)
        (run / "benign_cells.jsonl").write_text("")
        index, _ = pipeline._restatement_index("m", ["homoglyph"])
        assert not any(arm == "benign" for (arm, _, _) in index)


class TestTheCensus:
    def test_three_kinds_two_arms(self):
        assert pipeline._census(guards=1, families=2, n_prompts=100) == 1200

    def test_it_scales_linearly_in_every_term(self):
        base = pipeline._census(guards=1, families=1, n_prompts=10)
        assert pipeline._census(guards=2, families=1, n_prompts=10) == 2 * base
        assert pipeline._census(guards=1, families=3, n_prompts=10) == 3 * base
        assert pipeline._census(guards=1, families=1, n_prompts=50) == 5 * base


class TestEndToEnd:
    """`main()` on stubs. CPU, no weights, no network, sub-second."""

    def _run(self, tmp_path, monkeypatch, verdicts):
        ids = ["p0", "p1", "p2", "p3"]
        monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
        _write_run(tmp_path / "runs", "basemodel", "r1", ["homoglyph"], ids, benign=True)

        prompts = [FakePrompt(i, f"text for {i}") for i in ids]
        monkeypatch.setattr(pipeline, "load_contrast_sets", lambda *a, **k: (prompts, prompts))
        monkeypatch.setattr(pipeline, "guard_working_tree", lambda device, allow_dirty: "TREE")
        monkeypatch.setattr(pipeline, "load_model", lambda config: object())
        monkeypatch.setattr(
            pipeline, "resolve_run_paths", lambda *a, **k: (tmp_path, tmp_path, "run")
        )
        captured: dict = {}

        def fake_provenance(config, seed, extra, *, tree):
            captured["extra"] = extra
            captured["tree"] = tree
            return {"extra": extra}

        monkeypatch.setattr(pipeline, "capture_provenance", fake_provenance)
        monkeypatch.setattr(
            pipeline, "write_run_record", lambda d, r, cells: (captured.update(cells=cells), d)[1]
        )

        calls = {"n": 0}

        def fake_read(loaded, payloads, batch_size):
            calls["n"] += 1
            kind = verdicts.pop(0)
            return ([type("R", (), {"unsafe": flag})() for flag in kind], None)

        monkeypatch.setattr(pipeline, "read_verdicts", fake_read)
        code = pipeline.main(
            ["--guard", "wildguard", "--base-model", "basemodel",
             "--families", "homoglyph", "--n-prompts", "4", "--allow-cpu"]
        )
        return code, captured, calls["n"]

    def test_it_completes_and_records_every_cell(self, tmp_path, monkeypatch):
        # harmful: cipher 1/4 blocked, restatement 4/4, plaintext 4/4;
        # benign: cipher 0/4, restatement 0/4, plaintext 0/4.
        verdicts = [
            [True, False, False, False], [True] * 4, [True] * 4,
            [False] * 4, [False] * 4, [False] * 4,
        ]
        code, captured, n_calls = self._run(tmp_path, monkeypatch, verdicts)
        assert code == 0
        assert n_calls == 6, "three payload kinds on each of two arms"
        assert len(captured["cells"]) == 8, "four prompts on each arm"
        summary = captured["extra"]["summaries"][0]
        assert summary["harmful_repair"] == pytest.approx(0.75)
        assert summary["harmful_headroom"] == pytest.approx(0.75)
        assert summary["harmful_repair_fraction"] == pytest.approx(1.0)
        assert summary["benign_false_positive_cost"] == pytest.approx(0.0)

    def test_the_provenance_of_each_restatement_reaches_the_record(self, tmp_path, monkeypatch):
        verdicts = [[False] * 4 for _ in range(6)]
        _, captured, _ = self._run(tmp_path, monkeypatch, verdicts)
        assert captured["extra"]["decoder"] == "basemodel"
        assert captured["extra"]["restatement_sources"]["benign"]["homoglyph"] == "r1"
        assert captured["tree"] == "TREE", "the dirty-tree guard's result must be threaded through"

    def test_zero_headroom_gives_None_not_a_silent_zero(self, tmp_path, monkeypatch):
        """The defect class this repo has fixed four times, in its own arithmetic.

        A condition the guard already blocks at its plaintext rate has no room to
        improve, so the repair FRACTION is undefined. Returning 0.0 would read as
        'the pipeline did not help here', which is a measurement, when the truth
        is that the quantity does not exist.
        """
        verdicts = [
            [True] * 4, [True] * 4, [True] * 4,
            [False] * 4, [False] * 4, [False] * 4,
        ]
        _, captured, _ = self._run(tmp_path, monkeypatch, verdicts)
        summary = captured["extra"]["summaries"][0]
        assert summary["harmful_headroom"] == pytest.approx(0.0)
        assert summary["harmful_repair_fraction"] is None

    def test_a_pipeline_that_blocks_everything_shows_its_benign_cost(self, tmp_path, monkeypatch):
        """The failure mode the benign arm exists to catch, made visible as a number."""
        verdicts = [
            [False] * 4, [True] * 4, [True] * 4,
            [False] * 4, [True] * 4, [False] * 4,
        ]
        _, captured, _ = self._run(tmp_path, monkeypatch, verdicts)
        summary = captured["extra"]["summaries"][0]
        assert summary["harmful_repair"] == pytest.approx(1.0), "looks like a perfect repair"
        assert summary["benign_false_positive_cost"] == pytest.approx(1.0), (
            "and the benign arm shows it blocked everything, which is the whole point"
        )

    def test_the_decoder_recovery_flag_is_carried_per_prompt(self, tmp_path, monkeypatch):
        """Repair conditioned on an actual decode needs the flag on every cell."""
        verdicts = [[False] * 4 for _ in range(6)]
        _, captured, _ = self._run(tmp_path, monkeypatch, verdicts)
        assert all("decoder_recovered" in cell for cell in captured["cells"])
        assert all(cell["decoder_recovered"] is True for cell in captured["cells"])


class TestThePresetWiring:
    """The schema change, pinned where a future entrypoint would silently skip it."""

    def test_the_guard_target_set_has_exactly_one_home(self):
        """Both `tasks()` and the preset test must read the SAME set.

        They enumerated it separately for one commit, which is how the model-config
        lookup came to reject a guard preset that `tasks()` had rendered correctly.
        """
        from internals_safety.config import GUARD_TARGET_ENTRYPOINTS

        assert "decoder_guard_pipeline" in GUARD_TARGET_ENTRYPOINTS
        assert "as6_guard_probe" in GUARD_TARGET_ENTRYPOINTS
        source = (Path(__file__).resolve().parent.parent / "src" / "internals_safety"
                  / "config.py").read_text()
        assert source.count('"as6_guard_probe", "decoder_guard_pipeline"') <= 1, (
            "a second literal listing of the guard entrypoints is the drift this "
            "constant exists to prevent"
        )

    def test_the_decoder_is_required_and_has_no_default(self):
        """A pipeline with a guessed decoder is a number with no provenance."""
        from pathlib import Path as P

        from internals_safety.config import load_preset

        # Built from a shipped preset so `resources` and every other required
        # field come from a real declaration; only `decoder` is removed.
        shipped = load_preset("decoder_pipeline_wildguard")
        preset = shipped.model_copy(update={"decoder": None})
        with pytest.raises(ValueError, match="needs `decoder:`"):
            preset.tasks(P("outputs"))

    def test_both_shipped_presets_render_the_decoder_flag(self):
        from pathlib import Path as P

        from internals_safety.config import load_preset

        for name, decoder in (
            ("decoder_pipeline_wildguard", "mistral_7b_instruct"),
            ("decoder_pipeline_llama_guard", "llama3_1_8b_instruct"),
        ):
            (argv,) = load_preset(name).tasks(P("outputs"))
            assert "--base-model" in argv
            assert argv[argv.index("--base-model") + 1] == decoder
            assert "--guard" in argv and "--model" not in argv

    def test_the_shipped_presets_only_name_rungs_the_gap_run_covers(self):
        """The fail-closed benign check trips for the WHOLE run, so one uncovered
        rung would waste the job rather than degrade it."""
        from internals_safety.config import load_preset

        for name in ("decoder_pipeline_wildguard", "decoder_pipeline_llama_guard"):
            preset = load_preset(name)
            # Each pipeline is checked against ITS OWN decoder's gap preset, not
            # against one of them: the two decoders are different models and a
            # shared check would pass while one of them was uncovered.
            covered = set(load_preset(f"comprehension_gap_{preset.decoder}").families or [])
            assert covered, f"comprehension_gap_{preset.decoder} must name explicit families"
            assert set(preset.families or []) <= covered, (
                f"{name} names a rung comprehension_gap_{preset.decoder} does not cover; "
                "the fail-closed benign check trips for the whole run, so this wastes the job"
            )
