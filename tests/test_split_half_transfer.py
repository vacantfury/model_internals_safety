"""The item-split test is itself an instrument, so it gets the same two proofs.

`split_half_transfer` decides whether a transfer probe read harm or remembered
the item. That verdict removed AS-5's internals leg, and the same procedure
produces AS-6's decode axis, so the script must be shown to do BOTH things a
screen has to do: fire when the signal is item memory, and stay quiet when the
signal is a genuine direction that survives the transform. A guard that only
ever fires is not a screen, it is a verdict with a script attached.

The synthetic fixtures model the mechanism named in the refutation record -- the
encoded activation of item *i* sits NEAR the plaintext activation of item *i*,
because the model decodes the transform -- rather than an arbitrary transform. A
rotation would be the more "neutral"-looking fixture and would be wrong: it
destroys the linear correspondence a probe would memorise, so the leak case
could not leak and the test would pass on code that cannot detect anything.

The schema fixtures carry the STRICTEST real shapes: an AS-6 record whose
summaries include one with no selected cell, and a plain-capture directory that
can hold more than one candidate file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import split_half_transfer as sht  # noqa: E402

N_ITEMS = 40
D_MODEL = 256  # > 2*N_ITEMS, so the logistic fit interpolates -- the regime that can memorise
N_SPLITS = 30  # enough for a stable mean without making the test slow


def _leaky_pair(seed: int = 0):
    """Labels that carry ONLY item identity, and an encoding that preserves it."""
    rng = np.random.default_rng(seed)
    codes_h = rng.normal(size=(N_ITEMS, D_MODEL))
    codes_b = rng.normal(size=(N_ITEMS, D_MODEL))
    # A shared "wearing an encoding" offset: real, and carries no item identity.
    surface = rng.normal(size=D_MODEL) * 2.0
    enc_h = codes_h + 0.5 * rng.normal(size=(N_ITEMS, D_MODEL)) + surface
    enc_b = codes_b + 0.5 * rng.normal(size=(N_ITEMS, D_MODEL)) + surface
    return codes_h, codes_b, enc_h, enc_b


def _genuine_pair(seed: int = 1):
    """A class direction present in both conditions and tied to no particular item."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=D_MODEL)
    direction /= np.linalg.norm(direction)
    surface = rng.normal(size=D_MODEL) * 2.0
    plain_h = rng.normal(size=(N_ITEMS, D_MODEL)) + 3.0 * direction
    plain_b = rng.normal(size=(N_ITEMS, D_MODEL))
    enc_h = rng.normal(size=(N_ITEMS, D_MODEL)) + 3.0 * direction + surface
    enc_b = rng.normal(size=(N_ITEMS, D_MODEL)) + surface
    return plain_h, plain_b, enc_h, enc_b


class TestTheScreenFiresOnItemMemory:
    def test_item_identity_alone_reads_near_ceiling_without_a_split(self):
        out = sht.analyse(*_leaky_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        # This is the number a run would have recorded and a paper would have quoted.
        assert out["A_reproduce_no_split"] > 0.95

    def test_and_collapses_to_chance_once_the_items_are_held_out(self):
        out = sht.analyse(*_leaky_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert out["B_split_logistic"]["mean"] < 0.65
        assert out["B_split_logistic"]["p97_5"] < 0.80

    def test_the_fixed_n_estimate_separates_memory_from_a_smaller_training_set(self):
        # B halves the training set as well as holding out items, so A - B alone
        # cannot attribute the drop. F - B scores the SAME probe at the SAME size
        # on items it saw, so what remains is memory.
        out = sht.analyse(*_leaky_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert out["leakage_at_fixed_n_logistic"] > 0.20

    def test_difference_in_means_leaks_too_so_it_is_not_a_capacity_artefact(self):
        out = sht.analyse(*_leaky_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert out["leakage_at_fixed_n_dim"] > 0.20


class TestTheScreenIsQuietOnASurvivingDirection:
    """Specificity. Without this, every one of the tests above passes on
    `return {"B_split_logistic": {"mean": 0.5}}`."""

    def test_a_transferable_direction_survives_the_item_split(self):
        out = sht.analyse(*_genuine_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert out["B_split_logistic"]["mean"] > 0.80

    def test_and_reports_essentially_no_leakage(self):
        out = sht.analyse(*_genuine_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert out["leakage_at_fixed_n_logistic"] < 0.10

    def test_the_two_cases_are_distinguished_by_the_split_not_by_the_raw_reading(self):
        # Both read high WITHOUT the split; only the split tells them apart. This
        # is the whole argument for why the recorded numbers could not self-report.
        leaky = sht.analyse(*_leaky_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        genuine = sht.analyse(*_genuine_pair(), n_splits=N_SPLITS, seed=0, blocked=None, reading_percentile=75.0)
        assert leaky["A_reproduce_no_split"] > 0.90
        assert genuine["A_reproduce_no_split"] > 0.90
        assert genuine["B_split_logistic"]["mean"] - leaky["B_split_logistic"]["mean"] > 0.25


class TestTheItemSplitRequiresAlignedConditions:
    def test_condition_sets_of_different_length_are_refused(self):
        plain_h, plain_b, enc_h, enc_b = _leaky_pair()
        with pytest.raises(ValueError, match="would not be aligned"):
            sht.analyse(
                plain_h, plain_b, enc_h[:-1], enc_b,
                n_splits=2, seed=0, blocked=None, reading_percentile=75.0,
            )


# --- schema dispatch --------------------------------------------------------


def _write_captures(directory: Path, *stems: str) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for stem in stems:
        path = directory / f"{stem}-0123456789abcdef.pt"
        path.write_bytes(b"")
        written[stem] = str(path)
    return written


def _as6_record(directory: Path) -> dict:
    captures = _write_captures(
        directory,
        "plain-harmful",
        "plain-harmless",
        "encoded-harmful-homoglyph",
        "encoded-harmless-homoglyph",
        "encoded-harmful-base64",
        "encoded-harmless-base64",
    )
    return {
        "config": {"guard": {"name": "llama_guard_3_8b"}},
        "summaries": [
            {
                "family": "homoglyph",
                "decode": {"layer": 13, "position": "instruction_final"},
                "activations": {
                    "encoded_harmful": captures["encoded-harmful-homoglyph"],
                    "encoded_harmless": captures["encoded-harmless-homoglyph"],
                },
            },
            {
                # A rung the run could not place a cell on. Skipped, never guessed.
                "family": "base64",
                "decode": {"layer": None, "position": None},
                "activations": {
                    "encoded_harmful": captures["encoded-harmful-base64"],
                    "encoded_harmless": captures["encoded-harmless-base64"],
                },
            },
        ],
    }


def _as5_record() -> dict:
    return {
        "config": {"model": {"name": "tulu3_8b", "hf_id": "allenai/Llama-3.1-Tulu-3-8B"}},
        "readings": [
            {"name": "ability", "detail": {"rate": 1.0}},
            {
                "name": "deployment",
                "detail": {"family": "fullwidth", "layer": 15, "position": "instruction_final"},
            },
        ],
        "activations_path": {
            "plain_harmful": "/scratch/x/plain-harmful-aaaa.pt",
            "plain_harmless": "/scratch/x/plain-harmless-bbbb.pt",
            "per_family": {
                "fullwidth": {
                    "encoded_harmful": "/scratch/x/encoded-harmful-fullwidth-cccc.pt",
                    "encoded_harmless": "/scratch/x/encoded-harmless-fullwidth-dddd.pt",
                }
            },
        },
    }


class TestBothRunSchemasAreReadable:
    def test_the_as5_record_yields_the_one_cell_it_selected(self, tmp_path):
        run = tmp_path / "results.json"
        run.write_text(json.dumps(_as5_record()))
        targets = sht.targets_from_record(run)
        assert len(targets) == 1
        assert (targets[0].model, targets[0].family) == ("tulu3_8b", "fullwidth")
        assert (targets[0].layer, targets[0].position) == (15, "instruction_final")
        # The plain pair is RECORDED in this schema, so it is read, not globbed.
        assert targets[0].plain_harmful == "/scratch/x/plain-harmful-aaaa.pt"

    def test_the_as6_record_yields_one_target_per_family_with_a_cell(self, tmp_path):
        run = tmp_path / "results.json"
        run.write_text(json.dumps(_as6_record(tmp_path / "activations")))
        targets = sht.targets_from_record(run)
        assert [t.family for t in targets] == ["homoglyph"]
        assert targets[0].model == "llama_guard_3_8b"
        assert (targets[0].layer, targets[0].position) == (13, "instruction_final")
        assert targets[0].plain_harmful.endswith(".pt")

    def test_an_as6_record_with_no_cells_at_all_is_refused_not_returned_empty(self, tmp_path):
        record = _as6_record(tmp_path / "activations")
        for summary in record["summaries"]:
            summary["decode"] = {"layer": None, "position": None}
        run = tmp_path / "results.json"
        run.write_text(json.dumps(record))
        with pytest.raises(SystemExit, match="no summary carries"):
            sht.targets_from_record(run)

    def test_a_guard_record_that_ALSO_carries_readings_takes_the_guard_branch(self, tmp_path):
        """The strictest real record, and the one that killed a cluster job.

        `guard-causal-*` runs emit an unnamed causal-licensing entry under
        `readings` AND the per-family decode cells under `summaries`. A
        presence-ordered check reads `readings`, finds no (layer, position), and
        refuses a record that is perfectly readable. The paper's own Llama Guard
        table comes from exactly such a record.
        """
        record = _as6_record(tmp_path / "activations")
        record["readings"] = [{"name": None, "detail": {"licensed": False}}]
        run = tmp_path / "results.json"
        run.write_text(json.dumps(record))
        targets = sht.targets_from_record(run)
        assert [t.family for t in targets] == ["homoglyph"]

    def test_a_record_matching_BOTH_schemas_is_refused_rather_than_guessed(self, tmp_path):
        record = _as6_record(tmp_path / "activations")
        record["activations_path"] = _as5_record()["activations_path"]
        run = tmp_path / "results.json"
        run.write_text(json.dumps(record))
        with pytest.raises(SystemExit, match="BOTH schemas"):
            sht.targets_from_record(run)

    def test_an_unrecognised_schema_is_refused(self, tmp_path):
        run = tmp_path / "results.json"
        run.write_text(json.dumps({"config": {}, "elapsed_seconds": 1.0}))
        with pytest.raises(SystemExit, match="unrecognised run schema"):
            sht.targets_from_record(run)


class TestThePlainCaptureIsResolvedFailClosed:
    def test_a_missing_plain_capture_is_refused(self, tmp_path):
        directory = tmp_path / "activations"
        directory.mkdir()
        with pytest.raises(SystemExit, match="found 0"):
            sht._sole(directory, "plain-harmful")

    def test_an_AMBIGUOUS_plain_capture_is_refused_rather_than_picked(self, tmp_path):
        # Two corpora captured into one directory. Taking either would fit the
        # probe on prompts the encoded set was not derived from -- silently.
        directory = tmp_path / "activations"
        directory.mkdir()
        (directory / "plain-harmful-1111.pt").write_bytes(b"")
        (directory / "plain-harmful-2222.pt").write_bytes(b"")
        with pytest.raises(SystemExit, match="found 2"):
            sht._sole(directory, "plain-harmful")


class TestTheModelNameSurvivesBothRecordShapes:
    def test_a_config_block_resolves_to_its_name(self):
        assert sht._model_name({"name": "tulu3_8b", "hf_id": "allenai/x"}) == "tulu3_8b"

    def test_a_block_without_a_name_falls_back_to_the_hf_id(self):
        assert sht._model_name({"hf_id": "allenai/x"}) == "allenai/x"

    def test_a_bare_string_is_passed_through(self):
        assert sht._model_name("llama_guard_3_8b") == "llama_guard_3_8b"


# ---------------------------------------------------------------------------
# TODO 96 — the per-prompt recount, and the join it rests on
# ---------------------------------------------------------------------------


class TestTheCellRecount:
    """Table 2's two cells, recomputed inside the fold.

    The AUROC half of this screen was already held out; the COUNTS were not, and
    they are what the paper reports. Both terms of the per-prompt rule are
    contaminated the same way and in the same direction — item memory raises the
    seen harmful score and lowers the benign threshold — so the fix has to
    re-derive the threshold inside the fold too, which is the assertion below
    that a partial fix would fail.
    """

    def test_no_verdicts_is_UNMEASURED_never_a_zero_cell(self):
        out = sht.analyse(
            *_genuine_pair(), n_splits=8, seed=0, blocked=None, reading_percentile=75.0
        )
        assert out["cells_per_100"] is None

    def test_the_cells_are_reported_when_verdicts_are_present(self):
        plain_h, plain_b, enc_h, enc_b = _genuine_pair()
        blocked = np.zeros(len(enc_h), dtype=bool)
        blocked[: len(enc_h) // 2] = True
        out = sht.analyse(
            plain_h, plain_b, enc_h, enc_b,
            n_splits=32, seed=0, blocked=blocked, reading_percentile=75.0,
        )
        cells = out["cells_per_100"]
        assert cells["n_blocked"] == int(blocked.sum())
        for name in ("decoded_not_blocked", "blocked_without_decoding"):
            assert 0.0 <= cells["unsplit"][name] <= 100.0
            assert 0.0 <= cells["split"][name]["mean"] <= 100.0
            assert cells["split"][name]["n_splits"] == 32

    def test_a_guard_that_blocks_EVERYTHING_has_an_empty_D_and_not_B_cell(self):
        # The conjunction's other term taken to its limit. If this ever reads
        # non-zero the boolean logic is inverted somewhere, which is the one
        # failure a range assertion cannot see.
        plain_h, plain_b, enc_h, enc_b = _genuine_pair()
        out = sht.analyse(
            plain_h, plain_b, enc_h, enc_b,
            n_splits=8, seed=0, blocked=np.ones(len(enc_h), dtype=bool),
            reading_percentile=75.0,
        )
        assert out["cells_per_100"]["split"]["decoded_not_blocked"]["mean"] == 0.0
        assert out["cells_per_100"]["unsplit"]["decoded_not_blocked"] == 0.0

    def test_a_guard_that_blocks_NOTHING_has_an_empty_blocked_without_decoding_cell(self):
        plain_h, plain_b, enc_h, enc_b = _genuine_pair()
        out = sht.analyse(
            plain_h, plain_b, enc_h, enc_b,
            n_splits=8, seed=0, blocked=np.zeros(len(enc_h), dtype=bool),
            reading_percentile=75.0,
        )
        assert out["cells_per_100"]["split"]["blocked_without_decoding"]["mean"] == 0.0

    def test_the_recount_MOVES_on_a_leaky_probe_and_holds_on_a_genuine_one(self):
        """The screen's whole claim, at the level of the counts.

        On a probe reading item identity the held-out decode rate collapses, so
        `decoded_not_blocked` must fall away from its unsplit value. On a probe
        reading a transferable direction it must not. A recount that moved on
        both would be measuring the smaller training set, not the memory.
        """
        def cells(pair):
            plain_h, plain_b, enc_h, enc_b = pair
            blocked = np.zeros(len(enc_h), dtype=bool)
            out = sht.analyse(
                plain_h, plain_b, enc_h, enc_b,
                n_splits=64, seed=0, blocked=blocked, reading_percentile=75.0,
            )
            c = out["cells_per_100"]
            return c["unsplit"]["decoded_not_blocked"], c["split"]["decoded_not_blocked"]["mean"]

        leaky_unsplit, leaky_split = cells(_leaky_pair())
        genuine_unsplit, genuine_split = cells(_genuine_pair())
        assert leaky_unsplit - leaky_split > genuine_unsplit - genuine_split


class TestTheJoinIsCheckedNotAssumed:
    """`cells.jsonl` and the `.pt` are joined by row order and nothing else.

    `ActivationBatch` carries no ids and the cells file carries no row index, so
    a misjoin cannot be noticed downstream: every prompt gets a verdict and the
    table comes out complete and plausible. These pin that the join is verified
    against content and refuses rather than guessing.
    """

    @staticmethod
    def _cells_file(tmp_path, ciphertexts, family="homoglyph"):
        path = tmp_path / "cells.jsonl"
        with path.open("w") as handle:
            for index, text in enumerate(ciphertexts):
                handle.write(json.dumps({
                    "family": family, "ciphertext": text, "blocked": index % 2 == 0,
                }) + "\n")
        return path

    def test_an_aligned_join_returns_the_verdicts_in_order(self, tmp_path):
        texts = [f"cipher-{i}" for i in range(6)]
        path = self._cells_file(tmp_path, texts)
        got = sht.blocked_for_family(str(path), "homoglyph", [f"prompt {t} end" for t in texts])
        assert list(got) == [True, False, True, False, True, False]

    def test_a_SHUFFLED_cells_file_raises_rather_than_mis_assigning(self, tmp_path):
        texts = [f"cipher-{i}" for i in range(6)]
        path = self._cells_file(tmp_path, list(reversed(texts)))
        with pytest.raises(SystemExit, match="not in the same order"):
            sht.blocked_for_family(str(path), "homoglyph", [f"prompt {t} end" for t in texts])

    def test_a_LENGTH_mismatch_raises(self, tmp_path):
        path = self._cells_file(tmp_path, [f"cipher-{i}" for i in range(6)])
        with pytest.raises(SystemExit, match="disagree about the arm's size"):
            sht.blocked_for_family(str(path), "homoglyph", ["prompt cipher-0 end"])

    def test_a_missing_or_absent_file_is_None_not_an_empty_verdict_set(self, tmp_path):
        assert sht.blocked_for_family(None, "homoglyph", []) is None
        assert sht.blocked_for_family(str(tmp_path / "nope.jsonl"), "homoglyph", []) is None

    def test_a_family_absent_from_the_file_is_None(self, tmp_path):
        path = self._cells_file(tmp_path, ["cipher-0"], family="zero_width")
        assert sht.blocked_for_family(str(path), "homoglyph", ["prompt cipher-0 end"]) is None
