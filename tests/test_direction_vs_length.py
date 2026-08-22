"""Tests for `scripts/direction_vs_length.py` — TODO 28's offline comparison.

The property that matters most here is HELD-OUT-NESS. This script exists because
a fitted direction scored on the items it was fitted from withdrew AS-5's
internals leg on 2026-08-21 (`probe_transfer`, reported 0.938-0.995, held out
0.618-0.811). A script written to investigate that class of confound is exactly
where the same defect would be least expected and most embarrassing, so the
no-signal test below is the point of this file rather than an extra.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.directions import difference_in_means

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "direction_vs_length", ROOT / "scripts" / "direction_vs_length.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


def batch(tensor: torch.Tensor, messages: list[str]) -> ActivationBatch:
    n_layers, n_positions = tensor.shape[1], tensor.shape[2]
    return ActivationBatch(
        tensor=tensor,
        layers=list(range(n_layers)),
        positions=["instruction_final", "last"][:n_positions],
        site="resid_pre",
        model_name="test",
        user_messages=messages,
    )


def noise(n: int, seed: int, d: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 1, d, generator=g)


class TestTheDirectionAurocIsHeldOut:
    def test_pure_noise_does_not_separate(self, script):
        """No signal exists, so a probe that never saw the item must read ~chance.

        An IN-SAMPLE difference-in-means on two noise clouds is reliably above
        chance at this n, because the mean difference is fitted to the very
        samples it then scores. This test is the mutation guard: point the fit at
        the full set instead of the training folds and it fails.
        """
        pos, neg = batch(noise(60, 1), [""] * 60), batch(noise(60, 2), [""] * 60)
        auroc, _, _, _ = script.held_out_direction_auroc(pos, neg, 0, "instruction_final", 5, 0)
        assert 0.30 < auroc < 0.70, f"held-out noise separated at {auroc}"

    def test_the_in_sample_version_would_have_passed_a_weaker_bar(self, script):
        """Pins that the previous test is actually load-bearing.

        If in-sample scoring were also ~0.5 here, the test above would pass under
        the defect it exists to catch and would be decoration.
        """
        pos, neg = batch(noise(60, 1), [""] * 60), batch(noise(60, 2), [""] * 60)
        direction = difference_in_means(pos, neg, 0, "instruction_final")
        scores = np.concatenate(
            [
                direction.project(pos.select(0, "instruction_final")).numpy(),
                direction.project(neg.select(0, "instruction_final")).numpy(),
            ]
        )
        labels = np.concatenate([np.ones(60), np.zeros(60)])
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score(labels, scores) > 0.75, "in-sample noise fit is not inflated here"

    def test_real_signal_is_still_recovered(self, script):
        """Held-out must not mean blind: a genuine offset reads near ceiling."""
        pos = batch(noise(60, 3) + 3.0, [""] * 60)
        neg = batch(noise(60, 4), [""] * 60)
        auroc, _, _, _ = script.held_out_direction_auroc(pos, neg, 0, "instruction_final", 5, 0)
        assert auroc > 0.95


class TestDegenerateCellsAreCoverageNotExceptions:
    def test_identical_classes_do_not_crash(self, script):
        """`difference_in_means` documents that a zero-norm cell must be discarded
        by the caller. Item 28 already paid for the version where it raised four
        frames down inside a forward hook and took the whole run with it."""
        same = torch.ones(20, 1, 1, 8)
        auroc, raw_norm, degenerate, _ = script.held_out_direction_auroc(
            batch(same, [""] * 20), batch(same.clone(), [""] * 20), 0, "instruction_final", 5, 0
        )
        assert np.isnan(auroc)
        assert raw_norm == pytest.approx(0.0)
        assert degenerate == 5


class TestLengthExposure:
    def test_constant_scores_return_nan_not_a_correlation(self, script):
        """Guarding only the lengths let a constant-score cell reach spearmanr."""
        assert np.isnan(script.length_rho(np.ones(10), np.arange(10, dtype=float)))

    def test_constant_lengths_return_nan(self, script):
        assert np.isnan(script.length_rho(np.arange(10, dtype=float), np.ones(10)))

    def test_a_confound_running_either_way_scores_the_same(self, script):
        """Absolute value, matching `length_auroc`'s two-sided construction: a
        negative correlation is exactly as exploitable as a positive one."""
        lengths = np.arange(20, dtype=float)
        assert script.length_rho(lengths, lengths) == pytest.approx(
            script.length_rho(-lengths, lengths)
        )

    def test_nan_scores_are_excluded_rather_than_propagated(self, script):
        scores = np.arange(20, dtype=float)
        scores[:3] = np.nan
        assert script.length_rho(scores, np.arange(20, dtype=float)) == pytest.approx(1.0)


class TestSubsetKeepsFieldsConsistent:
    def test_messages_follow_the_tensor(self, script):
        source = batch(noise(6, 5), [f"m{i}" for i in range(6)])
        taken = script.subset(source, np.array([4, 1]))
        assert taken.user_messages == ["m4", "m1"]
        assert taken.tensor.shape[0] == 2
        assert torch.equal(taken.tensor[0], source.tensor[4])
