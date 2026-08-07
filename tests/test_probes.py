"""Probe-layer tests on synthetic activations.

Synthetic on purpose: these assert that the *instruments* behave as claimed —
a separable signal is found, an absent one is not, and a shuffled-label control
sits at chance. Whether a real model separates harmful from harmless is an
empirical question for the pilot, not something a test can or should pin.

The two that matter most are `test_probe_transfer_*`: measurement #2 is entirely
built on fit-here-evaluate-there, and if transfer reported signal on structureless
activations, every (D) cell in the paper would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from internals_safety.config import ProbeConfig
from internals_safety.measurements.deployment import measure_deployment
from internals_safety.measurements.recognition import (
    HARMFULNESS_POSITION,
    REFUSAL_POSITION,
    measure_recognition,
)
from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.linear import (
    permutation_null_max_auroc,
    permutation_p_value,
)
from internals_safety.probes.directions import (
    cosine_similarity,
    difference_in_means,
    projection_scores,
    sweep_directions,
)
from internals_safety.probes import linear
from internals_safety.probes.linear import probe_sweep, probe_transfer
from internals_safety.probes.overlap import (
    decision_threshold,
    overlap_coefficient,
    pairwise_overlap,
    summarize_projections,
)

LAYERS = [0, 1]
POSITIONS = [HARMFULNESS_POSITION, REFUSAL_POSITION]
D_MODEL = 16
N = 60
CONFIG = ProbeConfig(seed=0, test_fraction=0.3, auroc_threshold=0.70)


def make_batch(tensor: torch.Tensor) -> ActivationBatch:
    return ActivationBatch(
        tensor=tensor,
        layers=list(LAYERS),
        positions=list(POSITIONS),
        site="resid_pre",
        model_name="synthetic",
        user_messages=[f"prompt {index}" for index in range(tensor.shape[0])],
    )


def cluster(offset: float, generator: torch.Generator) -> ActivationBatch:
    """A batch whose first feature is shifted by `offset` at every cell."""
    tensor = torch.randn(N, len(LAYERS), len(POSITIONS), D_MODEL, generator=generator)
    tensor[..., 0] += offset
    return make_batch(tensor)


def wide_cluster(offset: float, generator: torch.Generator) -> ActivationBatch:
    """`cluster` at the pilot's real per-class sample size.

    N=60 is too small to separate "significant" from "strong": the null over the
    max sits high enough that anything detectable also clears 0.70, so the two
    concepts cannot be told apart at that size.
    """
    wide_n = 200
    tensor = torch.randn(wide_n, len(LAYERS), len(POSITIONS), D_MODEL, generator=generator)
    tensor[..., 0] += offset
    return ActivationBatch(
        tensor=tensor,
        layers=list(LAYERS),
        positions=list(POSITIONS),
        site="resid_pre",
        model_name="synthetic",
        user_messages=[f"prompt {index}" for index in range(wide_n)],
    )


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


class TestDirections:
    def test_direction_points_from_negative_to_positive(self, generator):
        positive = cluster(4.0, generator)
        negative = cluster(-4.0, generator)
        direction = difference_in_means(positive, negative, layer=0, position=POSITIONS[0])

        assert direction.vector.shape == (D_MODEL,)
        assert pytest.approx(1.0, abs=1e-5) == float(direction.vector.norm())
        # The planted signal lives in feature 0 and nowhere else.
        assert direction.vector[0] > 0.9
        assert direction.raw_norm > 1.0
        assert direction.n_positive == N and direction.n_negative == N

    def test_projections_separate_the_classes(self, generator):
        positive = cluster(4.0, generator)
        negative = cluster(-4.0, generator)
        direction = difference_in_means(positive, negative, layer=0, position=POSITIONS[0])

        assert projection_scores(positive, direction).mean() > projection_scores(
            negative, direction
        ).mean()

    def test_identical_classes_give_a_near_zero_raw_norm(self, generator):
        """A direction with no separation behind it is numerical noise pointing
        somewhere arbitrary; raw_norm is how a caller detects that."""
        shared = cluster(0.0, generator)
        direction = difference_in_means(shared, shared, layer=0, position=POSITIONS[0])
        assert direction.raw_norm == pytest.approx(0.0, abs=1e-5)
        assert torch.allclose(direction.vector, torch.zeros(D_MODEL))

    def test_sweep_covers_every_cell(self, generator):
        positive, negative = cluster(3.0, generator), cluster(-3.0, generator)
        directions = sweep_directions(positive, negative)
        assert len(directions) == len(LAYERS) * len(POSITIONS)
        assert {(d.layer, d.position) for d in directions} == {
            (layer, position) for layer in LAYERS for position in POSITIONS
        }

    def test_cosine_of_a_direction_with_itself_is_one(self, generator):
        positive, negative = cluster(3.0, generator), cluster(-3.0, generator)
        direction = difference_in_means(positive, negative, layer=0, position=POSITIONS[0])
        assert cosine_similarity(direction, direction) == pytest.approx(1.0, abs=1e-5)

    def test_mismatched_capture_grids_are_rejected(self, generator):
        positive = cluster(1.0, generator)
        odd = make_batch(torch.randn(N, len(LAYERS), len(POSITIONS), D_MODEL))
        odd.layers = [5, 6]
        with pytest.raises(ValueError, match="same layers"):
            sweep_directions(positive, odd)


class TestLinearProbes:
    def test_separable_data_reads_signal_and_the_control_does_not(self, generator):
        results = probe_sweep(cluster(4.0, generator), cluster(-4.0, generator), CONFIG)
        assert len(results) == len(LAYERS) * len(POSITIONS)
        for result in results:
            assert result.auroc > 0.95
            assert result.control_auroc < 0.8
            assert result.selectivity > 0.15
            assert result.reads_signal(CONFIG.auroc_threshold)

    def test_structureless_data_reads_no_signal(self, generator):
        """The load-bearing negative: without this, every absent measurement in
        the paper would still report a number."""
        results = probe_sweep(cluster(0.0, generator), cluster(0.0, generator), CONFIG)
        for result in results:
            assert not result.reads_signal(CONFIG.auroc_threshold)

    def test_probe_transfer_finds_content_that_is_present(self, generator):
        plain_positive, plain_negative = cluster(4.0, generator), cluster(-4.0, generator)
        encoded_positive, encoded_negative = cluster(3.0, generator), cluster(-3.0, generator)

        transfer, control = probe_transfer(
            plain_positive,
            plain_negative,
            encoded_positive,
            encoded_negative,
            layer=0,
            position=POSITIONS[0],
            config=CONFIG,
        )
        assert transfer > 0.95
        assert transfer - control > 0.15

    def test_probe_transfer_reports_chance_when_content_is_absent(self, generator):
        """Regime (D) is exactly this reading: ability present, content absent
        from the attack forward pass."""
        plain_positive, plain_negative = cluster(4.0, generator), cluster(-4.0, generator)
        encoded_positive, encoded_negative = cluster(0.0, generator), cluster(0.0, generator)

        transfer, _ = probe_transfer(
            plain_positive,
            plain_negative,
            encoded_positive,
            encoded_negative,
            layer=0,
            position=POSITIONS[0],
            config=CONFIG,
        )
        assert 0.35 < transfer < 0.65


class TestOverlapMetric:
    def test_identical_distributions_fully_overlap(self):
        scores = np.random.default_rng(0).normal(size=500)
        assert overlap_coefficient(scores, scores) == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_distributions_do_not_overlap(self):
        rng = np.random.default_rng(0)
        assert overlap_coefficient(rng.normal(-50, 0.1, 500), rng.normal(50, 0.1, 500)) < 0.01

    def test_partial_overlap_lands_between(self):
        rng = np.random.default_rng(0)
        value = overlap_coefficient(rng.normal(0, 1, 2000), rng.normal(1, 1, 2000))
        assert 0.3 < value < 0.9

    def test_threshold_separates_the_two_classes(self):
        rng = np.random.default_rng(0)
        harmful = rng.normal(3, 0.5, 500)
        harmless = rng.normal(-3, 0.5, 500)
        threshold = decision_threshold(harmful, harmless)
        assert -3 < threshold < 3

    def test_summary_signs_distance_by_side_of_the_boundary(self):
        above = summarize_projections("base64", np.full(10, 2.0), threshold=0.0)
        below = summarize_projections("morse", np.full(10, -2.0), threshold=0.0)
        assert above.mean_distance_to_threshold > 0 and above.fraction_above_threshold == 1.0
        assert below.mean_distance_to_threshold < 0 and below.fraction_above_threshold == 0.0

    def test_pairwise_overlap_covers_each_unordered_pair(self):
        rng = np.random.default_rng(0)
        scores = {name: rng.normal(size=200) for name in ("base64", "rot13", "morse")}
        pairs = pairwise_overlap(scores)
        assert len(pairs) == 3
        assert ("base64", "rot13") in pairs


class TestRecognitionAndDeployment:
    def test_recognition_detects_represented_harm(self, generator):
        result = measure_recognition(cluster(4.0, generator), cluster(-4.0, generator), CONFIG)
        assert result.recognized
        assert result.best_at(HARMFULNESS_POSITION).auroc > 0.95
        # The refusal position is carried as a contrast, not as the readout.
        assert result.at_position(REFUSAL_POSITION)
        assert result.direction_at(0, HARMFULNESS_POSITION).layer == 0

    def test_recognition_is_absent_when_nothing_separates(self, generator):
        result = measure_recognition(cluster(0.0, generator), cluster(0.0, generator), CONFIG)
        assert not result.recognized

    def test_deployment_detects_decoded_content(self, generator):
        curve = measure_deployment(
            "base64",
            cluster(4.0, generator),
            cluster(-4.0, generator),
            cluster(3.0, generator),
            cluster(-3.0, generator),
            CONFIG,
            strata=None,  # synthetic clusters carry no texts to stratify by
        )
        assert curve.deployed
        assert curve.best().transfer_auroc > 0.95
        assert len(curve.results) == len(LAYERS) * len(POSITIONS)

    def test_deployment_absent_is_the_didnt_decode_reading(self, generator):
        curve = measure_deployment(
            "base64",
            cluster(4.0, generator),
            cluster(-4.0, generator),
            cluster(0.0, generator),
            cluster(0.0, generator),
            CONFIG,
            strata=None,  # synthetic clusters carry no texts to stratify by
        )
        assert not curve.deployed


class TestPermutationLicensing:
    """The licensing gate — a permutation test, not a fixed AUROC cut.

    These pin the two failures the old `auroc >= 0.70 on any of ~33 cells` gate
    had in opposite directions: it discarded real-but-weak signal, while leaving
    the layer search uncorrected. The null is over the MAX AUROC across layers,
    so both are handled by one test.
    """

    # Small on purpose — these assert the statistic's behaviour, not its
    # resolution, and every draw refits every layer.
    CONFIG = ProbeConfig(seed=0, test_fraction=0.3, auroc_threshold=0.70, n_permutations=40)

    def test_structureless_activations_are_not_licensed(self, generator):
        """The failure that would invalidate every recognition claim."""
        harmful = cluster(0.0, generator)
        harmless = cluster(0.0, generator)
        result = measure_recognition(harmful, harmless, self.CONFIG)
        assert result.p_value > self.CONFIG.alpha
        assert not result.recognized

    def test_a_separable_signal_is_licensed(self, generator):
        harmful = cluster(3.0, generator)
        harmless = cluster(0.0, generator)
        result = measure_recognition(harmful, harmless, self.CONFIG)
        assert result.p_value <= self.CONFIG.alpha
        assert result.recognized

    def test_the_null_is_over_the_max_so_it_sits_above_chance(self, generator):
        """Family-wise control has to actually bite.

        A per-cell null centres on 0.5. A null over the max of several cells must
        sit ABOVE that, or selecting the best layer would be uncorrected — which
        is exactly what would make a lowered per-cell cut worse than the old one.
        """
        harmful = cluster(0.0, generator)
        harmless = cluster(0.0, generator)
        nulls = permutation_null_max_auroc(
            harmful, harmless, HARMFULNESS_POSITION, self.CONFIG
        )
        assert len(nulls) == self.CONFIG.n_permutations
        assert float(np.median(nulls)) > 0.5

    def test_the_p_value_can_never_be_zero(self):
        """A finite permutation set cannot produce p=0; reporting one overstates
        the evidence in a paper table (Phipson & Smyth 2010)."""
        assert permutation_p_value(1.0, np.zeros(200)) == pytest.approx(1 / 201)

    def test_licensing_fails_closed_without_a_p_value(self, generator):
        """A capture missing the harmfulness position must never license."""
        batch = cluster(3.0, generator)
        stripped = ActivationBatch(
            tensor=batch.tensor[:, :, 1:, :],
            layers=list(LAYERS),
            positions=[REFUSAL_POSITION],
            site="resid_pre",
            model_name="synthetic",
            user_messages=list(batch.user_messages),
        )
        result = measure_recognition(stripped, stripped, self.CONFIG)
        assert np.isnan(result.p_value)
        assert not result.recognized, "NaN must fail closed, never license"

    def test_significance_and_effect_size_are_reported_separately(self):
        """A weak-but-real signal must be licensed AND visibly weak.

        This is the case the old fixed cut got wrong. At n=200 per class a small
        offset yields AUROC 0.627 with p=0.010 — decisively above chance, and
        decisively below the 0.70 effect-size bar that used to be the licensing
        gate. Under that gate this signal was discarded as "no recognition".

        The numbers are not incidental: Llama `zero_width` in the phase-0 pilot
        read AUROC 0.617 at n=200 on a rung with deployment 200/200, i.e. right
        here. Collapsing significance and magnitude into one number is what made
        that read as absence rather than as a weak-but-present readout.
        """
        config = ProbeConfig(
            seed=0, test_fraction=0.3, auroc_threshold=0.70, n_permutations=200
        )
        generator = torch.Generator().manual_seed(0)
        harmful = wide_cluster(0.8, generator)
        harmless = wide_cluster(0.0, generator)
        result = measure_recognition(harmful, harmless, config)

        assert result.recognized, "must beat the shuffled-label null"
        assert not result.meets_effect_size_bar, "and must still read as weak"
        assert result.observed_max_auroc < result.threshold
        assert result.p_value < 0.05


class TestBlasPinning:
    """Regression, 2026-08-05: the sweep that died at the 8 h wall.

    Multi-threaded BLAS on these small (140 x 4096) fits is catastrophically
    slower than single-threaded — 3,680 ms vs 118 ms per fit, measured on the
    cluster's real activations. The 8-CPU allocation gave BLAS 8 threads by
    default, which turned a ~12 min permutation null into a ~6.5 h one and
    finished one rung of seven in eight hours.

    These tests pin the FIX in place, because the failure is invisible locally:
    everything still returns correct answers, just ~31x slower.
    """

    ENTRY_POINTS = [
        "fit_probe",
        "probe_sweep",
        "probe_transfer_detail",
        "probe_transfer",
        "crossval_scores",
        "permutation_null_max_auroc",
        "permutation_null_max_transfer_auroc",
    ]

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_every_fitting_entry_point_pins_blas(self, name):
        function = getattr(linear, name)
        assert getattr(function, "__wrapped__", None) is not None, (
            f"{name} fits probes but is not wrapped by single_threaded_blas; "
            "unpinned it costs ~31x more CPU and will miss the cluster wall"
        )

    def test_the_pin_is_actually_applied_while_the_body_runs(self):
        from threadpoolctl import threadpool_info

        seen = {}

        @linear.single_threaded_blas
        def probe_the_pool():
            seen["limits"] = {entry["num_threads"] for entry in threadpool_info()
                              if entry["user_api"] == "blas"}

        probe_the_pool()
        # No BLAS backend visible in this environment is not a failure of the
        # decorator; a backend visible and NOT pinned to 1 is.
        assert seen["limits"] <= {1}
