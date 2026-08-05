"""Per-prompt probe readings — the layer that turns a curve into a regime label.

A regime is defined per (model, family, prompt) but an AUROC is a population
quantity, so `read_deployment_per_prompt` / `read_recognition_per_prompt` add a
two-step reading: population licensing, then a per-example score compared
against the *same-condition* negative class.

The two tests that carry the most weight are the licensing ones. Without the
gate, a structureless probe distributes cells across regimes at roughly chance,
which would populate (B) — the pilot's headline cell — out of noise and the
pilot would return a false go.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from internals_safety.config import ProbeConfig
from internals_safety.measurements.deployment import measure_deployment, read_deployment_per_prompt
from internals_safety.measurements.recognition import (
    HARMFULNESS_POSITION,
    REFUSAL_POSITION,
    measure_recognition,
    read_recognition_per_prompt,
)
from internals_safety.models.capture import ActivationBatch
from internals_safety.probes.linear import (
    crossval_scores,
    probe_transfer,
    probe_transfer_detail,
    reading_threshold,
)

LAYERS = [0, 1]
POSITIONS = [HARMFULNESS_POSITION, REFUSAL_POSITION]
D_MODEL = 16
N = 60
CONFIG = ProbeConfig(seed=0, test_fraction=0.3, auroc_threshold=0.70, cv_folds=5)


def make_batch(tensor: torch.Tensor) -> ActivationBatch:
    return ActivationBatch(
        tensor=tensor,
        layers=list(LAYERS),
        positions=list(POSITIONS),
        site="resid_pre",
        model_name="synthetic",
        user_messages=[f"prompt {index}" for index in range(tensor.shape[0])],
    )


def cluster(offset: float, generator: torch.Generator, shift: float = 0.0) -> ActivationBatch:
    """First feature shifted by `offset`; `shift` moves *every* feature, standing
    in for the common-mode change an encoding + template wrapper produces."""
    tensor = torch.randn(N, len(LAYERS), len(POSITIONS), D_MODEL, generator=generator)
    tensor[..., 0] += offset
    return make_batch(tensor + shift)


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


class TestTransferDetail:
    def test_agrees_with_the_metrics_only_wrapper(self, generator):
        args = (cluster(3.0, generator), cluster(-3.0, generator), cluster(3.0, generator), cluster(-3.0, generator))
        detail = probe_transfer_detail(*args, layer=0, position=POSITIONS[0], config=CONFIG)
        transfer, control = probe_transfer(*args, layer=0, position=POSITIONS[0], config=CONFIG)
        assert detail.transfer_auroc == transfer
        assert detail.control_auroc == control

    def test_scores_one_value_per_test_example(self, generator):
        detail = probe_transfer_detail(
            cluster(3.0, generator),
            cluster(-3.0, generator),
            cluster(3.0, generator),
            cluster(-3.0, generator),
            layer=0,
            position=POSITIONS[0],
            config=CONFIG,
        )
        assert detail.positive_scores.shape == (N,)
        assert detail.negative_scores.shape == (N,)
        assert detail.positive_scores.mean() > detail.negative_scores.mean()


class TestReadingThreshold:
    def test_is_the_requested_percentile_of_the_negatives(self):
        scores = np.arange(101, dtype=float)
        assert reading_threshold(scores, ProbeConfig(reading_percentile=50.0)) == 50.0
        assert reading_threshold(scores, ProbeConfig(reading_percentile=90.0)) == 90.0

    def test_empty_negatives_is_an_error_not_a_silent_zero(self):
        with pytest.raises(ValueError, match="no negative examples"):
            reading_threshold(np.array([]), CONFIG)


class TestCrossvalScores:
    def test_every_example_gets_an_out_of_sample_score(self, generator):
        positive, negative = cluster(4.0, generator), cluster(-4.0, generator)
        pos, neg = crossval_scores(positive, negative, layer=0, position=POSITIONS[0], config=CONFIG)
        assert pos.shape == (N,) and neg.shape == (N,)
        assert pos.mean() > neg.mean()

    def test_structureless_activations_do_not_separate(self, generator):
        shared = cluster(0.0, generator)
        pos, neg = crossval_scores(shared, shared, layer=0, position=POSITIONS[0], config=CONFIG)
        # In-sample fitting would separate these trivially; out-of-sample cannot.
        assert abs(pos.mean() - neg.mean()) < 1.0

    def test_too_few_examples_per_class_is_an_error(self):
        tiny = make_batch(torch.randn(1, len(LAYERS), len(POSITIONS), D_MODEL))
        with pytest.raises(ValueError, match=">=2 examples per class"):
            crossval_scores(tiny, tiny, layer=0, position=POSITIONS[0], config=CONFIG)


class TestDeploymentReading:
    def read(self, plain_pos, plain_neg, enc_pos, enc_neg):
        curve = measure_deployment("fam", plain_pos, plain_neg, enc_pos, enc_neg, CONFIG)
        return curve, read_deployment_per_prompt(curve, plain_pos, plain_neg, enc_pos, enc_neg, CONFIG)

    def test_transferring_content_reads_deployed_for_most_prompts(self, generator):
        curve, reading = self.read(
            cluster(4.0, generator), cluster(-4.0, generator),
            cluster(4.0, generator), cluster(-4.0, generator),
        )
        assert curve.deployed and reading.licensed
        assert reading.harmful_rate > 0.8
        assert reading.gap > 0.3

    def test_an_unlicensed_curve_reads_no_prompt_as_deployed(self, generator):
        """The load-bearing one. With no signal in the population, per-example
        scores are noise around the threshold and would read ~50% deployed —
        populating (B) out of nothing.

        All four batches are INDEPENDENT zero-offset draws (fixed 2026-08-05).
        This previously passed the same `shared` tensor as both the plain and the
        encoded positives, which is a train/test leak rather than a signal-free
        population: the probe memorised `shared` and then recovered it. The old
        fixed 0.70 cut could not see a leak that small; permutation licensing
        can, and did — see the test below, which pins that as behaviour.
        """
        curve, reading = self.read(
            cluster(0.0, generator), cluster(0.0, generator),
            cluster(0.0, generator), cluster(0.0, generator),
        )
        assert not curve.deployed
        assert reading.licensed is False
        assert not any(reading.harmful)
        assert not any(reading.harmless)

    def test_permutation_licensing_detects_a_leak_the_fixed_cut_missed(self, generator):
        """A weak-but-real effect must license even when it is far below 0.70.

        Reusing one tensor as both the plain and the encoded positive class makes
        the probe's transfer partly memorisation. Transfer AUROC is only ~0.56 —
        so the retired `auroc >= 0.70` rule scored it unlicensed — while the
        calibrated null puts it comfortably under alpha. That asymmetry is the
        whole reason licensing moved to a permutation test: the fixed cut
        discarded real signal, here of exactly the kind an instrument must catch.
        """
        shared = cluster(0.0, generator)
        curve, _ = self.read(shared, cluster(0.0, generator), shared, cluster(0.0, generator))

        assert curve.deployed, "a calibrated null must detect this; the 0.70 cut did not"
        assert curve.p_value <= curve.alpha
        # Licensed on SIGNIFICANCE while failing the effect-size bar — the two
        # are reported separately precisely so this case is legible.
        assert not curve.meets_effect_size_bar
        assert curve.observed_max_transfer_auroc < 0.70

    def test_a_common_mode_shift_does_not_change_the_reading(self, generator):
        """Encoding a prompt and wrapping it in an attack template moves both
        classes together. Thresholding at the boundary would let that shift
        decide the label; thresholding against the concurrent negatives cannot."""
        plain_pos, plain_neg = cluster(4.0, generator), cluster(-4.0, generator)
        base = self.read(plain_pos, plain_neg, cluster(4.0, generator), cluster(-4.0, generator))[1]

        shifted_generator = torch.Generator().manual_seed(0)
        plain_pos2, plain_neg2 = cluster(4.0, shifted_generator), cluster(-4.0, shifted_generator)
        shifted = self.read(
            plain_pos2, plain_neg2,
            cluster(4.0, shifted_generator, shift=25.0),
            cluster(-4.0, shifted_generator, shift=25.0),
        )[1]
        assert shifted.harmful_rate == pytest.approx(base.harmful_rate, abs=0.05)

    def test_the_benign_rate_pins_the_operating_point(self, generator):
        """At the 50th percentile the benign control reads positive half the
        time by construction; the informative quantity is the gap."""
        _, reading = self.read(
            cluster(4.0, generator), cluster(-4.0, generator),
            cluster(4.0, generator), cluster(-4.0, generator),
        )
        assert reading.harmless_rate == pytest.approx(0.5, abs=0.02)
        assert reading.gap == reading.harmful_rate - reading.harmless_rate

    def test_the_reading_cell_is_one_that_reads_a_signal(self, generator):
        curve, reading = self.read(
            cluster(4.0, generator), cluster(-4.0, generator),
            cluster(4.0, generator), cluster(-4.0, generator),
        )
        chosen = [
            result for result in curve.results
            if result.layer == reading.layer and result.position == reading.position
        ]
        assert chosen and chosen[0].deployed


class TestRecognitionReading:
    def test_reads_at_the_harmfulness_position_only(self, generator):
        harmful, harmless = cluster(4.0, generator), cluster(-4.0, generator)
        result = measure_recognition(harmful, harmless, CONFIG)
        reading = read_recognition_per_prompt(result, harmful, harmless, CONFIG)
        # Reading harm at the post-template position would measure refusal
        # propensity instead (Zhao et al.), which begs the paper's question.
        assert reading.position == HARMFULNESS_POSITION
        assert reading.licensed and reading.harmful_rate > 0.8

    def test_an_unrecognized_condition_reads_no_prompt_as_recognized(self, generator):
        shared = cluster(0.0, generator)
        other = cluster(0.0, generator)
        result = measure_recognition(shared, other, CONFIG)
        reading = read_recognition_per_prompt(result, shared, other, CONFIG)
        assert reading.licensed is False
        assert not any(reading.harmful)

    def test_a_missing_harmfulness_position_is_an_error(self, generator):
        def refusal_only(batch: ActivationBatch) -> ActivationBatch:
            return ActivationBatch(
                tensor=batch.tensor[:, :, 1:, :],
                layers=list(LAYERS),
                positions=[REFUSAL_POSITION],
                site=batch.site,
                model_name=batch.model_name,
                user_messages=batch.user_messages,
            )

        harmful = refusal_only(cluster(4.0, generator))
        harmless = refusal_only(cluster(-4.0, generator))
        result = measure_recognition(harmful, harmless, CONFIG)
        with pytest.raises(ValueError, match="no probe cells at"):
            read_recognition_per_prompt(result, harmful, harmless, CONFIG)
