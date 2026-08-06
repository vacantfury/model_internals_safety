"""The two controls that gate any I5/I6 claim.

Build plan §4 lists both as NOT BUILT and names what each defeats. One of them
turned out to be degenerate in our regime, and these tests are what establish
that rather than assert it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from internals_safety.config import CausalLicenseConfig, ProbeConfig
from internals_safety.measurements.causal_license import (
    CausalEvidence,
    matched_norm_random_direction,
    random_direction_null,
)
from internals_safety.probes.linear import control_task_selectivity


def evidence(bypass: float, layer: int = 4) -> CausalEvidence:
    """Causal evidence whose bypass score is exactly `bypass`."""
    return CausalEvidence(
        layer=layer,
        position="instruction_final",
        refusal_before=1.0,
        refusal_after_ablation=1.0 - bypass,
        harmless_refusal_before=0.0,
        harmless_refusal_after_addition=0.5,
        kl=0.01,
    )


class TestMatchedNormRandomDirection:
    """Defeats: steering "working" because you perturbed anything."""

    def test_the_norm_is_matched_exactly(self):
        """The whole point. Ablation and addition scale with what is written, so
        an unmatched random vector tests magnitude rather than direction."""
        direction = torch.randn(512)
        random = matched_norm_random_direction(direction)
        assert float(random.norm()) == pytest.approx(float(direction.norm()), rel=1e-5)

    def test_it_is_not_the_direction_it_matches(self):
        direction = torch.nn.functional.normalize(torch.randn(512), dim=0)
        cosine = float(direction @ matched_norm_random_direction(direction))
        # Two random directions in 512-d are near-orthogonal with overwhelming
        # probability; anything large would mean the draw is not isotropic.
        assert abs(cosine) < 0.3

    def test_it_is_reproducible_from_a_generator(self):
        generator = torch.Generator().manual_seed(7)
        first = matched_norm_random_direction(torch.ones(64), generator)
        generator = torch.Generator().manual_seed(7)
        assert torch.equal(first, matched_norm_random_direction(torch.ones(64), generator))

    def test_a_zero_direction_is_refused(self):
        with pytest.raises(ValueError, match="zero direction"):
            matched_norm_random_direction(torch.zeros(16))

    def test_dtype_survives(self):
        assert matched_norm_random_direction(torch.ones(32, dtype=torch.float16)).dtype is torch.float16


class TestRandomDirectionNull:
    def test_a_direction_that_beats_every_random_one_licenses(self):
        null = random_direction_null(evidence(0.9), [evidence(0.1) for _ in range(19)])
        assert null.licensed
        assert null.p_value == pytest.approx(1 / 20)

    def test_a_direction_random_ones_match_does_not_license(self):
        """The failure this control exists to catch: the intervention did
        something, but so does perturbing anything by the same amount."""
        null = random_direction_null(evidence(0.5), [evidence(0.5) for _ in range(19)])
        assert not null.licensed
        assert null.margin == pytest.approx(0.0)

    def test_the_smallest_p_value_is_one_over_n_plus_one_never_zero(self):
        """A null that has never produced an equal value has not proved it
        cannot."""
        assert random_direction_null(evidence(1.0), [evidence(0.0)] * 9).p_value == pytest.approx(0.1)

    def test_an_undrawn_null_licenses_nothing(self):
        null = random_direction_null(evidence(0.9), [])
        assert not null.licensed
        assert null.p_value != null.p_value  # NaN


class TestControlTaskSelectivity:
    """Hewitt & Liang (EMNLP 2019) — and why it does not transfer to us."""

    @pytest.fixture
    def measured(self) -> object:
        generator = np.random.default_rng(0)
        n, d = 120, 512
        features = generator.normal(size=(n, d)).astype(np.float32)
        features[: n // 2] += 0.25
        labels = np.r_[np.ones(n // 2), np.zeros(n // 2)]
        return control_task_selectivity(
            torch.from_numpy(features), torch.from_numpy(labels), ProbeConfig(seed=0)
        )

    def test_the_real_task_separates(self, measured):
        assert measured.real_auroc > 0.7

    def test_the_control_task_is_at_chance_out_of_sample(self, measured):
        """**The measured finding.** Their control task assigns a random label
        per word TYPE, so the type recurs across the split and memorisation
        shows at test time. Our inputs are one-off activation vectors — nothing
        recurs, so a random labelling has nothing that transfers."""
        assert measured.is_degenerate

    def test_the_probe_memorises_any_labelling_in_sample(self, measured):
        """d >> n makes the training set linearly separable for ANY labels, at
        any regularisation. Fine for held-out evaluation, but it means a
        train-set number is never evidence."""
        assert measured.memorises_the_training_set

    def test_selectivity_therefore_carries_no_extra_information(self, measured):
        """It reduces to `real_auroc - 0.5`, so the capacity sweep their method
        prescribes has nothing to select on. Recorded here so the degeneracy is
        a checked fact rather than a remembered one."""
        assert measured.selectivity == pytest.approx(measured.real_auroc - 0.5, abs=0.10)

    def test_the_degeneracy_flag_would_clear_if_a_control_task_ever_transferred(self):
        """Guards the flag itself: it must key on the measurement, not be
        hard-coded true."""
        from internals_safety.probes.linear import ControlTaskSelectivity

        informative = ControlTaskSelectivity(
            real_auroc=0.9, control_task_train_auroc=1.0,
            control_task_test_auroc=0.8, regularization_c=1.0,
        )
        assert not informative.is_degenerate
        assert informative.selectivity == pytest.approx(0.1)
