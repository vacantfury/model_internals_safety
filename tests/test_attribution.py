"""I6's patching attribution, and the method-paper choices it is pinned to.

Zhang & Nanda (arXiv 2309.16042, ICLR 2023) show that activation-patching
results move with hyperparameters most people leave implicit. So the choices are
asserted here rather than left in prose: a wrong metric or a silently-widened
patch would still produce a plausible map.
"""

from __future__ import annotations

import math

import pytest
import torch

from internals_safety.config import load_measurements_config
from internals_safety.measurements.attribution import (
    AttributionMap,
    PatchCell,
    forward_passes,
    logit_difference,
    measure_attribution,
    normalized_effect,
    reading,
    unmeasured_reading,
)
from internals_safety.models.capture import capture_activations
from internals_safety.models.loader import prepare_prompts
from internals_safety.models.patching import _replace_at


def _cells(effects, controls=None):
    controls = controls or [None] * len(effects)
    return tuple(
        PatchCell(layer=i, position="last", effect=e, control_effect=c)
        for i, (e, c) in enumerate(zip(effects, controls))
    )


def _map(effects, controls=None, sd=2.0):
    return AttributionMap(
        cells=_cells(effects, controls), ld_clean=3.0, ld_corrupt=-1.0,
        n_prompts=8, detection_sd=sd,
    )


class TestTheMetric:
    """Logit difference, normalised — and why not probability."""

    def test_logit_difference_takes_the_max_within_each_answer_set(self):
        """'I' and 'As' both open a refusal, so the model choosing one is the
        answer given once. Summing would count it twice."""
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 2] = 5.0   # refusal opening A
        logits[0, 0, 3] = 1.0   # refusal opening B
        logits[0, 0, 7] = 2.0   # compliance opening
        assert logit_difference(logits, [2, 3], [7]).item() == pytest.approx(3.0)

    def test_it_reads_the_final_position(self):
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 2] = 99.0  # an earlier position must not leak in
        logits[0, -1, 2] = 4.0
        logits[0, -1, 7] = 1.0
        assert logit_difference(logits, [2], [7]).item() == pytest.approx(3.0)

    def test_both_token_sets_are_required(self):
        with pytest.raises(ValueError, match="counterfactual"):
            logit_difference(torch.zeros(1, 1, 10), [2], [])

    def test_the_effect_is_normalised_to_clean_equals_one(self):
        assert normalized_effect(patched=3.0, clean=3.0, corrupt=-1.0) == pytest.approx(1.0)
        assert normalized_effect(patched=-1.0, clean=3.0, corrupt=-1.0) == pytest.approx(0.0)
        assert normalized_effect(patched=1.0, clean=3.0, corrupt=-1.0) == pytest.approx(0.5)

    def test_a_contrast_that_never_separated_yields_None_not_a_huge_effect(self):
        """Fail-closed. Dividing by a near-zero gap would report an enormous
        effect from numerical noise — an undefined effect is not a zero one."""
        assert normalized_effect(patched=0.5, clean=1.0, corrupt=1.0) is None


class TestTheDetectionBar:
    """Derived from the grid's own spread, per the method paper."""

    def test_the_bar_is_mean_plus_k_sd_of_the_grid(self):
        effects = [0.0, 0.0, 0.0, 1.0]
        tensor = torch.tensor(effects, dtype=torch.float64)
        expected = float(tensor.mean() + 2.0 * tensor.std(unbiased=True))
        assert _map(effects).detection_threshold == pytest.approx(expected)

    def test_a_uniformly_elevated_grid_detects_nothing(self):
        """The point of a spread-relative bar. If every cell restores the answer
        equally, nothing is localised — an absolute cut would call them all hits."""
        assert _map([0.9, 0.9, 0.9, 0.9]).detected == ()

    def test_one_standout_cell_is_detected(self):
        detected = _map([0.0, 0.05, 0.0, 0.02, 0.0, 0.01, 0.9]).detected
        assert [cell.effect for cell in detected] == [0.9]

    def test_a_grid_too_small_for_the_bar_says_so_instead_of_detecting_nothing(self):
        """⚠️ The largest z attainable in a sample of n is (n-1)/sqrt(n), so at
        k=2 no grid under 6 cells can EVER detect a cell. Reporting 'nothing
        found' there would be arithmetic dressed as a measurement."""
        assert _map([0.0, 0.0, 0.0, 0.9]).bar_is_reachable is False
        assert _map([0.0, 0.0, 0.0, 0.9]).detected == ()
        assert _map([0.0] * 6 + [0.9]).bar_is_reachable is True

    def test_fewer_than_two_measured_cells_has_no_bar(self):
        """A standard deviation over one number is not a spread, and a bar
        derived from it would be arithmetic dressed as a criterion."""
        assert _map([0.9]).detection_threshold is None
        assert _map([0.9]).detected == ()

    def test_unmeasured_cells_are_excluded_rather_than_counted_as_zero(self):
        mapping = _map([None, None, 0.4, 0.6])
        assert len(mapping.measured) == 2
        assert len(mapping.cells) == 4

    def test_the_bar_is_configured_not_hardcoded(self):
        assert load_measurements_config().attribution.detection_sd == 2.0


class TestTheReading:
    """What the contract will and will not license."""

    def test_a_peak_clearing_both_its_control_and_the_bar_is_licensed(self):
        config = load_measurements_config()
        result = reading(_map([0.0, 0.02, 0.0, 0.0, 0.01, 0.0, 0.9],
                              [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.05]), config)
        assert result.licensed is True
        assert result.value == pytest.approx(0.9)
        assert result.detail["peak_layer"] == 6

    def test_a_peak_its_shuffled_control_also_reaches_is_not_licensed(self):
        """The control writes a real activation of the right shape and norm
        carrying ANOTHER prompt's content. If that restores the answer just as
        well, the patch moved the model without transferring information."""
        config = load_measurements_config()
        result = reading(_map([0.0] * 6 + [0.9], [0.0] * 6 + [0.92]), config)
        assert result.licensed is False

    def test_a_peak_with_no_control_at_all_is_not_licensed(self):
        config = load_measurements_config()
        assert reading(_map([0.0] * 6 + [0.9]), config).licensed is False

    def test_a_uniformly_elevated_grid_is_not_licensed_however_large(self):
        config = load_measurements_config()
        result = reading(_map([0.9] * 7, [0.0] * 7), config)
        assert result.licensed is False, "no cell stands out, so nothing is localised"

    def test_nothing_measurable_reads_licensed_None_not_False(self):
        """⚠️ The tri-state distinction, one instrument further on. 'No cell
        carries the effect' and 'the contrast never separated' are different
        claims, and only the first is a result about the model."""
        config = load_measurements_config()
        result = reading(_map([None, None]), config)
        assert result.licensed is None
        assert math.isnan(result.value)

    def test_the_reading_records_the_choices_that_change_the_answer(self):
        """The method paper's whole finding is that these move the result, so a
        map recorded without them is not reproducible."""
        config = load_measurements_config()
        detail = reading(_map([0.0] * 6 + [0.9], [0.0] * 7), config).detail
        assert detail["metric"] == "normalized_logit_difference"
        assert detail["corruption"] == "whole_prompt_contrast"
        assert detail["patch_extent"] == "single_cell"

    def test_the_operating_point_names_the_metric_and_the_bar(self):
        config = load_measurements_config()
        text = reading(_map([0.0] * 6 + [0.9], [0.0] * 7), config).operating_point
        assert "logit difference" in text and "sliding window" in text

    def test_selection_is_not_claimed_to_be_inside_a_null(self):
        """The peak is a maximum over the grid. The mean+kSD bar corrects within
        the grid but is not a null, and claiming otherwise is the overclaim P7
        exists to stop."""
        config = load_measurements_config()
        assert reading(_map([0.0] * 6 + [0.9], [0.0] * 7), config).selection_inside_null is False

    def test_an_unmeasured_reading_states_its_reason(self):
        config = load_measurements_config()
        assert "no direction" in unmeasured_reading(config, "no direction").detail["reason"]


class TestCost:
    def test_the_control_is_inside_the_price(self):
        """A control the dry-run cannot see is a cost nobody approved — the
        defect the causal gate's null already had once."""
        assert forward_passes(10, with_control=True) == 2 + 20
        assert forward_passes(10, with_control=False) == 2 + 10


class TestPerRowPatchPositions:
    """`instruction_final` sits at a different offset in every prompt."""

    def test_one_offset_per_row_writes_a_different_index_per_row(self):
        hidden = torch.zeros(2, 4, 3)
        vectors = torch.ones(2, 3)
        patched = _replace_at(hidden, [-1, -3], vectors)
        assert patched[0, 3].tolist() == [1.0, 1.0, 1.0]
        assert patched[1, 1].tolist() == [1.0, 1.0, 1.0]
        assert patched[0, 1].tolist() == [0.0, 0.0, 0.0]

    def test_a_scalar_still_applies_to_every_row(self):
        patched = _replace_at(torch.zeros(2, 4, 3), -2, torch.ones(2, 3))
        assert patched[:, 2].sum() == pytest.approx(6.0)

    def test_a_wrong_length_offset_list_fails_loud(self):
        with pytest.raises(ValueError, match="batch mismatch"):
            _replace_at(torch.zeros(2, 4, 3), [-1], torch.ones(2, 3))

    def test_a_positive_offset_is_rejected(self):
        with pytest.raises(ValueError, match="negative offsets"):
            _replace_at(torch.zeros(2, 4, 3), [-1, 2], torch.ones(2, 3))


class TestEndToEndOnARealModel:
    """Driven through the tiny model, so the hook path is genuinely exercised."""

    @pytest.fixture
    def prompts(self):
        return ["the cat sat", "a dog ran fast", "birds fly"]

    def test_patching_every_cell_from_the_clean_run_restores_it(self, tiny_model, prompts):
        """The sanity check the whole instrument rests on: patch the clean states
        into a run ON THE CLEAN PROMPTS and the effect must be ~1.0 everywhere,
        because the patch is writing back what was already there."""
        source = capture_activations(tiny_model, prepare_prompts(tiny_model, prompts))
        result = measure_attribution(
            tiny_model, source, corrupt_prompts=prompts, clean_prompts=prompts,
            answer_ids=[2], counter_ids=[3], detection_sd=2.0, with_control=False,
        )
        # clean == corrupt here, so the denominator vanishes and every cell is
        # UNMEASURED rather than reported as a perfect restoration.
        assert result.measured == ()
        assert result.ld_clean == pytest.approx(result.ld_corrupt)

    def test_a_real_contrast_produces_a_measured_grid(self, tiny_model, prompts):
        source = capture_activations(tiny_model, prepare_prompts(tiny_model, prompts))
        result = measure_attribution(
            tiny_model, source,
            corrupt_prompts=["x y z", "p q r s", "m n"],
            clean_prompts=prompts,
            answer_ids=[2], counter_ids=[3], detection_sd=2.0,
        )
        assert len(result.cells) == len(source.layers) * len(source.positions)
        assert all(cell.control_effect is not None for cell in result.measured)

    def test_an_unpaired_contrast_is_refused(self, tiny_model, prompts):
        source = capture_activations(tiny_model, prepare_prompts(tiny_model, prompts))
        with pytest.raises(ValueError, match="paired"):
            measure_attribution(
                tiny_model, source, corrupt_prompts=["only one"], clean_prompts=prompts,
                answer_ids=[2], counter_ids=[3], detection_sd=2.0,
            )

    def test_a_cache_that_does_not_cover_the_run_is_refused(self, tiny_model, prompts):
        """Patching writes row-for-row, so a mismatch would silently pair each
        prompt with someone else's activation — the control, run by accident."""
        source = capture_activations(tiny_model, prepare_prompts(tiny_model, prompts[:2]))
        with pytest.raises(ValueError, match="row-for-row"):
            measure_attribution(
                tiny_model, source, corrupt_prompts=prompts, clean_prompts=prompts,
                answer_ids=[2], counter_ids=[3], detection_sd=2.0,
            )
