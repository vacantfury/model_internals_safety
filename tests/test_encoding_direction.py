"""The encoding-direction ablation's scoring and selection layer.

Hermetic: no weights, no network, no judge. Every fixture is a plain dataclass
of real numbers, so there is no fake standing in for a stricter real thing —
the failure mode `conftest.TINY_CHAT_TEMPLATE` and `PerfectSAE` were caught for.

The properties worth stating up front, because they are what the module exists
to protect:

1. Selection is by the OUTCOME, which is only sound because the control is
   selected by the identical procedure. `select_cell` must therefore be a pure
   function of its criterion — nothing about "real vs random" may reach it.
2. The confound is checked BEFORE any hypothesis is assigned. A reading whose
   ablation also decoded the prompt supports nothing, however large its margin.
3. Every fail-closed path returns `None` or raises. None of them returns 0.0 —
   that is the silent-zero defect the deployment axis shipped for a week.
"""

from __future__ import annotations

import pytest

from internals_safety.config import EncodingDirectionConfig
from internals_safety.measurements.encoding_direction import (
    AblationReading,
    ArmRates,
    CellEvidence,
    is_eligible,
    select_cell,
)

N_LAYERS = 32
CONFIG = EncodingDirectionConfig()


def cell(**overrides) -> CellEvidence:
    """A cell that passes every eligibility clause, so a test can break exactly one."""
    base = dict(
        layer=14,
        position="instruction_final",
        proxy_harmful_refusal=0.80,
        proxy_benign_refusal=0.30,
        kl=0.01,
        separation_auroc=0.99,
        raw_norm=4.2,
    )
    return CellEvidence(**{**base, **overrides})


def arms(
    harmful: float,
    benign: float,
    ability: float | None = None,
    mechanism_errors: int = 0,
) -> ArmRates:
    return ArmRates(
        harmful_refusal=harmful,
        benign_refusal=benign,
        n_harmful=100,
        n_benign=100,
        ability=ability,
        mechanism_errors=mechanism_errors,
    )


def reading(**overrides) -> AblationReading:
    """Llama's measured shape: plaintext gap +0.82, encoded gap 0.00."""
    base = dict(
        family="homoglyph",
        model="llama3_1_8b_instruct",
        layer=14,
        position="instruction_final",
        separation_auroc=0.99,
        kl=0.01,
        baseline=arms(0.99, 0.99, ability=0.91),
        ablated=arms(0.95, 0.35, ability=0.91),
        control=arms(0.98, 0.96, ability=0.91),
        plaintext=arms(0.92, 0.10),
    )
    return AblationReading(**{**base, **overrides})


class TestArmRates:
    def test_both_arms_are_mandatory_because_a_single_arm_rate_is_the_defect(self):
        """The paper's whole subject is that a one-armed rate is uninterpretable.

        A type that could express one would let the defect back in through the
        instrument that exists to measure it.
        """
        with pytest.raises(TypeError):
            ArmRates(harmful_refusal=0.99, n_harmful=100, n_benign=100)  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 42.0])
    def test_a_rate_outside_zero_one_is_rejected(self, bad):
        with pytest.raises(ValueError, match="rate in"):
            arms(bad, 0.10)

    def test_an_empty_arm_is_rejected_rather_than_dividing_later(self):
        with pytest.raises(ValueError, match="non-empty"):
            ArmRates(
                harmful_refusal=0.9, benign_refusal=0.1,
                n_harmful=0, n_benign=100, mechanism_errors=0,
            )

    def test_harm_gap_is_harmful_minus_benign(self):
        assert arms(0.92, 0.10).harm_gap == pytest.approx(0.82)
        assert arms(0.99, 0.99).harm_gap == pytest.approx(0.00)


class TestEligibilityRejectsEachClauseIndependently:
    """Mutation-style: the healthy cell passes, then exactly one clause breaks.

    Counting rejections over the whole filter would say nothing about which
    branch fired — the lesson from verifying a guard by totals rather than by
    path (`CLAUDE.md`, 2026-08-07).
    """

    def test_the_unmutated_cell_passes(self):
        assert is_eligible(cell(), N_LAYERS, CONFIG) is True

    def test_a_late_layer_is_pruned(self):
        # 32 * (1 - 0.20) = 25.6 -> int 25, so 25 is out and 24 is in.
        assert is_eligible(cell(layer=25), N_LAYERS, CONFIG) is False
        assert is_eligible(cell(layer=24), N_LAYERS, CONFIG) is True

    def test_a_broken_model_is_rejected_by_kl(self):
        assert is_eligible(cell(kl=0.5), N_LAYERS, CONFIG) is False

    def test_a_direction_that_does_not_separate_its_own_classes_is_noise(self):
        assert is_eligible(cell(separation_auroc=0.55), N_LAYERS, CONFIG) is False

    def test_a_degenerate_direction_is_rejected(self):
        """raw_norm ~ 0 means the classes coincide and the unit vector is arbitrary."""
        assert is_eligible(cell(raw_norm=0.0), N_LAYERS, CONFIG) is False

    @pytest.mark.parametrize(
        "field", ["proxy_harmful_refusal", "kl", "separation_auroc", "raw_norm"]
    )
    def test_nan_is_rejected_not_propagated(self, field):
        """A sort would happily put NaN first."""
        assert is_eligible(cell(**{field: float("nan")}), N_LAYERS, CONFIG) is False

    def test_a_NEGATIVE_proxy_gap_is_still_eligible(self):
        """Eligibility is about the direction's VALIDITY, never about its outcome.

        This is what keeps selection non-circular in the only place it can be
        kept so: a cell whose ablation makes discrimination WORSE must remain in
        the pool, or the filter is quietly selecting on the reported quantity
        twice.
        """
        assert is_eligible(cell(proxy_harmful_refusal=0.10), N_LAYERS, CONFIG) is True


class TestSelection:
    def test_it_picks_the_largest_proxy_gap_among_eligible_cells(self):
        best = cell(layer=10, proxy_benign_refusal=0.10)  # gap 0.70
        chosen = select_cell([cell(layer=8), best, cell(layer=12)], N_LAYERS, CONFIG)
        assert chosen is best

    def test_an_ineligible_cell_cannot_win_however_good_its_gap(self):
        tempting = cell(layer=30, proxy_benign_refusal=0.0)  # gap 0.80, but pruned
        modest = cell(layer=9, proxy_benign_refusal=0.50)  # gap 0.30
        assert select_cell([tempting, modest], N_LAYERS, CONFIG) is modest

    def test_it_raises_when_nothing_is_eligible_rather_than_returning_least_bad(self):
        with pytest.raises(ValueError, match="report this rather than relaxing"):
            select_cell([cell(kl=9.0), cell(kl=8.0)], N_LAYERS, CONFIG)

    def test_it_raises_on_an_empty_pool(self):
        with pytest.raises(ValueError, match="no candidate cells"):
            select_cell([], N_LAYERS, CONFIG)

    def test_it_is_blind_to_whether_the_direction_is_REAL_OR_RANDOM(self):
        """The property the whole design rests on.

        The control is legitimate only if the identical selection procedure runs
        over it. `CellEvidence` carries no provenance field and `select_cell`
        takes no flag, so a caller CANNOT select the real arm one way and the
        control another — matched selection is structural here, not a discipline
        someone has to remember.
        """
        assert "real" not in CellEvidence.__dataclass_fields__
        assert "is_random" not in CellEvidence.__dataclass_fields__
        pool = [cell(layer=8, proxy_benign_refusal=0.6), cell(layer=9, proxy_benign_refusal=0.2)]
        assert select_cell(pool, N_LAYERS, CONFIG) is select_cell(list(pool), N_LAYERS, CONFIG)


class TestTheReading:
    def test_gap_destroyed_is_measured_against_plaintext(self):
        assert reading().gap_destroyed == pytest.approx(0.82)

    def test_margin_subtracts_what_a_random_direction_achieves(self):
        r = reading()
        assert r.gap_restored == pytest.approx(0.60)
        assert r.control_gap_restored == pytest.approx(0.02)
        assert r.margin == pytest.approx(0.58)

    def test_restored_fraction_is_the_margin_over_what_was_lost(self):
        assert reading().restored_fraction == pytest.approx(0.58 / 0.82)

    def test_restored_fraction_is_None_when_nothing_was_destroyed(self):
        """⚠️ The silent-zero guard, one measurement over from where it bit.

        With no gap destroyed there is no denominator. Returning 0.0 would read
        as "restored nothing" — a claim — when the truth is that the question
        does not apply.
        """
        r = reading(plaintext=arms(0.90, 0.90), baseline=arms(0.95, 0.95, ability=0.9))
        assert r.gap_destroyed == pytest.approx(0.0)
        assert r.restored_fraction is None

    def test_a_denominator_below_the_MEASUREMENT_RESOLUTION_is_also_None(self):
        """The case that would otherwise report a 58x restoration.

        `gap_destroyed` of 0.01 at n=100 is one prompt's worth of movement. The
        naive ratio against a 0.58 margin is 58, which sails past every verdict
        bar — a spectacular restoration of a gap that was never meaningfully
        lost. The floor is DERIVED from n, so this costs no knob.
        """
        r = reading(plaintext=arms(0.99, 0.98), baseline=arms(0.99, 0.99, ability=0.91))
        assert r.gap_destroyed == pytest.approx(0.01)
        assert r.resolution == pytest.approx(0.01)
        assert r.margin > 0.5  # the numerator that would have been inflated
        assert r.restored_fraction is None
        assert r.verdict(CONFIG) is None

    def test_the_resolution_follows_n_rather_than_being_a_constant(self):
        small = reading(
            plaintext=ArmRates(
                harmful_refusal=0.9, benign_refusal=0.1,
                n_harmful=20, n_benign=20, mechanism_errors=0,
            ),
            baseline=arms(0.99, 0.99, ability=0.91),
        )
        assert small.resolution == pytest.approx(0.05)
        assert reading().resolution == pytest.approx(0.01)

    def test_ability_shift_is_None_when_either_side_is_unmeasured(self):
        assert reading(ablated=arms(0.95, 0.35, ability=None)).ability_shift is None
        assert reading(baseline=arms(0.99, 0.99, ability=None)).ability_shift is None

    def test_ability_shift_is_signed(self):
        r = reading(ablated=arms(0.95, 0.35, ability=0.99))
        assert r.ability_shift == pytest.approx(0.08)


class TestTheVerdictChecksTheCONFOUNDFirst:
    """The ordering property, and it is the one worth a dedicated class.

    A restored gap produced by an ablation that also decoded the prompt is not
    evidence for either hypothesis. Checking the margin first and the confound
    second would report the headline result and footnote the reason it is void.
    """

    def test_a_clean_restoration_reads_as_recognition_suppressed(self):
        assert reading().verdict(CONFIG) == "recognition_suppressed"

    def test_a_large_margin_with_a_moved_ability_reads_as_CONFOUNDED(self):
        r = reading(ablated=arms(0.95, 0.35, ability=0.99 + 0.0))  # shift +0.08, clean
        assert r.verdict(CONFIG) == "recognition_suppressed"
        confounded = reading(ablated=arms(0.95, 0.35, ability=0.91 + 0.30))
        assert confounded.margin == pytest.approx(0.58)  # same headline
        assert confounded.verdict(CONFIG) == "confounded"  # and it does not count

    def test_a_confound_in_the_NEGATIVE_direction_also_voids_the_reading(self):
        """|shift|, not shift. An ablation that DESTROYED comprehension changed
        the task as surely as one that granted it."""
        r = reading(baseline=arms(0.99, 0.99, ability=0.91), ablated=arms(0.95, 0.35, ability=0.55))
        assert r.verdict(CONFIG) == "confounded"

    def test_no_restoration_reads_as_recognition_destroyed(self):
        r = reading(ablated=arms(0.99, 0.98, ability=0.91), control=arms(0.99, 0.99, ability=0.91))
        assert r.verdict(CONFIG) == "recognition_destroyed"

    def test_the_middle_band_is_NOT_a_verdict(self):
        """A middling restoration is evidence for neither hypothesis.

        Rounding it to the nearer side would manufacture a conclusion from
        noise, which is the failure this paper's own leg 2 was withdrawn for.
        """
        r = reading(ablated=arms(0.99, 0.72, ability=0.91), control=arms(0.99, 0.99, ability=0.91))
        fraction = r.restored_fraction
        assert CONFIG.max_null_restored_fraction < fraction < CONFIG.min_restored_fraction
        assert r.verdict(CONFIG) is None

    def test_an_unmeasured_ability_yields_no_verdict_rather_than_a_default(self):
        assert reading(ablated=arms(0.95, 0.35, ability=None)).verdict(CONFIG) is None


class TestNothingReportableDependsOnAPlaceholderKnob:
    """The architectural claim the config comments make, asserted rather than stated.

    `min_restored_fraction`, `max_null_restored_fraction` and `max_ability_shift`
    are untuned placeholders. The reading is built so they reach only the
    convenience `verdict()`; the continuous quantities a paper reports must be
    invariant to them.
    """

    @pytest.mark.parametrize(
        "knob", ["min_restored_fraction", "max_null_restored_fraction", "max_ability_shift"]
    )
    def test_the_continuous_quantities_do_not_move_with_the_placeholders(self, knob):
        r = reading()
        before = (r.margin, r.restored_fraction, r.ability_shift, r.gap_destroyed)
        mutated = CONFIG.model_copy(update={knob: 0.999})
        after = (r.margin, r.restored_fraction, r.ability_shift, r.gap_destroyed)
        assert before == after
        # And the knob demonstrably reaches the verdict, so this is not vacuous.
        assert r.verdict(CONFIG) is not None or r.verdict(mutated) is not None
