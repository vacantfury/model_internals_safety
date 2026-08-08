"""The loader's scale ASSUMPTION must be recorded, not trusted.

**Why this file exists.** `LlamaScopeSAE._normalise` scales activations by
`sqrt(d_model) / input_norm`, where `input_norm` is the checkpoint's
`dataset_average_activation_norm`. That is a claim about our activations — that
they arrive with that average norm — and nothing checked it. Job `9008803` read
`variance_explained` of -915 to -1760 on Llama-3.1-8B-**Base**, the model the
dictionary was fitted on, while a matched-shape RANDOM dictionary scored +0.13
to +0.21 against it. A trained dictionary cannot lose to random on its own
training distribution, so the input scale was the remaining suspect, and it was
unmeasured: the run record carried MSE and variance but never the one number
that would say whether the dictionary was being fed the space it expects.

**The fixture rule this file obeys.** `RandomDictionary` declares neither
`input_norm` nor `trained_on`, so a test built only on it would exercise the
absent branch of both `getattr` calls and never the present one — a fixture more
permissive than the real thing, which CLAUDE.md records four instances of. So
the scale tests here use **`LlamaScopeSAE` itself**, constructed from small hand
-built tensors. No download, no GPU: it is the real class, and it is the strict
one, because it is the only one that normalises.
"""

from __future__ import annotations

import torch

from internals_safety.config import SAEConfig
from internals_safety.measurements.sae_reconstruction import (
    RandomDictionary,
    _substitute,
    ReconstructionQuality,
    loaded_model_name,
    mean_activation_norm,
    measure_reconstruction,
    reading,
    scored_positions,
    tokenize_for_reconstruction,
)
from internals_safety.models.loader import prepare_prompts
from internals_safety.models.sae_loader import LlamaScopeSAE

CONFIG = SAEConfig(
    trained_on="a config string that must LOSE to the checkpoint's own claim",
    min_kl_recovered=0.8,
    min_transfer_ratio=0.8,
)


def tiny_scope_sae(d_model: int, *, input_norm: float, d_sae: int = 32) -> LlamaScopeSAE:
    """A real `LlamaScopeSAE` at toy scale — the strict fixture.

    `input_norm` is the knob under test: it is the only thing standing between
    raw activations and the space the dictionary was fitted in.
    """
    generator = torch.Generator().manual_seed(0)
    return LlamaScopeSAE(
        encoder_weight=torch.randn(d_sae, d_model, generator=generator) * 0.05,
        encoder_bias=torch.zeros(d_sae),
        decoder_weight=torch.randn(d_model, d_sae, generator=generator) * 0.05,
        decoder_bias=torch.zeros(d_model),
        trained_on="the-checkpoints-own-claim",
        hook_point="blocks.0.hook_resid_post",
        jump_relu_threshold=0.0,
        input_norm=input_norm,
        output_norm=input_norm,
        d_model=d_model,
        d_sae=d_sae,
        nominal_top_k=50,
    )


class TestMeanActivationNorm:
    def test_it_is_the_mean_l2_norm_per_vector(self):
        activations = torch.tensor([[3.0, 4.0], [0.0, 1.0]])  # norms 5 and 1
        assert mean_activation_norm(activations) == 3.0

    def test_it_is_not_the_norm_of_the_mean(self):
        """Two opposite vectors average to zero but each has norm 1. Collapsing
        the two would report ~0 for a distribution that is nowhere near zero."""
        activations = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        assert mean_activation_norm(activations) == 1.0


class TestTheRatioIsReportedNotAssumed:
    def _quality(self, **kwargs) -> ReconstructionQuality:
        base = dict(
            layer=1, n_prompts=1, mse=0.0, variance_explained=0.0,
            l0=1.0, kl_sae=0.1, kl_ablated=1.0,
        )
        return ReconstructionQuality(**{**base, **kwargs})

    def test_the_ratio_is_observed_over_declared(self):
        quality = self._quality(observed_activation_norm=70.0, declared_input_norm=14.0)
        assert quality.norm_ratio == 5.0

    def test_a_missing_observation_is_none_not_one(self):
        """None means "not measured". Returning 1.0 would assert agreement that
        was never checked — the silent-`False` shape this repo has paid for
        three times in this very instrument."""
        quality = self._quality(observed_activation_norm=None, declared_input_norm=14.0)
        assert quality.norm_ratio is None

    def test_a_dictionary_that_makes_no_scale_claim_is_none(self):
        quality = self._quality(observed_activation_norm=70.0, declared_input_norm=None)
        assert quality.norm_ratio is None

    def test_a_zero_declared_norm_does_not_divide_by_zero(self):
        quality = self._quality(observed_activation_norm=70.0, declared_input_norm=0.0)
        assert quality.norm_ratio is None


class TestTheDiagnosticActuallyFires:
    """A mismatched `input_norm` must SHOW as a ratio far from 1.

    This is the mutation: the diagnostic is only worth its line count if a wrong
    scale is visible in it. Two dictionaries differing ONLY in `input_norm` are
    run over the same model, and the ratio must move by exactly that factor
    while the observed norm — a property of the model, not the dictionary —
    must not move at all.
    """

    def _run(self, tiny_model, input_norm: float) -> ReconstructionQuality:
        return measure_reconstruction(
            tiny_model,
            tiny_scope_sae(tiny_model.model.config.hidden_size, input_norm=input_norm),
            ["hi", "a somewhat longer prompt with more tokens"],
            layer=1,
            config=CONFIG,
            batch_size=2,
        )

    def test_a_wrong_declared_norm_moves_the_ratio_and_only_the_ratio(self, tiny_model):
        honest = self._run(tiny_model, input_norm=1.0)
        wrong = self._run(tiny_model, input_norm=10.0)

        assert honest.observed_activation_norm is not None
        assert abs(
            honest.observed_activation_norm - wrong.observed_activation_norm
        ) < 1e-4, "the observed norm is a property of the MODEL and must not track the dictionary"
        assert honest.norm_ratio is not None and wrong.norm_ratio is not None
        assert abs(honest.norm_ratio / wrong.norm_ratio - 10.0) < 1e-3

    def test_the_ratio_reaches_the_run_record(self, tiny_model):
        """A diagnostic that never leaves the dataclass diagnoses nothing —
        `results.json` is where a human reads this."""
        record = reading(self._run(tiny_model, input_norm=1.0), CONFIG)
        assert record.detail["norm_ratio"] is not None
        assert record.detail["observed_activation_norm"] is not None
        assert record.detail["declared_input_norm"] == 1.0


class TestTheNormIsScoredOverTheKeptPositionsOnly:
    """Checked against an INDEPENDENTLY captured ground truth, not against a
    batched-vs-unbatched invariant.

    **The invariant version of this test was vacuous and was caught by mutating
    the code it guards.** Restoring the unmasked reduction left it green: in this
    fixture pad positions carry ~zero activation, so a norm summed over pads
    divided by the kept count is numerically indistinguishable from the masked
    one. `tiny_model` therefore CANNOT express pad contamination in a norm
    statistic — a fifth instance of the fixture rule, found the way the rule says
    to find it. What it can express is BOS, which is an ordinary real token here,
    so the assertion is pinned on the quantity the fixture can actually carry and
    is compared against a value computed outside the function under test.
    """

    def _capture(self, model, prompts, layer):
        """The same hidden states `measure_reconstruction` reduces over."""
        rendered = [p.text for p in prepare_prompts(model, prompts, positions=[])]
        encoded = tokenize_for_reconstruction(model.tokenizer, rendered, render_chat=True)
        inputs = {key: value.to(model.device) for key, value in encoded.items()}
        grabbed: list[torch.Tensor] = []

        def grab(hidden):
            grabbed.append(hidden.detach().float().cpu())
            return hidden

        with torch.inference_mode():
            with _substitute(model, layer, grab):
                model.model(**inputs)
        return grabbed[0], encoded["input_ids"], encoded["attention_mask"]

    def test_the_reported_norm_is_the_mean_over_kept_positions(self, tiny_bos_model):
        # `tiny_bos_model`, not `tiny_model`: since BOS is dropped BY IDENTITY
        # (2026-08-08) a fixture with no BOS drops nothing, and this test's own
        # vacuity guard would fire. The quantity under test only exists on a
        # model that has a BOS — which is every model the SAE gate runs on.
        tiny_model = tiny_bos_model
        prompts = ["hi", "a much longer prompt with many more tokens", "middling one"]
        sae = tiny_scope_sae(tiny_model.model.config.hidden_size, input_norm=1.0)
        quality = measure_reconstruction(
            tiny_model, sae, prompts, layer=1, config=CONFIG, batch_size=3
        )

        hidden, ids, mask = self._capture(tiny_model, prompts, layer=1)
        flat = hidden.reshape(-1, hidden.shape[-1])
        kept = scored_positions(
            ids, mask, bos_token_id=tiny_model.tokenizer.bos_token_id
        ).reshape(-1)
        every_real = mask.bool().reshape(-1)

        assert int(kept.sum()) < int(every_real.sum()), "nothing was dropped — test is vacuous"
        over_kept = float(flat[kept].norm(dim=-1).mean())
        including_bos = float(flat[every_real].norm(dim=-1).mean())
        assert abs(over_kept - including_bos) > 1e-6, (
            "BOS costs nothing in this fixture, so the test cannot discriminate"
        )

        assert abs(quality.observed_activation_norm - over_kept) < 1e-4, (
            f"reported {quality.observed_activation_norm}, kept-only mean is "
            f"{over_kept}, BOS-inclusive mean is {including_bos} — the reduction "
            "is not seeing the mask"
        )


class TestProvenanceIsRecordedNotPlaceholdered:
    def test_evaluated_on_names_the_model_that_ran(self, tiny_model):
        quality = measure_reconstruction(
            tiny_model,
            RandomDictionary(
                d_model=tiny_model.model.config.hidden_size, n_features=32, k=4,
                generator=torch.Generator().manual_seed(0),
            ),
            ["hi"],
            layer=1,
            config=CONFIG,
            batch_size=1,
        )
        assert quality.evaluated_on == tiny_model.config.hf_id
        assert "TODO" not in loaded_model_name(quality)

    def test_a_hand_built_quality_says_so_rather_than_naming_a_model(self):
        quality = ReconstructionQuality(
            layer=1, n_prompts=1, mse=0.0, variance_explained=0.0,
            l0=1.0, kl_sae=0.1, kl_ablated=1.0,
        )
        assert "not recorded" in loaded_model_name(quality)

    def test_the_checkpoints_trained_on_beats_the_config_string(self, tiny_model):
        """The config knob is hand-maintained; the checkpoint's is the artifact's
        own claim. A mislabelled config is exactly the error that would make this
        gate compare a dictionary against the wrong baseline."""
        quality = measure_reconstruction(
            tiny_model,
            tiny_scope_sae(tiny_model.model.config.hidden_size, input_norm=1.0),
            ["hi"],
            layer=1,
            config=CONFIG,
            batch_size=1,
        )
        assert reading(quality, CONFIG).detail["trained_on"] == "the-checkpoints-own-claim"

    def test_the_config_still_answers_when_the_dictionary_is_silent(self, tiny_model):
        """`RandomDictionary` declares nothing, and the record must not go blank
        — absent is a fallback to the config, never an empty field."""
        quality = measure_reconstruction(
            tiny_model,
            RandomDictionary(
                d_model=tiny_model.model.config.hidden_size, n_features=32, k=4,
                generator=torch.Generator().manual_seed(0),
            ),
            ["hi"],
            layer=1,
            config=CONFIG,
            batch_size=1,
        )
        assert reading(quality, CONFIG).detail["trained_on"] == CONFIG.trained_on
