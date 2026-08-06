"""I1 — the Patchscopes decode measurement.

Hermetic: mechanism, shapes, control construction and failure modes on the tiny
model. The semantic question — does a state captured on `zero_width` ciphertext
actually decode to plaintext — needs real weights and is the validation gate in
build plan §3.1, not a unit test.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import DecodeLensConfig
from internals_safety.measurements.decode_lens import (
    LensCurve,
    LensReading,
    content_token_ids,
    decode_rate,
    derange,
    read_layer,
    sweep_layers,
)
from internals_safety.models.capture import capture_activations
from internals_safety.models.loader import prepare_prompts


@pytest.fixture
def captured(tiny_model, messages):
    prompts = prepare_prompts(tiny_model, messages, positions=["instruction_final", "last"])
    return capture_activations(
        tiny_model, prompts, layers=[0, 1, 2], positions=["instruction_final", "last"]
    )


def test_derangement_leaves_no_row_scored_against_itself():
    items = ["a", "b", "c", "d"]
    assert all(new != old for new, old in zip(derange(items), items))


def test_derangement_is_a_permutation_not_a_resample():
    items = ["a", "b", "c"]
    assert sorted(derange(items)) == sorted(items)


def test_derangement_needs_two_items():
    with pytest.raises(ValueError, match="at least two"):
        derange(["only"])


def test_content_tokens_cover_both_spacings(tiny_model):
    """BPE gives " word" and "word" different ids; which appears depends on
    what precedes it in the scaffold, so both must be scored."""
    ids = content_token_ids(tiny_model, "assistant")
    bare = set(tiny_model.tokenizer.encode("assistant", add_special_tokens=False))
    spaced = set(tiny_model.tokenizer.encode(" assistant", add_special_tokens=False))
    assert bare <= ids and spaced <= ids


def test_content_tokens_drop_stopwords_when_asked(tiny_model):
    with_stop = content_token_ids(tiny_model, "user system")
    without = content_token_ids(tiny_model, "user system", stopwords=frozenset({"system"}))
    assert without < with_stop


def test_content_tokens_strip_punctuation(tiny_model):
    assert content_token_ids(tiny_model, "user.") == content_token_ids(tiny_model, "user")


def test_read_layer_returns_one_reading_per_prompt(tiny_model, captured, messages):
    readings = read_layer(
        tiny_model, captured, layer=1, position="last", plaintexts=messages,
        config=DecodeLensConfig(batch_size=2),
    )
    assert len(readings) == len(messages)
    assert all(reading.layer == 1 for reading in readings)


def test_readings_are_probabilities(tiny_model, captured, messages):
    readings = read_layer(
        tiny_model, captured, layer=0, position="last", plaintexts=messages,
        config=DecodeLensConfig(),
    )
    for reading in readings:
        assert 0.0 <= reading.matched <= 1.0
        assert 0.0 <= reading.mismatched <= 1.0


def test_batching_does_not_change_the_readings(tiny_model, captured, messages):
    """A measurement that depends on batch composition is not a measurement."""
    one = read_layer(
        tiny_model, captured, layer=2, position="last", plaintexts=messages,
        config=DecodeLensConfig(batch_size=1),
    )
    many = read_layer(
        tiny_model, captured, layer=2, position="last", plaintexts=messages,
        config=DecodeLensConfig(batch_size=8),
    )
    for a, b in zip(one, many):
        assert a.matched == pytest.approx(b.matched, abs=1e-5)
        assert a.mismatched == pytest.approx(b.mismatched, abs=1e-5)


def test_read_layer_needs_two_prompts_for_the_control(tiny_model, captured):
    with pytest.raises(ValueError, match="at least two prompts"):
        read_layer(
            tiny_model, captured, layer=0, position="last", plaintexts=["only one"],
            config=DecodeLensConfig(),
        )


def test_read_layer_rejects_a_plaintext_count_mismatch(tiny_model, captured):
    with pytest.raises(ValueError, match="activations but"):
        read_layer(
            tiny_model, captured, layer=0, position="last", plaintexts=["a", "b"],
            config=DecodeLensConfig(),
        )


def test_sweep_returns_one_curve_per_prompt_over_every_layer(tiny_model, captured, messages):
    curves = sweep_layers(
        tiny_model, captured, position="last", plaintexts=messages, config=DecodeLensConfig()
    )
    assert len(curves) == len(messages)
    assert all(len(curve.readings) == len(captured.layers) for curve in curves)


def test_sweep_rejects_an_uncaptured_layer(tiny_model, captured, messages):
    with pytest.raises(ValueError, match="were not captured"):
        sweep_layers(
            tiny_model, captured, position="last", plaintexts=messages,
            config=DecodeLensConfig(layers=[0, 99]),
        )


def test_margin_is_matched_minus_control():
    assert LensReading(layer=4, matched=0.30, mismatched=0.08).margin == pytest.approx(0.22)


def test_a_cell_decodes_only_when_it_clears_the_margin():
    reading = LensReading(layer=4, matched=0.30, mismatched=0.28)
    assert not reading.decodes(min_margin=0.05)
    assert reading.decodes(min_margin=0.01)


def test_the_curve_reports_its_peak_layer_not_a_mean():
    """§3.1 wants the curve — decoding is expected to peak intermediate-late,
    and a mean would wash out exactly that structure."""
    curve = LensCurve(
        readings=[
            LensReading(layer=0, matched=0.10, mismatched=0.09),
            LensReading(layer=18, matched=0.40, mismatched=0.05),
            LensReading(layer=31, matched=0.20, mismatched=0.10),
        ]
    )
    assert curve.peak_layer == 18
    assert curve.best.margin == pytest.approx(0.35)


def test_decode_rate_counts_prompts_clearing_the_margin():
    hot = LensCurve(readings=[LensReading(layer=1, matched=0.5, mismatched=0.1)])
    cold = LensCurve(readings=[LensReading(layer=1, matched=0.11, mismatched=0.10)])
    assert decode_rate([hot, cold, cold], min_margin=0.05) == pytest.approx(1 / 3)


def test_decode_rate_refuses_an_empty_run():
    """Returning 0.0 would read as 'nothing decoded here' — the silent-zero
    failure this repo has already fixed twice (instrument_layer §1.5)."""
    with pytest.raises(ValueError, match="no curves"):
        decode_rate([], min_margin=0.05)
