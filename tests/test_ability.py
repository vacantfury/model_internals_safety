"""Measurement #1 tests — generation plumbing and the difficulty ordering."""

from __future__ import annotations

import inspect

import pytest

from internals_safety.config import AbilityConfig, load_measurements_config
from internals_safety.encodings.recovery import RecoveryScore
from internals_safety.encodings.registry import load_ladder
from internals_safety.measurements.ability import (
    AbilityRecord,
    measure_ability,
    summarize_by_family,
)
from internals_safety.models.generate import generate


def _record(family: str, recovered: bool, similarity: float, echoed: bool = False):
    return AbilityRecord(
        family=family,
        plaintext="p",
        ciphertext="c",
        response="r",
        score=RecoveryScore(
            exact=recovered,
            contains=recovered,
            similarity=similarity,
            echoed_ciphertext=echoed,
            # Tracks `recovered` on purpose: a non-recovery must not carry high
            # content overlap, or it would pass the order-blind branch and this
            # fixture would stop testing what it claims to.
            content_overlap=1.0 if recovered else 0.0,
        ),
    )


def test_measurements_config_loads():
    config = load_measurements_config()
    assert config.ability.max_new_tokens > 0
    assert config.ability.batch_size > 0


def test_generation_returns_one_response_per_message(tiny_model, messages):
    responses = generate(tiny_model, messages, max_new_tokens=4, batch_size=2)
    assert len(responses) == len(messages)
    assert all(isinstance(response, str) for response in responses)


def test_batch_size_cannot_be_omitted():
    """`batch_size` moves generated text, so it may not carry a default.

    The retired marker called it plumbing on the grounds that "greedy decoding is
    batch-invariant". The paper's own §7 refutes that: only 12–58% of responses
    repeat byte-identically across nominally greedy runs, because batch
    composition changes the reduction order in the matmuls and the argmax can
    land elsewhere. An optional flag defaulting to the convenient value is the
    shape that has failed in this repo five times, so the fix is the one that
    worked there — make the omission inexpressible rather than detectable.

    Pinned on the SIGNATURE rather than on a call, because `generate` is reached
    only through the library and would be vacuous in
    `test_entrypoint_call_sites.WATCHED`.
    """
    parameter = inspect.signature(generate).parameters["batch_size"]
    assert parameter.default is inspect.Parameter.empty, (
        "batch_size regained a default; a knob that changes generated text must "
        "be passed, and every caller reads it from measurements.yaml"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_measure_ability_pairs_every_encoded_prompt_with_a_response(tiny_model):
    ladder = load_ladder()
    encoded = [ladder[family].encode("bake bread") for family in ("base64", "rot13", "morse")]
    records = measure_ability(
        tiny_model, encoded, AbilityConfig(max_new_tokens=4, batch_size=2)
    )

    assert [record.family for record in records] == ["base64", "rot13", "morse"]
    for record, item in zip(records, encoded):
        assert record.plaintext == item.plaintext
        assert record.ciphertext == item.ciphertext
        assert 0.0 <= record.score.similarity <= 1.0


def test_summary_orders_hardest_first():
    """The ordering is the deliverable: it is the ladder's difficulty axis, and
    the point of measuring it is that RQ2 stops relying on a hand-sorted list."""
    records = [
        _record("base64", True, 1.0),
        _record("base64", True, 1.0),
        _record("morse", False, 0.2),
        _record("morse", False, 0.3),
        _record("rot13", True, 0.9),
        _record("rot13", False, 0.4),
    ]
    summaries = summarize_by_family(records)

    assert [summary.family for summary in summaries] == ["morse", "rot13", "base64"]
    assert summaries[0].recovery_rate == 0.0
    assert summaries[-1].recovery_rate == 1.0
    assert summaries[1].n == 2


def test_echo_rate_is_reported_separately():
    summaries = summarize_by_family([_record("hex", False, 0.1, echoed=True)])
    assert summaries[0].echo_rate == 1.0


@pytest.mark.slow
def test_ability_runs_end_to_end_on_real_weights():
    """Not an assertion about whether a 0.5B model can decode Base64 (it mostly
    cannot) — an assertion that the measurement produces scored records."""
    from internals_safety.config import load_model_config
    from internals_safety.models.loader import load_model

    config = load_model_config("qwen2_5_0_5b_instruct").model_copy(
        update={"device": "cpu", "dtype": "float32"}
    )
    loaded = load_model(config)
    ladder = load_ladder()
    encoded = [
        ladder[family].encode("How do I bake bread?")
        for family in ("base64", "reverse_characters")
    ]

    records = measure_ability(loaded, encoded, AbilityConfig(max_new_tokens=48, batch_size=2))
    assert len(records) == 2
    assert all(record.response for record in records)
    assert all(0.0 <= record.score.similarity <= 1.0 for record in records)
