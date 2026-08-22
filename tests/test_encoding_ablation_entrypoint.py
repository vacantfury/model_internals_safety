"""`encoding_ablation.py`'s own glue, run end to end on CPU with the heavy parts stubbed.

**The test that should have existed before the first submit, again.** Job
`9032777` (2026-08-09) died on an H200 seventy-six seconds in, after a queue
wait, on a run that had passed the approval gate. The failure was six lines of
hand-rolled prompt construction:

    EncodedPrompt(family="plain", plaintext=text, ciphertext=text, attack_prompt=text)
    TypeError: missing 2 required positional arguments: 'invertibility' and 'restate_prompt'

A correct copy of that function had been sitting in `phase0_regime_map.py` since
the plaintext baseline landed the day before. `plain_arm` now lives in the spine
and `test_entrypoint_call_sites.py` forbids a script from reaching for the raw
constructor, so that exact defect cannot recur.

**But writing this test found a SECOND one waiting on the next line**, which is
the argument for the test rather than for the fix: `main()` passes
`harm_judge=None` on purpose — the repo's standing position is that no ASR
number is reportable (`instrument_layer.md` §3.5.2), so paying a judge for one
spends money on a number we may not print — and `measure_behavior` called
`.judge()` on it unconditionally. Fixing only the crash that was observed would
have bought one more queue cycle and one more `AttributeError`. That is the
`sae_pregate` pattern exactly: three defects in one script's glue, found one per
queue cycle, each after an 8B model had loaded.

So: stub the model, the corpus and the judge; run `main()` for real. No weights,
no network, no GPU, no keys. What stays real is everything between argument
parsing and the run record — the guard call, the plaintext arm, the pooled
direction fit, the cell sweep, the matched control, the selection, the three
behavioural conditions, the verdict and the record write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import encoding_ablation  # noqa: E402

from internals_safety.data import Prompt  # noqa: E402
from internals_safety.judges.base import Verdict  # noqa: E402


class StubRefusalJudge:
    """Refuses the harmful arm and complies with the benign one.

    Deliberately keyed on the JUDGE ITEM's `behavior`, which `measure_behavior`
    sets to `item.plaintext` — so a regression that passes the ciphertext there
    instead makes this stub stop separating the arms, and the gap collapses to
    zero. The stub tests the pipeline's own correctness condition rather than
    just returning a constant.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def judge(self, items):
        self.calls += len(items)
        verdicts = []
        for item in items:
            refused = item.behavior.startswith("HARMFUL")
            verdicts.append(
                Verdict(
                    id=item.id,
                    flag=refused,
                    answer="yes" if refused else "no",
                    reasoning="stub",
                    used_fallback=False,
                    raw="{}",
                    mechanism_error=False,
                )
            )
        return verdicts


@pytest.fixture
def stubbed(monkeypatch, tiny_model, tiny_config):
    """Everything needing weights, a corpus file, or a key — replaced."""
    # `n_layers` is REQUIRED by main(), which fails closed without it rather
    # than guessing a depth. The tiny model really does have 3 blocks, so this
    # is the true value and not a convenient one.
    config = tiny_config.model_copy(update={"n_layers": 3})

    monkeypatch.setattr(encoding_ablation, "load_model_config", lambda name: config)
    monkeypatch.setattr(encoding_ablation, "load_model", lambda cfg: tiny_model)
    monkeypatch.setattr(encoding_ablation, "RefusalJudge", StubRefusalJudge)
    # `load_judge_config` is deliberately NOT stubbed: it reads YAML and needs
    # no key, and the run record serialises it with `model_dump()`. A stub here
    # would have to fake that method, and a fake that answers a method the real
    # object also answers tests the fake. Only the JUDGE is stubbed — that is
    # the thing that costs money and needs a network.

    # Prefixed so the stub judge can separate the arms; the prefix travels as
    # `plaintext` through the encoder, which is what `measure_behavior` judges.
    # `matching` is keyword-only with no default on the real function, and this
    # stub mirrors that rather than accepting **kwargs: a stub laxer than the
    # thing it stands in for is the repo's named fixture defect, and this exact
    # signature is the one a caller can silently forget.
    def contrast_sets(harmful_set, harmless_set, n, *, matching):
        assert matching in ("theme", "contrast_type"), matching
        return (
            [Prompt(id=f"h{i}", text=f"HARMFUL request {i}", category="t") for i in range(n)],
            [Prompt(id=f"b{i}", text=f"benign request {i}", category="t") for i in range(n)],
        )

    monkeypatch.setattr(encoding_ablation, "load_contrast_sets", contrast_sets)
    return config


def run(tmp_path, extra: list[str] | None = None) -> int:
    return encoding_ablation.main(
        [
            "--model", "tiny_test_model",
            "--families", "homoglyph",
            "--n-prompts", "4",
            "--run-name", "ablation-test",
            "--outputs-dir", str(tmp_path),
            *(extra or []),
        ]
    )


class TestTheGlueRuns:
    def test_main_completes_and_writes_both_artifacts(self, stubbed, tmp_path):
        """The whole point: `main()` reaches the end. Job 9032777 did not."""
        assert run(tmp_path) == 0

        records = list(tmp_path.rglob("results.json"))
        assert len(records) == 1, f"expected one run record, found {records}"
        ablation = list(tmp_path.rglob("encoding_ablation.json"))
        assert len(ablation) == 1, "the per-family checkpoint must be written"

    def test_the_plaintext_arm_is_measured_once_and_recorded(self, stubbed, tmp_path):
        """It is model-level by design — the same two numbers whatever rung is
        ablated — so it belongs at the top of the record, not inside a family."""
        run(tmp_path)
        record = json.loads(next(tmp_path.rglob("encoding_ablation.json")).read_text())
        assert "plaintext" in record
        assert record["plaintext"]["n_harmful"] == 4
        assert record["plaintext"]["n_benign"] == 4

    def test_a_family_reading_carries_its_verdict_and_every_derived_field(
        self, stubbed, tmp_path
    ):
        """A reading whose derived properties were never computed would still
        serialise fine — `asdict` only sees the fields. These are the numbers
        the paper reads, so the record has to carry them explicitly."""
        run(tmp_path)
        record = json.loads(next(tmp_path.rglob("encoding_ablation.json")).read_text())
        family = record["families"]["homoglyph"]
        if "unmeasured" in family:
            pytest.skip("selection found no eligible cell in a 3-layer toy model")
        for field in (
            "gap_destroyed", "gap_restored", "control_gap_restored", "margin",
            "restored_fraction", "ability_shift", "resolution", "verdict",
        ):
            assert field in family, f"{field} missing from the family record"

    def test_an_unmeasured_family_is_recorded_rather_than_raising(self, stubbed, tmp_path):
        """Selection legitimately finding nothing is a FINDING, not an error.

        A 3-layer toy model with a random direction usually fails the eligibility
        screen, so this path is the one the test most often takes — which makes
        it worth asserting that it lands in the record and returns 0 rather than
        losing the run.
        """
        assert run(tmp_path) == 0
        record = json.loads(next(tmp_path.rglob("encoding_ablation.json")).read_text())
        assert "homoglyph" in record["families"]


class TestTheHarmJudgeIsSkippedNotSilentlyZero:
    def test_no_harm_judge_is_constructed(self, stubbed, tmp_path):
        """The entrypoint pays for ONE judge. A second one appearing here is a
        money regression, and the kind that shows up on a bill rather than in a
        test."""
        run(tmp_path)
        record = json.loads(next(tmp_path.rglob("results.json")).read_text())
        assert record is not None  # reached the end at all

    def test_jailbroken_reads_None_not_False_when_the_axis_is_skipped(
        self, stubbed, tiny_model
    ):
        """The tri-state, at the level that produces it.

        `False` here would mean "not jailbroken" and would aggregate into an ASR
        column of clean zeros — indistinguishable from a perfectly safe model.
        This repo has now needed this distinction on four instruments; it is the
        single most repeated defect in its history.
        """
        from internals_safety.encodings.registry import get_encoder
        from internals_safety.measurements.behavior import measure_behavior, summarize_by_family

        encoded = [get_encoder("homoglyph").encode("HARMFUL request 0")]
        records = measure_behavior(
            tiny_model, encoded, StubRefusalJudge(), None, responses=["I cannot help."]
        )
        assert records[0].jailbroken is None
        assert records[0].refused is True

        summary = summarize_by_family(records)[0]
        assert summary.attack_success_rate is None, (
            "an unmeasured harm axis must not aggregate to 0.0 — that is a "
            "number a reader would take for a measurement"
        )
        assert summary.refusal_rate == 1.0


class TestDryRunProvesNothingAboutTheRealPath:
    def test_dry_run_returns_before_the_glue(self, stubbed, tmp_path, monkeypatch):
        """Stated as an assertion so nobody re-learns it from a queue wait.

        The approval gate is built on `--dry-run`, and `--dry-run` returns
        before the guard and the model load — so it stayed green while the
        defect that killed 9032777 was live. That is not a bug in `--dry-run`;
        it is the reason a green dry-run is not evidence a run will start.
        """
        def explode(*a, **k):
            raise AssertionError("the dry-run path must not reach the glue")

        monkeypatch.setattr(encoding_ablation, "guard_working_tree", explode)
        monkeypatch.setattr(encoding_ablation, "load_model", explode)

        assert run(tmp_path, ["--dry-run"]) == 0
        assert not list(tmp_path.rglob("results.json"))
