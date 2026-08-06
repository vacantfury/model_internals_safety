"""The layer count: declared per model, verified against the checkpoint.

Regression home for `N_LAYERS_ASSUMED = 32` (retired 2026-08-06). It priced the
per-layer instruments — I1 costs one forward pass per layer, I3 one unembedding
readout per layer — for every model alike, and its comment justified itself with
"both pilot models are 32 layers".

That was false. Against the checkpoints' own config.json: Llama-3.1-8B is 32,
**Qwen2.5-7B is 28**, **Qwen2.5-0.5B is 24**. So the estimate the experiment-run
approval gate saw over-priced both Qwen models — by 14% and 33% — while the
constant's stated worry was the opposite direction, under-pricing deeper models.

The fix has two halves and both are asserted here: the count is a per-model FACT
in the model's own config, and the moment real weights load it is reconciled
against them, because a declared copy of a fact that lives elsewhere drifts.
"""

from __future__ import annotations

import pytest

from internals_safety.config import ModelConfig, load_guard_config, load_model_config
from internals_safety.models.loader import attach, verify_declared_layers
from internals_safety.paths import CONF_DIR


def _config_names(subdir: str) -> list[str]:
    return sorted(path.stem for path in (CONF_DIR / subdir).glob("*.yaml"))


@pytest.mark.parametrize("name", _config_names("models"))
def test_every_shipped_model_declares_its_layer_count(name):
    """Undeclared means the estimator refuses to price per-layer instruments, so
    a config shipped without one is a run that cannot be costed."""
    assert load_model_config(name).n_layers, f"{name} declares no n_layers"


@pytest.mark.parametrize("name", _config_names("guards"))
def test_every_shipped_guard_declares_its_layer_count(name):
    """AS-6 inherits the same estimate path — a guard is a causal LM."""
    assert load_guard_config(name).n_layers, f"{name} declares no n_layers"


def test_the_declared_counts_are_the_ones_read_off_the_checkpoints():
    """Pinned as VALUES, not just as present.

    The point of the whole fix is that these three are not all 32; a test that
    only checked "some number is declared" would pass on the broken world.
    Sources are each checkpoint's config.json `num_hidden_layers`.
    """
    assert load_model_config("llama3_1_8b_instruct").n_layers == 32
    assert load_model_config("qwen2_5_7b_instruct").n_layers == 28
    assert load_model_config("qwen2_5_0_5b_instruct").n_layers == 24


def test_they_are_not_all_the_same_number():
    """The retired constant's premise, asserted directly so it cannot return."""
    declared = {load_model_config(n).n_layers for n in _config_names("models")}
    assert len(declared) > 1, "if these ever agree, re-check before trusting one constant"


class TestReconciliationAgainstRealWeights:
    """`verify_declared_layers` — the half that catches drift.

    Driven through `attach`, which is the path `load_model` takes, so the check
    cannot be bypassed by loading a model the ordinary way.
    """

    def test_a_mismatch_fails_loud_and_says_the_estimate_is_stale(self, tiny_model, tiny_config):
        wrong = tiny_config.model_copy(update={"n_layers": tiny_model.n_layers + 7})
        with pytest.raises(ValueError, match="priced a different model"):
            attach(tiny_model.model, tiny_model.tokenizer, wrong)

    def test_an_undeclared_count_is_silent_rather_than_guessed(self, tiny_model):
        """`None` is the honest not-checked state. It must not be filled in here —
        the estimator reports the gap itself, which is where a human sees it."""
        assert tiny_model.config.n_layers is None
        verify_declared_layers(tiny_model)  # no raise

    def test_a_correct_declaration_passes(self, tiny_model, tiny_config):
        right = tiny_config.model_copy(update={"n_layers": tiny_model.n_layers})
        assert attach(tiny_model.model, tiny_model.tokenizer, right).n_layers == tiny_model.n_layers
