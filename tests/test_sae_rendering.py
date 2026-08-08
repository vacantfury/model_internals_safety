"""The pre-gate's Base arm must be readable on the DISTRIBUTION it was fitted on.

`conf/models/llama3_1_8b_base.yaml` borrows the Instruct sibling's chat template
so the two pre-gate arms differ only by MODEL — correct for reading the transfer
gap, and it silently disables the Base arm's other job. Llama Scope's
dictionaries were fitted on plain text and a base checkpoint never saw a chat
template in training, so a templated Base run reads the dictionary on a
distribution neither it nor the model has met, then reports the result as
evidence about `models/sae_loader.py`.

Measured consequence: the templated run reports a mean activation norm of 23.705
against the checkpoint's declared 13.8125.

`render_chat` is DEFINITIONAL, not a tunable — it decides what the number is
about. These tests pin that it reaches the code from the preset, that it changes
what is measured, and that a preset cannot declare it on an entrypoint that
would drop it.
"""

from __future__ import annotations

import pytest
import torch

from internals_safety.config import PresetConfig
from internals_safety.measurements.sae_reconstruction import _render, measure_reconstruction

from test_sae_scale_diagnostic import CONFIG, tiny_scope_sae

PROMPTS = ["hi", "a somewhat longer prompt with more tokens in it"]


class TestTheRenderingActuallyChanges:
    def test_plain_text_is_the_prompts_verbatim(self, tiny_model):
        assert _render(tiny_model, PROMPTS, render_chat=False) == PROMPTS

    def test_the_template_adds_something(self, tiny_model):
        """If the fixture's template were a no-op this whole file would be
        vacuous — the two paths must actually differ."""
        templated = _render(tiny_model, PROMPTS, render_chat=True)
        assert templated != PROMPTS
        assert all(len(t) > len(p) for t, p in zip(templated, PROMPTS))

    def test_the_two_paths_measure_different_activations(self, tiny_model):
        """The point of the run: same weights, same prompts, different
        distribution reaching the dictionary."""
        sae = tiny_scope_sae(tiny_model.model.config.hidden_size, input_norm=1.0)
        kwargs = dict(layer=1, config=CONFIG, batch_size=2)
        templated = measure_reconstruction(
            tiny_model, sae, PROMPTS, render_chat=True, **kwargs
        )
        plain = measure_reconstruction(
            tiny_model, sae, PROMPTS, render_chat=False, **kwargs
        )
        assert templated.observed_activation_norm != plain.observed_activation_norm, (
            "rendering did not reach the measurement — both arms scored the same tokens"
        )


class TestThePresetCarriesItIntoTheArgv:
    def _preset(self, **overrides) -> PresetConfig:
        base = dict(
            entrypoint="sae_pregate",
            description="d",
            gates="g",
            target="llama3_1_8b_base",
            sae_layers=[17],
            resources={"partition": "gpu", "cpus": 8, "mem": "64G", "time": "01:00:00"},
        )
        return PresetConfig(**{**base, **overrides})

    def test_render_chat_false_emits_the_flag(self):
        argv = self._preset(render_chat=False).tasks("/out")[0]
        assert "--plain-text" in argv

    def test_the_default_does_not(self):
        assert "--plain-text" not in self._preset().tasks("/out")[0]

    def test_every_layer_task_carries_it(self):
        """A three-layer preset is three array tasks; a flag on only the first
        would run two of them on the wrong distribution and report one run."""
        tasks = self._preset(render_chat=False, sae_layers=[17, 19, 21]).tasks("/out")
        assert len(tasks) == 3
        assert all("--plain-text" in argv for argv in tasks)


class TestAPresetCannotDeclareItWhereItWouldBeDROPPED:
    def test_an_entrypoint_that_ignores_it_refuses(self):
        """**The truthiness sweep cannot police this field**, because its
        meaningful value is False — `if getattr(self, name)` waves through
        `render_chat: false` on every entrypoint. A silently dropped declaration
        means the approved artifact says plain text and the job runs templated,
        which is precisely the confusion this flag exists to end.
        """
        with pytest.raises(ValueError, match="render_chat"):
            PresetConfig(
                entrypoint="phase0_regime_map",
                description="d",
                gates="g",
                target="llama3_1_8b_instruct",
                families="all",
                render_chat=False,
                resources={"partition": "gpu", "cpus": 8, "mem": "64G", "time": "01:00:00"},
            )

    def test_the_default_true_is_accepted_everywhere(self):
        """True is the status quo and must not become a per-entrypoint error."""
        preset = PresetConfig(
            entrypoint="phase0_regime_map",
            description="d",
            gates="g",
            target="llama3_1_8b_instruct",
            families="all",
            resources={"partition": "gpu", "cpus": 8, "mem": "64G", "time": "01:00:00"},
        )
        assert preset.render_chat is True


class TestTheShippedPresetSaysWhatItClaims:
    def test_the_plain_preset_is_actually_plain(self):
        from internals_safety.config import load_preset

        preset = load_preset("sae_pregate_base_plain")
        assert preset.render_chat is False
        assert preset.target == "llama3_1_8b_base"
        assert all("--plain-text" in argv for argv in preset.tasks("/out"))

    def test_the_templated_preset_is_unchanged(self):
        from internals_safety.config import load_preset

        preset = load_preset("sae_pregate_base")
        assert preset.render_chat is True
        assert preset.sae_layers == load_preset("sae_pregate_base_plain").sae_layers, (
            "the two presets must differ in RENDERING and nothing else"
        )
