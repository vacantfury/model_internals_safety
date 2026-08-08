"""The BOS convention guard (`models/loader.verify_bos_convention`).

Repo-wide tokenisation runs `add_special_tokens=False` on the premise that a chat
template emits BOS itself. That premise is a property of the checkpoints tried so
far, and Tulu 3 breaks it: `bos_token = '<|begin_of_text|>'`, template emits none.
Under the default a Tulu arm runs BOS-less against a Llama arm that does not —
inside the one comparison the arm exists to make, and with nothing failing.

So the omission is made inexpressible. These tests pin that a checkpoint in that
state CANNOT attach until the config decides, and that both decisions do what
they say. The `slow` class at the bottom pins the real artifacts, in the style of
`test_real_guard_tokenizers.py`: tokenizers only, no weights, seconds.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from internals_safety.config import ModelConfig
from internals_safety.models.loader import template_emits_bos, verify_bos_convention

WITH_BOS = "{{ bos_token }}{% for m in messages %}<|im_start|> {{ m['content'] | trim }}\n{% endfor %}"
WITHOUT_BOS = "{% for m in messages %}<|user|> {{ m['content'] | trim }}\n{% endfor %}"


def _tokenizer(*, template: str, bos: str | None) -> PreTrainedTokenizerFast:
    """A tokenizer whose BOS declaration and template are independently set.

    That independence IS the fixture's job: the defect is precisely a checkpoint
    where the two disagree, so a fixture that cannot express disagreement would
    certify rather than test.
    """
    vocab = {"[UNK]": 0, "[PAD]": 1, "<|im_start|>": 2, "<|user|>": 3, "<|begin_of_text|>": 4}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    kwargs = {"unk_token": "[UNK]", "pad_token": "[PAD]"}
    if bos is not None:
        kwargs["bos_token"] = bos
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    tokenizer.chat_template = template
    return tokenizer


def _config(**kwargs) -> ModelConfig:
    return ModelConfig(name="probe_model", hf_id="<in-process>", **kwargs)


class TestTheQuestionIsTriState:
    def test_no_template_is_undefined_not_false(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")
        tokenizer.chat_template = None
        assert template_emits_bos(tokenizer) is None

    def test_no_bos_token_is_undefined_not_false(self):
        # Qwen2.5-7B-Instruct's real state: bos_token is None, so the whole
        # question does not arise. `None`, never `False` — an unmeasured axis is
        # not a negative reading.
        assert template_emits_bos(_tokenizer(template=WITHOUT_BOS, bos=None)) is None

    def test_emitting_and_omitting_are_distinguished(self):
        assert template_emits_bos(_tokenizer(template=WITH_BOS, bos="<|begin_of_text|>")) is True
        assert template_emits_bos(_tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")) is False


class TestTheOmissionIsInexpressible:
    """The core guard: undeclared + disagreeing == cannot attach."""

    def test_undeclared_disagreement_raises(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")
        with pytest.raises(ValueError, match="never emits it"):
            verify_bos_convention(tokenizer, _config())

    def test_the_error_names_both_ways_out(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")
        with pytest.raises(ValueError) as caught:
            verify_bos_convention(tokenizer, _config())
        message = str(caught.value)
        assert "prepend_bos_to_chat_template: true" in message
        assert "prepend_bos_to_chat_template: false" in message

    def test_declaring_true_makes_the_template_emit_bos(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")
        verify_bos_convention(tokenizer, _config(prepend_bos_to_chat_template=True))
        assert template_emits_bos(tokenizer) is True

    def test_declaring_false_leaves_the_template_alone(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<|begin_of_text|>")
        before = tokenizer.chat_template
        verify_bos_convention(tokenizer, _config(prepend_bos_to_chat_template=False))
        assert tokenizer.chat_template == before
        assert template_emits_bos(tokenizer) is False


class TestADeclarationThatCannotBeHonouredFails:
    """A config asserting something false is worse than one asserting nothing."""

    def test_true_on_a_template_that_already_emits_bos_raises(self):
        # The Mistral v0.3 defect: template's literal '<s>' plus a second BOS
        # gives ['<s>', '<s>', '[INST]', ...]. Prepending here would recreate it.
        tokenizer = _tokenizer(template=WITH_BOS, bos="<|begin_of_text|>")
        with pytest.raises(ValueError, match="double BOS"):
            verify_bos_convention(tokenizer, _config(prepend_bos_to_chat_template=True))

    def test_true_with_no_bos_token_raises(self):
        tokenizer = _tokenizer(template=WITHOUT_BOS, bos=None)
        with pytest.raises(ValueError, match="no BOS token to prepend"):
            verify_bos_convention(tokenizer, _config(prepend_bos_to_chat_template=True))

    def test_silence_is_correct_when_the_template_already_emits(self):
        tokenizer = _tokenizer(template=WITH_BOS, bos="<|begin_of_text|>")
        before = tokenizer.chat_template
        verify_bos_convention(tokenizer, _config())
        assert tokenizer.chat_template == before


class TestTheGuardIsReachedByAttach:
    """A guard the loader never calls is documentation, not enforcement.

    Named for the failure this repo recorded four times in one day — a rule
    adopted, tested and documented that still governed nothing because the live
    entrypoint did not consult it.
    """

    def test_attach_refuses_an_undeclared_disagreeing_checkpoint(self, tiny_model, tiny_config):
        from internals_safety.models.loader import attach

        tokenizer = tiny_model.tokenizer
        original_bos, original_template = tokenizer.bos_token, tokenizer.chat_template
        try:
            tokenizer.bos_token = "<|im_start|>"
            tokenizer.chat_template = WITHOUT_BOS
            with pytest.raises(ValueError, match="never emits it"):
                attach(tiny_model.model, tokenizer, tiny_config)
        finally:
            tokenizer.bos_token = original_bos
            tokenizer.chat_template = original_template


class TestTheGuardFieldIsNotShadowed:
    """Two different BOS questions, one word — kept apart by name.

    ⚠️ REGRESSION. This guard was first written with the field called
    `prepend_bos`, and `GuardConfig` **inherits** `ModelConfig`, so the new field
    silently OVERRODE the guard layer's own `prepend_bos` — which means something
    else entirely: whether a `literal`-format renderer emits BOS itself, because
    the checkpoint ships no chat template (WildGuard). Result: every guard test
    with `prepend_bos: true` hit "there is no BOS token to prepend", 13 errors.

    The two are genuinely different questions and both must stay askable:
      * `GuardConfig.prepend_bos` — our renderer, for a guard with NO template.
      * `ModelConfig.prepend_bos_to_chat_template` — someone else's template,
        which has one and omits BOS from it.
    """

    def test_the_two_fields_are_distinct_attributes(self):
        from internals_safety.config import GuardConfig

        assert "prepend_bos" in GuardConfig.model_fields
        assert "prepend_bos_to_chat_template" in GuardConfig.model_fields
        assert "prepend_bos" not in ModelConfig.model_fields

    def test_a_templateless_guard_that_prepends_bos_still_verifies(self):
        """Against the REAL committed config, not a hand-made stand-in.

        WildGuard is the actual case: `prompt_style: literal`, no chat template,
        `prepend_bos: true`. A hand-built GuardConfig could drift from it; the
        shipped file cannot.
        """
        from internals_safety.config import load_guard_config

        guard = load_guard_config("wildguard")
        assert guard.prepend_bos is True
        assert guard.prepend_bos_to_chat_template is None

        tokenizer = _tokenizer(template=WITHOUT_BOS, bos="<s>")
        tokenizer.chat_template = None
        verify_bos_convention(tokenizer, guard)  # must not raise


def committed_model_names() -> list[str]:
    """Every model config in `conf/models/`, DERIVED — never hand-listed.

    ⚠️ A hand-written slate goes stale silently and keeps reporting green. The
    peer session's version of this table was written on the morning of
    2026-08-08 and the Tulu ladder added two configs the same afternoon, so it
    was covering 5 of 8 while passing. Deriving the row set is what makes "add a
    row when a model joins the slate" stop being a rule anyone has to remember.
    """
    from internals_safety.paths import CONF_DIR

    return sorted(path.stem for path in (CONF_DIR / "models").glob("*.yaml"))


@pytest.mark.slow
@pytest.mark.parametrize("model_name", committed_model_names())
class TestTheRealSlate:
    """Pins the artifacts, not our mechanism. Tokenizers only — no weights.

    Every committed model config is checked against the real tokenizer it names.
    If a vendor changes a template, this is where it surfaces — before a queue
    wait, not after.

    ⚠️ SIBLING FILE: `tests/test_real_bos_handling.py` pins that exactly ONE BOS
    reaches the model. The two are opposite ends of one axis and neither subsumes
    the other — that file catches a DOUBLE BOS (`add_special_tokens=True` against
    a post-processor that already prepends one), this one catches ZERO BOS (a
    template that never emits it under `add_special_tokens=False`). The shared
    invariant, and the fact that reading only the template gave a wrong answer in
    EACH direction on the same day, is stated once in
    `text_docs/shared/model_slate.md` §3.1 and restated in neither.
    """

    def test_the_config_declaration_matches_the_real_tokenizer(self, model_name):
        """The config must not lie about the checkpoint, and must not omit.

        This is the whole guard applied to committed reality: load the real
        tokenizer, run `verify_bos_convention` against the real config, and
        require it to accept. A config declaring `true` on a template that
        already emits BOS fails here, as does one declaring nothing on a template
        that emits none — which is the Tulu case that started this.
        """
        from transformers import AutoTokenizer

        from internals_safety.config import load_model_config

        config = load_model_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        verify_bos_convention(tokenizer, config)

    def test_after_the_guard_exactly_zero_is_impossible(self, model_name):
        """Post-guard, a chat template either emits BOS or the model has none.

        The zero-BOS end of the invariant, asserted on the state a run actually
        sees rather than on the shipped template — `verify_bos_convention`
        rewrites the template when the config says to, so checking the shipped
        one would test a model no run will ever use.
        """
        from transformers import AutoTokenizer

        from internals_safety.config import load_model_config

        config = load_model_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        if tokenizer.chat_template is None or tokenizer.bos_token is None:
            pytest.skip(f"{model_name}: no chat template or no BOS token — question undefined")
        verify_bos_convention(tokenizer, config)
        emits = template_emits_bos(tokenizer)
        assert emits is True, (
            f"{model_name} renders no BOS even after the guard ran; "
            f"prepend_bos_to_chat_template={config.prepend_bos_to_chat_template!r}"
        )
