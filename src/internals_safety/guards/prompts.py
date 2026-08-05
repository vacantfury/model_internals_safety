"""Render a guard's classification prompt and locate its verdict token.

Why this module exists. `models.loader.render_chat` fails closed when a
checkpoint ships no chat template, and that is not an edge case here — WildGuard
ships none, so half of AS-6's slate is unreachable without a second rendering
path. Llama Guard 3 does ship one, and it hard-wires the safety task, so for that
guard the shipped template *is* the contract and must be used as-is.

Two positions matter, and the spine already resolves both:

    instruction_final  last token of the payload   -> did the guard REPRESENT it?
    last               the verdict-formation token -> did it CONVERT that to a block?

Reading the verdict. A guard's output is one token, so it is read from logits in
a single forward pass rather than generated — Meta's own model card defines the
Llama Guard classifier score that way, and it cleanly separates "the output
format broke" from "the verdict moved", which generation does not. Guards differ
in *where* that token sits: Llama Guard 3 answers with it first, WildGuard emits
`Harmful request: <verdict>` and so needs its preamble teacher-forced on. Both
are the same mechanism with a different `verdict_prefix`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from transformers import PreTrainedTokenizerBase

from internals_safety.config import GuardConfig, PositionName
from internals_safety.models.loader import (
    LoadedModel,
    PreparedPrompt,
    render_chat,
    resolve_position,
)


@dataclass(frozen=True)
class VerdictTokens:
    """The two token ids whose probabilities are the guard's classifier score.

    `piece` fields carry what the tokenizer actually produced, so a caller (or a
    run record) can show that the resolution landed where it was meant to instead
    of asserting it did.
    """

    safe_id: int
    unsafe_id: int
    safe_piece: str
    unsafe_piece: str


def render_guard_prompt(
    tokenizer: PreTrainedTokenizerBase,
    config: GuardConfig,
    payload: str,
) -> str:
    """Render one payload into `config`'s classification prompt.

    `payload` is what the attack carries — plaintext or a rung's ciphertext. It
    is substituted verbatim, which the position resolver then relies on to find
    the payload's last token.
    """
    if config.prompt_style == "chat_template":
        return render_chat(tokenizer, payload, config.system_prompt)

    assert config.prompt_template is not None  # guaranteed by GuardConfig validation
    rendered = config.prompt_template.format(
        prompt=payload, response=config.response_placeholder
    )
    if not config.prepend_bos:
        return rendered
    bos = tokenizer.bos_token
    if not bos:
        raise ValueError(
            f"guard {config.name!r} sets prepend_bos but its tokenizer has no bos_token; "
            "leave prepend_bos false or the prompt gains a literal 'None'"
        )
    return bos + rendered


def prepare_guard_prompts(
    loaded: LoadedModel,
    payloads: Iterable[str],
    positions: Sequence[PositionName] | None = None,
) -> list[PreparedPrompt]:
    """Render each payload and resolve every readout position.

    The guard twin of `models.loader.prepare_prompts`; its output feeds the same
    `tokenize_batch` / `capture_activations` path unchanged.
    """
    config = loaded.config
    if not isinstance(config, GuardConfig):
        raise TypeError(
            f"prepare_guard_prompts needs a GuardConfig, got {type(config).__name__}; "
            "load the guard via config.load_guard_config"
        )
    wanted = list(positions) if positions is not None else list(config.capture.positions)
    prepared: list[PreparedPrompt] = []
    for payload in payloads:
        rendered = render_guard_prompt(loaded.tokenizer, config, payload)
        prepared.append(
            PreparedPrompt(
                user_message=payload,
                text=rendered,
                positions={
                    name: resolve_position(loaded.tokenizer, rendered, payload, name)
                    for name in wanted
                },
            )
        )
    return prepared


def verdict_context(config: GuardConfig, rendered: str) -> str:
    """The text whose FINAL position carries the verdict distribution.

    Equal to the rendered prompt when the verdict is the first generated token,
    and the prompt plus the guard's fixed preamble otherwise.
    """
    return rendered + config.verdict_prefix


def _first_continuation_token(
    tokenizer: PreTrainedTokenizerBase, context: str, continuation: str
) -> tuple[int, str]:
    """Id of the first token of `continuation` when it follows `context`.

    Resolved by re-tokenising rather than by vocabulary lookup, because the id of
    a label depends on what precedes it — " yes" and "yes" are different tokens in
    every BPE vocabulary this project touches, and picking the wrong one reads a
    probability that means nothing.

    Fails loud when the tokenizer merges across the boundary, because then no
    single token corresponds to the label and any id returned would be a guess.
    """
    context_ids = list(tokenizer(context, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(context + continuation, add_special_tokens=False)["input_ids"])

    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError(
            f"tokenizer merges across the boundary between the verdict prefix and "
            f"{continuation!r}: the label's tokenisation is not an extension of the "
            "prefix's, so no single token id is the verdict. Move the whitespace "
            "between verdict_prefix and the label, or pick a label that stands alone."
        )
    if len(full_ids) == len(context_ids):
        raise ValueError(f"continuation {continuation!r} tokenises to nothing after the prefix")

    token_id = int(full_ids[len(context_ids)])
    return token_id, tokenizer.convert_ids_to_tokens(token_id)


def resolve_verdict_tokens(
    tokenizer: PreTrainedTokenizerBase,
    config: GuardConfig,
    rendered: str,
) -> VerdictTokens:
    """Resolve the safe/unsafe token ids in the context they are actually read in.

    Takes the rendered prompt rather than the bare prefix so resolution happens in
    the true context. The ids must then be held FIXED across a run — comparing
    P(unsafe) between prompts is only a comparison if it is the same token — so a
    caller resolves once and reuses; `VerdictTokens` is frozen to make that the
    easy path.
    """
    context = verdict_context(config, rendered)
    safe_id, safe_piece = _first_continuation_token(tokenizer, context, config.safe_token)
    unsafe_id, unsafe_piece = _first_continuation_token(tokenizer, context, config.unsafe_token)
    if safe_id == unsafe_id:
        raise ValueError(
            f"guard {config.name!r}: safe and unsafe labels resolve to the same token "
            f"{safe_id} ({safe_piece!r}). The verdict would be unreadable — the labels "
            "differ only past their first token, so verdict_prefix must extend further."
        )
    return VerdictTokens(
        safe_id=safe_id,
        unsafe_id=unsafe_id,
        safe_piece=safe_piece,
        unsafe_piece=unsafe_piece,
    )
