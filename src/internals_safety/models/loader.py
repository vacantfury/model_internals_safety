"""Load a HF causal LM and prepare prompts for instrumented forward passes.

Two jobs, both prerequisites for every measurement in the paper:

1. **Load** a model/tokenizer pair with resolved dtype+device, in inference
   mode, and expose its decoder-layer modules (the hook points capture.py uses).
2. **Prepare prompts** — render the chat template and resolve the *token
   positions* the measurements read out at.

On positions. Batched capture needs a position index that is valid across a
padded batch, so tokenization is **left-padded** and positions are stored as
*negative* offsets from the end of the sequence. Left padding makes negative
indices padding-invariant: -1 is the last real token for every row regardless
of how much padding precedes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from internals_safety.config import ModelConfig, PositionName, load_model_config

# Attribute paths to the decoder-layer ModuleList, tried in order. Covers the
# Llama/Qwen/Mistral/Gemma family (model.layers) plus the older GPT-2/NeoX and
# encoder-decoder-style layouts.
_LAYER_LIST_PATHS: tuple[str, ...] = (
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "transformer.blocks",
)


@dataclass(frozen=True)
class PreparedPrompt:
    """A rendered chat prompt plus the token positions to read out at."""

    user_message: str
    text: str
    # position name -> negative index from the end of this prompt's token
    # sequence. Padding-invariant under left padding.
    positions: dict[str, int]


@dataclass
class LoadedModel:
    """A model in inference mode plus everything the capture layer needs."""

    config: ModelConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    layers: Sequence[torch.nn.Module]
    device: torch.device
    dtype: torch.dtype

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)


class ScheduledJobOnCPU(RuntimeError):
    """Raised when a batch job resolves to CPU — see `resolve_device`."""


def in_scheduled_job() -> bool:
    """Whether this process is running inside a SLURM allocation."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def resolve_device(name: str, allow_cpu_in_job: bool = False) -> torch.device:
    """Pick the device, and refuse a silent CPU fallback inside a batch job.

    `auto` falling back to CPU is correct on a laptop and a defect on the
    cluster, where it is never what the submitter asked for: the job requested a
    GPU, got one allocated, and then ran without it. That failure is quiet — the
    only trace is a `UserWarning` buried in the log — and it costs more than
    speed. `provenance.RESULT_BEARING_DEVICES` is CUDA-only, so a CPU fallback
    ALSO downgrades the dirty-tree guard from refuse to warn: one silent
    degradation disables two safety properties at once.

    Seed (2026-08-02): the first cluster run resolved to CPU because the torch
    build was cu130 against a 12.8 driver. It was caught by reading the log, not
    by the pipeline, which is exactly the problem.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if in_scheduled_job() and not allow_cpu_in_job:
        raise ScheduledJobOnCPU(
            "running inside SLURM job "
            f"{os.environ.get('SLURM_JOB_ID')} but CUDA is unavailable, so this would "
            "run on CPU with a GPU allocated and idle.\n"
            "Most likely the torch build does not match the node's driver — check\n"
            "    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'\n"
            "and compare torch's CUDA version against `nvidia-smi`. A cu130 wheel "
            "cannot run on a CUDA 12.x driver.\n"
            "Pass --allow-cpu if a CPU run is genuinely intended."
        )
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype, defaulting per device.

    `auto` means: bfloat16 on CUDA (what the cluster runs), float32 elsewhere.
    MPS and CPU stay in float32 deliberately — local runs are correctness smoke
    tests, and reduced precision there would add a difference between the
    machine we debug on and the machine that produces results.
    """
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def find_layer_modules(model: PreTrainedModel) -> Sequence[torch.nn.Module]:
    """Return the decoder-layer ModuleList, or raise with a usable message."""
    for path in _LAYER_LIST_PATHS:
        node: object = model
        for attribute in path.split("."):
            node = getattr(node, attribute, None)
            if node is None:
                break
        if isinstance(node, torch.nn.ModuleList) and len(node) > 0:
            return node
    raise RuntimeError(
        f"could not locate decoder layers on {type(model).__name__}; "
        f"tried {_LAYER_LIST_PATHS}. Add this architecture's path to _LAYER_LIST_PATHS."
    )


def _from_pretrained(config: ModelConfig, dtype: torch.dtype) -> PreTrainedModel:
    """Load weights, tolerating the transformers `torch_dtype` -> `dtype` rename."""
    kwargs = {"trust_remote_code": config.trust_remote_code}
    try:
        return AutoModelForCausalLM.from_pretrained(config.hf_id, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(config.hf_id, torch_dtype=dtype, **kwargs)


def attach(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
) -> LoadedModel:
    """Wrap an already-constructed model+tokenizer as a LoadedModel.

    Separate from `load_model` so tests can drive the whole capture stack with a
    tiny randomly-initialised model and no download.
    """
    device = resolve_device(config.device)
    dtype = next(model.parameters()).dtype
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Left padding: see the module docstring — it is what makes negative
    # position indices valid across a padded batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # **Borrowing a chat template is how a BASE model becomes comparable to its
    # Instruct sibling, and it is a measurement decision, not plumbing.**
    #
    # I4's pre-gate runs one Llama Scope dictionary against the model it was
    # trained on (Llama-3.1-8B-Base) and against our target (Instruct). Base
    # ships no chat template, so `render_chat` fails closed and the arm cannot
    # run at all. The tempting fix — let the Base arm see bare text — silently
    # makes the comparison meaningless: the "transfer gap" would then bundle a
    # formatting difference in with the model difference, and formatting moves
    # residual activations at exactly the positions we read.
    #
    # So both arms see the SAME rendered text and the model is the only variable.
    # Sound here specifically because Base and Instruct are the same checkpoint
    # family with the same tokenizer — never a licence to render one model's
    # prompts through an unrelated model's template.
    if config.chat_template_from is not None:
        if tokenizer.chat_template is not None:
            raise ValueError(
                f"{config.name} declares chat_template_from={config.chat_template_from!r} but its "
                "own tokenizer already ships a template; overwriting a real template with a "
                "borrowed one would silently change what every prior run measured"
            )
        donor_config = load_model_config(config.chat_template_from)
        donor = AutoTokenizer.from_pretrained(
            donor_config.hf_id, trust_remote_code=donor_config.trust_remote_code
        )
        if donor.chat_template is None:
            raise ValueError(f"donor {config.chat_template_from} has no chat template to lend")
        if donor.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"donor {config.chat_template_from} has vocab {donor.vocab_size} against "
                f"{config.name}'s {tokenizer.vocab_size}; a borrowed template is only meaningful "
                "between checkpoints that tokenise it identically"
            )
        tokenizer.chat_template = donor.chat_template

    loaded = LoadedModel(
        config=config,
        model=model,
        tokenizer=tokenizer,
        layers=find_layer_modules(model),
        device=device,
        dtype=dtype,
    )
    verify_declared_layers(loaded)
    return loaded


def verify_declared_layers(loaded: LoadedModel) -> None:
    """Reconcile `config.n_layers` against the model that actually loaded.

    The declared count exists so the pre-run cost estimate can price
    layer-proportional instruments without touching weights — which means it is
    a copy of a fact living somewhere else, and copies drift. So the moment the
    real model is in hand, they are compared.

    Fails LOUD rather than trusting either side: a mismatch means the estimate
    the approval gate saw priced a different model than the one about to run,
    and which of the two is wrong is not something this function can decide.

    Silent when nothing is declared — that is the honest "not checked" state,
    and the estimator reports the gap itself rather than substituting a number.
    """
    declared = loaded.config.n_layers
    if declared is None or declared == loaded.n_layers:
        return
    raise ValueError(
        f"{loaded.config.name}: config declares n_layers={declared} but the loaded "
        f"checkpoint ({loaded.config.hf_id}) has {loaded.n_layers}. The pre-run cost "
        f"estimate used the declared value, so it priced a different model than this "
        f"one. Fix the config from the checkpoint's config.json `num_hidden_layers`, "
        f"then re-cost — do not proceed on the estimate already approved."
    )


def load_model(config: ModelConfig) -> LoadedModel:
    """Load the model named by `config` in inference mode."""
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    model = _from_pretrained(config, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id, trust_remote_code=config.trust_remote_code
    )
    return attach(model, tokenizer, config)


def render_chat(
    tokenizer: PreTrainedTokenizerBase,
    user_message: str,
    system_prompt: str | None = None,
) -> str:
    """Render one user turn through the model's chat template.

    `add_generation_prompt=True`: the rendered text ends where the assistant is
    about to speak, so the final token is the one whose residual stream carries
    the model's state at the moment of deciding what to say.
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    if tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer has no chat template; instruction-tuned checkpoints are required "
            "(the four measurements are defined over chat-formatted attack prompts)"
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def instruction_final_offset(
    tokenizer: PreTrainedTokenizerBase, rendered: str, user_message: str
) -> int:
    """Negative index of the last token of the user message inside `rendered`.

    This is the recognition readout site (measurement #3): the last position at
    which the instruction itself is what the model is representing, before the
    template's end-of-turn and assistant-header tokens.

    Uses the fast tokenizer's character offsets when available and falls back to
    prefix re-tokenisation otherwise. Both paths assume the template inserts the
    user content verbatim, which is checked here rather than assumed.
    """
    try:
        content_start = rendered.rindex(user_message)
    except ValueError as error:  # pragma: no cover - template mangled the content
        raise ValueError(
            "user message does not appear verbatim in the rendered chat prompt; "
            "position resolution cannot be trusted for this tokenizer"
        ) from error
    content_end = content_start + len(user_message)

    encoded = tokenizer(rendered, add_special_tokens=False)
    n_tokens = len(encoded["input_ids"])

    if tokenizer.is_fast:
        offsets = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        candidates = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end <= content_end
        ]
        if candidates:
            return max(candidates) - n_tokens

    n_prefix = len(tokenizer(rendered[:content_end], add_special_tokens=False)["input_ids"])
    if n_prefix == 0:
        raise ValueError("empty instruction prefix; cannot resolve instruction_final")
    return (n_prefix - 1) - n_tokens


def resolve_position(
    tokenizer: PreTrainedTokenizerBase,
    rendered: str,
    user_message: str,
    position: PositionName,
) -> int:
    if position == "last":
        return -1
    if position == "instruction_final":
        return instruction_final_offset(tokenizer, rendered, user_message)
    raise ValueError(f"unknown position name: {position!r}")


def prepare_prompts(
    loaded: LoadedModel,
    user_messages: Iterable[str],
    positions: Sequence[PositionName] | None = None,
) -> list[PreparedPrompt]:
    """Render each message and resolve every requested readout position."""
    wanted = list(positions) if positions is not None else list(loaded.config.capture.positions)
    prepared: list[PreparedPrompt] = []
    for message in user_messages:
        rendered = render_chat(loaded.tokenizer, message, loaded.config.system_prompt)
        prepared.append(
            PreparedPrompt(
                user_message=message,
                text=rendered,
                positions={
                    name: resolve_position(loaded.tokenizer, rendered, message, name)
                    for name in wanted
                },
            )
        )
    return prepared


def tokenize_batch(
    loaded: LoadedModel, prompts: Sequence[PreparedPrompt], with_position_ids: bool = True
):
    """Left-pad a batch of rendered prompts onto the model's device.

    `add_special_tokens=False` throughout: the chat template already emits BOS
    where the architecture wants one, and adding a second would shift every
    position index by one.

    Two measured exceptions to that invariant, both in AS-6's guard slate and
    both pinned in `tests/test_real_guard_tokenizers.py`. WildGuard ships no chat
    template at all, so its renderer emits BOS itself (`GuardConfig.prepend_bos`).
    Llama Guard 3's template emits a stray leading SPACE before
    `<|begin_of_text|>`, so its token 0 is whitespace and BOS is token 1 —
    harmless here because positions are negative offsets from the end, and kept
    rather than normalised because `apply_chat_template` is also how that model's
    published numbers were produced.

    `position_ids` are built from the attention mask rather than left to
    default to `arange(seq_len)`. Without this, left padding shifts every real
    token's position index by that row's pad count, so an activation would
    depend on which batch a prompt happened to land in. RoPE's relative
    geometry hides most of that, but nothing guarantees it in general — and a
    measurement that changes with batch composition is not a measurement.

    `with_position_ids=False` for generation: `generate()` maintains its own
    mask-aware position bookkeeping across decoding steps, and handing it a
    prefilled `position_ids` fights that rather than helping it.
    """
    encoded = loaded.tokenizer(
        [prompt.text for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
    if with_position_ids:
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    return inputs
