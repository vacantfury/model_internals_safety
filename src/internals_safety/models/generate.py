"""Batched generation — the behaviour side of every measurement.

Greedy by default. The paper's headline quantities are regime *labels* built by
combining four measurements on the same prompt, so a sampled response would put
decoding noise inside the label rather than merely inside a number. Sampling
stays available (`temperature`/`top_p`) for anything that needs a distribution
over responses, and the seed then belongs in the run record.
"""

from __future__ import annotations

from typing import Sequence

import torch

from internals_safety.models.loader import LoadedModel, prepare_prompts, tokenize_batch


def generate(
    loaded: LoadedModel,
    user_messages: Sequence[str],
    max_new_tokens: int,
    # NO DEFAULT, keyword-only. `batch_size` is not plumbing: the retired
    # marker read "throughput only — greedy decoding is batch-invariant",
    # and the paper's own §7 refutes it. Only 12–58% of responses repeat
    # byte-identically across nominally greedy runs, because batch
    # composition changes the reduction order in the matmuls and the argmax
    # can land elsewhere. A knob that moves generated text may not carry a
    # convenient default that every caller silently inherits — that shape has
    # failed in this repo five times (`strata`, `device`, `inherited`, the
    # control floor, and now this one). Every caller reads it from YAML
    # (`measurements.ability.batch_size`, `measurements.behavior.batch_size`),
    # so omitting it is a TypeError rather than an inherited 8.
    *,
    batch_size: int,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Render each message as a chat turn and return the assistant text only."""
    responses: list[str] = []
    for start in range(0, len(user_messages), batch_size):
        chunk = list(user_messages[start : start + batch_size])
        prompts = prepare_prompts(loaded, chunk, positions=[])
        inputs = tokenize_batch(loaded, prompts, with_position_ids=False)
        prompt_length = inputs["input_ids"].shape[1]

        kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": loaded.tokenizer.pad_token_id,
        }
        if do_sample:
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p

        with torch.inference_mode():
            output = loaded.model.generate(**inputs, **kwargs)

        # Left padding means every row's completion starts at the same index.
        completions = output[:, prompt_length:]
        responses.extend(loaded.tokenizer.batch_decode(completions, skip_special_tokens=True))
    return responses
