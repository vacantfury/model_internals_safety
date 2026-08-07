"""Directional ablation and activation addition — the causal write operations.

Method provenance: **established literature, imported.** Arditi et al., *Refusal
in Language Models Is Mediated by a Single Direction*, **NeurIPS 2024** (arXiv
2406.11717); our copy is `other_repos/refusal_direction`. Their pipeline was
already the source of `probes/directions.py`'s difference-in-means estimator and
was named in `as5/s1_idea_check.md` at S1 as the phase-2 plan ("the Arditi trio:
directional ablation for necessity, activation addition for sufficiency"). This
module is that plan's mechanism.

**Why it is being built now rather than at phase 2.** Reading their
`select_direction.py` (2026-08-06) showed the trio is not only a validation step
— it is how they *choose* the direction in the first place. Ours is chosen by
AUROC against a permutation null, which tests whether a separation is real and
never whether it is the RIGHT separation; a direction that separates harmful from
benign by character length passes it and would fail theirs. So the causal test
belongs upstream of the correlational one. Full argument: `instrument_layer.md`
§6.3.1.

**The two operations differ in shape, and the difference matters.**

- *Ablation* removes the component along `r̂` from the residual stream at **every
  layer and every position**, including during generation. It is a claim about
  necessity: if behaviour is unchanged, the direction was not carrying it. Applying
  it at one site would test something much weaker, because the same information
  is present at many sites.
- *Addition* writes `coefficient * r̂` into **one layer**, at every position. It is
  a claim about sufficiency.

Both are implemented here rather than in `patching.py` because `patch_residual`
overwrites a single position with a supplied vector; these transform what is
already there, across positions. Same hook points, different algebra.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator, Sequence

import torch

from internals_safety.models.loader import LoadedModel


def project_out(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of `hidden` along `direction`.

    `direction` must be unit-norm — `probes.directions.difference_in_means`
    normalises and reports `raw_norm` separately, so a caller that has a
    near-degenerate direction can see it rather than silently projecting out
    numerical noise.
    """
    unit = direction.to(device=hidden.device, dtype=hidden.dtype)
    norm = float(unit.norm())
    if norm == 0.0:
        raise ValueError("cannot project out a zero direction")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"direction must be unit-norm, got {norm:.6f}")
    return hidden - (hidden @ unit).unsqueeze(-1) * unit


def _transform_output(output, transform):
    if isinstance(output, tuple):
        return (transform(output[0]), *output[1:])
    return transform(output)


@contextmanager
def _hook_layer_output(loaded: LoadedModel, layer: int, transform) -> Iterator[None]:
    handle = loaded.layers[layer].register_forward_hook(
        lambda _module, _args, output: _transform_output(output, transform)
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_direction(
    loaded: LoadedModel, direction: torch.Tensor, layers: Sequence[int] | None = None
) -> Iterator[None]:
    """Project `direction` out of the residual stream at every layer and position.

    The necessity test. `layers=None` means all of them, which is the default
    because a partial ablation tests a weaker claim than the one being made.
    """
    selected = list(range(loaded.n_layers)) if layers is None else list(layers)
    out_of_range = [index for index in selected if not 0 <= index < loaded.n_layers]
    if out_of_range:
        raise ValueError(f"layer indices {out_of_range} outside [0, {loaded.n_layers - 1}]")

    with ExitStack() as stack:
        for index in selected:
            stack.enter_context(
                _hook_layer_output(loaded, index, lambda hidden: project_out(hidden, direction))
            )
        yield


@contextmanager
def add_direction(
    loaded: LoadedModel,
    direction: torch.Tensor,
    layer: int,
    coefficient: float,
    mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Add `coefficient * direction` at one layer, at the masked positions.

    The sufficiency test. One layer rather than all: the claim is that writing
    the direction *somewhere* induces the behaviour, and applying it everywhere
    would confound the effect with its accumulated magnitude.

    **`mask` is `[batch, seq]` of bools, and its absence means EVERY position.
    SETTLED 2026-08-07 (TODO 33): the convention is PER CONDITION, and the
    default is right for the condition it defaults in.**

    The three-way comparison, read from the sources rather than argued:

    - **Arditi et al. (NeurIPS 2024)** — `activation += coeff * vector` over the
      whole `[batch, seq, d_model]`, no mask (`hook_utils.get_activation_addition_input_pre_hook`).
      **Every position.**
    - **CAA (arXiv 2312.06681)** — `mask = position_ids >= from_id` with
      `from_id = find_instruction_end_postion(...)`. **Instruction-end onward.**
      Tied to their multiple-choice format, where the thing being steered is the
      A/B answer.
    - **Ours** — every position by default, which **matches Arditi**, the paper
      this estimator was ported from.

    ⚠️ TODO 33 was filed reading CAA alone and framed ours as the outlier
    ("ours contaminates the input, CAA's does not"). Checking the closer source
    corrected that: ours matches our own method's paper, and CAA is the one that
    differs. Being unlike CAA is not by itself a defect.

    **The confound is real, and it is CONDITION-SCOPED.** Adding at every
    position writes the direction into the prompt as well as the response, so a
    positive result cannot separate "the direction causes the behaviour" from
    "the direction changed what the model read the prompt as". That second
    reading only *bites* where the prompt is something the model must decode:

    - **Plain prompts → every position (Arditi).** The causal gate runs on the
      plain contrast sets, where there is no ciphertext and nothing to decode, so
      the every-position form is both the grounded choice and the one that keeps
      us comparable to the paper we cite.
    - **Encoded prompts → `from_instruction_end` (CAA).** Here the confound is
      live: writing the direction into ciphertext could change the decode itself,
      which for a paper about decoding is the exact thing the design exists to
      rule out. A phase-2 sufficiency test on an encoded condition MUST pass a
      mask, and `from_instruction_end` builds it.

    A per-row mask rather than a slice because batches are ragged: with padding,
    "the tokens before the inversion question" is a different absolute index in
    every row, and a scalar slice would steer the wrong span in all but one.
    """
    if not 0 <= layer < loaded.n_layers:
        raise ValueError(f"layer {layer} outside [0, {loaded.n_layers - 1}]")
    unit = direction
    norm = float(unit.norm())
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"direction must be unit-norm, got {norm:.6f}")

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        delta = coefficient * unit.to(device=hidden.device, dtype=hidden.dtype)
        if mask is None:
            return hidden + delta
        if mask.shape != hidden.shape[:2]:
            # Loud rather than broadcast: a silently mis-shaped mask steers the
            # wrong tokens and still returns a plausible number.
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match the hidden state's "
                f"[batch, seq] = {tuple(hidden.shape[:2])}"
            )
        scale = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return hidden + scale * delta

    with _hook_layer_output(loaded, layer, transform):
        yield


def from_instruction_end(
    loaded: LoadedModel, prompts: Sequence[str], width: int | None = None
) -> torch.Tensor:
    """CAA's steering mask: `[batch, seq]`, True from instruction-end ONWARD.

    **The tool the settled convention requires, built at settle time rather than
    left to phase 2 to re-derive** (TODO 33, 2026-08-07). `add_direction`'s
    default is every-position, matching Arditi et al.; this builds the other
    convention, which is MANDATORY for a sufficiency test on an ENCODED
    condition — there, adding into the prompt could change the decode itself, and
    a paper about decoding cannot report a result that confounds the two.

    Mirrors CAA's `mask = position_ids >= from_id` with
    `from_id = find_instruction_end_postion(...)`, resolved through our own
    capture spine instead of their `END_STR` scan: `instruction_final` is a
    position name the spine already resolves per prompt against that prompt's own
    tokenization, so the boundary comes from the same machinery every measurement
    reads at, rather than from a second string-matching implementation that could
    drift from it.

    LEFT padding, matching the capture spine and `build_inversion_batch`: the
    final position must be a real token in every row. `width` defaults to the
    longest rendered prompt; pass it to match an existing batch's `input_ids`.
    """
    from internals_safety.models.loader import prepare_prompts

    lengths: list[int] = []
    boundaries: list[int] = []
    for prepared in prepare_prompts(loaded, prompts, positions=["instruction_final"]):
        tokens = loaded.tokenizer.encode(prepared.text, add_special_tokens=False)
        lengths.append(len(tokens))
        # `instruction_final` is a NEGATIVE index from the end of the rendered
        # prompt; the first steered position is the one after it.
        offset = prepared.positions["instruction_final"]
        boundaries.append(len(tokens) + offset)

    resolved_width = width if width is not None else max(lengths, default=0)
    if resolved_width < max(lengths, default=0):
        raise ValueError(
            f"width {resolved_width} is shorter than the longest rendered prompt "
            f"({max(lengths)}); a truncated mask steers the wrong span"
        )

    mask = torch.zeros((len(lengths), resolved_width), dtype=torch.bool)
    for index, (length, boundary) in enumerate(zip(lengths, boundaries)):
        pad = resolved_width - length
        mask[index, pad + boundary :] = True
    return mask
