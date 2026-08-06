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

    **`mask` is `[batch, seq]` of bools, and its absence means EVERY position —
    which is a convention worth stating rather than assuming (TODO 33).** CAA
    (arXiv 2312.06681) adds from the instruction-end position *onward*, including
    generated tokens; ours defaulted to everywhere, which under AS-5's setup
    writes the direction into the encoded prompt itself. Those are different
    interventions: theirs asks whether writing the direction into the model's own
    RESPONSE induces the behaviour, ours also changes what the model reads the
    prompt as — and for a paper about decoding, that is the confound the design
    exists to avoid. The mask makes the choice explicit at the call site instead
    of implicit in the default.

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
