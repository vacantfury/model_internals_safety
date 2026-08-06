"""Residual-stream patching — the write side of `capture.py`.

`capture.py` reads the residual stream at (layer, position); this module writes
one back. Deliberately the same hook points, because that module's docstring
states the reason: *"one code path for read and write means the intervention
writes at exactly the site the diagnosis read at."* A vector captured at
`resid_pre` layer 18 is patched back in at `resid_pre` layer 18, and
`tests/test_patching.py` pins the round trip — patch a state back into the pass
it came from and the logits must be unchanged.

Method provenance: this is the mechanism behind Patchscopes (Ghandeharioun et
al., **ICML 2024**, arXiv 2401.06102; our copy is
`other_repos/interpretability/patchscopes/code/patchscopes_utils.py`). Their
`set_hs_patch_hooks_llama` is plain HuggingFace forward hooks on
`model.model.layers[i]`, which is the same mechanism `capture.residual_hooks`
already uses — so this is an algorithm implemented inside our own spine, not a
framework adopted. Build plan §5 settled that; §3.1 records the read.

One thing taken from their implementation rather than invented: patching the
LAST layer requires skipping the final layer norm, because the model applies it
after the last decoder block and a patched-in `resid_post` would otherwise be
normalised a second time. They auto-enable `skip_final_ln` exactly when
`layer_source == layer_target == n_layers - 1`. We do not implement that path at
all — `patch_residual` rejects the final layer — because our capture site is
`resid_pre` by default and the case has no use here yet. Rejecting is honest;
silently patching it would be wrong.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from internals_safety.config import Site
from internals_safety.models.loader import LoadedModel


def _replace_at(hidden: torch.Tensor, position: int, vectors: torch.Tensor) -> torch.Tensor:
    """Write `vectors` [B, D] into `hidden` [B, T, D] at one position.

    `position` is a NEGATIVE offset from the end, matching `PreparedPrompt`'s
    convention throughout the capture layer — under left padding that counts
    back from each row's true last token regardless of pad width.
    """
    if position >= 0:
        raise ValueError(f"position must be a negative offset from the end, got {position}")
    if hidden.shape[0] != vectors.shape[0]:
        raise ValueError(
            f"batch mismatch: hidden has {hidden.shape[0]} rows, vectors {vectors.shape[0]}"
        )
    if hidden.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            f"d_model mismatch: hidden is {hidden.shape[-1]}, vectors {vectors.shape[-1]}"
        )
    index = hidden.shape[1] + position
    if index < 0:
        raise ValueError(f"position {position} falls before the start of a {hidden.shape[1]}-token sequence")

    patched = hidden.clone()
    patched[:, index, :] = vectors.to(device=hidden.device, dtype=hidden.dtype)
    return patched


@contextmanager
def patch_residual(
    loaded: LoadedModel,
    layer: int,
    position: int,
    vectors: torch.Tensor,
    site: Site = "resid_pre",
) -> Iterator[None]:
    """Overwrite the residual stream at (`layer`, `position`) with `vectors`.

    `vectors` is [n_prompts, d_model] and must line up row-for-row with the
    batch being run inside the context.
    """
    if not 0 <= layer < loaded.n_layers:
        raise ValueError(f"layer {layer} outside [0, {loaded.n_layers - 1}]")
    if site == "resid_post" and layer == loaded.n_layers - 1:
        # See the module docstring: this is the skip_final_ln case and we do not
        # implement it. Failing closed beats patching through a second norm.
        raise ValueError(
            "patching resid_post at the final layer needs final-layer-norm handling "
            "that this module does not implement"
        )
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be [n_prompts, d_model], got shape {tuple(vectors.shape)}")

    module = loaded.layers[layer]
    handles = []

    def pre_hook(_module, args, kwargs):
        if args:
            return (_replace_at(args[0], position, vectors), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": _replace_at(kwargs["hidden_states"], position, vectors)}
        raise RuntimeError("decoder layer received no hidden states positionally or as a kwarg")

    def post_hook(_module, _args, output):
        if isinstance(output, tuple):
            return (_replace_at(output[0], position, vectors), *output[1:])
        return _replace_at(output, position, vectors)

    try:
        if site == "resid_pre":
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        elif site == "resid_post":
            handles.append(module.register_forward_hook(post_hook))
        else:
            raise ValueError(f"unknown site: {site!r}")
        yield
    finally:
        for handle in handles:
            handle.remove()
