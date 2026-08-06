"""The logit lens — reading a residual state through the model's own unembedding.

One seam, because more than one instrument wants it: I3 (entropy dynamics) needs
a per-layer distribution, and any later token-identity readout needs the same
projection. Keeping it here means the final-norm handling is written once and
tested once.

**The final norm is not optional.** `capture.py` takes `resid_post` at the last
layer *before* the model's final norm — deliberately, so layers stay comparable
to each other — which means a captured state is NOT the vector the unembedding
expects. Applying `lm_head` to a raw residual skips the normalisation the model
itself applies and produces a distribution that is wrong in a way that looks
plausible: still peaked, still a valid distribution, systematically miscalibrated.
Entropy is exactly the statistic that failure would corrupt, so this module
applies the norm and pins it with a test.

**The lens is a tier-(B) method with a known ceiling, and the ceiling matters
for how its output may be read.** It is 542-citation established practice
(`tuned lens`, arXiv 2303.08112), but arXiv 2604.02608 **(C)** reports function
vectors that steer behaviour without being logit-lens readable. So a lens NULL is
not evidence of no representation — only a lens POSITIVE is evidence of a
readable one. Any instrument built on this seam inherits that asymmetry.
"""

from __future__ import annotations

import torch

from internals_safety.models.loader import LoadedModel


def final_norm(loaded: LoadedModel):
    """The model's pre-unembedding normalisation layer.

    Resolved by name across the shapes HuggingFace decoder models actually use,
    and raising rather than guessing when none matches — a silently skipped norm
    is the failure this module exists to prevent.
    """
    for path in (("model", "norm"), ("model", "final_layernorm"), ("transformer", "ln_f")):
        module = loaded.model
        for attribute in path:
            module = getattr(module, attribute, None)
            if module is None:
                break
        if module is not None:
            return module
    raise RuntimeError(
        f"could not locate a final norm on {type(loaded.model).__name__}; "
        "add its path here rather than skipping the norm"
    )


def unembed(loaded: LoadedModel, states: torch.Tensor) -> torch.Tensor:
    """[..., d_model] residual states -> [..., vocab] logits.

    Applies the final norm first, then the unembedding — the same two steps the
    model applies to its own last hidden state.
    """
    if states.shape[-1] != loaded.d_model:
        raise ValueError(
            f"expected d_model {loaded.d_model}, got {states.shape[-1]}"
        )
    norm = final_norm(loaded)
    parameters = list(loaded.model.parameters())
    device = parameters[0].device if parameters else states.device
    dtype = parameters[0].dtype if parameters else states.dtype

    with torch.inference_mode():
        normalised = norm(states.to(device=device, dtype=dtype))
        return loaded.model.lm_head(normalised).float()


def lens_distribution(loaded: LoadedModel, states: torch.Tensor) -> torch.Tensor:
    """[..., d_model] -> [..., vocab] probabilities."""
    return torch.softmax(unembed(loaded, states), dim=-1)
