"""I4's SAE loader — Llama Scope checkpoints as the two-method protocol.

**Why a thin loader rather than SAELens (the mature-tools law, argued not
assumed).** The house rule is to use a mature library rather than hand-roll, and
the exception it names is a concrete reason against. Measured 2026-08-06:
`sae-lens` resolves to **116 packages** — more than this entire repo's 103 — and
pulls in `transformer-lens` (a second model-loading framework beside our own
capture spine), plus `wandb`, `sentry-sdk`, `plotly`, `statsmodels` and `nltk`.
We need exactly two methods. A second model framework is not a dependency, it is
a second home for the thing I0 already owns, and this repo has spent three days
removing second homes.

What is left is not a reimplementation: a Llama Scope checkpoint is **four
tensors and a JSON**, and encode/decode are the two matrix products they define.

## The checkpoint, read from the artifact rather than assumed

Verified against `fnlp/Llama3_1-8B-Base-LXR-8x`, layer 15:

    encoder.weight   [d_sae, d_model]   (32768, 4096)  bf16
    encoder.bias     [d_sae]
    decoder.weight   [d_model, d_sae]   (4096, 32768)
    decoder.bias     [d_model]

## Three properties that would each silently corrupt the pre-gate

1. **`hook_point_in` is `blocks.N.hook_resid_POST`, and we capture `resid_pre`.**
   `resid_post` of block N is `resid_pre` of block N+1, so a Llama Scope SAE
   named for layer N must be read at OUR layer N+1. Off by one here does not
   error — it reconstructs a neighbouring layer badly and reads as "the
   dictionary does not transfer", which is the pre-gate's own verdict wearing
   our bug.

2. **`norm_activation: "dataset-wise"`.** The SAE was trained on activations
   rescaled so the dataset's average norm becomes `sqrt(d_model)`; the checkpoint
   carries that average (10.8125 at layer 15). Feeding raw activations in skips
   a ~5.9x scale factor.

   The convention is **derived, not guessed**, from two independent readings of
   the artifact. *First:* `jump_relu_threshold` is 0.3555, and under a
   normalise-to-unit-norm convention the reconstruction of a norm-1 vector from
   ~50 active features with decoder columns of norm ~1.54 needs feature
   activations near 0.09 — every one of which the threshold would zero, making
   the SAE output identically nothing. *Second:* `decoder.bias` approximates the
   mean of the normalised activations, and its norm is **4.81**; under unit-norm
   scaling that mean would sit near 0.08. Both point at `sqrt(d_model)`.

3. **`act_fn` is `jumprelu`, NOT TopK**, despite `top_k: 50` sitting in the same
   file as a training-schedule leftover. A TopK forward would produce exactly 50
   active features where the real dictionary produces however many clear 0.3555 —
   so `SAEConfig.control_k` must be matched to the OBSERVED L0, never to the
   nominal 50. `observed_l0` exists for that.

## The loader's own correctness is checked by the experiment we already planned

The pre-gate runs the dictionary against **Base** (the model it was trained on)
and against **Instruct** (our target). That design exists to test transfer — but
the Base arm doubles as a check on THIS FILE. If Base reconstruction is poor, the
loader is wrong, not the dictionary; a dictionary cannot fail to transfer to the
model it was fitted on. So the pre-gate cannot be read at all until the Base arm
passes, and that ordering is not optional.

⚠️ Their reference implementation (`OpenMOSS/Language-Model-SAEs`) is NOT cloned
locally — repo clones are the owner's call. The normalisation convention above is
derived from the artifact rather than read from their code, and the Base arm is
what would catch it being wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch

# Llama Scope names a layer's SAE by the block whose `resid_post` it reads.
# Our capture site is `resid_pre`, and `resid_post` of block N IS `resid_pre` of
# block N+1 — see the module docstring for why an off-by-one is invisible.
#
# definitional: an INDEX IDENTITY, not a tunable — `resid_post` of block N and
# `resid_pre` of block N+1 are the same tensor, so the offset is 1 by the
# architecture's definition and no data could move it. Tuning path: none exists,
# and that is the point; if a checkpoint ever names a different hook point,
# `our_layer` REFUSES rather than applying this offset to a site it was never
# derived for.
RESID_POST_TO_OUR_RESID_PRE = 1


@dataclass(frozen=True)
class LlamaScopeSAE:
    """A Llama Scope dictionary, satisfying the `SparseAutoencoder` protocol.

    Frozen and stateless: `encode`/`decode` are pure functions of the tensors, so
    the same object is safe to reuse across the pre-gate's two arms.
    """

    encoder_weight: torch.Tensor  # [d_sae, d_model]
    encoder_bias: torch.Tensor  # [d_sae]
    decoder_weight: torch.Tensor  # [d_model, d_sae]
    decoder_bias: torch.Tensor  # [d_model]

    # From hyperparams.json — carried so a reading can state what it loaded.
    trained_on: str
    hook_point: str
    jump_relu_threshold: float
    input_norm: float
    output_norm: float
    d_model: int
    d_sae: int
    # Training-schedule leftover. Carried ONLY so a caller can see that it is not
    # the activation function; never used to select features here.
    nominal_top_k: int

    # `sparsity_include_decoder_norm` from hyperparams.json, and it is TRUE in
    # every Llama Scope checkpoint. Upstream gates on `hidden_pre * decoder_norm`
    # and divides back out afterwards (their `models/sae.py` encode); this loader
    # did NEITHER half until 2026-08-07, which is the pre-gate's real defect.
    #
    # Read from the artifact rather than assumed, because upstream's own default
    # is True and a checkpoint that ever set it False must not be silently gated.
    decoder_norm_gating: bool = True

    @property
    def our_layer(self) -> int:
        """The layer index in OUR `resid_pre` capture indexing.

        Raises rather than guessing when the hook point is not the
        `blocks.N.hook_resid_post` form this loader was verified against — a
        different hook point means a different mapping, and a silent default
        would put the dictionary on the wrong layer.
        """
        parts = self.hook_point.split(".")
        if len(parts) != 3 or parts[0] != "blocks" or parts[2] != "hook_resid_post":
            raise ValueError(
                f"unrecognised hook point {self.hook_point!r}; this loader maps only "
                "`blocks.N.hook_resid_post` onto our resid_pre capture, and any other "
                "site needs its mapping established before a reading means anything"
            )
        return int(parts[1]) + RESID_POST_TO_OUR_RESID_PRE

    def _normalise(self, activations: torch.Tensor) -> torch.Tensor:
        """Scale raw activations into the space the SAE was fitted in."""
        scale = (self.d_model**0.5) / self.input_norm
        return activations * scale

    def _denormalise(self, reconstruction: torch.Tensor) -> torch.Tensor:
        return reconstruction * (self.output_norm / (self.d_model**0.5))

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Features, under the checkpoint's OWN activation function (jumprelu).

        `apply_decoder_bias_to_pre_encoder` is false in every checkpoint verified,
        so the decoder bias is NOT subtracted before encoding. Asserted by the
        loader rather than assumed here — see `load_llama_scope_sae`.
        """
        # Coerce the ACTIVATIONS to the dictionary's device, never the reverse:
        # the weights are d_sae x d_model (~134M params for Llama Scope) and the
        # batch is [n, d_model], so moving the weights per call would be absurd.
        #
        # This seam exists because `measure_reconstruction` deliberately hands
        # the round trip `h.float().cpu()` while `sae_pregate.py` loads the
        # dictionary with `device=cuda` — a contradiction that killed job
        # 9006556 after it had loaded an 8B model and a 540 MB dictionary.
        # Fixing it HERE rather than at either call site makes the mismatch
        # unexpressible, which is the same lesson as `strata` earlier today.
        activations = activations.to(self.encoder_weight.device)
        weight = self.encoder_weight.to(activations.dtype)
        pre = self._normalise(activations) @ weight.T + self.encoder_bias.to(activations.dtype)
        # `sparsity_include_decoder_norm`, verified against upstream's own encode
        # (OpenMOSS/Language-Model-SAEs `models/sae.py`), which does exactly:
        #
        #     hidden_pre  = hidden_pre * decoder_norm()
        #     feature_acts = activation_function(hidden_pre)
        #     feature_acts = feature_acts / decoder_norm()
        #
        # The gate is applied in a decoder-norm-scaled space and the magnitudes
        # are scaled back afterwards. It changes WHICH features fire (a feature
        # with a large decoder column clears the threshold on a smaller
        # pre-activation) and it is not a no-op on the reconstruction either.
        # Their config docstring gives the reason: it "suppresses the training
        # dynamics that model tries to increase the decoder norm in exchange of a
        # smaller feature activation magnitude".
        norms = self.decoder_norms.to(pre.device, pre.dtype) if self.decoder_norm_gating else None
        if norms is not None:
            pre = pre * norms
        # JumpReLU: pass values above the threshold through UNCHANGED, zero the
        # rest. Not ReLU (which would keep small positives) and not TopK — the
        # checkpoint's `top_k: 50` really is a training-schedule leftover, now
        # CONFIRMED by `act_fn: 'jumprelu'` in hyperparams.json rather than
        # derived from the artifact as it was until 2026-08-07.
        features = torch.where(pre > self.jump_relu_threshold, pre, torch.zeros_like(pre))
        return features / norms if norms is not None else features

    @property
    def decoder_norms(self) -> torch.Tensor:
        """Per-FEATURE L2 norm of the decoder columns — `[d_sae]`.

        Upstream reduces `norm(W_D, dim=1)` over their `[d_sae, d_model]` layout.
        Ours is transposed (`[d_model, d_sae]`), so the same quantity is `dim=0`.
        """
        return self.decoder_weight.norm(p=2, dim=0)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.decoder_weight.device)
        weight = self.decoder_weight.to(features.dtype)
        reconstruction = features @ weight.T + self.decoder_bias.to(features.dtype)
        return self._denormalise(reconstruction)

    def observed_l0(self, activations: torch.Tensor) -> float:
        """Mean active features on real activations — the sparsity to MATCH.

        The random-dictionary control must match the SAE's real sparsity, and
        `nominal_top_k` is not it: this dictionary is jumprelu, so how many
        features fire is a property of the data, not of the config.
        """
        features = self.encode(activations)
        return float((features > 0).float().sum(dim=-1).mean())


def load_llama_scope_sae(
    weights_path: Path | str,
    hyperparams_path: Path | str,
    repo_id: str,
    device: str = "cpu",
) -> LlamaScopeSAE:
    """Build the dictionary from a downloaded checkpoint and its hyperparams.

    Takes PATHS rather than downloading, so the loader is testable against a
    synthetic checkpoint with no network — the same property that makes the
    pre-gate testable against protocol doubles.
    """
    from safetensors.torch import load_file

    hyperparams = json.loads(Path(hyperparams_path).read_text(encoding="utf-8"))
    tensors = load_file(str(weights_path))

    missing = {"encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"} - set(tensors)
    if missing:
        raise ValueError(f"checkpoint is missing {sorted(missing)}; not a Llama Scope SAE")

    # Fail LOUD on the two settings whose other value would change the forward
    # pass. A checkpoint that differs here is not unsupported-and-approximated,
    # it is a different dictionary, and approximating it would produce a
    # confident reconstruction number for a function nobody implemented.
    if hyperparams.get("act_fn") != "jumprelu":
        raise ValueError(
            f"act_fn is {hyperparams.get('act_fn')!r}, not 'jumprelu'; this loader "
            "implements the jumprelu forward only"
        )
    if hyperparams.get("apply_decoder_bias_to_pre_encoder"):
        raise ValueError(
            "this checkpoint subtracts the decoder bias before encoding, which this "
            "loader does not do — the reconstruction would be wrong by that bias"
        )
    if hyperparams.get("norm_activation") != "dataset-wise":
        raise ValueError(
            f"norm_activation is {hyperparams.get('norm_activation')!r}; the dataset-wise "
            "scaling is derived for that setting only and is not transferable"
        )

    norms = hyperparams["dataset_average_activation_norm"]
    return LlamaScopeSAE(
        encoder_weight=tensors["encoder.weight"].to(device).float(),
        encoder_bias=tensors["encoder.bias"].to(device).float(),
        decoder_weight=tensors["decoder.weight"].to(device).float(),
        decoder_bias=tensors["decoder.bias"].to(device).float(),
        # The pre-gate's whole subject: what this was fitted on, against what we
        # are about to read it on.
        trained_on=repo_id,
        hook_point=hyperparams["hook_point_in"],
        jump_relu_threshold=float(hyperparams["jump_relu_threshold"]),
        input_norm=float(norms["in"]),
        output_norm=float(norms["out"]),
        d_model=int(hyperparams["d_model"]),
        d_sae=int(hyperparams["d_sae"]),
        nominal_top_k=int(hyperparams.get("top_k", 0)),
        # Upstream's own default is True, so an ABSENT key must gate rather than
        # skip: a checkpoint that omits the flag is a checkpoint trained under
        # the default, and defaulting to False here would silently restore the
        # exact bug this line fixes.
        decoder_norm_gating=bool(hyperparams.get("sparsity_include_decoder_norm", True)),
    )


def download_llama_scope_sae(
    layer: int,
    repo_id: str = "fnlp/Llama3_1-8B-Base-LXR-8x",
    device: str = "cpu",
) -> LlamaScopeSAE:
    """Fetch one layer's dictionary from the Hub.

    `layer` is LLAMA SCOPE's index (the block whose `resid_post` it reads), not
    ours — passing our index here would load a neighbour. `LlamaScopeSAE.our_layer`
    converts in the other direction, and is what the pre-gate should read.
    """
    from huggingface_hub import hf_hub_download

    stem = f"Llama3_1-8B-Base-L{layer}R-8x"
    return load_llama_scope_sae(
        weights_path=hf_hub_download(repo_id, f"{stem}/checkpoints/final.safetensors"),
        hyperparams_path=hf_hub_download(repo_id, f"{stem}/hyperparams.json"),
        repo_id=f"{repo_id}/{stem}",
        device=device,
    )
