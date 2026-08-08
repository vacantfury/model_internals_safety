"""I4's mandatory pre-gate: does this SAE reconstruct OUR model's activations?

**Nothing in I4 is admissible until this passes**, and the reason is specific
rather than procedural. The SAE suites we would use are **Llama Scope**
(arXiv 2410.20526) and **Qwen-Scope** (arXiv 2605.11887), and Llama Scope is
trained on Llama-3.1-8B-***Base*** while our target is ***Instruct***. A
base-trained dictionary applied to an instruct-tuned model without this check
produces findings about the wrong model — and it fails silently, because a
dictionary always returns *some* reconstruction and *some* feature activations.

## Method provenance

**Cunningham, Ewart, Riggs, Huben & Sharkey, *Sparse Autoencoders Find Highly
Interpretable Features in Language Models*, ICLR 2023** (arXiv 2309.08600,
1432c) — I4's foundation, read at this build step. Two things taken from it:

1. **The validation triple.** They report reconstruction loss and *proportion of
   variance unexplained*, and — crucially — check the **downstream** effect by
   substituting the reconstruction back into the model: "replacing the residual
   stream activations in layer 2 of Pythia-70M with our reconstruction ...
   increases the perplexity on the Pile from 25 to 40."
2. **Which of the three actually matters.** Their own limitations section says
   they would rather be "minimizing the change in model outputs when replacing
   the activations with our reconstructed vectors, **rather than the
   reconstruction loss**". So the downstream term is the gate here and the other
   two are diagnostics — a dictionary can have respectable MSE and still wreck
   the distribution the model was about to emit.

Their baseline set (random directions, PCA, the neuron basis) is where this
module's negative control comes from: a **random dictionary of matched shape**,
run through the identical pipeline.

## The gate is relative, not absolute — and that is the fifth derived floor

An absolute KL bar would mean different things per model, per layer and per
corpus. So the downstream term is reported as **fraction of KL recovered**,
against the damage done by ablating the layer entirely:

    kl_recovered = 1 - KL(clean || sae) / KL(clean || zero-ablated)

1.0 is a perfect substitution, 0.0 is no better than deleting the layer, and
negative is *worse than deleting it* — which is a real outcome for an
out-of-distribution dictionary and the reason the quantity is not clamped.

## What this module deliberately does NOT do

It does not load an SAE. `SparseAutoencoder` is a two-method protocol
(`encode`/`decode`), so SAELens supplies the object in production and a test
double supplies it here. The measurement is therefore testable with no
download, no GPU and no 256-checkpoint suite — the same split as everywhere
else in this layer, and the reason it could be built before the adapter.

## Policy this gate enforces, from the build plan

I4 is a **descriptive/naming instrument**, never a control handle, unless the
matched-evaluation bar of arXiv 2607.10226 is met explicitly. The published
track record is the worst of the four instruments: SAE interventions recover
post-intervention (2606.18322), apparent SAE safety control arises from weak or
non-localized interventions (2607.10226), and features can be causally inert
despite good correlational recovery (2607.12166). A passing reconstruction
pre-gate licenses *naming what fires*. It licenses nothing causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

import torch
from contextlib import contextmanager

from internals_safety.config import MeasurementsConfig, SAEConfig
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.loader import LoadedModel, prepare_prompts

QUESTION = "does this SAE reconstruct our model's activations well enough to be read"
KIND: Kind = "correlational"


class SparseAutoencoder(Protocol):
    """The whole interface I4 needs from a dictionary.

    Two methods, on purpose. SAELens' `SAE` satisfies it, Llama Scope's
    checkpoints satisfy it through any thin wrapper, and a test double satisfies
    it in three lines — so the pre-gate is not coupled to a loader, a suite, or a
    download. Whatever supplies it, `decode(encode(x))` must be the round trip
    the model would see.
    """

    def encode(self, activations: torch.Tensor) -> torch.Tensor: ...

    def decode(self, features: torch.Tensor) -> torch.Tensor: ...


class RandomDictionary:
    """The negative control: a matched-shape dictionary that learned nothing.

    Taken from the foundational paper's own baseline set (random directions,
    PCA, the neuron basis). It reconstructs whatever a random overcomplete basis
    with the same sparsity happens to reconstruct, so the trained dictionary must
    beat it — otherwise "the SAE reconstructs our model" is a statement about
    linear algebra rather than about the dictionary.

    Rows are unit-normalised, matching the foundational paper's constraint
    (normalisation "prevents the model from reducing the sparsity loss term by
    increasing the size of the feature vectors").
    """

    def __init__(self, d_model: int, n_features: int, k: int, generator: torch.Generator):
        weights = torch.randn(n_features, d_model, generator=generator)
        self.weights = weights / weights.norm(dim=-1, keepdim=True)
        self.k = k

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        # Same coercion as the trained dictionary's seam: activations move to
        # the weights, not the reverse. Symmetric on purpose — a control that
        # breaks where the real dictionary works is not a matched control.
        activations = activations.to(device=self.weights.device, dtype=self.weights.dtype)
        scores = activations @ self.weights.T
        # Top-k, matching Llama Scope's TopK architecture rather than an L1
        # penalty: the control has to share the trained dictionary's SPARSITY,
        # since a dense control would reconstruct better for a reason that has
        # nothing to do with having learned features.
        kept = torch.zeros_like(scores)
        values, indices = scores.topk(min(self.k, scores.shape[-1]), dim=-1)
        return kept.scatter_(-1, indices, values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features.to(self.weights.device) @ self.weights


def variance_explained(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """1 - FVU, the complement of the paper's "proportion of variance unexplained".

    Centred on the batch mean, so a dictionary that only ever emits the mean
    activation scores 0 rather than looking respectable — the mean is free.
    """
    residual = (original - reconstruction).pow(2).sum()
    centred = (original - original.mean(dim=0, keepdim=True)).pow(2).sum()
    if centred <= 0:
        return float("nan")
    return float(1.0 - residual / centred)


def l0_sparsity(features: torch.Tensor) -> float:
    """Mean number of active features per activation — the paper's x-axis.

    Reported beside every reconstruction number because the two trade off
    smoothly ("the lack of a 'bump' or 'knee' in these plots"), so a
    reconstruction figure without its sparsity is uninterpretable.
    """
    return float((features != 0).float().sum(dim=-1).mean())


def scored_positions(attention_mask: torch.Tensor, *, drop_first_real: bool) -> torch.Tensor:
    """[batch, seq] bool: which positions may enter a reconstruction statistic.

    **Why this exists — it cost job `9008483` and produced a false refusal.**
    The pre-gate reduced over EVERY position of a padded batch, and both things
    it swept in are catastrophic for this particular metric:

    * **PAD.** `models/loader.py` sets `padding_side = "left"`, so the pad run
      sits at the START of every short row. The reduction summed pads into the
      error AND counted them in the denominator.
    * **BOS.** Llama-3.1 carries a massive-activation / attention-sink spike at
      the first real token, orders of magnitude above a normal residual. Llama
      Scope declares `dataset_average_activation_norm: 13.8125`; the norm implied
      by the failed run's `MSE 7.6M` at `variance_explained -936` is ~5,700.

    BOS is dropped by an EXPLICIT decision, not as a side effect: the attention
    mask marks it as real, so a pure padding fix would leave it in. Whether the
    spike belongs in the metric is a question about what the dictionary is being
    asked to reconstruct, and the answer is no — the dictionary was normalised
    against a dataset average that the spike would dominate, so scoring it asks
    the SAE to reconstruct the one activation its normalisation assumes away.

    `drop_first_real` finds the first UNMASKED position per row rather than
    index 0, because under left padding index 0 is a pad on every short row.
    """
    keep = attention_mask.bool().clone()
    if drop_first_real and keep.any():
        first_real = keep.float().argmax(dim=1)          # left padding: first 1
        keep[torch.arange(keep.shape[0], device=keep.device), first_real] = False
    return keep


@contextmanager
def _substitute(loaded: LoadedModel, layer: int, transform) -> Iterator[None]:
    """Apply `transform` to the residual stream entering `layer`.

    Whole-sequence, unlike `models/patching.py`, which writes one position per
    row. That is the difference between the two questions: patching asks what
    ONE cell carries, this asks whether the dictionary can stand in for the
    layer's input at all — which is the substitution the foundational paper
    measures perplexity under.
    """
    def pre_hook(_module, args, kwargs):
        if args:
            return (transform(args[0]), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": transform(kwargs["hidden_states"])}
        raise RuntimeError("decoder layer received no hidden states positionally or as a kwarg")

    handle = loaded.layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@dataclass(frozen=True)
class ReconstructionQuality:
    """One SAE's fitness to stand in for one model's layer."""

    layer: int
    n_prompts: int
    mse: float
    variance_explained: float
    l0: float
    kl_sae: float
    kl_ablated: float
    # The same three, from the matched-shape random dictionary.
    control_variance_explained: float | None = None
    control_kl_sae: float | None = None

    @property
    def kl_recovered(self) -> float | None:
        """Fraction of the layer's downstream contribution the SAE preserves.

        `None` when ablating the layer did nothing measurable — with no damage
        to recover from, the ratio is undefined and reporting 1.0 would credit
        the dictionary for a layer that does not matter.
        """
        if self.kl_ablated <= 1e-9:
            return None
        return float(1.0 - self.kl_sae / self.kl_ablated)

    @property
    def control_margin(self) -> float | None:
        """How far the trained dictionary beats the random one on variance."""
        if self.control_variance_explained is None:
            return None
        return self.variance_explained - self.control_variance_explained


def measure_reconstruction(
    loaded: LoadedModel,
    sae: SparseAutoencoder,
    prompts: Sequence[str],
    layer: int,
    config: SAEConfig,
    control: SparseAutoencoder | None = None,
    # plumbing(batch_size): throughput only — every read is per-prompt
    batch_size: int = 8,
    # constant: BOS carries Llama-3.1's massive-activation spike, which the
    # dictionary's dataset-wise normalisation assumes away — see scored_positions
    drop_bos: bool = True,
) -> ReconstructionQuality:
    """Run the three checks the foundational paper reports, on OUR model.

    Prompts are rendered through the chat template, like everywhere else on the
    write side: an SAE validated on bare instruction text has been validated on
    inputs the instruct-tuned target never receives, which is the same
    distribution error the pre-gate exists to catch one level up.

    **Only real, non-BOS positions are scored** (`scored_positions`). The first
    version reduced over every position of a left-padded batch and produced
    `variance_explained` of -936 to -1819 on the model the dictionary was FITTED
    ON — a false refusal that cost job `9008483`. Padding and the attention-sink
    spike were in both the numerator and the denominator.
    """
    rendered = [prompt.text for prompt in prepare_prompts(loaded, prompts, positions=[])]

    totals = {"sq_err": 0.0, "n_tokens": 0.0, "l0": 0.0, "kl_sae": 0.0, "kl_ablated": 0.0}
    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    control_reconstructions: list[torch.Tensor] = []
    kept_masks: list[torch.Tensor] = []

    def round_trip(dictionary: SparseAutoencoder, hidden: torch.Tensor) -> torch.Tensor:
        features = dictionary.encode(hidden.float())
        reconstruction = dictionary.decode(features)
        if reconstruction.shape != hidden.shape:
            # Fail here with the shapes named. Writing a mis-shaped tensor back
            # into the residual stream surfaces as a broadcast error inside
            # scaled_dot_product_attention several frames down, which says
            # nothing about the dictionary that caused it.
            raise ValueError(
                f"the SAE round trip changed shape: {tuple(hidden.shape)} -> "
                f"{tuple(reconstruction.shape)}. decode(encode(x)) must return x's "
                f"shape — a dictionary that reduces over the batch is reconstructing "
                f"the corpus, not the activation."
            )
        # DEVICE as well as dtype — the round trip returns on the CALLER's
        # device, whatever device the dictionary itself lives on.
        #
        # This completes the invariant the seam fix started (2026-08-07):
        # `encode`/`decode` move activations TO the dictionary, because its
        # weights are ~134M params and the batch is not; `round_trip` brings the
        # result BACK, because everything downstream here is deliberately CPU
        # (`capture_only` does `.detach().float().cpu()`, logits are `.cpu()`).
        #
        # Fixing only the first half is what killed job 9006846: encode then
        # succeeded on a CUDA dictionary and returned a CUDA reconstruction into
        # a CPU pipeline, so `hidden - reconstruction` raised one line later.
        # Every test passed throughout, because they all use a CPU dictionary
        # and the two halves are only distinguishable when it is not.
        return reconstruction.to(device=hidden.device, dtype=hidden.dtype), features

    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        encoded = loaded.tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}

        captured: list[torch.Tensor] = []

        def capture_only(hidden):
            captured.append(hidden.detach().float().cpu())
            return hidden

        with torch.inference_mode():
            with _substitute(loaded, layer, capture_only):
                clean_logits = loaded.model(**inputs).logits[:, -1, :].float().cpu()

            hidden = captured[0]
            reconstruction, features = round_trip(sae, hidden)

            # EVERY statistic below sees the same positions, and the denominator
            # is the kept count — not `batch * seq_len`. Masking only the
            # numerator would put a correct error over an inflated denominator
            # and land somewhere plausible-but-wrong, which is the same
            # half-an-invariant shape that killed job 9006846 one seam over.
            keep = scored_positions(
                encoded["attention_mask"].cpu(), drop_first_real=drop_bos
            )
            flat = keep.reshape(-1)
            kept_hidden = hidden.reshape(-1, hidden.shape[-1])[flat]
            kept_reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])[flat]

            originals.append(kept_hidden)
            reconstructions.append(kept_reconstruction)
            totals["sq_err"] += float((kept_hidden - kept_reconstruction).pow(2).sum())
            totals["n_tokens"] += float(flat.sum())
            totals["l0"] += l0_sparsity(
                features.reshape(-1, features.shape[-1])[flat]
            ) * hidden.shape[0]
            kept_masks.append(flat)

            with _substitute(loaded, layer, lambda h: round_trip(sae, h.float().cpu())[0].to(h.device, h.dtype)):
                sae_logits = loaded.model(**inputs).logits[:, -1, :].float().cpu()
            with _substitute(loaded, layer, lambda h: torch.zeros_like(h)):
                ablated_logits = loaded.model(**inputs).logits[:, -1, :].float().cpu()

            reference = torch.log_softmax(clean_logits, dim=-1)
            for name, other in (("kl_sae", sae_logits), ("kl_ablated", ablated_logits)):
                totals[name] += float(
                    torch.nn.functional.kl_div(
                        torch.log_softmax(other, dim=-1), reference,
                        log_target=True, reduction="sum",
                    )
                )

            if control is not None:
                # Filtered with the SAME mask: a control scored over different
                # positions than the dictionary is not a matched control.
                control_reconstructions.append(
                    round_trip(control, hidden)[0].reshape(-1, hidden.shape[-1])[flat]
                )

    stacked = torch.cat(originals)
    n = float(len(rendered))
    return ReconstructionQuality(
        layer=layer,
        n_prompts=len(rendered),
        mse=totals["sq_err"] / max(totals["n_tokens"], 1.0),
        variance_explained=variance_explained(stacked, torch.cat(reconstructions)),
        l0=totals["l0"] / n,
        kl_sae=totals["kl_sae"] / n,
        kl_ablated=totals["kl_ablated"] / n,
        control_variance_explained=(
            variance_explained(stacked, torch.cat(control_reconstructions))
            if control_reconstructions
            else None
        ),
    )


def loaded_model_name(quality: ReconstructionQuality) -> str:
    """Placeholder until the loader carries the target's name into the reading.

    Left explicit rather than dropped: `trained_on` vs `evaluated_on` IS the
    pre-gate's whole subject, so a record showing only one of the pair is a
    record that cannot answer the question it was written to answer.
    """
    return "set by the caller once the SAE loader lands (TODO 55)"


def unmeasured_reading(reason: str) -> Reading:
    return Reading(
        instrument="sae_reconstruction",
        kind=KIND,
        value=float("nan"),
        operating_point="SAE reconstruction pre-gate not run — see detail.reason",
        licensed=None,
        detail={"reason": reason},
    )


def reading(quality: ReconstructionQuality, config: SAEConfig) -> Reading:
    """The pre-gate's verdict. `licensed=True` means I4 MAY be read on this model.

    Three conditions, and the ordering of importance is the paper's own: the
    downstream term is the gate, variance explained is a diagnostic that must
    still beat the random-dictionary control, and MSE is reported but decides
    nothing on its own.

    ⚠️ `licensed=True` here licenses NAMING, never a causal claim — see the
    module docstring's policy note. It is a gate on admissibility, not evidence
    about the model.
    """
    recovered = quality.kl_recovered
    if recovered is None:
        return unmeasured_reading(
            f"ablating layer {quality.layer} changed the output distribution by "
            f"KL {quality.kl_ablated:.2e}, so there is no downstream contribution "
            "for a reconstruction to recover — the gate is undefined here, not passed"
        )
    margin = quality.control_margin
    passes = (
        recovered >= config.min_kl_recovered
        and quality.variance_explained >= config.min_variance_explained
        and margin is not None
        and margin > 0.0
    )
    return Reading(
        instrument="sae_reconstruction",
        kind=KIND,
        value=recovered,
        operating_point=(
            f"fraction of downstream KL recovered by substituting the SAE round trip "
            f"at layer {quality.layer} (1.0 = perfect, 0.0 = no better than deleting "
            f"the layer, negative = worse than deleting it); passes at "
            f">= {config.min_kl_recovered} with variance explained "
            f">= {config.min_variance_explained} and above a matched-shape random "
            "dictionary"
        ),
        licensed=bool(passes),
        control_reading=quality.control_variance_explained,
        control_margin=margin,
        detail={
            "layer": quality.layer,
            "mse": quality.mse,
            "variance_explained": quality.variance_explained,
            # Reported beside every reconstruction number: the two trade off
            # smoothly, so a reconstruction figure without its sparsity is
            # uninterpretable.
            "l0": quality.l0,
            "kl_sae": quality.kl_sae,
            "kl_ablated": quality.kl_ablated,
            "n_prompts": quality.n_prompts,
            # ⚠️ The whole reason this gate exists. Llama Scope is trained on
            # Llama-3.1-8B-Base and our target is Instruct.
            "trained_on": config.trained_on,
            "evaluated_on": loaded_model_name(quality),
        },
    )
