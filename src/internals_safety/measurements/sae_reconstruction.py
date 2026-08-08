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

import json
from dataclasses import dataclass
from pathlib import Path
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


def mean_activation_norm(activations: torch.Tensor) -> float:
    """Mean L2 norm per activation vector — the quantity the loader ASSUMES.

    **Why this is recorded rather than derived.** `LlamaScopeSAE._normalise`
    scales by `sqrt(d_model) / input_norm`, where `input_norm` is the checkpoint's
    `dataset_average_activation_norm`. That is a claim about OUR activations: it
    asserts they arrive with that average norm, so scaling lands them at
    `sqrt(d_model)` — the space the dictionary was fitted in. Nothing checked it.

    A mismatch here does not error. It feeds the encoder vectors at the wrong
    scale, which surfaces two layers downstream as an inflated L0 (too many
    features clear the jumprelu threshold) and a catastrophically negative
    `variance_explained` — i.e. as "the dictionary does not transfer", which is
    the pre-gate's own verdict wearing our bug. Job `9008803` read
    `variance_explained` of -915 to -1760 on the model the dictionary was FITTED
    ON, with a random dictionary scoring +0.13 to +0.21 against it; a trained
    dictionary cannot lose to random on its own training distribution, so the
    input scale was the first thing that had to be measured rather than assumed.

    Reported beside `declared_input_norm` so the ratio is readable directly off
    the run record instead of being reconstructed from MSE afterwards.
    """
    return float(activations.norm(dim=-1).mean())


def l0_sparsity(features: torch.Tensor) -> float:
    """Mean number of active features per activation — the paper's x-axis.

    Reported beside every reconstruction number because the two trade off
    smoothly ("the lack of a 'bump' or 'knee' in these plots"), so a
    reconstruction figure without its sparsity is uninterpretable.
    """
    return float((features != 0).float().sum(dim=-1).mean())


def _render(loaded: LoadedModel, prompts: Sequence[str], render_chat: bool) -> list[str]:
    """Prompt text as the model will see it.

    **The Base arm of the pre-gate must NOT be chat-templated, and that is not a
    formatting preference.** Llama Scope's dictionaries were fitted on plain
    text and the Base checkpoint never saw a chat template in training, so a
    templated Base run reads the dictionary on a distribution neither it nor the
    model has ever met — and then reports poor reconstruction as evidence about
    `models/sae_loader.py`, which is the one thing the Base arm exists to test.

    `conf/models/llama3_1_8b_base.yaml` borrows the Instruct sibling's template
    to hold the input text fixed across the two arms. That is correct for
    reading the TRANSFER gap and it silently disables the loader check; the two
    purposes pull opposite ways and the arm can only serve one at a time.

    Evidence it matters: the templated Base run measures a mean activation norm
    of 23.7 against the checkpoint's declared 13.8 — special tokens the base
    model never trained on are exactly what inflates that.
    """
    if render_chat:
        return [prompt.text for prompt in prepare_prompts(loaded, prompts, positions=[])]
    return list(prompts)


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
      Scope declares `dataset_average_activation_norm: 13.8125` **at layer 17**
      — the value is PER LAYER (17.125 and 21.5 at 19 and 21), not global — while
      the norm implied by the failed run's `MSE 7.6M` at `variance_explained -936`
      is ~5,700, orders of magnitude above every one of them.

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

    # The loader's central ASSUMPTION, measured. `None` = not recorded (a test
    # double that never saw real activations), never "matches".
    observed_activation_norm: float | None = None
    # What the checkpoint declared it would be. `None` for any dictionary that
    # does not normalise, which is the honest reading for the random control.
    declared_input_norm: float | None = None
    # The target's own identity. `trained_on` vs `evaluated_on` IS this gate's
    # whole subject, so a record carrying one without the other cannot answer
    # the question it was written to answer.
    evaluated_on: str | None = None
    # What the CHECKPOINT says it was fitted on, read from hyperparams.json
    # rather than from our config. The config knob is a hand-maintained string
    # and this one is the artifact's own claim; where both exist the artifact
    # wins, because a mislabelled config is exactly the error that would make
    # this gate compare a dictionary against the wrong baseline.
    trained_on: str | None = None
    # Which rule selected the features. Two readings from one run differ ONLY in
    # this, so a record without it cannot say which reading it is.
    selection: str | None = None

    @property
    def norm_ratio(self) -> float | None:
        """Observed activation norm over the checkpoint's declared average.

        **1.0 is the only value under which the reconstruction numbers mean what
        they say.** Far from 1.0 and the dictionary is being fed a distribution
        it was never fitted on, so every downstream figure is measuring our
        scaling rather than its transfer. Deliberately not clamped and
        deliberately not a gate: it is reported so the failure is legible, and
        turning it into a pass/fail condition is a separate decision that would
        change what `licensed` means.
        """
        if self.observed_activation_norm is None or not self.declared_input_norm:
            return None
        return self.observed_activation_norm / self.declared_input_norm

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
    # definitional: which DISTRIBUTION the dictionary is read on. Not a knob for
    # tuning a number — it decides what the number is ABOUT. Tuning path: none;
    # the preset declares it and `render_chat: false` is the Base arm's own
    # training distribution. See `scripts/sae_pregate.py`.
    render_chat: bool = True,
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
    rendered = _render(loaded, prompts, render_chat)

    totals = {
        "sq_err": 0.0, "n_tokens": 0.0, "l0": 0.0, "kl_sae": 0.0, "kl_ablated": 0.0,
        # Summed over the SAME kept positions as everything else, for the same
        # reason: a norm averaged over pads and the BOS spike describes a
        # distribution the dictionary was never asked to reconstruct.
        "norm_sum": 0.0,
    }
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
            totals["norm_sum"] += float(kept_hidden.norm(dim=-1).sum())
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
        observed_activation_norm=(
            totals["norm_sum"] / totals["n_tokens"] if totals["n_tokens"] else None
        ),
        # `getattr` rather than a protocol method: `SparseAutoencoder` is
        # deliberately two methods, and a dictionary that does not normalise
        # (the random control) has no declared norm to report. Absent reads as
        # None — "this dictionary makes no scale claim" — never as agreement.
        declared_input_norm=getattr(sae, "input_norm", None),
        evaluated_on=loaded.config.hf_id,
        trained_on=getattr(sae, "trained_on", None),
        selection=getattr(sae, "selection", None),
    )


def observed_sparsity(
    loaded: LoadedModel,
    sae: SparseAutoencoder,
    prompts: Sequence[str],
    layer: int,
    *,
    # plumbing(batch_size): throughput only — the L0 is a per-position mean
    batch_size: int = 8,
    # constant: must MATCH `measure_reconstruction`'s masking, since the whole
    # point is that the control is sized on the positions the run will score
    drop_bos: bool = True,
    # definitional: must MATCH `measure_reconstruction`'s rendering for the same
    # reason — a control sized on a different distribution is not matched.
    render_chat: bool = True,
) -> float:
    """Mean L0 over EXACTLY the positions `measure_reconstruction` will score.

    **The random control's sparsity must match the dictionary's, and matching it
    to the wrong distribution makes the margin uninterpretable in both
    directions** — a control that is too dense is unbeatable, one that is too
    sparse is beaten by anything.

    It existed as `LlamaScopeSAE.observed_l0` over `positions=["last"]` on 16
    prompts, while the run reduces over every real non-BOS position of all of
    them. Job 9008803 recorded both in ONE results.json and they disagreed by ~4x
    (167 vs 549) — found by the peer session reading the artifact. The last token
    of a chat-templated prompt is simply not drawn from the same distribution as
    the body.

    So this walks the same positions through the same mask as the reduction. It
    is not a second implementation of that rule: `scored_positions` is the single
    home, and this calls it.
    """
    rendered = _render(loaded, prompts, render_chat)
    total, counted = 0.0, 0.0
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        encoded = loaded.tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
        captured: list[torch.Tensor] = []

        with torch.inference_mode():
            with _substitute(loaded, layer, lambda h: (captured.append(h.detach().float().cpu()), h)[1]):
                loaded.model(**inputs)

        hidden = captured[0]
        keep = scored_positions(encoded["attention_mask"].cpu(), drop_first_real=drop_bos)
        kept = hidden.reshape(-1, hidden.shape[-1])[keep.reshape(-1)]
        if not kept.numel():
            continue
        features = sae.encode(kept)
        total += float((features > 0).float().sum(dim=-1).sum())
        counted += float(kept.shape[0])
    return total / counted if counted else 0.0


def ceiling_from(path: str | Path, layer: int) -> float:
    """The ceiling arm's variance explained, for THIS layer.

    **Two guards, and both exist because the failure would be silent.**

    * **The source must BE a ceiling arm.** Chaining a target reading as a
      ceiling would compound one transfer ratio on top of another and produce a
      confident number describing nothing. `detail.arm` says which it is.
    * **The layer must MATCH.** The ceiling is per layer — Base measures
      0.698 / 0.708 / 0.723 at 18 / 20 / 22 — so a ceiling read from the wrong
      layer shifts the floor by a few percent and never errors.

    Raises rather than defaulting on every failure. A missing ceiling is not a
    ceiling of 0.0, which would license anything that reconstructs at all.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    readings = [r for r in record.get("readings", []) if r.get("instrument") == "sae_reconstruction"]
    if len(readings) != 1:
        raise ValueError(
            f"{path} holds {len(readings)} sae_reconstruction readings; the ceiling is ambiguous"
        )
    detail = readings[0].get("detail", {})
    if detail.get("arm") != "ceiling":
        raise ValueError(
            f"{path} is a {detail.get('arm')!r} arm, not a ceiling — a target reading "
            "used as a ceiling compounds two transfer ratios"
        )
    if detail.get("layer") != layer:
        raise ValueError(
            f"{path} is layer {detail.get('layer')}, this run is layer {layer}; the "
            "ceiling is per layer and a mismatch shifts the floor silently"
        )
    variance = detail.get("variance_explained")
    if variance is None:
        raise ValueError(f"{path} carries no variance_explained to use as a ceiling")
    return float(variance)


def loaded_model_name(quality: ReconstructionQuality) -> str:
    """The target's identity, as recorded by the measurement itself.

    `trained_on` vs `evaluated_on` IS the pre-gate's whole subject, so a record
    showing only one of the pair cannot answer the question it was written to
    answer. This used to return a standing placeholder string; the loader has
    landed, so the name now comes from the model that was actually run, and the
    placeholder survives only for a `ReconstructionQuality` built by hand
    without one — where it says so rather than naming a model.
    """
    if quality.evaluated_on:
        return quality.evaluated_on
    return "not recorded — ReconstructionQuality built without a target model"


def unmeasured_reading(reason: str) -> Reading:
    return Reading(
        instrument="sae_reconstruction",
        kind=KIND,
        value=float("nan"),
        operating_point="SAE reconstruction pre-gate not run — see detail.reason",
        licensed=None,
        detail={"reason": reason},
    )


def reading(
    quality: ReconstructionQuality,
    config: SAEConfig,
    ceiling: float | None = None,
) -> Reading:
    """The pre-gate's verdict. `licensed=True` means I4 MAY be read on this model.

    **Two arms, two questions — and conflating them inverted the instrument.**

    * **Ceiling arm** (`ceiling is None`): the dictionary against the model it
      was FITTED on. The question is whether our loader works, so the bar is
      that it reconstructs *at all* — positive variance explained, above its own
      matched random control, with the downstream KL term clearing its bar.
      There is nothing above this arm to compare it to; it IS the comparison.
    * **Target arm** (`ceiling` supplied): the same dictionary on a model it was
      not fitted on. The question is transfer, so the bar is a FRACTION of what
      the ceiling arm achieved.

    An absolute `min_variance_explained` was applied to both until 2026-08-07,
    and it failed the ceiling arm: the guessed 0.75 sat above the measured
    ceiling of 0.698-0.723, so the run whose job is to SET the bar was being
    judged against a guess at it. The KL term keeps an absolute bar because it
    is already relative by construction — a fraction of the layer's own
    downstream contribution — while variance explained is not.

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
    # A missing control is never a passed one: without it "the dictionary
    # reconstructs" is a statement about linear algebra, not about training.
    beats_control = margin is not None and margin > 0.0
    if ceiling is None:
        floor = 0.0
        passes = (
            recovered >= config.min_kl_recovered
            and quality.variance_explained > floor
            and beats_control
        )
    else:
        floor = ceiling * config.min_transfer_ratio
        passes = (
            recovered >= config.min_kl_recovered
            and quality.variance_explained >= floor
            and beats_control
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
            + (
                "> 0 (CEILING arm — the model the dictionary was fitted on, judged on "
                "reconstructing at all rather than against a bar it is meant to set)"
                if ceiling is None
                else f">= {floor:.4f} = {config.min_transfer_ratio} of the "
                     f"{ceiling:.4f} ceiling (TARGET arm)"
            )
            + " and above a matched-shape random dictionary"
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
            # The loader's scale assumption, measured against what the
            # checkpoint declared. `norm_ratio` far from 1.0 means every figure
            # above describes our scaling rather than the dictionary's transfer,
            # so these three are read BEFORE the verdict, not after it.
            "observed_activation_norm": quality.observed_activation_norm,
            "declared_input_norm": quality.declared_input_norm,
            "norm_ratio": quality.norm_ratio,
            # Which feature-selection rule produced this reading. Two readings
            # from one run differ only here.
            # Which arm this is, and what it was judged against. A reading that
            # cannot say whether it set the bar or was measured against one is
            # not interpretable later.
            "arm": "ceiling" if ceiling is None else "target",
            "ceiling_variance_explained": ceiling,
            "variance_floor_applied": floor,
            # ⚠️ The whole reason this gate exists. Llama Scope is trained on
            # Llama-3.1-8B-Base and our target is Instruct. The checkpoint's own
            # claim wins over the config string when it has one.
            "trained_on": quality.trained_on or config.trained_on,
            "evaluated_on": loaded_model_name(quality),
        },
    )
