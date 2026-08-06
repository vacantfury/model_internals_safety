"""I1 — the Patchscopes decode measurement.

**The question, which is not the question measurements #2 and #3 ask.** Those ask
whether harmful content is linearly *separable* in the residual stream. This asks
whether the **plaintext tokens themselves** become predictable from an
intermediate state captured on the *encoded* prompt — "did the model decode it",
not "is harm decodable by us".

Why that difference is the whole point. The deployment probe was measured reading
raw character length at AUROC 0.654 on 12 of 15 rungs
(`text_docs/as5/pilot_rebaseline.md` §5), because length is content-correlated and
survives every encoder. A probe can exploit any separating feature. This
instrument cannot exploit length: its readout is probability mass on *specific
plaintext tokens*, and a longer prompt does not make the word "ransomware" more
likely unless the state actually encodes it.

**Method provenance.** Ghandeharioun et al., *Patchscopes*, **ICML 2024** (arXiv
2401.06102) — established literature, imported. The target prompt in
`DecodeLensConfig` is theirs verbatim. The specific application to ciphertext is
Fang & Marks (arXiv 2512.01222), which is tier **(C)**, 4 citations, unrefereed:
it motivates this instrument but does not license it, so the validation gate
below is load-bearing rather than ceremonial and the paper must say we are
replicating an unrefereed result on our own rungs.

**The built-in negative control is what makes this better than a probe.** Every
reading is scored twice — once against its own plaintext, once against a
*mismatched* plaintext from another prompt in the same condition. The mismatched
score is what surface features, prompt length and scaffold artifacts produce on
their own, so the margin between them is the decoding signal. A probe has no
equivalent: its null is a label shuffle, which tests significance and not
sufficiency (`instrument_layer.md` §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from internals_safety.config import DecodeLensConfig, Site
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.capture import ActivationBatch
from internals_safety.models.loader import LoadedModel
from internals_safety.models.patching import patch_residual


# P1 — the question this instrument answers and no other on the roster does.
# Module-level so `assert_distinct_questions` can check the roster rather than
# trusting three docstrings to stay distinct.
QUESTION = "did the model recover the plaintext tokens from the encoded prompt, in situ"
KIND: Kind = "correlational"
OPERATING_POINT = (
    "mean probability mass on the prompt's own plaintext content tokens, minus "
    "the same mass on a same-condition mismatched plaintext (rotated pairing)"
)


@dataclass(frozen=True)
class LensReading:
    """One (prompt, layer) cell of the decode measurement."""

    layer: int
    # Probability mass the patched pass puts on this prompt's own plaintext
    # content tokens.
    matched: float
    # The same quantity against a mismatched plaintext — the negative control.
    mismatched: float
    # True when this prompt's plaintext yielded NO content tokens, so both scores
    # are 0.0 by construction. That is unmeasurable, not un-decoded, and without
    # the flag it reads as a confident zero margin — the same silent-False shape
    # as the deployment defect, one instrument over.
    vacuous: bool = False

    @property
    def margin(self) -> float:
        """How far the real plaintext beats the control. The reported quantity."""
        return self.matched - self.mismatched

    def decodes(self, min_margin: float) -> bool:
        return self.margin >= min_margin


@dataclass(frozen=True)
class LensCurve:
    """The per-layer curve for one prompt — never collapsed to a scalar here.

    §3.1 wants the curve because decoding is expected to peak in intermediate-late
    layers; a single number would hide exactly the structure that distinguishes
    computed recovery from lexical presence (`instrument_layer.md` §3.1).
    """

    readings: list[LensReading]

    @property
    def best(self) -> LensReading:
        return max(self.readings, key=lambda reading: reading.margin)

    @property
    def peak_layer(self) -> int:
        return self.best.layer


def content_token_ids(
    loaded: LoadedModel, text: str, stopwords: frozenset[str] | None = None
) -> set[int]:
    """Token ids of `text`'s content words.

    Content words rather than all tokens because the scaffold's own continuation
    is full of function words and punctuation: scoring those would credit the
    control and the real plaintext equally, which is precisely the confound this
    measurement exists to avoid.
    """
    words = [word.strip(".,!?;:'\"()[]") for word in text.split()]
    keep = [word for word in words if word and (stopwords is None or word.lower() not in stopwords)]
    ids: set[int] = set()
    for word in keep:
        # Both spacings: BPE tokenisers give " word" and "word" different ids and
        # which one appears depends on what precedes it in the scaffold.
        for variant in (word, f" {word}"):
            ids.update(loaded.tokenizer.encode(variant, add_special_tokens=False))
    return ids


def derange(items: Sequence) -> list:
    """Rotate by one, so no element keeps its own index.

    This is the negative control's pairing: every prompt is scored against a
    plaintext that is not its own, drawn from the SAME condition so the encoding,
    the attack template and the length distribution all stay fixed and only the
    pairing moves. A rotation rather than a random permutation because a random
    one can fix a point, and a fixed point silently scores a row against itself —
    which would inflate the control and shrink the margin toward zero, i.e. fail
    in the direction that hides a real result.
    """
    if len(items) < 2:
        raise ValueError("a derangement needs at least two items")
    return list(items[1:]) + list(items[:1])


def _mass_on(probabilities: torch.Tensor, token_ids: Sequence[int]) -> float:
    """Total probability the distribution puts on a token set."""
    if not token_ids:
        return 0.0
    index = torch.tensor(sorted(token_ids), device=probabilities.device)
    return float(probabilities.index_select(0, index).sum())


def read_layer(
    loaded: LoadedModel,
    activations: ActivationBatch,
    layer: int,
    position: str,
    plaintexts: Sequence[str],
    config: DecodeLensConfig,
    site: Site = "resid_pre",
) -> list[LensReading]:
    """Patch every prompt's state at (`layer`, `position`) and score the readout.

    One target forward pass per batch, not per prompt — the target prompt is
    identical across rows, so the batch is the same scaffold repeated with a
    different patched vector in each row.
    """
    if len(plaintexts) < 2:
        raise ValueError("the mismatched control needs at least two prompts to permute")

    vectors = activations.select(layer, position)
    if vectors.shape[0] != len(plaintexts):
        raise ValueError(
            f"{vectors.shape[0]} activations but {len(plaintexts)} plaintexts"
        )

    matched_ids = [content_token_ids(loaded, text) for text in plaintexts]
    control_ids = derange(matched_ids)

    encoded = loaded.tokenizer(
        [config.target_prompt] * min(config.batch_size, vectors.shape[0]),
        return_tensors="pt",
        add_special_tokens=False,
    )

    readings: list[LensReading] = []
    for start in range(0, vectors.shape[0], config.batch_size):
        chunk = vectors[start : start + config.batch_size]
        inputs = {
            "input_ids": encoded["input_ids"][: chunk.shape[0]].to(loaded.device),
            "attention_mask": encoded["attention_mask"][: chunk.shape[0]].to(loaded.device),
        }
        with torch.inference_mode(), patch_residual(
            loaded, layer, config.target_position, chunk, site=site
        ):
            logits = loaded.model(**inputs, use_cache=False).logits

        probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
        for row in range(chunk.shape[0]):
            index = start + row
            readings.append(
                LensReading(
                    layer=layer,
                    matched=_mass_on(probabilities[row], sorted(matched_ids[index])),
                    mismatched=_mass_on(probabilities[row], sorted(control_ids[index])),
                    # Both scores are 0.0 with nothing to score against, which is
                    # unmeasurable rather than a measured absence of decoding.
                    vacuous=not matched_ids[index] or not control_ids[index],
                )
            )
    return readings


def sweep_layers(
    loaded: LoadedModel,
    activations: ActivationBatch,
    position: str,
    plaintexts: Sequence[str],
    config: DecodeLensConfig,
    site: Site = "resid_pre",
) -> list[LensCurve]:
    """Per-prompt curves across every configured layer."""
    layers = config.layers if config.layers is not None else activations.layers
    unknown = [layer for layer in layers if layer not in activations.layers]
    if unknown:
        raise ValueError(f"layers {unknown} were not captured; captured: {activations.layers}")

    by_layer = {
        layer: read_layer(loaded, activations, layer, position, plaintexts, config, site=site)
        for layer in layers
    }
    return [
        LensCurve(readings=[by_layer[layer][index] for layer in layers])
        for index in range(len(plaintexts))
    ]


def decode_rate(curves: Sequence[LensCurve], min_margin: float) -> float:
    """Fraction of prompts whose best layer clears the control margin."""
    if not curves:
        raise ValueError("no curves to summarise")
    return sum(curve.best.decodes(min_margin) for curve in curves) / len(curves)


def reading(
    curves: Sequence[LensCurve],
    config: DecodeLensConfig,
    *,
    layer: int | None = None,
    length_null_margin: float | None = None,
    selection_inside_null: bool = False,
    detail: dict | None = None,
) -> Reading:
    """This instrument's condition-level verdict, with its evidence attached.

    P2 comes free here and that is the instrument's main advantage over a probe:
    the mismatched pairing is a built-in negative control, so `control_reading`
    is a measured quantity rather than something a caller must remember to run.

    P3 and P7 do NOT come free, so they are keyword arguments with fail-closed
    defaults. `length_null_margin=None` means the null was not computed and the
    reading is not reportable — an uncomputed control must never read as a
    cleared one. `selection_inside_null` defaults False because `layer=None`
    takes each prompt's argmax over the layer grid, which is an uncorrected
    multiple comparison until the null says otherwise.

    `licensed` is tri-state on a real unmeasurable case: if every prompt's
    plaintext yielded no content tokens, both scores are 0.0 by construction and
    a zero margin means "nothing to score", not "nothing was decoded".
    """
    if not curves:
        raise ValueError("no curves to summarise")

    cells = [
        curve.best if layer is None else _at_layer(curve, layer) for curve in curves
    ]
    vacuous = [cell for cell in cells if cell.vacuous]
    scored = [cell for cell in cells if not cell.vacuous]

    if not scored:
        licensed: bool | None = None
        matched = mismatched = 0.0
    else:
        matched = sum(cell.matched for cell in scored) / len(scored)
        mismatched = sum(cell.mismatched for cell in scored) / len(scored)
        licensed = (matched - mismatched) >= config.min_control_margin

    return Reading(
        instrument="decode_lens",
        kind=KIND,
        value=matched,
        operating_point=OPERATING_POINT,
        licensed=licensed,
        control_reading=mismatched,
        control_margin=config.min_control_margin,
        length_null_margin=length_null_margin,
        selection_inside_null=selection_inside_null,
        detail={
            "n_prompts": len(curves),
            "n_vacuous": len(vacuous),
            "layer": layer,
            "peak_layers": sorted({curve.peak_layer for curve in curves}) if layer is None else [layer],
            **(detail or {}),
        },
    )


def _at_layer(curve: LensCurve, layer: int) -> LensReading:
    for cell in curve.readings:
        if cell.layer == layer:
            return cell
    raise ValueError(f"layer {layer} not in curve; have {[c.layer for c in curve.readings]}")
