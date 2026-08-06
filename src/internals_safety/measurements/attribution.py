"""I6's attribution half — activation patching over the (layer, position) grid.

`interventions.py` + `causal.py` answer *does this direction cause refusal*;
this answers *which cells carry it*. Same toolkit, different question, and the
build plan lists them as one instrument because a causal claim needs both: a
direction with no location is not a mechanism, and a location with no
intervention is a correlation.

## Method provenance — established, and read before building

**Zhang & Nanda, *Towards Best Practices of Activation Patching in Language
Models*, ICLR 2023** (arXiv 2309.16042, 294c). Read at this build step under the
rule that a methods search precedes the build it governs; it was one of the seven
tier-(A) papers the arXiv-only sweep missed. Three of its findings changed what
this module does, and one of them changed it in our favour.

### 1. Corruption: symmetric token replacement, not Gaussian noise — satisfied by construction

Their headline recommendation is STR over Gaussian noising, because GN puts the
model off-distribution and "could even lead to unreliable or illusory results";
they show GN and STR localise *different* components on the same task.

**We never had the choice, and that is worth stating rather than claiming
credit:** our clean and corrupted runs are two real prompt sets — plain-harmful
and plain-harmless — so the corrupted run is in-distribution by construction. No
noise is ever added to an embedding here.

**But our corruption is STRONGER than their STR, and that has a consequence.**
They swap *key tokens* inside an otherwise identical sentence; we swap the whole
prompt. That is closer to the `p_ABC` corruption they analyse in §4.2, where all
task-relevant information is removed — and that regime is exactly where they show
the probability metric breaks. Which brings us to:

### 2. Metric: logit difference, normalised — and the paper's failure case IS our case

They recommend logit difference over probability, because "probability must fail
to detect negative model components, if corruption reduces the correct token
probability to near zero" — measured, under whole-prompt corruption P(answer)
fell to 5e-4 and probability detected neither negative head, while logit
difference still did.

Our corrupted run is a *benign* prompt, on which refusal mass is near the floor.
So the failure they demonstrate is not a hypothetical for us; it is the regime we
operate in. This module therefore reads

    LD = max logit(refusal openings) - max logit(compliance openings)

normalised as they specify, `(LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)`,
so 1.0 is fully-restored refusal and 0.0 is the corrupted run's own level.

**⚠️ This DIVERGES from `causal.py`, deliberately, and the two reasons do not
conflict.** There the metric is refusal *probability*, because that module's gate
takes a *fraction* of the refusal removed and a fraction of a log-odds is
undefined. Here nothing takes a fraction, and the paper's argument against
probability applies at full strength. Same repo, two metrics, each chosen by what
its own consumer needs — the numbers are not comparable across the two and no
report should place them side by side.

### 3. Single-site patching only — sliding windows inflate the peak

They measure sliding-window patching producing 1.40x-1.75x the peak of summed
single-layer effects, and attribute it to non-linear joint effects: a window can
suppress corrupted information flow, or jointly perform a computation no single
layer does. `measure_attribution` patches ONE cell at a time. A window is not
offered at all, because offering it would invite reporting the larger number.

### 4. The degree of freedom they flag that we must declare

Their Appendix F shows that *which* tokens are corrupted changes which components
are found — corrupting `S1, IO` finds the Name Movers that corrupting `S2` misses.
We do not corrupt tokens at all, but the analogous freedom is real and is the
POSITION grid: `instruction_final` and `last` are different questions (this
repo's own layer diagnostic separates them), so both are swept and reported
separately rather than pooled into one "best cell".

## The detection bar is derived, not chosen

A cell counts as carrying the effect when it exceeds `mean + k*SD` over the grid's
own effects — the paper's own rule ("2 standard deviation away from the mean").
Derived from the run's own distribution rather than set as an absolute effect
size, which is the fifth instance of the pattern this repo now treats as a rule
(`instrument_build_plan.md` §4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from internals_safety.config import MeasurementsConfig
from internals_safety.measurements.contract import Kind, Reading
from internals_safety.models.capture import ActivationBatch
from internals_safety.models.loader import LoadedModel, prepare_prompts
from internals_safety.models.patching import patch_residual
from internals_safety.pairing import derange

QUESTION = "which (layer, position) cells causally carry the refusal decision"
KIND: Kind = "causal"


def logit_difference(
    logits: torch.Tensor, answer_ids: Sequence[int], counter_ids: Sequence[int]
) -> torch.Tensor:
    """LD(r, r') per prompt — [n_prompts].

    Max over each token set rather than sum: the sets are alternative openings of
    the SAME answer ("I" / "As" both begin a refusal), so the model picking one
    is the answer being given once, not twice.
    """
    if not answer_ids or not counter_ids:
        raise ValueError("both an answer and a counterfactual token set are required")
    final = logits[:, -1, :]
    return final[:, list(answer_ids)].max(dim=-1).values - final[:, list(counter_ids)].max(dim=-1).values


def normalized_effect(patched: float, clean: float, corrupt: float) -> float | None:
    """The patching effect on the paper's scale: 1 restored, 0 corrupted.

    `None` when the clean and corrupted runs do not separate — if the contrast
    itself produced no gap there is nothing for a patch to restore, and dividing
    by that gap would report a large effect from a numerically tiny denominator.
    Fail-closed: an undefined effect is not a zero effect.
    """
    denominator = clean - corrupt
    if abs(denominator) < 1e-6:
        return None
    return float((patched - corrupt) / denominator)


@dataclass(frozen=True)
class PatchCell:
    """One (layer, position) cell's patching effect."""

    layer: int
    position: str
    effect: float | None
    # The same patch with the source rows deranged — see `measure_attribution`.
    control_effect: float | None = None

    @property
    def margin(self) -> float | None:
        """How far the real patch beats its own shuffled control."""
        if self.effect is None or self.control_effect is None:
            return None
        return abs(self.effect) - abs(self.control_effect)


@dataclass(frozen=True)
class AttributionMap:
    """Every cell's effect, plus the two baselines they are all measured against."""

    cells: tuple[PatchCell, ...]
    ld_clean: float
    ld_corrupt: float
    n_prompts: int
    detection_sd: float

    @property
    def measured(self) -> tuple[PatchCell, ...]:
        return tuple(cell for cell in self.cells if cell.effect is not None)

    @property
    def detection_threshold(self) -> float | None:
        """`mean + k*SD` over the grid's own effects (Zhang & Nanda's rule).

        `None` on fewer than two measured cells: a standard deviation over one
        number is not a spread, and a bar derived from it would be arithmetic
        dressed as a criterion.
        """
        values = [cell.effect for cell in self.measured]
        if len(values) < 2:
            return None
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(tensor.mean() + self.detection_sd * tensor.std(unbiased=True))

    @property
    def bar_is_reachable(self) -> bool:
        """Can ANY cell clear `mean + k*SD` on a grid this size?

        ⚠️ Not a formality. The largest z-score attainable in a sample of size n
        is `(n-1)/sqrt(n)` — a lone outlier inflates the very SD it is measured
        against. At the paper's k=2 that means **no grid smaller than 6 cells can
        detect anything**, whatever the effects are: n=4 caps at 1.5, n=5 at 1.79.

        Real grids here are 32 layers x 2 positions, so this never binds in
        production — but a bar that cannot be cleared returning "nothing
        detected" would be arithmetic reported as a measurement, which is the
        exact confusion the tri-state discipline exists to prevent. `reading`
        treats an unreachable bar as UNMEASURED.
        """
        n = len(self.measured)
        return n >= 2 and (n - 1) / math.sqrt(n) > self.detection_sd

    @property
    def detected(self) -> tuple[PatchCell, ...]:
        threshold = self.detection_threshold
        if threshold is None or not self.bar_is_reachable:
            return ()
        return tuple(cell for cell in self.measured if cell.effect > threshold)

    @property
    def peak(self) -> PatchCell | None:
        return max(self.measured, key=lambda cell: cell.effect, default=None)


def forward_passes(n_cells: int, with_control: bool = True) -> int:
    """Passes over the prompt sets this costs, for `--dry-run`.

    Two shared baselines (clean and corrupted) plus one per cell, doubled when
    the deranged control runs. Stated as a function for the same reason as
    everywhere else here: the approval gate prices the run from it, and the
    control must be inside the price — a control the estimate cannot see is a
    cost nobody approved.
    """
    return 2 + n_cells * (2 if with_control else 1)


def _run_logits(
    loaded: LoadedModel, prompts: Sequence[str], batch_size: int
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        encoded = loaded.tokenizer(
            list(prompts[start : start + batch_size]), return_tensors="pt", padding=True
        )
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            rows.append(loaded.model(**inputs).logits[:, -1:, :].float().cpu())
    if not rows:
        raise ValueError("no prompts supplied")
    return torch.cat(rows, dim=0)


def _patched_logits(
    loaded: LoadedModel,
    prompts: Sequence[str],
    offsets: Sequence[int],
    layer: int,
    vectors: torch.Tensor,
    site: str,
    batch_size: int,
) -> torch.Tensor:
    """Corrupted run with one cell overwritten from the clean cache."""
    rows: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        encoded = loaded.tokenizer(
            list(prompts[start:stop]), return_tensors="pt", padding=True
        )
        inputs = {key: value.to(loaded.device) for key, value in encoded.items()}
        with patch_residual(
            loaded, layer, list(offsets[start:stop]), vectors[start:stop], site=site
        ):
            with torch.inference_mode():
                rows.append(loaded.model(**inputs).logits[:, -1:, :].float().cpu())
    return torch.cat(rows, dim=0)


def measure_attribution(
    loaded: LoadedModel,
    clean_source: ActivationBatch,
    corrupt_prompts: Sequence[str],
    clean_prompts: Sequence[str],
    answer_ids: Sequence[int],
    counter_ids: Sequence[int],
    detection_sd: float,
    # plumbing(batch_size): throughput only — every read is per-prompt
    batch_size: int = 8,
    with_control: bool = True,
) -> AttributionMap:
    """Patch each cached (layer, position) into the corrupted run and score it.

    `clean_source` holds the CLEAN run's activations — captured already, so this
    costs no extra clean forward passes. `corrupt_prompts` is the run those
    states are written into, row-aligned with the cache.

    **The negative control patches the same cell with the source rows
    DERANGED.** It writes a real, in-distribution activation of the right shape
    and norm carrying another prompt's content, so anything it restores is the
    intervention's own artefact rather than transferred information. Same logic
    as the matched-norm random direction one module over, and it reuses the same
    derangement primitive — a rotation fixes no point, so no row keeps its own
    state.
    """
    if len(corrupt_prompts) != len(clean_prompts):
        raise ValueError(
            f"the contrast must be paired: {len(clean_prompts)} clean prompts against "
            f"{len(corrupt_prompts)} corrupted ones"
        )
    n_source = clean_source.tensor.shape[0]
    if n_source != len(corrupt_prompts):
        raise ValueError(
            f"cached clean states cover {n_source} prompts but the corrupted run has "
            f"{len(corrupt_prompts)}; patching writes row-for-row"
        )

    # ⚠️ RENDER BOTH SETS THROUGH THE CHAT TEMPLATE, and this is not a detail.
    # The cached states were captured by `capture_or_load`, which renders; the
    # position offsets are counted back from the end of the RENDERED sequence.
    # Tokenising the bare instruction instead would patch a state captured at
    # "final token of the user message" into whatever token happens to sit at
    # that offset in an untemplated string — and would measure refusal on an
    # input the instruction-tuned model never sees, since refusal is largely a
    # chat-format behaviour. `models/capture.py` owns this convention; every
    # write-side instrument has to follow it or its intervention lands somewhere
    # else than its diagnosis read.
    prepared = prepare_prompts(loaded, corrupt_prompts, positions=clean_source.positions)
    corrupt_rendered = [prompt.text for prompt in prepared]
    clean_rendered = [
        prompt.text
        for prompt in prepare_prompts(loaded, clean_prompts, positions=clean_source.positions)
    ]
    ld_clean = float(
        logit_difference(
            _run_logits(loaded, clean_rendered, batch_size), answer_ids, counter_ids
        ).mean()
    )
    ld_corrupt = float(
        logit_difference(
            _run_logits(loaded, corrupt_rendered, batch_size), answer_ids, counter_ids
        ).mean()
    )

    order = derange(list(range(n_source)))
    cells: list[PatchCell] = []
    for layer in clean_source.layers:
        for position in clean_source.positions:
            vectors = clean_source.select(layer, position)
            offsets = [prompt.positions[position] for prompt in prepared]

            def score(source: torch.Tensor) -> float:
                return float(
                    logit_difference(
                        _patched_logits(
                            loaded, corrupt_rendered, offsets, layer, source,
                            clean_source.site, batch_size,
                        ),
                        answer_ids,
                        counter_ids,
                    ).mean()
                )

            effect = normalized_effect(score(vectors), ld_clean, ld_corrupt)
            control = (
                normalized_effect(score(vectors[order]), ld_clean, ld_corrupt)
                if with_control
                else None
            )
            cells.append(PatchCell(layer, position, effect, control))

    return AttributionMap(
        cells=tuple(cells),
        ld_clean=ld_clean,
        ld_corrupt=ld_corrupt,
        n_prompts=n_source,
        detection_sd=detection_sd,
    )


def unmeasured_reading(config: MeasurementsConfig, reason: str) -> Reading:
    """No cell was measurable — the contrast produced no gap to restore."""
    return Reading(
        instrument="attribution",
        kind=KIND,
        value=float("nan"),
        operating_point="patching attribution not run — see detail.reason",
        licensed=None,
        detail={"reason": reason, "n_cells": 0, "n_measured": 0},
    )


def reading(attribution: AttributionMap, config: MeasurementsConfig) -> Reading:
    """The peak cell's effect, licensed on its own control and the grid's spread.

    Licensed requires three things, and each defeats a different way of being
    wrong: a defined effect (the contrast separated at all), a peak that clears
    the derangement control (the patch transferred information rather than
    perturbing anything), and a peak that clears the derived detection bar (the
    cell stands out from the grid rather than the whole grid being elevated).
    """
    peak = attribution.peak
    if peak is None:
        return unmeasured_reading(config, "no cell produced a defined effect")
    if not attribution.bar_is_reachable:
        # The bar is arithmetically unreachable on a grid this small, so
        # "nothing detected" would be a property of n, not of the model.
        return unmeasured_reading(
            config,
            f"{len(attribution.measured)} measured cells cannot reach a "
            f"{attribution.detection_sd}-SD bar (max attainable z is "
            f"(n-1)/sqrt(n)); widen the grid or lower the bar deliberately",
        )

    threshold = attribution.detection_threshold
    margin = peak.margin
    clears_control = margin is not None and margin > 0.0
    clears_bar = threshold is not None and peak.effect > threshold

    return Reading(
        instrument="attribution",
        kind=KIND,
        value=float(peak.effect),
        operating_point=(
            "normalised logit difference at the strongest single (layer, position) "
            "cell — LD = max logit(refusal openings) - max logit(compliance "
            "openings), scaled so 1.0 is the clean run and 0.0 the corrupted one; "
            f"a cell counts as detected above mean + {attribution.detection_sd} SD "
            "of the grid's own effects (Zhang & Nanda, ICLR 2023). Single-cell "
            "patching, never a sliding window, which inflates the peak 1.4-1.75x"
        ),
        licensed=bool(clears_control and clears_bar),
        control_reading=peak.control_effect,
        control_margin=margin,
        # The peak is a MAXIMUM over the grid, so it is a selection. The
        # mean+k*SD bar corrects within the grid, but it is not a null and
        # claiming otherwise here would be the exact overclaim P7 exists to
        # stop. A permutation null over the grid is the honest upgrade and is
        # filed rather than faked.
        selection_inside_null=False,
        detail={
            "peak_layer": peak.layer,
            "peak_position": peak.position,
            "detection_threshold": threshold,
            "n_detected": len(attribution.detected),
            "n_measured": len(attribution.measured),
            "n_cells": len(attribution.cells),
            "ld_clean": attribution.ld_clean,
            "ld_corrupt": attribution.ld_corrupt,
            "n_prompts": attribution.n_prompts,
            # Recorded per the method paper: these two choices change which cells
            # are found, so a map without them is not reproducible.
            "metric": "normalized_logit_difference",
            "corruption": "whole_prompt_contrast",
            "patch_extent": "single_cell",
        },
    )
