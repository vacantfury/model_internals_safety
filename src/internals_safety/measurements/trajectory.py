"""I2 — processing-trajectory features.

**Question:** does the *evolution across layers* carry evidence that no single
best cell carries? Distinct from I1 (token identity) and from measurements #2/#3
(one cell, chosen by argmax over the grid).

**Provenance, stated plainly because the rule requires it.** The instrument is
justified by OUR OWN data and costs nothing: the per-layer curves already exist
on every run, and the layer-index diagnostic is validated on both models
independently (`instrument_layer.md` §3.1 — late-mid + `instruction_final` means
content recovered by computation; early + `last` means content present at the
surface). The literature raised its *rank*, not its *justification*:

- arXiv 2605.00269 **(C, unrefereed, 0 citations)** reports that under
  length-matched evaluation attention-derived scores collapse to chance while
  trajectory features retain signal (0.721 avg AUROC, 0.850 jailbreak), and
  names a *vocabulary-transparency spectrum* — which is our own lexical-
  transparency reclassification of `reverse_words` under a different name.
- arXiv 2605.02958 **(C)**, arXiv 2607.18348 **(C)** — refusal is constructed
  across layers; residual-stream geometry across depth as a vocabulary.

Under the house rule a (C) may motivate but not drive, so **no claim from this
module may rest on those papers**; the promotion is conditional on a replication
we run on our own cached activations. That replication is the validation gate,
not this file.

**What the features are, and why these.** Everything here is computed from the
residual stream at ONE position across layers, so nothing depends on sequence
length directly — but the length null still applies, because content correlates
with length and these features read content. Three families:

- *norms* — how large the representation is at each layer.
- *step norms* — how far it moves between consecutive layers. This is the
  "relative displacement" the geometry paper names.
- *step cosines* — whether consecutive moves agree in direction. A trajectory
  that turns sharply mid-stack is doing something different from one that
  travels straight, and neither shows up in any single cell.

Deliberately NOT included: anything requiring a fitted direction. Those belong
with the probe layer, are subject to its licensing, and would smuggle a
correlational selection into a module whose point is to be label-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from internals_safety.models.capture import ActivationBatch

# Every feature block this module can emit, in a fixed order so that a feature
# matrix's columns mean the same thing across runs and models.
FEATURE_BLOCKS = ("norms", "step_norms", "step_cosines")


@dataclass(frozen=True)
class Trajectory:
    """One prompt-set's per-layer geometry at a single position.

    Each tensor is [n_prompts, n_layers - k] for the appropriate k, so a caller
    can always recover which layer transition a column refers to.
    """

    position: str
    layers: list[int]
    norms: torch.Tensor
    step_norms: torch.Tensor
    step_cosines: torch.Tensor

    @property
    def n_prompts(self) -> int:
        return int(self.norms.shape[0])

    def peak_step_layer(self) -> torch.Tensor:
        """Index into `layers` of each prompt's largest single move.

        The trajectory analogue of the layer-index diagnostic: where the
        representation changes most is where the work is being done.
        """
        return self.step_norms.argmax(dim=1)


def _at_position(batch: ActivationBatch, position: str) -> torch.Tensor:
    """[n_prompts, n_layers, d_model] for one position, in float32."""
    if position not in batch.positions:
        raise ValueError(f"position {position!r} not captured; have {batch.positions}")
    index = batch.positions.index(position)
    return batch.tensor[:, :, index, :].to(torch.float32)


def trajectory(batch: ActivationBatch, position: str) -> Trajectory:
    """Per-layer geometry of the residual stream at one position."""
    if len(batch.layers) < 3:
        # step_cosines needs two consecutive steps, so three layers is the
        # minimum at which every feature block is defined. Failing here beats
        # returning an empty tensor a caller would silently treat as a feature.
        raise ValueError(
            f"trajectory features need at least 3 captured layers, got {len(batch.layers)}"
        )

    states = _at_position(batch, position)
    steps = states[:, 1:, :] - states[:, :-1, :]
    step_norms = steps.norm(dim=-1)

    # Cosine between consecutive steps. Guarded against a zero step, which would
    # otherwise produce NaN and propagate into every downstream statistic; a
    # zero step means the representation did not move, and 0.0 (orthogonal) is
    # the honest encoding of "no direction to agree with".
    first, second = steps[:, :-1, :], steps[:, 1:, :]
    denominator = first.norm(dim=-1) * second.norm(dim=-1)
    dot = (first * second).sum(dim=-1)
    step_cosines = torch.where(denominator > 0, dot / denominator.clamp(min=1e-12), torch.zeros_like(dot))

    return Trajectory(
        position=position,
        layers=list(batch.layers),
        norms=states.norm(dim=-1),
        step_norms=step_norms,
        step_cosines=step_cosines,
    )


def feature_matrix(traj: Trajectory, blocks: tuple[str, ...] = FEATURE_BLOCKS) -> torch.Tensor:
    """[n_prompts, n_features] — the probe-ready view.

    Concatenated in `FEATURE_BLOCKS` order so a column index means the same
    thing across runs. `blocks` narrows the set, which is what an ablation over
    feature families needs: if the whole signal is carried by `norms`, that is a
    much weaker result than one carried by `step_cosines`, and the only way to
    tell is to fit each block alone.
    """
    unknown = [name for name in blocks if name not in FEATURE_BLOCKS]
    if unknown:
        raise ValueError(f"unknown feature blocks {unknown}; have {list(FEATURE_BLOCKS)}")
    if not blocks:
        raise ValueError("no feature blocks selected")
    return torch.cat([getattr(traj, name) for name in blocks], dim=1)


def feature_names(traj: Trajectory, blocks: tuple[str, ...] = FEATURE_BLOCKS) -> list[str]:
    """Column labels for `feature_matrix`, naming the layer each one came from.

    Without these a trajectory probe's coefficients are uninterpretable, and the
    whole point of §3.1's layer-index diagnostic is that WHICH layer carries the
    signal is the validity check.
    """
    names: list[str] = []
    for name in blocks:
        if name == "norms":
            names += [f"norm@L{layer}" for layer in traj.layers]
        elif name == "step_norms":
            names += [f"step@L{a}->L{b}" for a, b in zip(traj.layers, traj.layers[1:])]
        elif name == "step_cosines":
            names += [
                f"turn@L{b}" for b in traj.layers[1:-1]
            ]
        else:
            raise ValueError(f"unknown feature block {name!r}")
    return names
