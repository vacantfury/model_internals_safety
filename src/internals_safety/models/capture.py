"""Residual-stream capture — the measurement substrate.

Everything downstream reads activations through this module: the direction
work (probes/), the in-situ content probe (measurement #2), the recognition
readout (measurement #3), and later the interventions, which reuse the same
hook points to *write* instead of read.

Sites, in HF terms. Hooking a decoder layer's **input** gives `resid_pre` for
that layer; hooking its **output** gives `resid_post`. With
`output_hidden_states=True`, `hidden_states[i] == resid_pre(layer i)` and
`hidden_states[i + 1] == resid_post(layer i)`. The hook path is the one used
here — not `output_hidden_states` — because interventions need module hooks
anyway, and one code path for read and write means the intervention writes at
exactly the site the diagnosis read at. `tests/test_capture.py` pins the
equivalence against `output_hidden_states` so the hook points cannot drift.

Note on `resid_post` at the final layer: it is taken *before* the model's final
norm, so it is not the vector the unembedding sees. That is deliberate —
consistency across layers matters more here than comparability to logits.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch

from internals_safety.config import PositionName, Site
from internals_safety.models.loader import LoadedModel, PreparedPrompt, tokenize_batch


@dataclass
class ActivationBatch:
    """Captured activations plus everything needed to interpret them.

    `tensor` is [n_prompts, n_layers, n_positions, d_model], float32 on CPU.
    Only the requested positions are kept — never the full sequence — which is
    what keeps a full-ladder sweep's memory bounded.
    """

    tensor: torch.Tensor
    layers: list[int]
    positions: list[str]
    site: str
    model_name: str
    user_messages: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = (len(self.user_messages), len(self.layers), len(self.positions))
        if self.user_messages and tuple(self.tensor.shape[:3]) != expected:
            raise ValueError(f"tensor shape {tuple(self.tensor.shape)} does not match {expected}")

    @property
    def d_model(self) -> int:
        return int(self.tensor.shape[-1])

    def select(self, layer: int, position: PositionName | str) -> torch.Tensor:
        """Return [n_prompts, d_model] for one (layer, position) cell."""
        return self.tensor[:, self.layers.index(layer), self.positions.index(position), :]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tensor": self.tensor,
                "layers": self.layers,
                "positions": self.positions,
                "site": self.site,
                "model_name": self.model_name,
                "user_messages": self.user_messages,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ActivationBatch":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**payload)


def resolve_layers(loaded: LoadedModel, layers: Sequence[int] | str | None) -> list[int]:
    """Normalise a layer selection to a validated list of indices."""
    selection = loaded.config.capture.layers if layers is None else layers
    if selection == "all":
        return list(range(loaded.n_layers))
    resolved = [int(index) for index in selection]
    out_of_range = [index for index in resolved if not 0 <= index < loaded.n_layers]
    if out_of_range:
        raise ValueError(f"layer indices {out_of_range} outside [0, {loaded.n_layers - 1}]")
    return resolved


def _hidden_from_args(args: tuple, kwargs: dict) -> torch.Tensor:
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise RuntimeError("decoder layer received no hidden states positionally or as a kwarg")


@contextmanager
def residual_hooks(
    loaded: LoadedModel,
    layers: Sequence[int],
    site: Site,
    sink: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Capture the residual stream at `layers` into `sink` for one forward pass."""
    handles = []

    def make_pre_hook(index: int):
        def hook(_module, args, kwargs):
            sink[index] = _hidden_from_args(args, kwargs)
            return None

        return hook

    def make_post_hook(index: int):
        def hook(_module, _args, output):
            sink[index] = output[0] if isinstance(output, tuple) else output
            return None

        return hook

    try:
        for index in layers:
            module = loaded.layers[index]
            if site == "resid_pre":
                handles.append(module.register_forward_pre_hook(make_pre_hook(index), with_kwargs=True))
            elif site == "resid_post":
                handles.append(module.register_forward_hook(make_post_hook(index)))
            else:
                raise ValueError(f"unknown site: {site!r}")
        yield
    finally:
        for handle in handles:
            handle.remove()


def _gather_positions(
    hidden: torch.Tensor, prompts: Sequence[PreparedPrompt], positions: Sequence[str]
) -> torch.Tensor:
    """[B, T, D] -> [B, P, D], reading each prompt's own negative offsets.

    Valid because tokenisation is left-padded (loader.tokenize_batch): a
    negative index counts back from the true last token for every row.
    """
    seq_len = hidden.shape[1]
    index = torch.tensor(
        [[seq_len + prompt.positions[name] for name in positions] for prompt in prompts],
        device=hidden.device,
    )
    if (index < 0).any():
        raise ValueError("a resolved position falls before the start of its padded sequence")
    rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
    return hidden[rows, index]


def capture_activations(
    loaded: LoadedModel,
    prompts: Sequence[PreparedPrompt],
    layers: Sequence[int] | str | None = None,
    positions: Sequence[PositionName] | None = None,
    site: Site | None = None,
    batch_size: int | None = None,
) -> ActivationBatch:
    """Run `prompts` through the model and keep the residual stream at
    (layers x positions).

    No generation happens here — this is the single forward pass over the
    attack prompt, which is the condition every regime in the paper is defined
    over.
    """
    layer_indices = resolve_layers(loaded, layers)
    position_names = list(positions) if positions is not None else list(loaded.config.capture.positions)
    capture_site: Site = site if site is not None else loaded.config.capture.site
    size = batch_size if batch_size is not None else loaded.config.capture_batch_size

    missing = {
        name for prompt in prompts for name in position_names if name not in prompt.positions
    }
    if missing:
        raise ValueError(f"prompts were not prepared with positions {sorted(missing)}")

    chunks: list[torch.Tensor] = []
    for start in range(0, len(prompts), size):
        batch = list(prompts[start : start + size])
        inputs = tokenize_batch(loaded, batch)
        sink: dict[int, torch.Tensor] = {}
        with torch.inference_mode(), residual_hooks(loaded, layer_indices, capture_site, sink):
            loaded.model(**inputs, use_cache=False)
        per_layer = [
            _gather_positions(sink[index], batch, position_names).to("cpu", torch.float32)
            for index in layer_indices
        ]
        chunks.append(torch.stack(per_layer, dim=1))

    tensor = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty(0, len(layer_indices), len(position_names), loaded.d_model)
    )
    return ActivationBatch(
        tensor=tensor,
        layers=layer_indices,
        positions=position_names,
        site=capture_site,
        model_name=loaded.config.name,
        user_messages=[prompt.user_message for prompt in prompts],
    )
