"""Read a guard's verdict from logits, in one forward pass, without generating.

Meta's own model card defines the Llama Guard classifier score as the probability
of the first token, and `safe`/`unsafe` are single tokens. That buys three things
generation does not:

* **It costs nothing.** No sampling loop, no judge call — the verdict is the
  guard's own logits, which is why AS-6's phase 1 is a $0 experiment.
* **It is continuous.** `P(unsafe)` is a graded degradation signal, so a guard
  that is *drifting* toward permitting something is visible before it flips.
* **It separates "the format broke" from "the verdict moved".** Under a mangled
  or off-distribution prompt a generated string may stop parsing entirely, which
  is indistinguishable from a `safe` verdict at the string level. Here the two
  are different observations: `top_token` says what the guard actually wanted to
  emit, and the label probabilities say what it thought.

That third property is why `top_token` is recorded on every reading rather than
only when something looks wrong. It is also the standing empirical check on
`verdict_prefix`: if a guard's argmax first token is a newline rather than a
label, the prefix is set wrong and every number in the run is read one position
too early (TODO item 13). The check rides along at zero cost instead of needing
its own experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from internals_safety.config import GuardConfig
from internals_safety.guards.prompts import (
    VerdictTokens,
    render_guard_prompt,
    resolve_verdict_tokens,
    verdict_context,
)
from internals_safety.models.loader import LoadedModel, PreparedPrompt, tokenize_batch

# How many prompts past the first to re-resolve the verdict ids on. The ids must
# be identical across a run — comparing P(unsafe) between prompts is only a
# comparison if it is the same token — but re-resolving every prompt would
# tokenise each long ciphertext context four extra times for a check that has
# never varied within a rung. A spread sample catches a real drift.
# config: measurements.guard_verdict.id_stability_samples
_ID_STABILITY_SAMPLES = 4


@dataclass(frozen=True)
class VerdictReading:
    """One guard verdict, as probabilities rather than a parsed string."""

    p_unsafe: float
    p_safe: float
    # What the guard would actually have emitted first. Recorded always: it is
    # the evidence that the label probabilities were read at the right position.
    top_token: str
    top_prob: float

    @property
    def unsafe(self) -> bool:
        """The binary verdict, by the guard's own preference between the labels.

        Compared against each other rather than against an absolute cut: the two
        labels do not carry the whole probability mass (formatting tokens take
        some), so a fixed threshold would drift with how confident the guard is
        about *format* rather than about safety.
        """
        return self.p_unsafe > self.p_safe

    @property
    def label_mass(self) -> float:
        """How much probability the two labels hold between them.

        A diagnostic, not a verdict: near 1.0 means the guard is answering in the
        expected format at the expected position. A small value means it wanted
        to emit something else first — usually that `verdict_prefix` is wrong.
        """
        return self.p_unsafe + self.p_safe


def verdict_probability(logits: torch.Tensor, tokens: VerdictTokens) -> float:
    """Mean P(unsafe), RENORMALISED over the two labels, from final-position logits.

    The logits-shaped sibling of `VerdictReading.unsafe`, and it lives HERE rather
    than in the causal layer because what a verdict means is this module's
    business. The causal path needs a scalar computed from the same logits it
    hands to `kl_divergence`, which `read_verdicts` cannot supply — it returns
    per-prompt readings, not the tensor. Two call shapes, one definition; their
    agreement is pinned by test rather than assumed.

    **Renormalised, not raw mass.** The template has already committed the guard
    to answering, so the quantity of interest is P(unsafe | it answers). Raw
    softmax mass would let an intervention that merely diffuses probability
    across unrelated tokens read as "less unsafe" while the guard's preference
    between its own two labels never moved — and diffusing the distribution is
    exactly what an over-strong ablation does. The denominator removes that.

    **Consequence, stated because it is load-bearing:** this score cannot see
    distributional damage, so in the causal criteria the KL-on-benign term is the
    only thing separating a real direction from one that simply breaks the model.
    Do not drop the KL gate for guards.
    """
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    unsafe = probabilities[:, tokens.unsafe_id]
    safe = probabilities[:, tokens.safe_id]
    total = unsafe + safe
    # Fail loud rather than dividing by ~0. An empty denominator means the logits
    # were read where the guard was not about to answer, which is the defect that
    # once reported a guard blocking 98% as blocking 0.00 — a bug, not a verdict.
    if float(total.min()) <= 0.0:
        raise ValueError(
            "no probability mass on either label — logits read at the wrong position; "
            "check verdict_prefix and see verdict_format_health"
        )
    return float((unsafe / total).mean())


def label_mass_from_logits(logits: torch.Tensor, tokens: VerdictTokens) -> float:
    """Mean total mass on the two labels — the read-position health check.

    `VerdictReading.label_mass` one call shape over, for the same reason as
    above. Near 1.0 means the read is at the right place; near 0 means every
    verdict number from that pass is noise.
    """
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    return float((probabilities[:, tokens.unsafe_id] + probabilities[:, tokens.safe_id]).mean())


@torch.inference_mode()
def read_verdicts(
    loaded: LoadedModel,
    payloads: Sequence[str],
    batch_size: int | None = None,
) -> tuple[list[VerdictReading], VerdictTokens]:
    """Read one verdict per payload, batched, with no generation.

    Returns the readings and the resolved token pair, so a run record can state
    which two ids the probabilities refer to instead of leaving it implicit.
    """
    config = loaded.config
    if not isinstance(config, GuardConfig):
        raise TypeError(
            f"read_verdicts needs a GuardConfig, got {type(config).__name__}; "
            "load the guard via config.load_guard_config"
        )
    if not payloads:
        raise ValueError("no payloads to read verdicts for")

    rendered = [render_guard_prompt(loaded.tokenizer, config, payload) for payload in payloads]
    contexts = [verdict_context(config, text) for text in rendered]

    tokens = resolve_verdict_tokens(loaded.tokenizer, config, rendered[0])
    _check_id_stability(loaded, config, rendered, tokens)

    size = batch_size or config.capture_batch_size
    readings: list[VerdictReading] = []
    for start in range(0, len(contexts), size):
        chunk = contexts[start : start + size]
        prompts = [
            PreparedPrompt(user_message="", text=text, positions={}) for text in chunk
        ]
        inputs = tokenize_batch(loaded, prompts)
        logits = loaded.model(**inputs).logits
        # Left padding (loader.tokenize_batch) makes -1 the last REAL token for
        # every row regardless of that row's pad count — the same invariant the
        # capture layer relies on.
        final = logits[:, -1, :].float()
        probabilities = torch.softmax(final, dim=-1)

        top_probs, top_ids = probabilities.max(dim=-1)
        for row in range(probabilities.shape[0]):
            readings.append(
                VerdictReading(
                    p_unsafe=float(probabilities[row, tokens.unsafe_id]),
                    p_safe=float(probabilities[row, tokens.safe_id]),
                    top_token=loaded.tokenizer.convert_ids_to_tokens(int(top_ids[row])),
                    top_prob=float(top_probs[row]),
                )
            )
    return readings, tokens


def _check_id_stability(
    loaded: LoadedModel,
    config: GuardConfig,
    rendered: Sequence[str],
    tokens: VerdictTokens,
) -> None:
    """Fail loud if the verdict ids depend on which prompt precedes them.

    They should not — the prefix ends at a stable boundary — but "should not" is
    exactly the assumption that produced the ability defect earlier in this
    project, and the check costs a handful of tokenizer calls per rung.
    """
    if len(rendered) <= 1:
        return
    step = max(1, len(rendered) // (_ID_STABILITY_SAMPLES + 1))
    for index in range(step, len(rendered), step):
        other = resolve_verdict_tokens(loaded.tokenizer, config, rendered[index])
        if (other.safe_id, other.unsafe_id) != (tokens.safe_id, tokens.unsafe_id):
            raise ValueError(
                f"guard {config.name!r}: verdict token ids are not stable across prompts "
                f"({tokens.safe_id}/{tokens.unsafe_id} at prompt 0 vs "
                f"{other.safe_id}/{other.unsafe_id} at prompt {index}). P(unsafe) would "
                "not be comparable between prompts."
            )


def verdict_format_health(readings: Sequence[VerdictReading]) -> dict:
    """Whether the verdict was read at the right position, as a reportable number.

    Summarised per rung rather than per prompt because that is the granularity at
    which `verdict_prefix` is right or wrong. A low `mean_label_mass` with a
    consistent `most_common_top_token` that is not a label is the signature of a
    misplaced prefix, and it is a run-invalidating condition rather than a
    finding — so it belongs in the results record, not in a log line.
    """
    if not readings:
        return {"n": 0}
    counts: dict[str, int] = {}
    for reading in readings:
        counts[reading.top_token] = counts.get(reading.top_token, 0) + 1
    top_token, top_count = max(counts.items(), key=lambda item: item[1])
    return {
        "n": len(readings),
        "mean_label_mass": sum(r.label_mass for r in readings) / len(readings),
        "most_common_top_token": top_token,
        "most_common_top_token_share": top_count / len(readings),
        "distinct_top_tokens": len(counts),
    }
