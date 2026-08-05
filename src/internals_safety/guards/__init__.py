"""Content guards as objects of study (AS-6).

AS-5 measures the transmit->decode->represent->convert chain inside the *target*.
AS-6 measures the same chain inside the *defense*. The instrument layer is shared
— capture, probes, regimes are all reused unchanged — and this package holds the
one thing a guard needs that a target does not: a way to be addressed at all.
"""

from internals_safety.guards.prompts import (
    VerdictTokens,
    prepare_guard_prompts,
    render_guard_prompt,
    resolve_verdict_tokens,
    verdict_context,
)

__all__ = [
    "VerdictTokens",
    "prepare_guard_prompts",
    "render_guard_prompt",
    "resolve_verdict_tokens",
    "verdict_context",
]
