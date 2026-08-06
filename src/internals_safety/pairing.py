"""Pairing primitives — the derangements every negative control is built on.

**Why this is at the package root rather than inside `measurements/`.** A
derangement is not a measurement; it is a list operation with a stated
statistical property. Two measurement modules now need it — `decode_lens` scores
each activation against a mismatched plaintext, and `ability_control` scores each
response against one — and `tests/test_package_structure.py` deliberately forbids
a pure control module and an instrument from importing each other. Extracting the
shared primitive one level up satisfies both rules with one home for the rule
itself, which is what the spine criterion asks for: a construction whose
divergence in ONE caller would be a defect.

**The property that matters, and why rotation rather than a random permutation.**
A control pairing must fix no point. A random permutation can, and a fixed point
silently scores a row against ITSELF — which inflates the control reading and
shrinks the margin toward zero, i.e. fails in the direction that hides a real
result. Rotation by one is deterministic, needs no seed, and cannot fix a point
for n >= 2.
"""

from __future__ import annotations

from typing import Sequence


def derangement(n: int) -> list[int]:
    """Index permutation of length `n` fixing no point. Rotation by one."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    return [(index + 1) % n for index in range(n)]


def derange(items: Sequence) -> list:
    """Rotate by one, so no element keeps its own index.

    This is the negative control's pairing: every prompt is scored against a
    reference that is not its own, drawn from the SAME condition so the encoding,
    the attack template and the length distribution all stay fixed and only the
    pairing moves.
    """
    return [items[index] for index in derangement(len(items))]


def quantile_strata(values: Sequence[float], n_bins: int) -> list[int]:
    """Rank-based quantile bins — the one binning rule in the codebase.

    Quantile rather than equal-width: encoder outputs are long-tailed (`binary`
    runs ~9x the plaintext), and equal-width bins would put almost every example
    in one bin, silently degrading a matched null back into a free one. Ties are
    handled by ranking rather than by bin edges, so a corpus with many identical
    lengths still spreads across bins instead of collapsing.

    Honest about the knob: `n_bins` IS one. More bins means a stricter test, and
    a result should be stable across a broad range rather than tuned to one
    count — report that stability rather than asserting it.
    """
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: (values[index], index))
    rank = [0] * count
    for position, index in enumerate(order):
        rank[index] = position
    bins = max(1, min(n_bins, count))
    return [(rank[index] * bins) // count for index in range(count)]


def stratified_derangement(values: Sequence[float], n_bins: int) -> list[int | None]:
    """A derangement that pairs each item only with one in its own stratum.

    **This is what makes a control MATCHED rather than free.** A free derangement
    holds the condition fixed but lets the mismatched reference have any length;
    if a scorer is riding length, the mismatched pairing draws references of the
    wrong length and scores low, so the margin looks healthy and the confound
    survives. Pairing within a length stratum removes exactly that escape: the
    mismatched reference is the same length as the real one, so anything the
    scorer reads from length alone is present in BOTH arms and cancels.

    Returns `None` at every index whose stratum has fewer than two members —
    those items cannot be paired without leaving the stratum, and quietly pairing
    them across bins would turn the matched null back into a free one for exactly
    the extreme-length items most likely to carry the confound. The caller drops
    them and reports how many, which is the honest handling: a control computed
    over 94 of 100 prompts is a different claim from one computed over 100.
    """
    strata = quantile_strata(values, n_bins)
    members: dict[int, list[int]] = {}
    for index, stratum in enumerate(strata):
        members.setdefault(stratum, []).append(index)

    partner: list[int | None] = [None] * len(strata)
    for group in members.values():
        if len(group) < 2:
            continue
        for offset, index in enumerate(group):
            partner[index] = group[(offset + 1) % len(group)]
    return partner
