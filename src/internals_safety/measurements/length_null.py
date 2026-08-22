"""The length null model — the control §8's format-decorrelation cannot provide.

Found empirically on 2026-08-05, by the re-licensing run, and it is the reason
this module exists rather than being a precaution someone thought prudent.

**What happened.** Permutation licensing moved deployment from 2/15 to 14/15
rungs on Llama-3.1-8B, which looks like the regime map growing. It was not. Raw
character length ALONE separates the JBB harmful corpus from the benign one at
AUROC 0.6544 (86.0 vs 73.8 mean characters). Every encoder in the ladder is
monotone in length, so that separation survives encoding essentially intact —
ciphertext-length AUROC lands at 0.649-0.657 on every rung. The newly-licensed
rungs sat at mean deployment AUROC 0.659 against a length baseline of 0.654:
indistinguishable. The signal was real and significant (p about 0.005) and it was
length, not decoded content.

**Why the existing controls miss it.** §8's format-decorrelation control sends
benign content through the *same* encoder, which defeats a probe firing on "this
looks encoded" — both classes look equally encoded. Length is different in kind:
it is a property *of the content* that the encoder preserves, so it passes
straight through a control designed for properties of the format. The
characters-per-token null (Broken-Token) is likewise a different control: it
addresses tokenizer fragmentation, not content length, and holding surface form
constant does not hold content length constant.

**Why the permutation test does not cover it either.** Permutation licensing
controls the multiple comparison over the (layer x position) grid. It says
whether a separation is larger than chance; it says nothing about WHAT separates.

**The rule this module implements.** Report the observed AUROC beside the length
AUROC on the SAME prompts, and require a margin. On the pilot's own numbers that
separates cleanly rather than by taste: the two rungs with genuine decode signal
beat the null by 0.19 and 0.29 (`reverse_words` 0.844, `zero_width` 0.945) while
every confounded rung beats it by about 0.005.

Binding on BOTH papers. AS-6's guard-side probes inherit it at their first line
of code rather than as a retrofit — the same mistake would be cheaper to make
there, because a guard that "represents the harm" is exactly the claim a length
confound would manufacture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import roc_auc_score

from internals_safety.pairing import quantile_strata


def length_auroc(positive_texts: Sequence[str], negative_texts: Sequence[str]) -> float:
    """AUROC of raw character length separating the two classes.

    **Two-sided on purpose.** A directional AUROC would score a confound that runs
    the other way (benign longer than harmful) as 0.35 and call it harmless, when
    a linear probe can exploit either direction equally. The null is therefore
    `max(a, 1 - a)`: the best any length-only classifier could do.

    Returns NaN when either class is empty or all lengths are identical, so the
    caller's margin is NaN and any comparison against it fails closed rather than
    silently licensing.
    """
    if not positive_texts or not negative_texts:
        return float("nan")

    lengths = [len(text) for text in positive_texts] + [len(text) for text in negative_texts]
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    if len(set(lengths)) <= 1:
        return float("nan")

    directional = float(roc_auc_score(labels, lengths))
    return max(directional, 1.0 - directional)


@dataclass(frozen=True)
class LengthNull:
    """The length baseline for one encoding family, plus the plaintext baseline.

    Both are carried because the pair is the diagnosis. If `encoded_auroc` tracks
    `plain_auroc`, the encoder preserved the content-length signal — which is what
    the pilot measured on every rung, and what makes this a ladder-wide confound
    rather than one rung's quirk.
    """

    family: str
    plain_auroc: float
    encoded_auroc: float
    mean_positive_chars: float
    mean_negative_chars: float
    n_positive: int
    n_negative: int

    def margin(self, observed_auroc: float) -> float:
        """How far an observed AUROC clears the length baseline for this rung.

        Measured against the ENCODED baseline, not the plaintext one: the probe
        reads activations produced from ciphertext, so ciphertext length is the
        confound actually available to it.
        """
        return observed_auroc - self.encoded_auroc

    def beats_null(self, observed_auroc: float, min_margin: float) -> bool:
        """Whether `observed_auroc` clears the baseline by at least `min_margin`.

        Fails CLOSED on NaN — an unmeasurable baseline never licenses. Written as
        an explicit comparison rather than `>=` on a NaN-propagating expression so
        that intent is visible at the call site.
        """
        computed = self.margin(observed_auroc)
        if computed != computed:  # NaN
            return False
        return computed >= min_margin


def length_strata(
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    n_bins: int,
) -> "np.ndarray":
    """Quantile bins of character length, for a length-MATCHED permutation null.

    Order matches the concatenated test set the probe layer builds — positives
    first, then negatives — because the null permutes labels in that order.

    The binning rule itself lives in `internals_safety.pairing.quantile_strata`,
    shared with the stratified derangement that makes `ability_control` matched.
    One home, because two callers binning length by different rules is how a
    "length-matched" claim becomes two different claims wearing one name.

    Honest about the knob: `n_bins` IS one, and it is milder than the margin
    threshold it replaces — more bins means a stricter test, and the result should
    be stable across a broad range rather than tuned. Report that stability rather
    than asserting it (`conf/measurements.yaml` names the check).
    """
    import numpy as np

    lengths = [float(len(text)) for text in positive_texts]
    lengths += [float(len(text)) for text in negative_texts]
    if not lengths:
        return np.empty(0, dtype=int)
    return np.array(quantile_strata(lengths, n_bins), dtype=int)


def measure_length_null(
    family: str,
    plain_positive: Sequence[str],
    plain_negative: Sequence[str],
    encoded_positive: Sequence[str],
    encoded_negative: Sequence[str],
) -> LengthNull:
    """Compute the length baseline from the exact texts a run sent to the model.

    Takes texts rather than a corpus name so it cannot drift from what was
    actually encoded: the ciphertexts here must be the same strings that produced
    the activations the probe was fitted and read on.
    """
    positive_lengths = [len(text) for text in plain_positive]
    negative_lengths = [len(text) for text in plain_negative]
    return LengthNull(
        family=family,
        plain_auroc=length_auroc(plain_positive, plain_negative),
        encoded_auroc=length_auroc(encoded_positive, encoded_negative),
        mean_positive_chars=(
            sum(positive_lengths) / len(positive_lengths) if positive_lengths else float("nan")
        ),
        mean_negative_chars=(
            sum(negative_lengths) / len(negative_lengths) if negative_lengths else float("nan")
        ),
        n_positive=len(plain_positive),
        n_negative=len(plain_negative),
    )


# ---------------------------------------------------------------------------
# The RATE-scale null (2026-08-21, TODO 84 #12/#21).
#
# `LengthNull` above answers P3 for a reading whose `value` is an AUROC. Two
# readings on the roster carry a RATE instead — `behavior` (an attack-success
# rate) and `refusal` (a difference of two refusal rates) — and both were being
# handed `LengthNull.margin()`, which subtracts a character-length AUROC from
# them. A rate and an AUROC are not the same scale, so that difference is not a
# measurement: it is negative by construction whenever the rate is small, which
# withheld every behaviour-axis reading in all 31 runs on disk for a reason that
# never examined the data.
#
# The mistake was PREDICTED IN WRITING, in a comment sitting in the same list
# literal as one of the two violating calls ("that one compares a rate against a
# character-length AUROC, which is not the same scale"). This repo has now paid
# for the lesson three times: a note predicting a defect is not a guard against
# it. Hence `tests/test_entrypoint_call_sites.py`, which makes the call
# unrepresentable rather than the instance detectable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLengthNull:
    """P3 for a rate-scale reading: does the effect survive length matching?

    The probe-side null asks whether an AUROC beats what character length alone
    achieves. A rate difference has no such baseline classifier, so the analogue
    is a **length-matched permutation**: bin both arms by ciphertext length, and
    permute the harmful/benign labels WITHIN bins. Length is held constant by
    construction, so the resulting distribution of gaps is what length alone can
    manufacture on this corpus, and the margin is how far the observed gap
    clears it.

    Binning uses `quantile_strata`, the same rule as `length_strata` — one home,
    because two callers binning length differently is how a "length-matched"
    claim becomes two claims wearing one name.
    """

    family: str
    observed_gap: float
    stratified_gap: float
    null_quantile: float
    n_bins: int
    n_strata_used: int
    n_positive: int
    n_negative: int

    @property
    def margin(self) -> float:
        """How far the length-matched gap clears the length-matched null.

        Magnitudes on both sides: the null is two-sided (a permutation can throw
        the gap either way) and a reading's direction is the `claim`'s business,
        never this control's. NaN propagates rather than being swallowed, so an
        unmeasurable null fails closed at `clears_length_null`.
        """
        return abs(self.stratified_gap) - self.null_quantile

    @property
    def matching_shift(self) -> float:
        """How far length matching MOVED the gap. Reported, never gated on.

        A large shift with a surviving margin is still a pass; it says the raw
        number was partly length and the remainder is not. Collapsing the two
        into one gate would hide which happened.
        """
        return self.stratified_gap - self.observed_gap


def stratified_rate_gap(
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    positive_flags: Sequence[bool],
    negative_flags: Sequence[bool],
    n_bins: int,
) -> tuple[float, int]:
    """Length-stratified difference of two rates, and how many bins contributed.

    Bins with only one arm present contribute nothing — a stratum containing no
    benign item cannot say anything about a difference — and are excluded from
    the weights rather than counted as a zero difference.
    """
    lengths = [float(len(text)) for text in positive_texts]
    lengths += [float(len(text)) for text in negative_texts]
    if not lengths:
        return float("nan"), 0
    strata = list(quantile_strata(lengths, n_bins))
    cut = len(positive_texts)
    positive_strata, negative_strata = strata[:cut], strata[cut:]

    numerator = denominator = 0.0
    used = 0
    for stratum in sorted(set(strata)):
        pos = [f for f, s in zip(positive_flags, positive_strata) if s == stratum]
        neg = [f for f, s in zip(negative_flags, negative_strata) if s == stratum]
        if not pos or not neg:
            continue
        weight = float(len(pos) + len(neg))
        numerator += weight * (sum(pos) / len(pos) - sum(neg) / len(neg))
        denominator += weight
        used += 1
    if denominator == 0.0:
        return float("nan"), 0
    return numerator / denominator, used


def measure_rate_length_null(
    family: str,
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    positive_flags: Sequence[bool],
    negative_flags: Sequence[bool],
    *,
    n_bins: int,
    n_permutations: int,
    quantile: float,
    seed: int,
) -> RateLengthNull:
    """Fit the length-matched permutation null for one condition's rate gap.

    Every knob is keyword-only with no default. The omission that produced this
    module's sibling defect was a caller silently getting a passing value, so
    nothing here can be reached by forgetting it.
    """
    import numpy as np

    positive = [bool(flag) for flag in positive_flags]
    negative = [bool(flag) for flag in negative_flags]
    observed = (
        (sum(positive) / len(positive) - sum(negative) / len(negative))
        if positive and negative
        else float("nan")
    )
    stratified, used = stratified_rate_gap(
        positive_texts, negative_texts, positive, negative, n_bins
    )

    lengths = [float(len(text)) for text in positive_texts]
    lengths += [float(len(text)) for text in negative_texts]
    strata = np.array(quantile_strata(lengths, n_bins), dtype=int) if lengths else np.empty(0, int)
    flags = np.array(positive + negative, dtype=bool)
    labels = np.array([True] * len(positive) + [False] * len(negative), dtype=bool)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_permutations):
        shuffled = labels.copy()
        for stratum in np.unique(strata):
            where = np.flatnonzero(strata == stratum)
            shuffled[where] = rng.permutation(shuffled[where])
        numerator = denominator = 0.0
        for stratum in np.unique(strata):
            where = np.flatnonzero(strata == stratum)
            pos = flags[where][shuffled[where]]
            neg = flags[where][~shuffled[where]]
            if pos.size == 0 or neg.size == 0:
                continue
            weight = float(pos.size + neg.size)
            numerator += weight * (float(pos.mean()) - float(neg.mean()))
            denominator += weight
        draws.append(abs(numerator / denominator) if denominator else float("nan"))

    finite = [value for value in draws if value == value]
    null_quantile = float(np.quantile(finite, quantile)) if finite else float("nan")
    return RateLengthNull(
        family=family,
        observed_gap=observed,
        stratified_gap=stratified,
        null_quantile=null_quantile,
        n_bins=n_bins,
        n_strata_used=used,
        n_positive=len(positive),
        n_negative=len(negative),
    )
