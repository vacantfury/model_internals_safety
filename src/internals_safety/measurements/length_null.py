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

    Quantile bins rather than equal-width: encoder outputs are long-tailed
    (`binary` runs ~9x the plaintext), and equal-width bins would put almost every
    example in one bin, which silently degrades the matched null back into a free
    one. Ties are handled by ranking rather than by bin edges, so a corpus with
    many identical lengths still spreads across bins instead of collapsing.

    Honest about the knob: `n_bins` IS one, and it is milder than the margin
    threshold it replaces — more bins means a stricter test, and the result should
    be stable across a broad range rather than tuned. Report that stability rather
    than asserting it (`conf/measurements.yaml` names the check).
    """
    import numpy as np

    lengths = np.array(
        [len(text) for text in positive_texts] + [len(text) for text in negative_texts],
        dtype=float,
    )
    if lengths.size == 0:
        return np.empty(0, dtype=int)
    # Rank-based so that ties spread rather than piling into one edge.
    order = np.argsort(np.argsort(lengths, kind="stable"), kind="stable")
    bins = max(1, min(n_bins, lengths.size))
    return (order * bins // lengths.size).astype(int)


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
