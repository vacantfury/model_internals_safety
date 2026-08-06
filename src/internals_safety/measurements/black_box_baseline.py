"""The black-box baseline — P4, and the generalisation of the length null.

**What P4 asks.** Build plan §2: an internals claim must beat what a surface
classifier would have said. If a model reading only the *text* separates the two
classes as well as a probe reading the residual stream, the probe has
demonstrated nothing about the model's internals — it has rediscovered a
property of the strings. Named as missing in build plan §4 and traced to Pando
(arXiv 2604.11061); built 2026-08-06.

**Complementary to the length null, NOT a superset of it.** An earlier draft of
this file claimed the relationship was strict — "anything the length null
catches, this catches too". **Measurement refuted that** (2026-08-06, all 19
rungs at n=100/class): TF-IDF l2-normalises, so this classifier is blind to
length by construction, and the length null scored HIGHER on every single rung
by 0.03-0.14. The two controls read different surface properties — length here,
compositional n-gram structure there — and neither contains the other. Report
both; when they disagree, the gap says which surface property is doing the work.

**The baseline must be strong, and that is a deliberate choice against our own
interest** — a weak baseline makes every internals result look good. So the
classifier is a real one: TF-IDF over character n-grams, cross-validated rather
than single-split, capped so it cannot memorise documents.

**⚠️ On THIS corpus it is nonetheless weak, and the honest reading matters.**
The plaintext baseline is only 0.615, so there is little surface separability for
an encoder to destroy and correspondingly little power to discriminate rungs. A
failed screen here means "not established", never "the content is on the
surface". Full table and consequences: build plan §4.2.

**What clearing it does and does not license.** Beating this baseline means the
probe read something not present on the surface. It does NOT establish that what
it read was the decoded content — that is what the decode lens (I1) and the
ability-0 floor are for. P4 is necessary, not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

# Character n-grams rather than words: most rungs destroy word boundaries
# (base64, hex, zero_width), so a word tokenizer would see one long token and the
# baseline would be a strawman on exactly the rungs that matter most.
# FAIL-SAFE DEFAULT — live value is `controls.black_box_ngram_{min,max}`.
NGRAM_RANGE = (2, 5)
# ⚠️ TF-IDF l2-normalises by default, so this baseline is BLIND TO LENGTH by
# construction — a doubled document has the same normalised vector. That is why
# it must be reported ALONGSIDE the length null rather than as a superset of it:
# the two controls are complementary, not nested. Measured consequence
# (2026-08-06, all 19 rungs): the length null scores HIGHER on every single rung,
# by 0.03 to 0.14, so it is the stricter of the two here. Removing the
# normalisation to fold length in was considered and rejected — it would make the
# two controls redundant and hide which surface property is doing the work.
# Capped so the baseline cannot simply memorise each document. Without a cap,
# n-gram counts on ~200 texts exceed the sample size by orders of magnitude and
# cross-validation would be measuring a nearest-neighbour lookup.
# FAIL-SAFE DEFAULT — live value is `controls.black_box_max_features`.
MAX_FEATURES = 5000


def surface_auroc(
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    seed: int = 0,
    folds: int = 5,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    max_features: int = MAX_FEATURES,
) -> float:
    """Cross-validated AUROC of a character-n-gram classifier on raw text.

    **Cross-validated, not single-split.** A rung runs 100-200 prompts per class,
    so a 30% holdout leaves ~60 test points and an AUROC whose noise band is
    wider than the margins being argued about. Out-of-fold prediction uses every
    example for evaluation exactly once.

    **Two-sided, matching the length null**: the returned value is
    `max(a, 1 - a)`, because a probe can exploit a surface signal running in
    either direction and a directional AUROC of 0.35 is a 0.65 confound wearing a
    disguise.

    Returns NaN when the baseline cannot be fitted — an empty class, or fewer
    examples than folds — so the caller's margin is NaN and every comparison
    against it fails closed rather than silently licensing.
    """
    if not positive_texts or not negative_texts:
        return float("nan")

    texts = list(positive_texts) + list(negative_texts)
    labels = np.array([1] * len(positive_texts) + [0] * len(negative_texts))
    smallest_class = min(len(positive_texts), len(negative_texts))
    if smallest_class < 2:
        return float("nan")

    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range, max_features=max_features),
        LogisticRegression(max_iter=2000),
    )
    splits = min(folds, smallest_class)
    scores = cross_val_predict(
        model,
        texts,
        labels,
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    if len(set(scores)) <= 1:
        return float("nan")
    directional = float(roc_auc_score(labels, scores))
    return max(directional, 1.0 - directional)


@dataclass(frozen=True)
class BlackBoxBaseline:
    """What a text-only classifier achieves on one rung, plain and encoded.

    Both are carried for the same reason `LengthNull` carries both: the PAIR is
    the diagnosis. `plain_auroc` is how separable the corpus is before any
    encoding — the ceiling any surface method could reach — and `encoded_auroc`
    is how much of that the encoder preserved. A rung where the two are close
    preserved the surface signal essentially intact; a rung where `encoded_auroc`
    collapses to chance has genuinely hidden the content from the surface, which
    is the condition under which an internals claim is interesting at all.
    """

    family: str
    plain_auroc: float
    encoded_auroc: float
    n_positive: int
    n_negative: int

    def margin(self, observed_auroc: float) -> float:
        """How far an observed AUROC clears the surface baseline for this rung.

        Against the ENCODED baseline, not the plaintext one, for the same reason
        the length null does: the probe reads activations produced from
        ciphertext, so ciphertext surface structure is the confound actually
        available to it.
        """
        return observed_auroc - self.encoded_auroc

    def beats_baseline(self, observed_auroc: float, min_margin: float) -> bool:
        """Whether `observed_auroc` clears the baseline by at least `min_margin`.

        Fails CLOSED on NaN — an unmeasurable baseline never licenses.
        """
        computed = self.margin(observed_auroc)
        if computed != computed:  # NaN
            return False
        return computed >= min_margin

    @property
    def surface_loss(self) -> float:
        """How much surface separability the encoding destroyed.

        Measured against this corpus's own plaintext baseline rather than against
        an absolute cut. That is deliberate: an absolute threshold would be a
        magic number, and the first version of this property used one (0.60) that
        measurement showed split a cluster — 17 of 19 rungs land between 0.589
        and 0.617, so a cut anywhere in that band assigns rungs by noise.
        """
        return self.plain_auroc - self.encoded_auroc

    def hides_content_from_the_surface(self, min_loss: float) -> bool:
        """Whether this rung meaningfully removed content from the surface.

        The screen `instrument_layer.md` states in words — "a deployment probe
        measures decoding only when the encoding removes content from the lexical
        surface" — expressed relative to the corpus's own ceiling.

        ⚠️ **Weak on this corpus, and the number says so.** Measured 2026-08-06
        over all 19 rungs at n=100/class: the plaintext baseline is only 0.615,
        so there is little separability for an encoder to destroy and this screen
        has correspondingly little power. Only `base32` (0.514) and `vigenere`
        (0.529) clear a 0.05 loss; everything else sits within 0.03 of plaintext.
        Treat a False here as "not established", never as "the content is on the
        surface" — the length null remains the sharper screen (§4.2 of the build
        plan).
        """
        loss = self.surface_loss
        if loss != loss:  # NaN
            return False
        return loss >= min_loss


def measure_black_box_baseline(
    family: str,
    plain_positive: Sequence[str],
    plain_negative: Sequence[str],
    encoded_positive: Sequence[str],
    encoded_negative: Sequence[str],
    seed: int = 0,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    max_features: int = MAX_FEATURES,
) -> BlackBoxBaseline:
    """Compute the surface baseline from the exact texts a run sent to the model.

    Texts rather than a corpus name, for the reason `measure_length_null`
    documents: the ciphertexts must be the same strings that produced the
    activations the probe was fitted and read on, or the control drifts away from
    what it is controlling.
    """
    return BlackBoxBaseline(
        family=family,
        plain_auroc=surface_auroc(
            plain_positive, plain_negative, seed=seed,
            ngram_range=ngram_range, max_features=max_features,
        ),
        encoded_auroc=surface_auroc(
            encoded_positive, encoded_negative, seed=seed,
            ngram_range=ngram_range, max_features=max_features,
        ),
        n_positive=len(plain_positive),
        n_negative=len(plain_negative),
    )
