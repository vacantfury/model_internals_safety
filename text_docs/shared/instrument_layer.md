# The instrument layer — shared by AS-5 and AS-6

**This file is the canonical home for measurement-layer knowledge. AS-5 and AS-6
both CITE it; neither copies it.**

Founded 2026-08-05. The repo went two-paper the same day, and `CLAUDE.md` already
states the consequence: *"the two papers share the instrument layer, so a
probe/regime/coherence change is never AS-5-only."* Until now every instrument
finding was being written into `text_docs/as5/`, which would have made AS-6
re-derive facts already paid for. Anything below that is a property of the
MEASUREMENT rather than of a particular paper's object belongs here.

Why this matters more for AS-6 than for AS-5: **deployment is AS-6's central
quantity.** AS-5 can survive a weak deployment measurement by leaning on ability
and behaviour; AS-6 cannot, because "the guard decoded it but did not flag it" is
the entire paper. Every defect below is therefore inherited at AS-6's core, and
several must be settled before guard-side code is written, not after.

---

## 1. Settled defects and their fixes

Each of these was found by a real run and cost real time. Do not re-litigate them
from first principles; do re-check them if the pipeline changes.

### 1.1 Probe fits must be single-threaded (operational, landed `a0855fa`)

BLAS thread thrashing on small fits. Measured on real cached activations at the
phase-0 shape (140 × 4096, shuffled labels — the permutation null's workload):

    OMP_NUM_THREADS=1     118 ms/fit    ->  12.5 min per rung
    OMP_NUM_THREADS=8   3,680 ms/fit    ->   6.5 h  per rung

**Single-threaded is 31x faster than the 8 threads an 8-CPU allocation takes by
default.** The 2026-08-05 band sweep was killed at the 8 h wall having finished
one rung of seven, with `sacct` reporting 2d14h of CPU over an 8 h wall — all
eight cores saturated, doing almost nothing. `probes.linear.single_threaded_blas`
decorates all seven fitting entry points. It is decorated at the ENTRY POINTS,
not at `_fit`, because `crossval_scores` fits through sklearn's own
`cross_val_predict` and would bypass a choke-point wrapper.

*AS-6 inheritance:* guard capture will fit the same probes at the same shape.
The decorator is already applied library-wide; do not remove it to "parallelise".

### 1.2 Completed work must survive a job that does not (landed `a0855fa`)

`cells.jsonl` was 0 BYTES on both models after the killed sweep even though a
rung had COMPLETED — buffered writes, and `results.json` is written only after
the family loop. Eight GPU-hours of finished work were unrecoverable. Every
family now flushes + `fsync`s and appends a timed row to
`summaries.partial.jsonl`.

### 1.3 The ability binary is not exact-match (landed `0483fe8`, calibration OPEN)

`exact or contains` scored genuine decodes as failures on two-character slips,
manufacturing `deployment_without_ability` hard incoherences that instrument-
failed the decisive rungs. Replaced by three routes: exact/contains short-circuit;
similarity ≥ cut with a content-overlap veto; order-blind overlap. 439 cells
newly counted as ability.

**Still open, and kept as an explicit test** (`test_KNOWN_GAP_a_long_benign_tail_defeats_the_veto`):
the content-overlap veto fails its own motivating case — the ransomware
payload-drop scores overlap 0.75, above the 0.6 cut, because payload words are a
small minority of a long sentence's content tokens. A fraction-based veto is
diluted by sentence length.

### 1.4 The coherence check is split by behaviour (landed `0483fe8`)

`recognition_without_deployment` is HARD only when the cell **complied**. When it
refused, it is the (R) surface-refusal mechanism seen from the inside and gets a
soft `surface_recognition` flag. Evidence: of 142 such cells on Llama
`reverse_characters`, **141 refused and 1 complied** — the flag was deleting 71%
of a rung as "broken" while the instrument worked.

*AS-6 inheritance:* this rule is AS-6's core instrument, per `CLAUDE.md`. A guard
that flags without decoding is the guard-side analogue of surface refusal, and it
is a FINDING there, not an error.

### 1.5 Recognition is tri-state; deployment must become tri-state too

`None` = the probe was not licensed = unmeasured, NOT absent. An unlicensed probe
returning `False` asserts "harm is not represented" when the truth is "this
instrument could not read this cell". Recognition was fixed; deployment's
tri-state landed separately (`5faffa6`).

---

## 2. Licensing vs reading — the distinction the whole layer turns on

**These are two different questions and only one was built.**

- **Licensing (population): is there signal at all?** Answered by a permutation
  test over the MAX AUROC across layers, which puts the layer selection inside
  the null instead of leaving `any()` over ~33 cells as an uncorrected multiple
  comparison. This is correct and stays.
- **Reading (per cell): is there enough signal to label ONE prompt?** This was
  never built, and it is where the regime label is actually decided.

**Significance is not sufficiency.** At n=200 an AUROC of 0.63 clears a
shuffled-label null comfortably. Permutation licensing therefore passed exactly
the rungs whose per-cell reads were noise.

### 2.1 The per-cell read had a 50% false-positive rate by construction

`probes.reading_percentile` is 50.0, so `harmless_rate` came out at **EXACTLY
0.50 on all 14 rung-cells** of the band sweep. `assign_regime` consumes the
resulting boolean and never sees the gap — so the caveat the config comment
states plainly never reaches the label. On a strong probe the harmful read rate
reaches ~1.00 against that benign 0.50 and a positive read is informative; on a
weak one (0.63–0.68) the harmful rate is 0.63–0.76 against the same 0.50, so most
"deployed" cells are the operating point's own false positives.

### 2.2 The fix: a control-calibrated floor, derived not guessed

**Rungs whose ability is ~0 are a free negative control.** Whatever the
deployment probe reads there is by construction NOT decoded content, so its
transfer AUROC on those rungs is this instrument's surface-feature noise floor.

Measured (`scripts/recalibrate_deployment.py`, run `band2-20260805`):

| model | control rungs | floor |
|---|---|---|
| Llama-3.1-8B | reverse_characters, tag_block | **0.656** |
| Qwen2.5-7B | reverse_characters, tag_block | **0.671** |

The retired fixed cut (`auroc_threshold: 0.70`) was approximately right *by
luck*. It now has a stated basis. Applying the floor collapses the hard
incoherence that had instrument-failed three rungs: below-floor rungs read `U`
(unmeasured) instead of manufacturing `deployment_without_ability`.

*AS-6 inheritance — LOAD-BEARING:* AS-6 needs its own control rungs, i.e.
encodings the GUARD cannot decode, measured on the guard itself. Do not import
0.656 as a constant; the floor is instrument- and model-specific and must be
re-derived per model. **Design the guard-side ladder so it contains can't-decode
rungs on purpose** — they are not wasted cells, they are the calibration.

### 2.3 A non-circular criterion for choosing the operating point

Hard incoherence falls **monotonically** as the read tightens, on both models:

    Llama   pct50 0.028 -> pct95 0.018
    Qwen    pct50 0.136 -> pct95 0.088   (crosses below the 10% bar)

The coherence check endorses the strict read on its own internal consistency,
without assuming the answer we want. That is the argument to use; do not justify
an operating point by the regime counts it produces.

---

## 3. Validated diagnostics

### 3.1 The layer index discriminates lexical presence from computed recovery

Free, and available from any run that persists its curves. Best-cell selection
across the band sweep, both models independently:

| rung | Llama | Qwen | reads |
|---|---|---|---|
| zero_width | L18 / instruction_final | L19 / instruction_final | computed |
| homoglyph | L21 / instruction_final | L19 / instruction_final | computed |
| fullwidth | L18 / instruction_final | L21 / instruction_final | computed |
| reverse_words | L5 / last | L3 / last | lexical |
| combining_marks | L3 / last | L2 / last | lexical/surface |
| reverse_characters *(control)* | L2 / last | L20 / last | surface |
| tag_block *(control)* | L16 / last | L11 / last | surface |

**Late-mid layer (L18–22) at `instruction_final` = content recovered by
computation. Early layer at `last` = content present at the surface.** Three
character-level rungs land in the first cluster on both models independently.

Use it as a validity check on every deployment claim: a "decoded harmful content"
direction that is linearly readable at layer 2 is not decoded content.

### 3.2 A rung can be lexically transparent — `reverse_words` is, and is a CONTROL

`reverse_words` permutes word ORDER while leaving the words intact, so the
harmful tokens are literally present in the input. A content probe fires without
the model decoding anything, which is why it reads at L3–L5 rather than L18–L22.
Its transfer AUROC (0.844 / 0.870) passes the noise floor but **does not evidence
in-situ decoding.** Reclassified as a control, not a sound rung.

The general rule this instance teaches: **a deployment probe measures decoding
only when the encoding removes the content from the lexical surface.** Screen
every rung for this before counting its cells.

### 3.3 Invertibility is not comprehension

`tag_block` is exactly invertible — the property the deterministic band was built
on — and scores ability **0.00 on both models** (it encodes into the invisible
Unicode Tags block at preserved character count). Invertibility is a property of
the ENCODING; comprehension is a property of the MODEL. They came apart here for
the first time. `reverse_characters` is likewise a genuine can't-decode rung
(Llama restates "The quick brown fox jumps over the lazy dog").

---

## 4. Open, with the method already identified

### 4.1 Deployment should be measured by logit lens, not a transferred probe

The transferred content probe (fit plain-harmful vs plain-harmless, evaluated on
encoded) is the current measurement #2 and it is the weakest link in both papers.
It reads surface form on weak rungs and lexical presence on transparent ones.

**Method import: Fang & Marks, "Unsupervised decoding of encoded reasoning using
language model interpretability" (arXiv 2512.01222, Goodfire AI / Anthropic,
NeurIPS 2025 Mech-Interp workshop).** They fine-tune a reasoning model to do
chain-of-thought in ROT-13 and show **logit lens recovers the plaintext from
activations alone**, with accuracy peaking in intermediate-to-late layers
(~L58/80), because representations stay anchored to English in intermediate
layers (the semantic-hub effect). Their Appendix E cross-checks with concept
activation probes, peaking ~L20.

Why it is the right instrument here: it asks whether the **plaintext tokens
themselves** become predictable from intermediate residual states — a direct
operationalisation of "did it decode in situ", with a built-in negative control
(it must read nothing on `tag_block`). It takes the (B)/(D) split off the
operating point entirely.

Two notes: their direction is encoded *output* (model writes ROT-13) while ours is
encoded *input*, so the method transfers but the claim does not collide; and they
report being **unable to prompt any open-weights model to reason in ROT-13**
(Llama-3.3-70B, DeepSeek-R1, Qwen3-30B/32B) without fine-tuning — independent
corroboration that our `reverse_characters` ability of 0.00 is real.

Implementation constraint: logit lens needs **per-token** readout, and the spine
currently captures two positions. Caching every token's residual is ~5 GB per
condition; computing the lens inside the forward pass and storing only the
expected-plaintext-token probability per (position, layer) is ~640K floats.
Alignment is tractable exactly where it matters — `zero_width`, `homoglyph`,
`fullwidth` are all character-level and order-preserving.

*AS-6 inheritance:* the same measurement answers "did the GUARD decode it", which
is AS-6's novelty-bearing link. AS-6's scoop check found nobody has asked "did the
guard decode it?" as an internal measurement — and a format-locked chat template
(Llama Guard 3) blocks the generation-based route entirely, so a residual-stream
method is not merely better there, it is the only one available.

### 4.2 Probe POSITION may be wrong, and this is untested

Kwon 2026 (arXiv 2607.14147) cites **Doda 2026**: final-token probes miss many
jailbreaks because the evidence is **displaced to earlier tokens**. Both our
capture positions (`instruction_final`, `last`) are late. This is a live candidate
explanation for the Llama recognition anomaly (§5 below) and it is NOT settled —
testing it needs per-token capture, i.e. the same pipeline change §4.1 requires.

### 4.3 Method imports worth taking from Kwon 2026

Not a scoop (their attack is prefill/response-site; ours is prompt-modifying), and
useful to us in three ways:

1. **Nested-CV layer selection** — cleaner than argmax-plus-permutation for the
   multiple-comparison problem.
2. **XSTest as a lexical surface control** — scary-but-benign prompts. We validated
   FORMAT decorrelation (harmful-encoded vs benign-encoded traverse the identical
   pipeline) but never LEXICAL decorrelation. This is the standard instrument.
3. **Matched-norm random direction control** for any steering, plus a warning:
   they find the harm direction is "a read-out but not a selective write-handle",
   with no clean steering window, and no single refusal-decision direction (any
   one reads ≤ 0.73). That is a predicted-null risk for AS-5's Move C.

Positioning gift, worth citing: they name the boundary of their guarantee as
*"prompt-modifying attacks (suffixes, encodings, disguise) attenuate or displace
the prompt representation while prefill leaves it whole"* and mark that claim
**analytic, not shown** — their own encoding work is an appendix of "encoding
nulls". The attenuating side is precisely what this repo measures.

---

## 5. Known anomaly, unexplained

On Llama-3.1-8B the **recognition** probe licenses on `reverse_characters`
(ability 0.00, p=0.010) while failing to license on `zero_width` (1.00),
`fullwidth` (1.00) and `homoglyph` (0.91) — the three rungs it comprehends
perfectly. Qwen is mostly coherent by comparison but also licenses
`reverse_characters`. Harm "represented" in text the model provably cannot read,
on both models.

Candidate explanations, none tested: probe position displacement (§4.2); the
harmfulness direction firing on surface anomaly rather than harm (partly excluded
by construction, since both classes traverse the identical encoding pipeline);
small-sample licensing instability. **Do not build a claim on recognition until
this is resolved.**

---

## 6. The instrument inventory, and the rule for growing it

Written 2026-08-05 after a scoped literature search, in answer to a direct
question: *is the probing layer good enough, and should we add more methods
before the next experiments?*

### 6.1 What is actually implemented — one method family, not several

The honest inventory of `src/internals_safety/probes/` + the capture spine:

| module | what it is |
|---|---|
| `probes/linear.py` | logistic-regression probes; layer sweep; plain->encoded transfer; cross-val; permutation licensing (length-matched strata); per-cell reading threshold |
| `probes/directions.py` | difference-in-means directions, projection scores, cosine similarity |
| `probes/overlap.py` | overlap coefficient / projection summaries (the H4 metric) |
| `measurements/length_null.py` | the character-length baseline every probe number must clear |
| `models/capture.py` | residual-stream activations at **2 positions** (`instruction_final`, `last`) x ~33 layers |

Read as a whole this is **one method family: supervised linear read-out of the
residual stream at two late positions.** `directions` and `overlap` are the same
signal geometry expressed differently; the length null is a control on it, not an
independent instrument. Everything the two papers currently claim about internals
rests on that single family — and it has been instrument-failed twice in one week
(§2.1 operating point, §3.2 lexical transparency).

### 6.2 The adoption rule — more methods is NOT automatically better

The failures above were **not** caused by having too few methods. They were caused
by one method with no negative control and no sufficiency bar. Adding N methods
without controls yields N more ways to be confidently wrong, and multiplies the
forking-paths surface: at 6 methods x ~33 layers x 2 positions x 19 rungs,
*something* always separates.

**A new instrument is adopted only if it satisfies all three:**

1. **It answers a question no implemented method answers** — not a variant of
   "is harm linearly decodable here" (a nonlinear probe, a different classifier,
   a different pooling would all inherit the same confounds and add nothing).
2. **It has a negative control this ladder already provides free** — the ability-0
   rungs (`tag_block`, `reverse_characters`). An instrument that cannot be shown
   to read ~nothing on text the model provably cannot decode is not evidence.
3. **It clears the length null** (`measurements/length_null.py`) on the same data.

### 6.3 The real gap: everything implemented is CORRELATIONAL read-out

Four capability classes are absent, and they are absent in a way that bears
directly on claims both papers intend to make:

- **Token-level decode evidence.** "Is harm linearly decodable" is not "did the
  model decode the ciphertext". Only a lens answers the second.
- **Causal evidence.** AS-5's Move C is a *repair* claim. A repair claim cannot be
  made from a read-out — and Kwon 2026 already reports the harm direction is "a
  read-out but not a selective write-handle" (§4.3). Zero patching, ablation or
  steering exists in this repo today.
- **Feature-level evidence.** A firing direction cannot distinguish "harm feature"
  from "surface-anomaly feature". That distinction is exactly the unexplained
  recognition anomaly of §5.
- **Layer-trajectory / per-token structure.** The spine captures 2 late positions
  and the analysis keeps the argmax cell. §4.2's displacement hypothesis is
  untestable against it.

### 6.4 Ranked shortlist, with method imports and the control each brings

Ordered by evidence-per-unit-build, not by novelty.

**1. Logit lens / token-level decode readout** *(already §4.1; unchanged as the
top item)*. Method import: Fang & Marks 2512.01222. Two additions found in this
search: **LogitLens4LLMs (arXiv 2503.11667)** ports logit-lens tooling to modern
architectures (LLaMA-family included), which removes most of the plumbing risk;
and **TriLens (arXiv 2606.01033)** uses *per-layer logit-lens entropy* rather than
top-token identity, which is a strictly cheaper readout that needs no plaintext
alignment. Control: must read ~0 on `tag_block`.

**2. Layer trajectory instead of best-cell argmax.** Import: **arXiv 2605.02958,
"Tracing the Dynamics of Refusal"** — argues directly that static terminal/pooled
directions miss how refusal is *constructed across layers*, and gets more robust
jailbreak detection from the trajectory. We already compute the full (layer x
position) curve on every run and now persist it (`eaace85`), so this converts an
existing discard into a measurement. **Near-zero build cost; do it regardless of
what else is adopted.** It is also the first real test of §4.2.

**3. Entropy dynamics — an unsupervised, label-free signal class.** Import:
**arXiv 2606.25182, "What Intermediate Layers Know: Detecting Jailbreaks from
Entropy Dynamics"**. Load-bearing property: it trains no classifier on our labels,
so **the length confound that broke the supervised probe cannot enter it by the
same route**. That makes it a genuine independent instrument under rule 6.2(1),
not another read-out variant.

**4. SAE features — the largest new capability, and it is off-the-shelf for BOTH
our models.** **Llama Scope (arXiv 2410.20526)**: 256 TopK SAEs, every layer and
sublayer, 32K/128K features. **Qwen-Scope (arXiv 2605.11887)** for the Qwen side.
No SAE training needed. This is the instrument that can separate "harm feature"
from "surface-anomaly feature" and therefore the one with a real chance at §5.
  - **VERIFIED CAVEAT, must not be skipped:** Llama Scope is trained on
    **Llama-3.1-8B-*Base***, and our target is **Instruct**. The paper explicitly
    studies generalisation to fine-tuned models, so this is a transfer question
    to measure, not a blocker — but any SAE result must first show the SAE
    reconstructs Instruct activations acceptably, or the finding is about the
    wrong model.
  - *AS-6 inheritance, and it is a gift:* **Llama Guard 3 8B is itself a
    fine-tune of Llama-3.1-8B**, so the same base-model SAEs are a candidate
    instrument for the guard-side paper — same transfer question, one shared
    validation. Test it once, both papers use it.

**5. Causal methods — LAST, and only after 1-4 have established what to intervene
on.** Needed for Move C. Adopt with the field's own warnings wired in from the
start: **arXiv 2606.27510 ("The Curse of Multiple Mediators")** on hidden
interaction effects in activation patching, and **arXiv 2607.10226 ("When Are
Sparse Feature Interventions Actually Localized?")** on apparent SAE-steering
success arising from weak or non-localized interventions. Mechanism reference for
steering refusal specifically: **arXiv 2604.08524**. Related and worth reading for
the repo's unlearning half: **arXiv 2605.24614**, activation patching as an audit
of unlearning *depth*.

**Explicitly NOT adopted:** further supervised linear-probe variants (nonlinear
probes, alternative classifiers, alternative pooling). They fail rule 6.2(1) —
same confounds, new hyperparameters, no new question.

### 6.5 The scheduling fact that decides sequencing

**Instruments 1-4 all read the same forward pass.** Lens readout, layer
trajectory, entropy dynamics and SAE encoding are all functions of the residual
stream on the same inputs; none needs generation, none needs a judge call, none
needs training. Built together they cost **one** forward-pass-only run per model;
built one at a time they cost four.

So the sequencing answer is not "more lit search, then experiments" and not
"experiments now". It is: **build 1-4 against the cached data and the offline
controls, then spend one run.** The capture spine change (per-token readout) is
shared by all four and is the single real engineering item.
