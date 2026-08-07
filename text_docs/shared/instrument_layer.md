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

⚠️ **The table below is run-specific and does NOT port** — §2.4 shows the
statistic grows with the control-set size, so these two numbers belong to
`band2-20260805` and to no other run. Re-derive per run; never import them.

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

### 2.4 The floor statistic is n-dependent, and §2.2's table is not portable

**Found 2026-08-07** by re-licensing the whole pilot ladder (`relicense_all`, job
`8995184`, 30 CPU array tasks, 29 completed, $0). That run supplied a control set
of **11 rungs on Llama and 10 on Qwen** — the inert cipher band — against the
**two** rungs §2.2 had to work with. Three things came out of the larger set, and
the second is a defect in the fix §2.2 landed.

⚠️ **The licensing column of that run is under the RETIRED null — and the gap
was wider than first reported (FIXED 2026-08-07).** The length-matched null was
settled 2026-08-06 as THE licensing rule (`as6/phase1_map.md` §1) and threaded
into `measure_deployment` and AS-6's sweep the same day. It reached **neither
`relicense_probes.py` nor `phase0_regime_map.py`** — the main AS-5 entrypoint —
so *every AS-5 run until 2026-08-07 licensed deployment under the unmatched
null*, not merely the re-licensing job. **The conclusion above is unaffected,
because the screen compares AUROCs against control rungs and never consults a
p-value**, but no licensing COUNT from any of those runs may be reported.

**The fix is structural, not procedural:** `strata` is now a keyword-only
argument with **no default**, so omitting it is a `TypeError` rather than a
silently superseded test, and all three production callers build it. Pinned by
`tests/test_control_floor.py::TestTheSupersededNullCannotBeReachedByOmission` —
both the signature and the callers, because a later tidy-up restoring `= None`
would reopen the defect exactly the way it opened the first time.

*This is the settled-rule-not-threaded failure, third instance: a rule can be
adopted, tested and documented and still not reach every caller. The lesson is
that "thread it into the callers" is not the fix — **making the omission
inexpressible** is.*

**(a) Significance is not sufficiency — measured a second time, now within one
run.** Permutation licensing passes **14/15 rungs on Llama and 14/14 on Qwen**.
It is a real test and it is answering a different question: whether a separation
exists, never what produces it. On Qwen, twelve of fourteen rungs sit inside a
band 0.046 wide (0.638–0.683) and every one is "significant".

**(b) ⚠️ `control_floor`'s MAX is an extreme-value statistic, so the floor grows
with the size of the control set.** Same instrument, same models, different
number of can't-decode rungs in the run:

| model | controls | floor (max) | controls | floor (max) |
|---|---|---|---|---|
| Llama-3.1-8B | 2 (`band2`) | 0.656 | **11** (pilot ladder) | **0.674** |
| Qwen2.5-7B | 2 (`band2`) | 0.671 | **10** (pilot ladder) | **0.683** |

A rung's pass/fail therefore depends on *how many rungs the model happened to be
unable to decode in that run* — which is not a property of the rung. **Floors
from different runs are not comparable, and the §2.2 numbers must not be carried
into a run with a different ladder.** The docstring's rationale ("a rung that
read higher on nothing is the binding constraint") is sound for n=2 and becomes
an upward-biased estimator as n grows.

**ADOPTED 2026-08-07 (owner go): the rule is `mean + sigma*SD`**, sigma = 2.0 —
0.711 (Llama), 0.691 (Qwen). Derivation and both knobs:
`measurements/control_floor.py`, `controls.control_floor_sigma` /
`control_floor_min_controls`. One implementation now serves both scripts; it had
been written twice, and a floor computed two ways can disagree with itself.

**Sigma is a WINDOW, derived, not a number chosen.** The requirement *no control
rung may clear its own floor* forces **sigma ≥ 1.531** (Qwen binds; Llama alone
needs 0.846). The requirement *both genuine rungs still pass* caps it at
**6.17**. Nothing in `[1.53, 6.17]` changes any conclusion, so there is no
reachable setting that alters an answer — and **`hex` sits at k = 1.383, below
the bar the no-control-may-pass requirement imposes before 2.0 is chosen at
all.** That is what makes the adoption non-circular rather than answer-driven: it
was settled on a requirement that says nothing about which rungs we want to pass.
`sigma_bounds` recomputes both; **if they ever cross, the screen has no valid
setting and must report that rather than pick one.**

Below `min_controls` (5) the floor falls back to max and is **labelled a
`bound`** rather than a `distribution`. Not a refusal — the band run's
two-control screen was informative — but labelled, because 0.656 was a bound that
got copied as though it described the instrument.

**(c) The gate answer is robust to that choice, which is what makes it
reportable.** Only `zero_width` and `reverse_words` clear on both models:

    Llama   zero_width 0.945 (+0.272)   reverse_words 0.844 (+0.171)
    Qwen    zero_width 0.989 (+0.306)   reverse_words 0.870 (+0.187)

`hex` clears on Llama by **+0.017** and fails on Qwen by **−0.021** under max,
and fails both under mean+2SD. Every floor ever applied to it — the retired 0.70
guess, max, mean+2SD — has landed within ±0.02 of it. **A rung that sits inside
the noise of every bar and flips sign across models is not a measurable rung.**

**The two decisive cells are both on Llama, and they are the cleanest available
statement of the confound §2.1–§2.2 exist to contain** — established inside a
single run against its own control, needing no external length null:

| rung | ability (settled cuts) | deployment AUROC | vs control ceiling 0.674 |
|---|---|---|---|
| `hex` | **0.84** | 0.6908 | +0.017 |
| `unicode_escape` | **0.54** | 0.6610 | **−0.013** |

The model recovers the plaintext on **84 of 100** `hex` prompts, and the
deployment probe reads it at essentially the value it reads on rungs the model
cannot decode at all. On `unicode_escape` it decodes over half the time and the
probe reads *below* that ceiling. No appeal to a weak probe explains a monotone
failure like that: the probe is not reading decoding.

⚠️ **Ability under the settled cuts is not the number quoted in `CLAUDE.md`.**
Recomputed here: Llama `hex` **0.84** (not 72/100) and Qwen `hex` **0.01** (not
25/100). The Qwen figures on record came from a **0.60** similarity cut; the
settled cut is **0.75**. Under the settled rule `hex` and `unicode_escape` are
Llama-readable and Qwen-inert — which is itself why `hex` cannot anchor
anything, and it is a second reason beyond the floor disagreement.

*Reproduce:* `scripts/control_floor.py` — derives the control set from
recomputed ability rather than a listed one, prints both floors with their n, and
refuses to call a rung measurable when they disagree.

*AS-6 inheritance:* the guard-side ladder must carry enough can't-decode rungs to
estimate a control **distribution**, not just bound it — n=2 cannot yield an SD.

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

#### 6.3.1 The causal harness already exists, peer-reviewed — and it would have caught our length confound

*Settled 2026-08-06 by reading `other_repos/refusal_direction` (Arditi et al.,
arXiv 2406.11717, **NeurIPS 2024, 914c**). This is instrument knowledge, so it
lives here; both papers cite it.*

**⚠️ Correction to how this read was first reported.** It was framed as "the
paper the coverage sweep found we had missed." **Arditi was not missed** — it is
cited in `probes/directions.py` line 1, `measurements/recognition.py` line 3, and
four places in `as5/s1_idea_check.md`, including measurement #3's own table row
and the phase-2 plan ("the Arditi trio: directional ablation for necessity,
activation addition for sufficiency"). The estimator was ported from them
deliberately at S1. What the coverage sweep actually found is that the *instrument
build plan* omitted them. The findings below stand on their own — they came from
reading the source, not from the sweep — but the "we missed this" framing was
wrong and is withdrawn.

**What they do that we do not.** Their direction is a **difference in means**
between harmful and harmless activations (`generate_directions.py`), swept over
every layer and every position in the end-of-instruction token span
(`positions=range(-len(eoi_toks), 0)`). Then — the part that matters — they
**select the direction causally, not correlationally** (`select_direction.py`):

| their criterion | what it tests |
|---|---|
| `refusal_score` after ablation | ablate the direction → does refusal on harmful prompts actually **bypass**? |
| `induce_refusal_threshold` | add the direction → does it **induce** refusal on harmless prompts? |
| `kl_threshold = 0.1` | ablating it must leave harmless behaviour **unchanged** (KL vs baseline) |
| `prune_layer_percentage = 0.20` | discard directions from the last 20% of layers |

**Ours is purely correlational.** `conf/measurements.yaml` licenses on AUROC
against a permutation null, then reads at `reading_percentile: 50.0`. Nothing in
this repo ablates, steers, or checks that an intervention preserves unrelated
behaviour.

**⚠️ The consequence, and it is the sharpest thing this read produced: their
selection would have rejected our length direction automatically.** We found the
length confound (§5, `pilot_rebaseline.md` §5) the expensive way — a re-licensing
run, then a separate null model, now mandatory for both papers. A direction that
separates harmful from benign *by character length* cannot pass their filter: it
will not bypass refusal when ablated, and ablating it is unlikely to leave
harmless behaviour within KL 0.1. **A causal criterion is not a nicer validity
check than a correlational one — it is a different kind, and it screens confounds
a permutation test structurally cannot see.** Permutation licensing answers "is
this separation real?"; it never answers "is this separation *the thing*?"

**Three consequences.**

1. **I5/I6 stop being the last instruments and become the licensing layer.** §6.3
   lists causal evidence as a gap to close eventually; this read says the causal
   test belongs *upstream*, gating which direction is used at all. **What is new
   here is the SEQUENCING, not the technique** — `as5/s1_idea_check.md` already
   planned "the Arditi trio" for phase 2. The claim is that a validation scheduled
   *after* the correlational measurement should instead gate it.
2. **The estimator question is now live.** We fit a discriminative probe; they
   take a difference in means. Diff-in-means is the weaker classifier and that is
   the point — a fitted probe can exploit any separating feature, including
   length, while a mean difference is at least constrained to the direction the
   two populations differ along. Worth measuring both on our cached activations,
   which is offline and free.
3. **It answers I1's open source-position question by precedent.** §3.1 of the
   build plan left "which position to patch" open; their answer is to **sweep the
   whole post-instruction token span and select**, rather than fix positions in
   advance. Our spine's two fixed positions (`instruction_final`, `last`) are a
   design choice we made before knowing a sweep-and-select precedent existed.

**Contribution check for AS-5 — measurement #3 is NOT a reimplementation, but the
estimator axis is narrower than first reported.** We already compute
difference-in-means (`probes/directions.py`, ported from them); what rides on the
*fitted probe* is the licensing decision (`best_by_auroc` +
`permutation_null_max_auroc` in `measurements/recognition.py`), and measurement #2
(deployment) uses the fitted probe alone. So the estimator difference is real for
#2 and only partial for #3. Two genuine differences remain: selection
(permutation licensing vs causal filtering), and **input distribution** — they
ask whether a single direction mediates refusal on *readable* harmful prompts,
while AS-5 asks whether the harmfulness representation forms at all when the
prompt is ciphertext. The delta is real. What is *not* defensible is presenting
our licensing as state of the art: theirs is stronger and it is two years old.
Both papers must cite them, and AS-5 must say plainly which criterion it used and
why.

### 6.4 The roster, the sequencing, and the literature map

**Superseded and moved.** The ranked instrument roster that stood here — with each
instrument's question, method imports, validation gate, cost and per-paper use —
plus the build sequence and the annotated literature map, is now canonical in
**`text_docs/shared/instrument_build_plan.md`** (written 2026-08-05 after a
comprehensive literature sweep). It is a sibling of this file: **this file is what
the instrument layer has been FOUND to do; the build plan is what it WILL be.**

Keeping a second copy here would be dual truth, so this section is a pointer. Two
results from that sweep belong here as findings and are recorded above and below
rather than only in the plan:

- **The length confound is established literature, not just our observation.**
  arXiv 2605.00269 finds white-box OOD detection scores — naming CED, RAUQ and
  **WildGuard confidence** — structurally confounded by sequence length at
  |r| >= 0.61, collapsing to chance (~0.50) under length-matched evaluation. Our
  `measurements/length_null.py` and the length-matched permutation strata
  (`a1ae5f7`) are the same control, reached independently. §2's licensing-vs-
  reading distinction and this are the same lesson from two directions.
- **Our lexical-transparency reclassification of `reverse_words` (§3.2) has a
  published name and a published consequence.** The same paper formalises a
  **vocabulary-transparency spectrum** and reports that *trajectory* features —
  hidden-state evolution across layers — are the pathway that retains signal on
  covert-intent inputs sharing vocabulary with normal text (0.721 avg AUROC,
  0.850 on jailbreak), precisely where single-cell scores collapse. Our three
  sound rungs sit on that side of the spectrum. **This promotes trajectory
  analysis from a cheap extra to a first-class instrument**, and it costs no new
  capture because §2.2's re-read already persists the full curves.
- **Our two capture positions have an established interpretation.** arXiv
  2507.11878 studies exactly `t_inst` (last instruction token) and `t_post-inst`
  (last sequence token) — our `instruction_final` and `last` — and finds the first
  clusters by the instruction's intrinsic **harmfulness** while the second
  clusters by the model's **behaviour**. That is a free interpretation of
  measurements this repo already takes, and it is why recognition (read at
  `instruction_final`) and behaviour are not redundant.
