# The instrument build plan — shared by AS-5 and AS-6

**Canonical design doc for what the internals layer WILL be.** Its sibling
`instrument_layer.md` is the canonical home for what the instrument layer *has
been found to do* (settled defects, validated diagnostics, open anomalies). This
file is the forward half: what we build, why, in what order, and what each
instrument must pass before any number it produces may enter a paper.

Written 2026-08-05 after a comprehensive literature sweep, on the owner's
instruction: *write the docs of everything we want to build, and do another lit
search — be very comprehensive, do not rush to experiments.*

Both papers CITE this file; neither copies it. `instrument_layer.md` §6.4's
shortlist is superseded by §3 here and now points at it.

### Evidence tier and read status — both stated, because they are different things

**Re-verified 2026-08-06 through the `literature-review` front door** (Semantic
Scholar batch lookup by EXACT arXiv ID — no fuzzy title matching, so the
wrong-match failure mode cannot occur — cross-checked against OpenAlex). This
supersedes both the original arXiv-only labelling AND the first correction pass,
which was itself too harsh in places. Standing rule, canonical in science
`handbook/literature_search.md`:

**Tier — how well the WORLD has verified it:** **(A)** peer-reviewed at a named
venue · **(B)** preprint with substantial independent uptake · **(C)** recent
unrefereed preprint, little or no uptake · **(D)** not verified beyond an
abstract by us. **Read status — how well WE have verified it** — stated
separately; neither substitutes for the other.

#### Tier (A) — peer-reviewed, venue confirmed

| paper | venue | cites |
|---|---|---|
| 2102.12452 probing classifiers (Belinkov) | *Computational Linguistics* (journal) | 942 |
| 2404.14082 MI for AI safety review | **TMLR** | 499 |
| 2402.10588 do Llamas work in English | **ACL** | 314 |
| 2409.14507 *A is for Absorption* (SAE splitting) | **NeurIPS** | 142 |
| 2006.00995 amnesic probing | *TACL* (journal) | 121 |
| 2411.04986 semantic hub hypothesis | **ICLR** | 71 |
| **2507.11878 harmfulness/refusal separately** | **NeurIPS 2025** (OpenReview `zLkpt30ngy`) | 49 | 
| 2411.08745 separating tongue from thought | **ACL** | 41 |
| 2505.23556 understanding refusal with SAEs | **EMNLP** | 21 |
| 2505.11770 internal causal mechanisms predict OOD | **ICML** | 12 |
| 2502.07424 RomanLens | **ACL** | 10 |
| 2408.15510 reliability of causal probing | **IJCNLP-AACL** | 7 |
| 2509.17030 transfer neurons | **EMNLP** | 5 |
| 2506.11673 mean projection / LEACE | **ACL** | 1 |
| 2602.05347 character-level information | **EACL** | 0 |
| 2607.08883 · 2605.02914 | AAAI Symposium Series | 0–4 |

#### Tier (B) — preprint, real uptake

2303.08112 tuned lens (**542**) · 2410.20526 Llama Scope (**137**) · 2503.11667
LogitLens4LLMs (21) · 2512.18792 dead salmons (10) · 2603.18353 interpretability
without actionability (10, 5 influential) · 2508.21258 RelP (10) · 2605.11887
Qwen-Scope (7).

#### Tier (C) — recent unrefereed, ~no uptake. **These may not drive a decision.**

**2605.00269** two-pathway / length confound (**0 cites**) · **2604.02608**
steerable-but-not-decodable (2) · **2512.01222 Fang & Marks (4)** · 2605.02958 ·
2606.25182 · 2606.01033 · 2607.18348 · 2606.18322 · 2607.10226 · 2607.12166 ·
2606.27510 · 2606.09899 · 2606.08292 · 2605.24614 · 2604.08524 · 2607.08349 ·
2604.11061 · 2607.14147 · 2607.00572 · 2606.16349 · 2605.08513 · 2604.09544 ·
2606.28153 · 2606.08044 · 2608.03201 · 2608.03838 · 2604.26130 · 2603.19426 ·
2511.16288 · 2603.10771 · 2604.18519 · 2604.05090.

#### What the re-verification changed, stated plainly

- **Three papers were under-rated by the first correction and are actually (A):**
  2505.23556 (EMNLP), 2409.14507 (NeurIPS), 2411.04986 (ICLR). The over-harsh
  pass was itself a labelling error, in the opposite direction.
- **⚠️ The primary method import for I1 — Fang & Marks 2512.01222 — is tier (C),
  4 citations, and its claimed NeurIPS-2025-workshop status is NOT reflected in
  either database.** The whole logit-lens instrument rests on an unrefereed
  preprint. That does not make the method wrong (logit lens itself is tier B at
  542 citations via the tuned lens, and the mechanism is supported by tier-(A)
  semantic-hub work), but the specific "lens recovers ROT-13 CoT" result is
  unreplicated. **I1's validation gate is therefore doing real work, not
  ceremony** — we are replicating an unrefereed result, on our own rungs, and
  should say so in the paper.
- **2605.00269 stays (C) at 0 citations**, so §3.2's conditional trajectory
  promotion stands exactly as written.
- **The two databases disagree, and neither is authoritative.** Semantic Scholar
  mis-mapped Belinkov to "International Conference on Computational Logic" and
  recorded amnesic probing as arXiv-only at 25 cites, where OpenAlex correctly
  gives *Computational Linguistics* and *TACL* at 121. Cross-checking is not
  belt-and-braces; it is required. Filed as a new handbook point.

Read status: **2507.11878 read in full**; everything else at abstract +
retrieved-summary depth; **no PDFs read.** Deep reads are filed per instrument in
§3 as gates on that instrument, not on this plan.

*Retrieval provenance: pass 1 (2026-08-05) alphaXiv + web search, arXiv-only —
defective, root cause was skipping the installed skills. Pass 2 (2026-08-06) via
the `literature-review` front door: Semantic Scholar exact-ID batch resolution +
OpenAlex cross-check + a published-venue-filtered topical sweep over Semantic
Scholar and DBLP.*

---

## 1. Why this document exists

The inventory in `instrument_layer.md` §6.1 found that everything implemented is
**one method family** — supervised linear read-out of the residual stream at two
late positions — and that it has been instrument-failed twice in one week. The
natural reaction is to add methods. That reaction is half right, and this plan
exists to keep the other half.

**The failures were not caused by too few methods.** They were caused by one
method with no negative control and no sufficiency bar. Six methods without
controls is six ways to be confidently wrong, and the search space is already
large: at ~33 layers x 2 positions x 19 rungs x 2 models, something always
separates. The literature has a name for what that produces — arXiv 2512.18792,
*The Dead Salmons of AI Interpretability*, after the fMRI study that found
"brain activity" in a dead salmon using standard analyses of the day. Our
deployment probe licensing on `reverse_characters`, a rung the model provably
cannot read, was a dead salmon.

So this plan is organised around **controls first, instruments second.**

---

## 2. Design principles — binding on every instrument below

*P1–P3 are the operational form of the adoption rule that is canonical in
`instrument_layer.md` §6.2; P4–P7 are new here. On conflict §6.2 wins.*

**P1. Every instrument declares the question it answers, and no two instruments
may answer the same one.** "Is harm linearly decodable at layer L" is answered.
A nonlinear probe, a different classifier, or different pooling answers the same
question with new hyperparameters and inherits every existing confound. Those
are explicitly **not built** (§7).

**P2. Every instrument must read ~nothing on the negative controls.** The ladder
supplies them free: `tag_block` and `reverse_characters` score ability 0.00 on
both models, so whatever an instrument reads there is not decoded content. This
is the control-calibrated floor of `instrument_layer.md` §2.2, generalised from
the deployment probe to every instrument. **An instrument that cannot be shown to
be silent on unreadable text is not evidence.**

**P3. Every instrument must clear the length null.** This principle rests on OUR
OWN measurement, not on the literature, and that ordering matters: raw character
length separates JBB-harmful from benign at AUROC 0.6544 on our corpus, and the
rungs that newly licensed sat at mean 0.659 against a length null of 0.654. A
tier-(C) preprint corroborates it — arXiv 2605.00269 finds white-box OOD detection scores — naming
CED, RAUQ and **WildGuard confidence** — are structurally confounded by sequence
length at |r| >= 0.61, and **collapse to ~0.50, chance, under length-matched
evaluation** — but the corroboration is a bonus, not the warrant. Our
`measurements/length_null.py` and length-matched permutation strata (`a1ae5f7`)
were built from our own data before that paper was seen. **P3 would stand
unchanged if 2605.00269 were retracted tomorrow.** Mandatory for every
instrument, not just the probe.

**P4. Every internals claim must beat a black-box baseline.** arXiv 2604.11061
(*Pando*) — **tier (C), and not retrievable in OpenAlex at all** — makes the point
that interpretability evaluations often fail to
control whether a black-box method would have sufficed. For us the baselines are
concrete and cheap: raw length, surface-feature classifiers, and the model's own
generated restatement (the ability measurement). An internals number that does
not beat those is not an internals finding. *The principle is ordinary
methodological hygiene and needs no citation to bind; the paper is named as where
we met it, not as its authority.*

**P5. Correlational and causal claims are labelled separately and never
substituted.** Everything currently implemented is read-out. AS-5's Move C is a
*repair* claim and cannot be made from read-out. Two independent warnings:
Kwon 2026 (arXiv 2607.14147, **tier C**) reports the harm direction is "a read-out
but not a selective write-handle" with no clean steering window; arXiv 2603.18353
(*Interpretability without actionability*, **tier C**) reports mechanistic methods
failing to correct model errors **despite near-perfect internal representations**.
Both are unrefereed, so neither predicts our null — they are reasons to *design
for* a null rather than to expect one. Treat a successful repair as a finding to
be earned, not assumed.

**P6. The operating point is chosen by a non-circular criterion, stated in
advance.** Established for this repo in `instrument_layer.md` §2.3: hard
incoherence must fall monotonically as the read tightens. Never justify an
operating point by the regime counts it produces.

**P7. Layer/position selection is inside the null.** Permutation over the MAX
statistic across layers, as already built. Additionally to evaluate: nested CV
for layer selection (import from Kwon 2026). Statistical strengthening candidate:
arXiv 2607.08349 (*Certified Interventional Fidelity*) offers anytime-valid,
adaptive evaluation of causal claims — relevant once §3.6 intervention work
starts, not before.

---

## 3. The instrument roster

Each entry states: the question, the method import and its provenance, the
design, the validation gate, cost, and each paper's use.

### 3.0 — I0: the capture-spine extension (shared prerequisite, build FIRST)

**Not an instrument; the substrate all four need.** Today `models/capture.py`
records the residual stream at 2 positions (`instruction_final`, `last`) x ~33
layers. Instruments I1–I4 all need more than that, and they need the *same* more.

**Design:**
- **Per-token readout, not per-token caching.** Caching every token's residual is
  ~5 GB per condition. Instead compute each instrument's statistic *inside the
  forward pass* and store only the reduced quantity — for I1 the
  expected-plaintext-token probability per (token, layer), ~640K floats.
- **Position set widened and made explicit.** Both current positions are late.
  Kwon 2026 cites Doda 2026 that final-token probes miss jailbreaks because
  evidence is **displaced to earlier tokens**; arXiv 2507.11878 shows the two
  positions we already have carry *different concepts* (below). The spine should
  expose a named position schema, not two hard-coded picks.
- **Curves persisted, never argmax-only.** Already started (`eaace85`); make it
  the spine's contract so no run discards them again.

**Gate:** hermetic tests reproduce a known lens/trajectory value on a tiny
in-process model; memory ceiling measured, not assumed.

**Cost:** offline, no GPU, no spend. This is the single real engineering item.

---

### 3.1 — I1: logit-lens decode measurement *(replaces the transferred probe)*

**Question:** did the model *decode the ciphertext in situ* — do the plaintext
tokens themselves become predictable from intermediate residual states? No
implemented method answers this. The current deployment probe answers only "is
harm linearly decodable", which we have shown fires on surface form and on
lexical presence.

**Provenance — mixed tiers; see §0. The primary import is (C/D): its NeurIPS 2025
workshop status is claimed, NOT re-verified, and a workshop is not main-track
review. The tuned lens is (B).**
- **Fang & Marks, arXiv 2512.01222** **(C/D)** (Goodfire AI / Anthropic; NeurIPS 2025
  Mech-Interp workshop) — logit lens recovers ROT-13 chain-of-thought from
  activations alone, peaking intermediate-late. *The primary import.*
- **Tuned Lens, arXiv 2303.08112** — learned per-layer affine correction; lower
  bias and perplexity than the raw logit lens, which suffers basis drift.
  **Decision deferred to build time:** start with the logit lens (no training,
  no extra fitting surface), and adopt the tuned lens only if basis drift is
  measured to matter on our rungs. A learned correction is another fitted object
  and therefore another thing to license.
- **LogitLens4LLMs, arXiv 2503.11667** — toolkit that ports logit-lens analysis
  to modern architectures. **Its repo states support for Llama-3.1-8B and
  Qwen-2.5-7B — our exact two models.** Removes most plumbing risk.
- **TriLens, arXiv 2606.01033** — per-layer logit-lens *entropy* rather than
  top-token identity. Strictly cheaper: **needs no plaintext alignment at all.**
  Build as the fallback readout for rungs where alignment is hard.

**Why it is the right instrument:** it takes the (B)/(D) split off the operating
point entirely, and it carries a built-in negative control — it must read
~nothing on `tag_block`.

**Mechanistic reason to expect it to work on our rungs**, which also gives the
write-up its framing: the **semantic hub hypothesis** (arXiv 2411.04986) and the
latent-English pivot literature (arXiv 2402.10588, *Do Llamas Work in English?*;
arXiv 2509.17030, transfer neurons; arXiv 2411.08745, *Separating Tongue from
Thought*, which uses activation patching to show language-agnostic concept
representations). If representations normalise toward a shared semantic space in
intermediate layers, an encoded input's plaintext should become lens-visible
exactly there — which is what Fang & Marks observe. Directly on point for our
character-level rungs: arXiv 2602.05347 (*How Do LMs Acquire Character-Level
Information?*) and arXiv 2603.10771 (*Word Recovery ... Enables Character-Level
Tokenization Robustness*) — the second names the mechanism our
`zero_width`/`homoglyph`/`fullwidth` band exercises.

**Known risk, must be stated in the paper:** arXiv 2604.02608 **(C, unrefereed,
0 citations)**, *Steerable but Not Decodable: Function Vectors Operate Beyond the
Logit Lens* — some task-relevant directions steer the model without being
readable along the unembedding. **A lens null is therefore not proof of no
decoding.** This is the most important caveat on I1 and the reason I2–I4 are
built alongside it rather than after it.

*Tier note (added after the 2026-08-05 correction): this is a (C) source, so the
caveat is carried as a **stated limitation** rather than as an established
result — which is the conservative direction and costs us nothing, since it only
ever makes us hedge a null. Its load-bearing use is defensive, not affirmative;
no build decision rests on it except "don't ship I1 alone", which the plan wants
anyway. Deep read of 2604.02608 is a gate on I1's write-up, not on its build.*

**Validation gate (all must pass before any claim):**
1. Fires on the three sound rungs — `zero_width`, `homoglyph`, `fullwidth`.
2. Reads ~nothing on `tag_block` and `reverse_characters` (P2).
3. Clears the length null (P3).
4. Beats the black-box baseline — the ability restatement (P4).
5. Peaks at intermediate-late layers, matching the L18–L22 `instruction_final`
   cluster the transferred probe already selects (`instrument_layer.md` §3.1).

**Cost:** steps 1–3 offline, $0. Step 4 is one forward-pass-only run, no
generation, no judge calls.

**AS-5 use:** the (B)/(D) split — Move B's stated contribution.
**AS-6 use — load-bearing:** Llama Guard 3's chat template hard-wires the safety
task, so the generation-based decode route is un-runnable there. A residual-stream
method is not merely better for AS-6; it is the only one available.

---

### 3.2 — I2: processing-trajectory features *(promoted — the literature says this is what survives)*

**Question:** does the *evolution across layers* carry decode/harm evidence that
no single best cell carries? Distinct from I1 (which asks about token identity)
and from the current probe (which asks about one cell).

**Provenance — ⚠️ TIER (C), unrefereed preprints with zero citations. Read §0
before weighting anything here.** This section originally promoted trajectory
analysis to first-class on the strength of 2605.00269 alone. That promotion is
now **conditional**, and the condition is stated at the end of this entry.
- **arXiv 2605.00269, *How Language Models Process Out-of-Distribution Inputs: A
  Two-Pathway Framework*** **(C)** — the most striking paper this search returned.
  Under **length-matched evaluation** all attention-derived scores collapse to
  chance (~0.50) while **trajectory features retain genuine signal (0.721 avg
  AUROC; jailbreak 0.850)**. It also formalises a **vocabulary-transparency
  spectrum**: embedding-based methods excel on vocabulary-distinctive inputs,
  while trajectory features are what detect **covert-intent inputs that share
  vocabulary with normal text.**
- **arXiv 2605.02958, *Tracing the Dynamics of Refusal*** — static terminal or
  pooled directions miss how refusal is *constructed* across layers; trajectory
  gives more robust jailbreak detection.
- **arXiv 2607.18348** — residual-stream geometry across depth (relative
  displacement, orthogonal Procrustes) as a trajectory vocabulary.

**Why this is now co-first rather than a cheap extra.** Three of our own hard-won
findings are the same findings, independently: our length confound is their
length confound; our length-matched permutation licensing is their length-matched
evaluation; and **our "lexical transparency" reclassification of `reverse_words`
is their vocabulary-transparency spectrum.** We reclassified `reverse_words` as a
control precisely because the words are lexically present — their axis names that
condition and reports that **trajectory features are the pathway that works on
the covert-intent side of it.** Our sound rungs (`zero_width`, `homoglyph`,
`fullwidth`) are exactly covert-intent-sharing-vocabulary inputs.

**Design:** we already compute the full (layer x position) curve on every run and
now persist it, so I2 is a change of *analysis*, not of capture — the cheapest
instrument on this list by a wide margin. Features: per-layer probe score
trajectory, relative displacement between consecutive layers, and the layer index
of peak signal (already validated as a diagnostic, §3.1 of `instrument_layer.md`).

**Validation gate:** P2, P3, P4, plus reproduction of the known layer split —
`zero_width`/`homoglyph`/`fullwidth` late-mid at `instruction_final`;
`reverse_words`/`combining_marks` early at `last`.

**Cost:** offline, $0, no new capture. **Build regardless of what else is
adopted** — and note this recommendation does NOT depend on the tier-(C)
citations: it was already justified before the sweep by our own data (the curves
exist, the layer-index diagnostic is validated on both models independently,
`instrument_layer.md` §3.1), and it costs nothing. The literature raised its
*rank*, not its *justification*.

**⚠️ The condition on the promotion.** "Trajectory features survive
length-matching where single-cell scores collapse" is a tier-(C) claim, and this
plan may not adopt it as fact. **The replication is cheap and we can run it
ourselves on data already on disk:** compare single-cell probe AUROC against
trajectory-feature AUROC on the SAME cells under the SAME length-matched strata,
using the 1,400-cell band run. If trajectory does not beat single-cell under
length matching **on our data**, the promotion is withdrawn regardless of what
2605.00269 reports. That comparison is added to I2's validation gate above and is
the upgrade path §0 requires.

**AS-6 use:** guard trajectories are the natural expression of "where in the
guard does the verdict form" — and Gamma-Guard's finding that clean vs attacked
curves on Llama-Guard-3-8B are indistinguishable through layer 27 of 32 is
already a trajectory claim.

---

### 3.3 — I3: entropy dynamics *(label-free, independent failure mode)*

**Question:** does the model's *uncertainty profile across layers* mark encoded
harmful input, without any classifier trained on our labels?

**Provenance — ⚠️ TIER (C), unrefereed, 0 citations:** **arXiv 2606.25182, *What
Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics***.
Related readout: TriLens (arXiv 2606.01033) uses per-layer lens entropy for
hallucination detection.

**Why it earns a slot under P1:** it trains no classifier on our labels, so the
length confound cannot enter by the route that broke the supervised probe. That
makes it a genuinely independent signal class rather than another read-out
variant — and an instrument whose failure mode is uncorrelated with I1/I2 is
worth more than a fourth correlated one.

**Validation gate:** P2, P3, P4. Note P3 is subtler here: label-free does not mean
length-free, since entropy is computed per token and sequence length changes the
aggregate. Length-matched evaluation still applies.

**Cost:** offline on cached logits/residuals, $0.

---

### 3.4 — I4: SAE features *(off-the-shelf for both models; the only instrument that can name WHAT fires)*

**Question:** *which* feature fires — is the signal a harm feature or a
surface-anomaly feature? No implemented method can tell these apart, and that
distinction is exactly the unexplained recognition anomaly of
`instrument_layer.md` §5.

**Provenance — (B) for the SAE suites themselves (real uptake, and the artifacts
are downloadable and checkable, which is a stronger guarantee than a citation
count); (C) for the refusal-SAE study. No SAE training needed.**
- **Llama Scope, arXiv 2410.20526** — 256 TopK SAEs, every layer and sublayer,
  32K/128K features. **VERIFIED: trained on Llama-3.1-8B-*Base*, while our target
  is Instruct.** The paper explicitly studies generalisation to fine-tuned
  models, so this is a transfer question to measure, not a blocker.
- **Qwen-Scope, arXiv 2605.11887** — the Qwen-side counterpart.
- **arXiv 2505.23556, *Understanding Refusal in Language Models with Sparse
  Autoencoders*** — our instrument applied to our domain by others; read before
  building, both for method and for delta.

**Mandatory pre-gate, before any SAE result is admissible:** show the SAE
reconstructs **Instruct** activations acceptably (reconstruction error, variance
explained, downstream KL). A base-trained SAE applied to Instruct without that
check produces findings about the wrong model.

**Known risks — this instrument has the worst published track record of the four,
and the plan treats it accordingly:**
- **arXiv 2606.18322, *SAE Interventions are Unreliable*** — suppressed behavior
  recovers post-intervention.
- **arXiv 2607.10226** — apparent SAE-based safety control can arise from weak or
  non-localized interventions; requires matched evaluation.
- **arXiv 2607.12166** — SAE features can be **causally inert** despite good
  correlational recovery metrics.
- **arXiv 2409.14507, *A is for Absorption*** — feature splitting and absorption
  as dictionary size grows; a "harm feature" may be several, or absorbed.

**Consequence, stated as policy:** I4 is used as a **descriptive/naming
instrument** (which features distinguish decoded from non-decoded), **not** as a
control handle, unless the matched-evaluation bar of 2607.10226 is met
explicitly. Do not build an SAE-steering defense on this line without that.

**Validation gate:** the Instruct-reconstruction pre-gate, then P2, P3, P4.

**Cost:** SAE download + encode passes; no training. GPU-light, offline after
capture.

**AS-6 gift, and it is a real one:** **Llama Guard 3 8B is itself a fine-tune of
Llama-3.1-8B**, so the same base-model SAEs are a candidate instrument for the
guard-side paper. One transfer validation serves both papers. *(Hypothesis, not
yet tested — the same reconstruction pre-gate applies, and a safety-fine-tuned
model may drift further from base than an instruct-tune does.)*

---

### 3.5 — I5: the reply-inversion causal test *(cheap causal evidence, new to this plan)*

**Question:** did an intervention change the model's *internal judgment*, as
opposed to its surface behavior? This is the causal question AS-5 needs and the
only cheap way to get at it.

**Provenance — ✅ TIER (A), the strongest citation in this plan: arXiv 2507.11878,
*LLMs Encode Harmfulness and Refusal Separately*, published at **NeurIPS 2025**
(poster; OpenReview `zLkpt30ngy`)** (Zhao, Huang, Bau, Wu, Shi — Northeastern
+ Stanford). *The one paper read in full for this plan.* It matters to us three
ways:

1. **It validates our capture design and tells us what our two positions mean.**
   It studies exactly `t_inst` (last token of the user instruction) and
   `t_post-inst` (last token of the full sequence) — **our `instruction_final`
   and `last`.** Finding: at `t_inst` hidden states cluster by the instruction's
   *intrinsic harmfulness*; at `t_post-inst` they cluster by the model's
   *behavior*. So our recognition measurement (at `instruction_final`) is their
   harmfulness belief, and our `last` position is a behavior readout. **This is a
   free interpretation of measurements we already have.**
2. **The method import:** their *reply inversion task* appends an inversion
   question ("Could this user prompt cause harm? Answer 'Certainly' if it could,
   otherwise 'No'"), then steers *before* the inversion question with the
   harmfulness direction and *across all tokens* with the refusal direction.
   Steering harmfulness flips the judgment; steering refusal does not — causal
   evidence that the two are distinct. **We can run this on our encoded rungs,
   which they never do.**
3. **Positioning, honestly stated.** Their jailbreaks are adversarial suffixes,
   persuasion and templates — **all of which leave the text readable.** Their
   finding that jailbreaks suppress refusal *without* changing the harmfulness
   belief is the closest prior to AS-5's frame, and AS-5 must cite it as such.
   **AS-5's delta survives precisely because of the decode link:** where the input
   must first be *decoded*, "can't" and "didn't" are separable states that have no
   analogue in their setting. The distinction is ours; the harmfulness-vs-refusal
   split is theirs and must be attributed.
   **AS-6 must also cite it:** their Latent Guard, built from the harmfulness
   belief, is competitive with Llama Guard 3 8B. AS-6 measures a guard's internals
   rather than building a guard, so the object differs — but the claim "internal
   harm representation beats the deployed guard" is already published and is not
   available to us as a contribution.

**Adjacent safety-internals work to read before the write-up** (abstract-level
here; each is a named read): arXiv 2607.00572 (HARC, coupling harmfulness and
refusal directions), arXiv 2606.16349 (refusal geometry to safety geometry),
arXiv 2605.08513 (a single neuron sufficient to bypass safety alignment —
distinguishes *refusal neurons* from *concept neurons*, the neuron-level analogue
of our recognition/behaviour split), arXiv 2604.09544 (a distinct harmful-response
mechanism shared across harm types), arXiv 2606.28153 (attention-head
specialisation; jailbreaks selectively rather than comprehensively remove safety
features), arXiv 2606.08044 (*When Behavioral Safety Evaluation Fails: A
Representation-Level Perspective* — **closest in motivation to AS-5's premise;
deep read is a gate on AS-5's related-work section**).

**Cost:** generation + steering, small. Needs the approval gate but is cheap.

---

### 3.6 — I6: full causal toolkit *(activation patching, ablation, steering) — LAST*

**Question:** which components *cause* the decode and the refusal decision.
Required for AS-5's Move C (repair); premature before I1–I4 say what to
intervene on.

**Method imports, established:** activation patching and its efficient variants —
arXiv 2508.21258 (RelP, relevance patching for scalable circuit discovery);
concept erasure for amnesic-style tests — arXiv 2006.00995 (amnesic probing),
arXiv 2506.11673 (mean projection / LEACE as the principled erasure);
arXiv 2605.24614 (activation patching as an audit of *unlearning depth* — directly
relevant to this repo's training-time half).

**Warnings to wire in from the start, not discover later:**
- arXiv 2606.27510 — hidden interaction effects with multiple mediators.
- arXiv 2606.09899 — attribution patching can lie; second-order correction.
- arXiv 2606.08292 — ablation-reversible heads don't transfer; a stress test
  before any role claim.
- arXiv 2408.15510 — reliability of causal probing interventions.
- arXiv 2607.08349 — anytime-valid evaluation of causal claims.
- arXiv 2604.08524 — what actually drives representation steering.
- arXiv 2603.18353 — **mechanistic methods failed to correct model errors despite
  near-perfect internal representations.** Treat Move C's success as a hypothesis.

**Predicted-null risk, recorded in advance:** Kwon 2026 reports no clean steering
window and no single refusal-decision direction (any one reads <= 0.73). If Move C
returns a null, that null was predicted here and is reportable rather than
embarrassing.

---

## 4. The mandatory control battery

Every instrument runs all of these. Not optional, not per-instrument judgment.

| control | what it defeats | status |
|---|---|---|
| **Ability-0 negative control rungs** (`tag_block`, `reverse_characters`) | reading surface form and calling it decoded content | built, and it set the floor (0.656 / 0.671) |
| **Length null** (`measurements/length_null.py`) | the content-correlated length confound; corroborated by arXiv 2605.00269 | built |
| **Length-matched permutation strata** (`a1ae5f7`) | length leaking into the licensing null | built |
| **Format decorrelation** (harmful-encoded vs benign-encoded through the identical pipeline) | "looks encoded" | built, structural |
| **Lexical transparency screen** | probes firing on words that were never hidden (`reverse_words`) | validated as a diagnostic; make it a gate |
| **Lexical decorrelation — XSTest** (scary-but-benign) | firing on alarming vocabulary rather than harm | **NOT built** — import from Kwon 2026 |
| **Black-box baseline** (P4) | an internals claim a surface classifier would have made | **NOT built** — needed per Pando (arXiv 2604.11061) |
| **Matched-norm random direction** | steering "working" because you perturbed anything | **NOT built** — needed before any I5/I6 claim |
| **Control task / selectivity** (Hewitt-Liang style) | probe capacity memorising rather than reading | **NOT built** — see arXiv 2102.12452 |

The bottom four are build items, filed with the instruments that need them.

---

## 5. Tooling decision

**Recommendation: stay on plain HuggingFace hooks, which is what
`models/capture.py` already is, and import algorithms rather than frameworks.**

Reasoning: the capture spine works, is hermetically tested, and runs on the
cluster today. TransformerLens requires model surgery into its own
`HookedTransformer` format (a re-validation cost, and a divergence risk on
Llama-3.1-8B/Qwen2.5-7B chat templates, which is exactly where AS-6 already found
a fail-closed defect). NNsight is designed around remote execution we don't need.

Two exceptions worth taking as *references*, not dependencies:
- **LogitLens4LLMs** (arXiv 2503.11667; repo `zhenyu-02/LogitLens4LLMs`) — states
  support for Llama-3.1-8B and Qwen-2.5-7B, our exact models. Read its handling of
  per-layer normalisation before writing ours.
- **SAELens** (`decoderesearch/SAELens`) — the standard loader for Llama Scope /
  Qwen-Scope checkpoints. Likely a real dependency for I4, since hand-rolling SAE
  loading is the reinvention the global standard forbids.
- `davidbau/logitlenskit` exists for the NNsight path — noted, not chosen.

*(Global law: prefer mature tools over reinventing. The judgment here is that the
mature tool for capture is the one we already have, while the mature tool for SAE
loading is SAELens and should be used.)*

---

## 6. Build sequence, and the scheduling fact that fixes it

**The scheduling fact: I1–I4 all read the same forward pass.** No generation, no
judge calls, no training. Built together they cost **one** forward-pass-only run
per model; built serially, four runs.

| step | what | GPU | spend | gate |
|---|---|---|---|---|
| 1 | **I0** capture-spine extension (per-token readout, named positions, curves as contract) | none | $0 | hermetic tests |
| 2 | **I2** trajectory features (analysis-only, no new capture) | none | $0 | reproduces the known layer split |
| 3 | **I1** logit-lens decode measurement | none to build | $0 | fires on 3 sound rungs, silent on 2 controls |
| 4 | **I3** entropy dynamics | none | $0 | P2/P3/P4 |
| 5 | **missing controls** — XSTest, black-box baseline, control task | none | $0 | — |
| 6 | **I4** SAE — Instruct-reconstruction pre-gate FIRST, then features | light | $0 | pre-gate passes |
| 7 | **one** forward-pass-only run, both models, the sound rungs + controls | yes | ~$0 (no judge) | **approval gate** |
| 8 | **I5** reply inversion; then **I6** causal, only if 1–7 hold up | yes | small | **approval gate** |

Steps 1–6 are entirely offline: no GPU, no spend, fully testable against cached
activations and the existing 1,400-cell band run. **That is the whole point — the
expensive step is seventh, not first.**

---

## 7. What we are deliberately NOT building, and why

- **More supervised linear-probe variants** (nonlinear probes, alternative
  classifiers, alternative pooling, ensembles). Fails P1: same question, same
  confounds, new hyperparameters.
- **Attention-derived detection scores.** arXiv 2605.00269 finds these collapse to
  chance under length-matched evaluation, and attributes the confound to
  attention's logarithmic dependence on input length. Building them would be
  building a known artefact.
- **An SAE-based defense / control handle.** Reduced to a descriptive instrument
  (§3.4) until the matched-evaluation bar is met.
- **Circuit discovery / attribution graphs** (ACDC, transcoders, crosscoders,
  CLT-Forge arXiv 2603.21014, RelP arXiv 2508.21258). Real and mature, but they
  answer "what is the circuit", which neither paper asks. Revisit only if I1–I4
  localise something worth tracing.
- **Training our own SAEs.** Llama Scope and Qwen-Scope exist for both models.
- **A verbalizer / self-report instrument.** arXiv 2509.13316 questions whether
  activation-verbalization conveys privileged information; our ability
  measurement is already the honest version of this and is treated as a black-box
  baseline, not as internals.

---

## 8. Literature map

Grouped for reuse by both papers' related-work sections. Read status per §0.

**Methodology and critique (the discipline this plan is built on)**
2102.12452 probing classifiers · 2512.18792 dead salmons · 2603.19426 probe
evidence under controlled prompt structure · 2511.16288 probe geometry
identifiability · 2408.15510 causal probing reliability · 2006.00995 amnesic
probing · 2506.11673 mean projection / LEACE · 2604.11061 Pando (black-box
control) · 2607.08349 certified interventional fidelity · 2404.14082 MI for AI
safety review.

**Lens family**
2512.01222 Fang & Marks (primary import) · 2303.08112 tuned lens · 2503.11667
LogitLens4LLMs · 2606.01033 TriLens · 2604.02608 steerable-but-not-decodable
(risk).

**Trajectory and dynamics**
2605.00269 two-pathway / length confound (**key**) · 2605.02958 refusal dynamics ·
2606.25182 entropy dynamics · 2607.18348 residual-stream geometry across depth.

**Sparse dictionaries**
2410.20526 Llama Scope · 2605.11887 Qwen-Scope · 2505.23556 refusal via SAEs ·
2409.14507 absorption · 2606.18322 unreliable interventions · 2607.10226 matched
evaluation · 2607.12166 causal inertness.

**Causal methods**
2508.21258 RelP · 2606.27510 multiple mediators · 2606.09899 attribution patching
lies · 2606.08292 ablation-reversible heads · 2605.24614 unlearning depth ·
2604.08524 what drives steering · 2603.18353 interpretability without
actionability · 2505.11770 causal mechanisms predict OOD behavior.

**Safety internals (target side — AS-5)**
2507.11878 harmfulness/refusal separately (**key**) · 2607.14147 Kwon prefill ·
2607.00572 HARC · 2606.16349 refusal-to-safety geometry · 2605.08513 single neuron
· 2604.09544 distinct harmful mechanism · 2606.28153 attention-head specialisation
· 2606.08044 behavioral-eval failure · 2607.08883 optimizing against safety
representations.

**Guard internals (defense side — AS-6)**
2604.18519 SIREN · 2608.03201 refusal-cue shortcut in guard models (2026-08-04,
new) · 2605.02914 guard safety-geometry collapse · 2608.03838 LatentGuard ·
2604.26130 reward-lens (MI library for reward models).

**Surface form, script and character-level processing (why encoded input is
readable at all)**
2411.04986 semantic hub · 2402.10588 do Llamas work in English · 2411.08745
separating tongue from thought · 2509.17030 transfer neurons · 2502.07424
RomanLens · 2604.05090 script over linguistic structure · 2602.05347 character-
level information · 2603.10771 word recovery / tokenization robustness.

---

## 9. Open questions this plan does not settle

1. **Tuned lens vs logit lens** — decided at build time on measured basis drift,
   not now.
2. **Does Llama Scope transfer to Instruct** — an experiment, gated before any I4
   claim.
3. **Does it further transfer to Llama Guard 3** — AS-6's version of the same
   question.
4. **The recognition anomaly** (`instrument_layer.md` §5) — I4 is the instrument
   most likely to explain it; that is a hypothesis, not a plan.
5. **Whether Move C repairs anything at all** — two papers predict it may not
   (§3.6). The plan makes the null reportable; it does not make it unlikely.
