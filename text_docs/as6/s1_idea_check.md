# S1 idea check — AS-6, guard internals (opened 2026-08-05)

**Workflow stage:** S1 in progress. Scoop check DONE and FINAL (Level 3, Medium Overlap — open); feasibility spike DONE at the config level. What remains for S1: owner sign-off on the design skeleton below, then S2.

**Scope of record** is `text_docs/proposal.md` §"AS-6 — the second paper" — that section holds the object cut, the delta statement, and the full prior-art record. This memo does not repeat them; it turns them into an experiment.

**Namespace note.** This file opens `text_docs/as6/`. AS-5's artifacts stay at `text_docs/` root for now (moving them is a separate, mechanical pass); `conf/experiment/` and `outputs/` get the same split when AS-6 lands its first config.

---

## 1. The question, in one paragraph

Every existing measurement of a guard's failure is an **end-to-end rate** — DSR, ASR, bypass rate. That number structurally confounds three different events: the guard never *received* the evidence, the guard received it but never *decoded* it, or the guard decoded it and never *converted* that into a block. Those three call for three different repairs, and no published work tells you which one you are looking at. AS-6 opens the guard and asks which link broke.

**v1 scope:** text guards (LlamaGuard-3-8B, WildGuard), two links — **decode** and **convert**. Transmit is held constant by construction (§6.4), not assumed away.

---

## 2. Core research questions

- **RQ1 (decode).** When a guard fails on an encoded harmful prompt, is the plaintext content recoverable from the guard's own activations during its classification forward pass?
- **RQ2 (convert).** Where content *is* recoverable, is harm represented — and does the guard nonetheless emit `safe`?
- **RQ3 (attribution).** Does the per-rung mix of (never decoded) / (decoded, not converted) / (blocked) explain the end-to-end DSR curve better than the DSR curve explains itself — i.e. do two guards with the *same* DSR drop fail by different links?

RQ3 is the one that makes this a paper rather than a measurement. If two guards with matching end-to-end numbers turn out to fail at different links, the end-to-end number is demonstrably the wrong instrument, and that is the contribution.

---

## 3. What the pilot already tells us — AS-5's dead band is AS-6's populated cell

This is the most important design inheritance, and it inverts a null result into an asset.

AS-5's phase-0 pilot found the **twelve cipher rungs inert at 7–9B**: ability ≈ 0, mean decode-and-restate similarity 0.03–0.30, map uniformly (R). For AS-5 that band is *uninformative* — "can't, didn't, or wouldn't?" has one answer there, `can't`, and the interesting distinctions are unmeasurable.

**For AS-6 the identical finding is a positive result.** The guards are the same scale (both 32 × 4096, 7–8B class). If they also cannot decode the cipher band, then "the guard never decoded it" is not a dead cell — it is *the answer to RQ1 on that band*, and it says the repair for those rungs is decode-side (a pre-decoder, as DecipherGuard built) rather than boundary-side. The rungs AS-5 must discard are rungs AS-6 can report.

**And the comprehension band is where AS-6's money cell lives.** On `zero_width` AS-5 measured mean similarity 0.987 with B=5–13, S=86–95 — targets overwhelmingly decode *and refuse*. The four exact comprehension-preserving rungs added 2026-08-04 (`tag_block`, `homoglyph`, `fullwidth`, `combining_marks`) plus `zero_width` are the band where a guard should be able to read the payload — so a guard that outputs `safe` there is a **convert** failure, and that band is behaviourally unmeasured in the literature (DecipherGuard tested base64 / Caesar / Zulu, not surface-preserving transforms).

**Prediction that follows, and it is falsifiable before any expensive work:** the per-link attribution should be *bimodal across the ladder* — cipher band dominated by never-decoded, comprehension band dominated by decoded-but-not-converted, with the transition at the same place the target's comprehension transition sits. *Refuted if* attribution is flat across the ladder, which would mean the links are not separable by encoding depth and RQ3 loses its independent variable.

---

## 4. Hypotheses

- **G1 (the decode link is real and measurable inside the guard).** On at least one rung, plaintext content is linearly recoverable from the guard's residual stream at the payload-final token while the guard outputs `safe`. *Refuted if* no rung licenses a content probe at all — in which case AS-6 reduces to the convert link alone and must reshape, because the decode link is what carries the delta (proposal §AS-6).
- **G2 (represent-without-blocking exists guard-side).** There are cells where harm is linearly readable in the guard's activations before layer 27 and the guard's verdict is `safe`. This is the guard-side analogue of the target-side result established by *Knowing without Acting* and *LLMs Encode Harmfulness and Refusal Separately*. *Refuted if* the harm probe and the verdict agree everywhere — which would say guards fail by not-seeing, never by not-acting, and would collapse the two-link frame into one.
- **G3 (the links dissociate across guards).** LlamaGuard-3 and WildGuard, at matched end-to-end DSR on the same payloads, fail by different link mixes. *Refuted if* the mixes are indistinguishable — a graceful degradation, not a dead paper: the attribution is still novel, it just stops being guard-specific.
- **G4 (attribution predicts the repair).** Rungs attributed to decode-failure are repaired by a pre-decoder and not by boundary work; rungs attributed to convert-failure are the reverse. This is the crossed test, and DecipherGuard's published 18.6pp decode-side recovery is a *partial prior confirmation of one arm* — so the novel content is the **null cells**, not the positive one. *Refuted if* a pre-decoder repairs convert-attributed rungs, which would mean the attribution is not tracking what it claims.

G2 and G4's null cells are what no prior paper attempts.

---

## 5. Where to read — sited from prior measurement, not guessed

Gamma-Guard (EMNLP 2025) measured, on Llama-Guard-3-8B, that clean and attacked safe/unsafe probability curves are **indistinguishable through layer 27 of 32** and diverge only in the final five layers. That gives AS-6 two readout sites without a search:

| Link | Position | Layers | Reads |
|---|---|---|---|
| **decode / represent** | payload-final token (inside the policy template) | < 27 | did the guard recover the content, and is harm represented |
| **convert** | last token (verdict formation) | 27–32 | did that become a `unsafe` verdict |

**The sharp consequence:** if no decoded-harm signal is findable before layer 27, the "decoded but not converted" cell **cannot exist as described**, and G2 is refuted by construction. That is a real, cheap, early falsification test — it should be the first thing the harness prints.

The feasibility spike (proposal §AS-6) found that `loader.instruction_final_offset` already lands on the payload's final token even when it is buried inside LlamaGuard's policy preamble, and that `last` is now the verdict-formation token. **The two positions the AS-5 pilot already captures map onto AS-6's two links without new position code.**

---

## 6. Instruments

### 6.1 Objects of study (the guards)

| | `allenai/wildguard` | `meta-llama/Llama-Guard-3-8B` |
|---|---|---|
| architecture | `MistralForCausalLM` | `LlamaForCausalLM` |
| layers × d_model | 32 × 4096 | 32 × 4096 |
| chat template | **none — needs a renderer** | policy template, ships |
| gated on HF | yes | yes |

*(Verified 2026-08-05 from the live HF configs, not recalled.)* Both are covered by `loader._LAYER_LIST_PATHS`, so `capture.residual_hooks` attaches unchanged.

### 6.2 The independent variable

The **same ladder AS-5 built**, ordered by surface conspicuousness — which is legitimate reuse within one repo, not a cross-repo dependency. Priority band for the first sweep: the comprehension band (§3), because that is where the convert cell lives; the cipher band is cheap to add and is now *informative* rather than inert.

### 6.3 The measurements

| # | Quantity | Instrument | Tri-state? |
|---|---|---|---|
| 1 | **Guard decode** — did the guard recover the content? | in-situ content probe on the guard's classification forward pass, payload-final token, layers < 27 | **yes — mandatory** |
| 2 | **Guard recognition** — is harm represented? | harmfulness direction fit harmful-encoded vs benign-encoded, same construction as AS-5 | **yes — mandatory** |
| 3 | **Guard verdict** — did it block? | the guard's own emitted `safe`/`unsafe` token, parsed | n/a (behavioural) |
| 4 | **End-to-end** — the number the field reports | DSR on identical payloads, re-measured in this harness | n/a |

Measurement 4 exists only so the attribution can be bound to a number a reviewer recognises. Sibling numbers are **hypotheses, never measurements** — their pipelines differ upstream, and stitching them would be a confound a reviewer would rightly kill.

### 6.4 Inherited instrument law — non-negotiable, and paid for already

AS-5 paid for these in blood (TODO items 6–7, commits `0483fe8` / `16c3349`). AS-6 inherits them at the first line of code rather than rediscovering them:

1. **Unmeasured ≠ negative.** An unlicensed probe returns `None`, never `False`. Scoring an unlicensed guard probe as "never decoded" would manufacture exactly the conclusion this paper exists to *distinguish* — it is the single most dangerous failure mode available to AS-6, because the false reading is also the headline-friendly one.
2. **Expect licensing to be rare.** AS-5's recognition probe licensed on **4 of 30** (model, rung) cells at the 0.70 AUROC cut. Design the corpus and rung set so a licensing failure is survivable, and print licensing status beside every number.
3. **Coherence rules split by behaviour, not by flag.** Recognition-without-deployment is a hard failure only when the cell *complied*; when it refused it is `surface_recognition`, a soft flag. Guard-side: harm-represented-without-decode is hard only where the guard emitted `safe`.
4. **Ability is an upper bound, not the regime.** Whatever "can this guard decode X" test we run out-of-band bounds what the forward pass *could* do; it never assigns the cell.

### 6.5 Controls

1. **Format decorrelation (the load-bearing one).** Fit harmful-encoded vs **benign-encoded**, so both classes traverse the identical pipeline and a direction reading "looks encoded" cannot separate them. Structural, as in AS-5 — do not build a separate benign control, it already exists.
2. **Prompt length bounded below the guard's inspection window, per guard, by construction** — and stated as a scope condition. Prompt Overflow (arXiv 2605.23196) shows transmit is *not* free in text; if length is unbounded, a convert-attributed failure may really be a transmit failure.
3. **Control tasks** (Hewitt & Liang 2019) — report the gap over shuffled-label controls, never raw AUROC.
4. **Template-token confound.** The guard's policy preamble dominates the rendered prompt and is identical across conditions, which helps — but the probe must be shown not to be reading template position. Cheapest check: shuffle the payload's position within the conversation block.
5. **Probe-generalisation honesty.** *False Sense of Security* (arXiv 2509.03888) is the objection a reviewer will raise. Claims are about natural transfer, not adaptive adversaries; say so.

---

## 7. Risks and contingencies

- **WildGuard's missing chat template is a silent-confound risk, not just a build task.** With no template shipped, its published instruction format must be reproduced exactly; a subtly wrong format degrades the guard's accuracy and every link attribution downstream inherits the error — and it would look like a finding. **Mitigation: reproduce its clean-input DSR against the published WildGuard numbers before any encoded rung is run.** If clean DSR does not reproduce, the renderer is wrong and nothing else is trustworthy.
- **Probes may not license on guards at all** (G1 refuted). Contingency: the paper becomes a convert-link paper, which the scoop check says is *weaker than the taken zone* — so this is the risk that most needs early evidence. It is also cheap to test.
- **Both repos gated.** Licence acceptance on the running account is unverified; check before scheduling GPU.
- **Guards are the same scale as AS-5's targets**, so cost should be in the same order as phase 0 — but `conf/cost.yaml` is still tuned to the badly optimistic pre-pilot assumptions (measured ~4.5 h Llama / ~8+ h Qwen against a predicted 0.7–2.4 h). **Re-tune before costing any AS-6 run**; do not quote the old model at the approval gate.

---

## 8. Open decisions (for the owner)

1. **Does AS-6 join the 8/12 AIA abstract registration?** The scoop verdict now makes this answerable. It is a free, non-binding placeholder; the paper will not have results by 8/21 and the full-paper wall is expected to lapse, exactly as for AS-5.
2. **Sweep band for the first guard run** — comprehension band only (cheap, holds the money cell), or comprehension + cipher (adds the never-decoded cell, which §3 argues is now informative). Recommendation: **both**, since the cipher band is what makes the attribution bimodal and therefore what makes RQ3 answerable.
3. **A third guard?** Two is the minimum for G3. GuardReasoner-VL is already served in the sibling but is a VLM and belongs to the declared extension, not v1.
