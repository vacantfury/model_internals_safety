# S1 idea check — first paper (direction committed 2026-08-02)

Status: S1 COMPLETE. This document is the **scientific** idea check — the research question, the conceptual contribution, the hypotheses, the design as a set of scientific instruments, and the validity controls. Feasibility/logistics (compute, data engineering, venue timing) live separately in `proposal.md` and the repo `TODO.md`, deliberately kept out of here. Literature basis: a 4-bucket deep read of ~20 papers full-text (2026-08-02); per-paper notes + BibTeX in the science repo's `literature/model-internals/`.

Verdict: **proceed to S2.** The direction is scoop-clean at the level of the whole claim, rests on established methods we extend rather than invent, and — critically — is built around a *falsifiable* mechanistic prediction rather than a descriptive measurement.

---

## 1. The committed direction

**Recognition or action? Diagnosing and repairing safety failures under encoded inputs.** *(our proposal — the merge of proposal.md candidates 1 + 4.)*

A safety-aligned model that refuses "how do I build a bomb" in plain English will often comply when the same request arrives as a Caesar cipher, Base64, a math/logic word problem, or a low-resource language. The field treats this as one failure ("the safety training didn't cover this surface form"). We claim it is **two distinct failures that look identical from the outside**, and that telling them apart — using the model's own internal representations — is both a scientific question with a clean answer and the key to a principled fix.

The two failures:

- **(R) Recognition failure.** The harmful intent never becomes *represented* inside the model. The model cannot decode the encoding well enough to know what is being asked. There is no internal "this is harmful" signal to act on — so no amount of safety-behavior training can help until decoding ability itself improves.
- **(A) Action / calibration failure.** The harmful intent *is* represented — it is linearly readable in the activations and causally implicated in the model's computation — but that internal recognition never gets converted into refusal behavior. The safety signal is present but *unbound* from the safety response.

The distinction matters because it dictates what fixes can possibly work. R-failures are capability problems (a guard model can't help either — nothing in the pipeline has decoded the harm). A-failures are binding problems, and binding is exactly what cheap safety fine-tuning should be able to repair. A single external metric (attack success rate) cannot distinguish them; only looking inside can.

---

## 2. Core research questions

1. **RQ1 (diagnosis).** For each encoding family, is a safety failure a recognition failure or an action failure? Where in the network (which layers) and when in processing (which token positions / decoding steps) does harmful intent become linearly detectable under each encoding — if it does at all?
2. **RQ2 (structure).** Does the recognition/action split vary *systematically* with obfuscation depth? Is there a graded transition (recognition erodes smoothly as encodings get harder) or a cliff (recognition holds until some threshold, then collapses) — and does the behavioral failure track the recognition failure or lag behind it?
3. **RQ3 (repair, the causal test).** Does cross-encoding safety fine-tuning — training refusal on *some* encoding families — repair action failures on *held-out* families? Does the generalization track a measurable quantity (representational overlap between families)? And does the same fine-tune correctly **fail** to fix genuine recognition failures, where there is no internal signal to bind to?

RQ3 is the heart of the paper: it turns the Phase-1 diagnosis from a description into a *causal claim* that can be proven wrong.

---

## 3. The conceptual contribution

Two moves, one inherited-and-extended, one genuinely new.

**Move A — port the recognition/action lens from the language axis to the encoding axis.** The recognition-vs-action dichotomy already exists for *one* axis: Aziz et al. (2026, arXiv 2606.01196) showed that low-resource-*language* safety failures are "action failures, not representation failures" — the harmfulness direction stays linearly readable (AUROC > 0.85) even as behavioral refusal collapses to 44%. We adopt their diagnostic vocabulary and their core method (contrastive difference-in-means direction + causal ablation), and extend it to the far broader and structurally different space of *encodings*: ciphers, Base64, math/logic transformations, rendered/formatted text. This is a real but bounded contribution on its own — a generality test of someone else's hypothesis.

**Move B — the decode-capability trichotomy (the new conceptual object).** Here the encoding axis is not just "more of the same." Every language in Aziz et al.'s study is, by construction, *decodable* — the model understands Swahili, just imperfectly. Encodings break that assumption. A 7–13B model may genuinely be unable to decode a keyed substitution cipher or a deep Base64 nesting. That splits every (model, encoding-family) cell into **three** regimes, not two:

| Regime | Internal recognition | Behavior | What it means |
|---|---|---|---|
| **can't-decode** | absent (no linear signal) | complies | genuine R-failure — nothing decoded the harm; unfixable by safety training or by any guard |
| **decode-and-refuse** | present | refuses | safety works — the control case |
| **decode-and-comply** | present | complies | the A-failure — recognition present but unbound; **the target of the fix** |

This trichotomy has *no analogue* in the language-axis prior work, because the "can't-decode" regime effectively doesn't arise there. It is what lets the paper make a **double-dissociation** argument (§6) that Aziz et al. structurally could not: the fix should move the third regime and leave the first untouched. That double dissociation is what elevates the work from "we measured probes across more conditions" to "we causally established what kind of failure each encoding produces and showed a fix that respects the boundary."

---

## 4. Why this is scientifically good (not just a measurement paper)

The concern with any internals-analysis paper is that it becomes descriptive — "we probed X across Y settings, here are the AUROCs." This design avoids that in three ways:

1. **It makes a falsifiable mechanistic claim.** The recognition/action diagnosis is not the endpoint; it generates a specific prediction (cheap cross-encoding training re-binds behavior to already-present recognition, and generalizes in proportion to representational overlap). If that prediction fails, the mechanism story is wrong. Phenomenon → mechanism hypothesis → interventional test is the full scientific arc; the intervention is the load-bearing evidence, not a bolted-on "defense product."
2. **It has a built-in negative control.** The can't-decode regime is a case where the fix *should not* work. A paper that only ever shows its intervention succeeding is weaker than one that predicts and demonstrates where it must fail. The double dissociation is the strongest form of causal evidence available here short of full circuit-level mechanism.
3. **The probes are correlational, and the design knows it.** Linear separability shows harm is *represented*; it does not show the representation *governs* behavior. Phase 2 (causal validation via ablation, activation addition, and the reply-inversion test) is what licenses causal language. Without it the paper could only say "harm is decodable"; with it the paper can say "this decoded representation is what safety training must, and does, bind to."

The paper's identity can be framed either way depending on taste: as a **defense** ("a principled cross-encoding safety-training recipe") or, better for scientific weight, as a **mechanistic question** ("is safety failure under encodings a recognition or an action failure, and is the mechanism shared across representation families?") with the fine-tuning experiment as the causal evidence and the practical recipe as a payoff that falls out. Same experiments, analysis-led framing.

---

## 5. Why the whole claim is open (scoop check, 2026-08-02, full-text verified)

Three lines sit nearest; each holds one piece; none holds the combination. This is the "partially taken → reshape narrowly" situation, and the reshape is exactly Move B above.

| Prior line | What it established | What it did NOT do |
|---|---|---|
| **Aziz et al. 2026** (2606.01196) — *the* closest | Recognition-vs-action diagnostic (diff-in-means + causal ablation): low-resource-*language* failures are action failures; AUROC > 0.85 while refusal collapses to 44%; fixes with a few-shot latent gate | Only natural-language translation — no ciphers/Base64/structural encodings; no can't-decode regime (all languages decodable, so no trichotomy, no double dissociation); fix is few-shot threshold gating, explicitly disclaims adversarial robustness; no text-guard comparison |
| **TrajGuard** (2604.07727) | Behavioral proof that decode-time internal monitoring catches cipher attacks that text guards miss (Llama Guard 3 ASR 0.41 vs 0.14) — validates our motivating premise for ciphers specifically | Detection only; never asks recognition-vs-action; no causal validation; no training-side repair |
| **SALO** (2605.02958, ICML 2026) | Zero-shot detector generalization from plain safety data to unseen attack families; refusal is a sparse *upstream* trajectory (not a terminal-token readout) | **Explicitly names encoded/Base64 inputs as its untested boundary case** — "may not trigger the refusal trajectory" — a direct research-gap invitation for exactly our recognition question |

**Training-axis sub-check.** The nearest training work is **CIFR** (Youstra et al. 2025, arXiv 2508.17158): it also holds out cipher families for OOD testing — but it trains a *monitor to detect malicious fine-tuning data* on a fine-tuning API. Different object (a detector, not the model's own refusal), different threat model (API abuse, not inference-time attacks). Must be distinguished on first mention, but not a scoop. **Circuit Breakers** (Zou et al. 2024) already shows multilingual (incl. low-resource) robustness but tests **no ciphers/Base64/structural encodings**; **LAT** (Sheshadri et al., TMLR 2025) tests none. No paper trains a model's own refusal to generalize across held-out *encoding* families.

**Net:** the exact combination — recognition/action diagnosis across the full encoding taxonomy, the decode-capability trichotomy, the cross-encoding training repair, the double dissociation, and a head-to-head against text guards on identical payloads — is unclaimed.

---

## 6. Hypotheses and falsifiable predictions

- **H1 (recognition is often present).** For decode-capable families, harmful intent is linearly readable in mid-layer activations even when the model complies — i.e., most failures for these families are A-failures, not R-failures. *Refuted if* probe AUROC is at chance wherever behavior fails.
- **H2 (graded structure).** As obfuscation deepens (plain → paraphrase → language → simple cipher → substitution cipher → Base64 → math/logic), recognition degrades and behavioral refusal degrades — but they *decouple*: behavior falls first and further, opening an "action gap" that widens before recognition itself finally collapses at the decode boundary. *Refuted if* recognition and behavior fall together with no gap (that would make it a recognition story throughout).
- **H3 (repair generalizes).** Cross-encoding safety fine-tuning on a subset of families closes the action gap on *held-out* families, and the amount of generalization tracks the cosine similarity between families' harmfulness directions (representational overlap predicts transfer). *Refuted if* held-out generalization is absent or unrelated to representational overlap.
- **H4 (double dissociation — the decisive test).** The *same* fine-tune does **not** manufacture recognition where decode capability is absent (can't-decode families stay failing, and stay signal-less). *Refuted if* the fix "works" on can't-decode families — which would mean it is teaching surface pattern-matching, not binding recognition to behavior, and would collapse the whole conceptual frame.
- **H5 (mechanism of the fix).** Post-fix, the *recognition* signal is essentially unchanged while the *action gap* closes — the intervention re-binds behavior to existing representation rather than re-learning recognition. *Refuted if* the fix works by strengthening recognition (higher AUROC) rather than by re-coupling a stable representation to behavior.

H4 and H5 together are what no prior paper attempts, and what make the result a mechanism rather than a benchmark number.

---

## 7. Design as scientific instruments

**Models.** Qwen2.5-7B-Instruct, gemma-2-9b-it, Llama-3.1-8B-Instruct — deliberately the same three as Aziz et al. (2026) for direct comparability; optionally one ~3B model to probe whether the recognition/action gap shifts with scale (an axis no prior paper tests). These families are also the field-standard substrate for refusal-direction work (Arditi et al. 2024), so methods and baselines port for free.

**The independent variable: the encoding taxonomy.** A graded obfuscation ladder is the spine of the whole study — plain text → paraphrase → low-resource languages → simple ciphers (ROT13/Caesar) → substitution ciphers → Base64 → math/logic encodings. The ladder is what turns RQ2 from a yes/no into a curve. For every (model, family) cell we **first measure decode capability directly** (can the model reconstruct the plaintext at all, e.g. via a decode-and-restate probe), because that measurement is what assigns each cell to one of the three regimes — it is not a side check, it is the axis that makes Move B possible.

**Phase 1 — diagnosis (established recipes, two critical upgrades).** Difference-in-means harmfulness direction (Arditi et al. 2024), with two corrections drawn from the 2025–26 literature that most naive probing would get wrong:
- *(a) Probe harmfulness separately from refusal.* Zhao et al. (NeurIPS 2025) show these are distinct directions at different token positions (harmfulness at the instruction-final token; refusal at the post-instruction/template token). A probe placed naively risks measuring "will it refuse" rather than "is this harmful" — which would quietly beg the very question the paper asks. We read harmfulness at the instruction-final token and keep a parallel refusal probe as a contrast.
- *(b) Probe across positions and decoding steps, not one readout.* Under encodings the harmful "gist" may only become legible partway through the model's internal decode, not at a single final-token position (TrajGuard's masking→unmasking finding; and final-token probes drop 95%→64% on wrapped inputs, arXiv 2605.12726). We report per-layer × per-position recognition curves, not a scalar.

**Phase 2 — causal validation.** The Arditi trio (directional ablation for necessity; activation addition for sufficiency) applied to encoded inputs, plus the **reply-inversion test** (Zhao et al. 2025): steer along the harmfulness direction and confirm it flips the model's *own* judgment of harmfulness on an independent probe question — not merely that it emits a refusal token. This is what separates "the direction correlates with harm" from "the model uses this direction."

**Phase 3 — the repair experiment (the causal test of the whole story).** Lightweight safety fine-tuning on a *subset* of encoding families (encoded-harmful → refusal; encoded-benign → normal compliance, so over-refusal is controlled), with **held-out-family generalization as the headline measurement** and the can't-decode regime as the negative control (H4). Comparisons that make the result legible:
- against plain safety-data mixing (Qi et al. 2023's standard mitigation) — is cross-encoding structure actually doing anything beyond "more safety data"?
- against Circuit Breakers representation-rerouting — a representation-space method that never targets input encodings; if it wins on held-out encodings that is itself a finding about where representation-space fixes suffice.
- side-by-side with WildGuard / Llama Guard 3 on *identical* payloads — the external-guard-gap table that motivates the whole internals framing, which no prior paper runs on a shared payload set.
Utility and over-refusal tracked throughout (a capability/MMLU slice + XSTest) so any fix is shown not to break the model.

---

## 8. Validity controls (the reviewer arsenal — core to the idea, not logistics)

A probe claim of the form "trained on plain text, transfers to encoded attacks" invites a specific, fatal objection: *the probe is reading surface features (dangerous nouns, format cues, refusal-proneness), not harmfulness.* The five load-bearing defenses (full checklist in the science-repo notes):

1. **Format-decorrelation 2×2 (the single most important control).** Run *benign* content through the identical encoding pipeline and confirm the probe does **not** fire on "looks encoded"; run harmful content in untrained formats and confirm it still fires. Confounded probes have hit 94.5% false positives on format alone; only decorrelated designs recover the real signal (arXiv 2603.19426).
2. **Selectivity / control tasks** (Hewitt & Liang 2019) on both plain and encoded conditions — report the accuracy gap over random-label controls, never raw accuracy.
3. **Harmfulness ≠ refusal** — the parallel refusal-direction probe (Zhao et al. 2025); if a refusal probe "transfers" just as well, the harmfulness claim is undercut; if it transfers worse, that differential is itself evidence.
4. **Length matching** across encoded/plain conditions — encodings inflate token counts, a trivial confound the probe could exploit.
5. **Scope honesty on robustness** — claims are about *natural* zero-shot transfer, not adaptive adversaries. Obfuscated-activation attacks drive probe recall to zero when an attacker optimizes against a known probe (arXiv 2412.09565); the paper states this limitation plainly and may add a light adaptive stress test rather than over-claim.

Two further caveats we state rather than defend against: probe directions are **non-identifiable** (never claim "*the* harmfulness direction"; claim "a direction correlating with harm across tested conditions," backed by the Phase-2 causal check — arXiv 2602.06801); and high AUROC is **not** proof of a safety-meaningful representation (arXiv 2606.08044) — which is precisely why Phase 2 exists.

---

## 9. Anticipated objections + answers

- *"Why not just use Circuit Breakers / representation rerouting?"* — RR was never tested on ciphers/Base64/structural encodings; we run it as a baseline. Our diagnostic also explains *where* any representation-space method can work at all: only where recognition exists (the decode-capable regimes). If RR wins on held-out encodings, that is a publishable finding *of the diagnostic framework*, not a threat to it.
- *"Isn't this CIFR?"* — no. CIFR detects malicious fine-tuning *datasets* with an external monitor on a fine-tuning API. We repair the model's *own* inference-time refusal behavior. Same held-out-family benchmark idea; entirely different object and threat model.
- *"Your probe reads surface features, not harmfulness."* — the five controls in §8, plus the Phase-2 causal validation.
- *"Incremental over Aziz et al. 2026."* — new axis (structural encodings, not natural language); a new phenomenon cell (the can't-decode regime, which has no language analogue); a new fix class (training-side repair vs few-shot gating); the double-dissociation test they structurally cannot run; and the guard-comparison table. Framed honestly: *their* diagnostic vocabulary, *our* axis, trichotomy, and causal repair test.

---

## 10. Scientific risks and contingencies

- **Risk: at 7–13B scale, too many encoding families land in "can't-decode."** If the model can't decode most ciphers, the action-failure regime is thin and the fix has little to act on. *Contingency:* this is itself a finding (it sharpens the decode-boundary map), and the graded ladder guarantees a populated middle band (paraphrase, languages, ROT13-class) where recognition is present. The paper's spine survives even if the deep-cipher end is all R-failures.
- **Risk: recognition and behavior degrade together (no action gap).** Then H2 is refuted and the failures are recognition failures throughout — a *different* but still-publishable conclusion (it would say safety-behavior training is the wrong tool for encodings, decoding capability is the lever), and it would sharpen rather than kill the paper. The design measures the gap either way.
- **Risk: the double dissociation is muddy** (the fix leaks into can't-decode families by teaching surface patterns). *Contingency:* held-out *unseen* cipher families and the benign-through-the-same-pipeline control are designed to catch exactly this; a leak is detectable and reportable, not a silent failure.
- **Risk: probe-validity objections still land despite controls.** *Contingency:* the analysis-led framing degrades gracefully — even if a reviewer discounts the "abstract harmfulness" interpretation, the causal repair result (train on A, generalize to held-out B, fail on can't-decode C) stands on behavioral evidence independent of the probe interpretation.

The through-line: every major risk converts into a finding rather than a dead end, because the design measures the discriminating quantity in each case.

---

## 11. Expected contributions (what the paper claims)

1. A **recognition/action diagnosis across the encoding taxonomy** — the first to ask, per encoding family, whether safety fails because harm isn't recognized or isn't acted on.
2. The **decode-capability trichotomy** — a conceptual object with no language-axis analogue, and the **double dissociation** it enables.
3. A **cross-encoding safety-training recipe** that repairs action failures and generalizes to held-out families in proportion to representational overlap, with the negative control showing it correctly does *not* fix recognition failures.
4. The **guard-gap table** — internals vs WildGuard/Llama Guard 3 on identical encoded payloads — grounding the structural argument that external guards cannot catch what they never decode.
