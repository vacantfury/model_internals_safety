# S1 idea check — first paper (direction committed 2026-08-02; frame revised 2026-08-02, v2)

Status: S1 COMPLETE. This document is the **scientific** idea check — the research question, the conceptual contribution, the hypotheses, the design as a set of scientific instruments, and the validity controls. Feasibility/logistics (compute, data engineering, venue timing) live separately in `proposal.md` and the repo `TODO.md`, deliberately kept out of here. Literature basis: a 4-bucket deep read of ~20 papers full-text (2026-08-02); per-paper notes + BibTeX in the science repo's `literature/model-internals/`.

Verdict: **proceed to S2**, gated on the regime-map pilot (§7). The direction is scoop-clean at the level of the whole claim, rests on established methods we extend rather than invent, and is built around a *falsifiable* mechanistic prediction rather than a descriptive measurement.

**v2 revision (2026-08-02).** The v1 frame had two failure types (recognition vs action) and one intervention. Review found that its recognition measurement conflated two distinct things — whether a model *can* decode an encoding, and whether it *does* decode it while under attack. Separating them adds a fourth regime with no analogue anywhere in this literature, and turns the single dissociation into a crossed two-intervention × three-regime design. §12 records what changed and why.

---

## 1. The committed direction

**Can't, didn't, or wouldn't? Diagnosing and repairing safety failures under encoded inputs.** *(our proposal — the merge of proposal.md candidates 1 + 4, reframed at v2.)*

A safety-aligned model that refuses "how do I build a bomb" in plain English will often comply when the same request arrives as a Caesar cipher, Base64, a math/logic word problem, or a low-resource language. The field treats this as one failure ("the safety training didn't cover this surface form"). We claim it is **three distinct failures that look identical from the outside** — identical attack-success numbers, three different causes, three different fixes — and that telling them apart requires reading the model's internals, because no external metric can separate them.

**The three failures, and why they are not the same problem:**

- **(C) Capability failure — *can't decode*.** The model cannot invert the encoding at all. The harmful content never becomes available to any part of the computation. Nothing is there to recognize, so no safety intervention of any kind — training, steering, or an external guard — can help; only decoding ability itself can change this cell.
- **(D) Deployment failure — *didn't decode*.** The model demonstrably *can* decode the encoding when decoding is the task, but does not do so in the attack forward pass. The capability exists and goes unspent. Again nothing is recognized, but the fix is entirely different from (C): the model does not need new ability, it needs to be induced to use the ability it has before answering.
- **(B) Binding failure — *wouldn't refuse*.** The content is decoded, the harmful intent is represented — linearly readable and causally implicated in the computation — and that recognition never becomes refusal behavior. The safety signal is present but *unbound* from the safety response. This is the cell that cheap safety fine-tuning should be able to repair, and the only one where it should.

Plus the control cell, **(S) safe** — decoded, recognized, refused.

The distinction is the whole point: **the same observed attack success rate can arise from a missing ability, an unused ability, or an unconnected wire, and the appropriate response differs in each case.** A single external metric cannot distinguish them. Neither can a text guard — which is itself the argument for going inside the model.

---

## 2. Core research questions

1. **RQ1 (diagnosis).** For each (model, encoding-family) cell, which of the four regimes holds? Specifically: does the model possess the decoding ability; does it deploy that ability in the attack condition; is harmfulness represented; and does refusal follow? Where in the network and when in processing does each of these become measurable?
2. **RQ2 (structure).** How do the regimes lay out along an obfuscation ladder? Is the transition graded (each quantity erodes smoothly) or does it have thresholds — and in what order do capability, deployment, recognition, and behavior fail? The prediction that behavior fails *first*, opening a gap above the recognition boundary, is what makes the binding regime real rather than a relabeling.
3. **RQ3 (repair, the causal test).** Do two different interventions repair two different regimes, each failing where the theory says it must? Cross-encoding safety fine-tuning should close binding failures — including on *held-out* encoding families — and should do nothing for capability or deployment failures. Decode-elicitation should fix deployment failures and nothing else. Neither should touch capability failures.

RQ3 is the heart of the paper: it turns the diagnosis from a description into a set of causal claims that can each be proven wrong independently.

---

## 3. The conceptual contribution

Three moves: one inherited-and-extended, one new, one that only becomes available because of the second.

**Move A — port the recognition/action lens from the language axis to the encoding axis.** The recognition-vs-action dichotomy already exists for *one* axis: Aziz et al. (2026, arXiv 2606.01196) showed that low-resource-*language* safety failures are "action failures, not representation failures" — the harmfulness direction stays linearly readable (AUROC > 0.85) even as behavioral refusal collapses to 44%. We adopt their diagnostic vocabulary and their core method (contrastive difference-in-means direction + causal ablation), and extend it to the structurally different space of *encodings*: ciphers, Base64, math/logic transformations, rendered/formatted text. On its own this is a real but bounded contribution — a generality test of someone else's hypothesis.

**Move B — separate decoding *ability* from decoding *deployment*, yielding the four-regime taxonomy (the new conceptual object).** Every language in Aziz et al.'s study is decodable by construction — the model understands Swahili, just imperfectly — and, crucially, understanding it is not optional: comprehension of natural language is not a step the forward pass can skip. Encodings break both assumptions at once. A 7–13B model may genuinely be unable to invert a keyed substitution cipher (capability), and — the part no one has measured — a model that *can* invert ROT13 when instructed to may simply not do so when the instruction is "answer the following," because nothing in the prompt asks it to.

| Regime | Can decode | Decodes in situ | Harm represented | Behavior | What can fix it |
|---|---|---|---|---|---|
| **(C) can't decode** | ✗ | ✗ | ✗ | complies | nothing in the safety stack — only decoding ability |
| **(D) didn't decode** | ✓ | ✗ | ✗ | complies | decode-elicitation — the ability exists, unspent |
| **(B) decode & comply** | ✓ | ✓ | ✓ | complies | safety training — bind recognition to refusal |
| **(S) decode & refuse** | ✓ | ✓ | ✓ | refuses | nothing — the control cell |

The (D) regime is the contribution. It has no analogue on the language axis, because a model does not decline to spend compute on Swahili. It is invisible to every method in the current literature, all of which measure either behavior or a single internal readout. And it is not a curiosity: it would be the mechanistic explanation for a practical effect the field has observed and never grounded — that prompting a model to *decode first, then respond* changes attack success rates. Under the four-regime account that is not a prompting trick; it is moving cells from (D) to (B) or (S), and it should provably do nothing for (C).

**Move C — the crossed intervention design (available only because of Move B).** With three failure regimes and two mechanistically distinct interventions, the causal test becomes a 2 × 3 matrix in which **four of the six predicted cells are nulls**:

| | (C) can't decode | (D) didn't decode | (B) decode & comply |
|---|---|---|---|
| **cross-encoding safety fine-tuning** | no effect | no effect | **repairs** |
| **decode-elicitation** | no effect | **repairs** | no effect needed |

A paper that only ever shows its intervention succeeding is weak evidence. A paper that predicts in advance exactly where each of two interventions must fail, and then shows both patterns hold, is making a mechanism claim that could have failed in four independent ways. This is the strongest causal argument available short of full circuit-level analysis, and v1's design — one intervention, one dissociation — could not make it.

---

## 4. Why this is scientifically good (not just a measurement paper)

The concern with any internals-analysis paper is that it becomes descriptive — "we probed X across Y settings, here are the AUROCs." This design avoids that in four ways:

1. **It makes falsifiable mechanistic claims, and the interventions are the evidence.** Phenomenon → mechanism hypothesis → interventional test is the full scientific arc. The fine-tuning experiment is load-bearing evidence, not a bolted-on "defense product."
2. **Its negative controls are predicted, not discovered.** The four null cells of the Move-C matrix are stated before the experiments run. Predicted nulls are evidence; unpredicted ones are noise.
3. **The instruments cross-check each other.** The regimes are defined over four measurements with a built-in coherence constraint: harm cannot be represented where content was never decoded. Only four of eight logically possible measurement combinations are coherent, so an incoherent cell is a detected instrument failure rather than a silent one (§7).
4. **The probes are correlational, and the design knows it.** Linear separability shows harm is *represented*; it does not show the representation *governs* behavior. Phase 2 (ablation, activation addition, reply-inversion) is what licenses causal language.

Framing: the paper is best presented as a **mechanistic question** ("when safety fails under encodings, has the model failed to decode, failed to spend a decoding ability it has, or failed to act on what it decoded — and does each failure have its own fix?") with the interventions as causal evidence and the practical recipe as a payoff that falls out. The defense framing ("a principled cross-encoding safety-training recipe") is available from the same experiments but buys less scientific weight; recommendation on record is analysis-led.

---

## 5. Why the whole claim is open (scoop check, 2026-08-02, full-text verified)

Three lines sit nearest; each holds one piece; none holds the combination.

| Prior line | What it established | What it did NOT do |
|---|---|---|
| **Aziz et al. 2026** (2606.01196) — *the* closest | Recognition-vs-action diagnostic (diff-in-means + causal ablation): low-resource-*language* failures are action failures; AUROC > 0.85 while refusal collapses to 44%; fixes with a few-shot latent gate | Only natural-language translation — no ciphers/Base64/structural encodings; no capability or deployment regime (language comprehension is neither optional nor absent), so no four-regime taxonomy and no crossed design; fix is few-shot threshold gating, explicitly disclaims adversarial robustness; no text-guard comparison |
| **TrajGuard** (2604.07727) | Behavioral proof that decode-time internal monitoring catches cipher attacks that text guards miss (Llama Guard 3 ASR 0.41 vs 0.14) — and that models "decipher during generation," which is direct evidence the deployment question is live | Detection only; never asks which failure type is occurring; no causal validation; no training-side repair; never separates ability from deployment |
| **SALO** (2605.02958, ICML 2026) | Zero-shot detector generalization from plain safety data to unseen attack families; refusal is a sparse *upstream* trajectory (not a terminal-token readout) | **Explicitly names encoded/Base64 inputs as its untested boundary case** — "may not trigger the refusal trajectory" — which is precisely the C-vs-D-vs-B question, left open |

**Training-axis sub-check.** The nearest training work is **CIFR** (Youstra et al. 2025, arXiv 2508.17158): it also holds out cipher families for OOD testing — but it trains a *monitor to detect malicious fine-tuning data* on a fine-tuning API. Different object (a detector, not the model's own refusal), different threat model (API abuse, not inference-time attacks). Must be distinguished on first mention, but not a scoop. **Circuit Breakers** (Zou et al. 2024) already shows multilingual (incl. low-resource) robustness but tests **no ciphers/Base64/structural encodings**; **LAT** (Sheshadri et al., TMLR 2025) tests none. No paper trains a model's own refusal to generalize across held-out *encoding* families.

**Adjacent-axis sub-check** (added after the external idea check, §12). Two papers show a related shape on *other* axes; neither is a scoop:
- **Self-Jailbreaking** (Yong & Bach, ICLR 2026, arXiv 2510.20956) — after benign reasoning training, models stay internally aware of harmfulness yet reason themselves into compliance; mitigated by mixing in a small amount of safety-reasoning data. Our binding failure on the *reasoning-training* axis, with a fix of H4's shape. Converging evidence that the split generalizes beyond one axis; differs in cause (training-induced drift, not input surface form), and runs no held-out-family generalization test and no regime-specific null predictions.
- **DeRTa** (Yuan et al., ACL 2025, arXiv 2407.09121) — refusal-position bias: safety tuning teaches models to check at response *onset*, so refusal rarely fires later in a generation. Directly load-bearing for the (D) regime: even a model that decodes mid-generation may have no trained pathway to refuse at that point. Supports the trajectory readouts in §7 and is a candidate component of the decode-elicitation intervention.

**Deployment-regime sub-check.** No paper in any of the four buckets measures whether a model deploys a decoding ability it possesses. The nearest adjacent observations are TrajGuard's finding that deciphering happens *during generation* (behavioral, not internal, and never contrasted against instructed decoding) and the general CoT-safety literature's observation that reasoning traces change refusal rates (never mechanized as an ability-vs-deployment distinction). This is the clearest open space in the design.

**Net:** the exact combination — the four-regime diagnosis across the encoding taxonomy, the ability/deployment separation, the crossed two-intervention causal matrix with its four predicted nulls, the cross-encoding training repair with held-out families, and a head-to-head against text guards on identical payloads — is unclaimed.

---

## 6. Hypotheses and falsifiable predictions

- **H1 (binding failures exist and are common).** For families the model decodes in situ, harmful intent is linearly readable in mid-layer activations even when the model complies — most failures in that band are (B), not (C) or (D). *Refuted if* probe AUROC is at chance wherever behavior fails.
- **H2 (the deployment regime is real — the new claim).** There exist (model, family) cells where decode-and-restate accuracy is high but in-situ content readout is absent, and these cells are behaviorally distinguishable: an explicit decode-then-answer prompt restores refusal in (D) cells while leaving (C) cells unchanged. *Refuted if* ability and in-situ deployment coincide everywhere — in which case the taxonomy collapses back to v1's three regimes, which is a graceful degradation, not a dead paper.
- **H3 (graded structure, with an action gap).** As obfuscation deepens (plain → paraphrase → language → simple cipher → substitution cipher → Base64 → math/logic), the four quantities fail in order and not together: refusal behavior degrades first and furthest, opening an "action gap" that widens before recognition, deployment, and finally ability give way. *Refuted if* behavior and recognition fall together with no gap — which would make the whole story a recognition story and eliminate regime (B).
- **H4 (repair generalizes).** Cross-encoding safety fine-tuning on a subset of families closes the action gap on *held-out* families, and the amount of generalization tracks **representational overlap** between families. *The overlap metric is pre-registered as the projection-score distribution overlap* (how far each family's harmfulness projections sit from the refusal threshold), **not** direction cosine: Wang et al. (NeurIPS 2025, arXiv 2505.17306) show refusal directions are near-parallel across languages — an English-extracted vector bypasses refusal elsewhere — and Aziz et al. find the low-resource failure is a threshold/projection shift with the direction intact. If the same universality holds across encodings, direction cosine is degenerate (all ≈ 1) with no variance to correlate against. Cosine is retained as a secondary check, and measured near-degeneracy is itself reportable: it would say encodings move a family *along* a shared harmfulness axis rather than rotating it. *Refuted if* held-out generalization is absent, or unrelated to every overlap metric measured.
- **H5 (the crossed dissociation — the decisive test).** The safety fine-tune repairs (B) and leaves (C) and (D) failing; decode-elicitation repairs (D) and leaves (C) failing. Four predicted nulls, each independently falsifiable. *Refuted if* either intervention "works" outside its regime — most consequentially if the fine-tune fixes (C) families, which would mean it teaches surface pattern-matching ("refuse anything that looks encoded") rather than binding recognition to behavior, and would collapse the conceptual frame.
- **H6 (mechanism of the fix).** Post-fix, the *recognition* signal is essentially unchanged while the action gap closes — the intervention re-binds behavior to existing representation rather than re-learning recognition. *Refuted if* the fix works by strengthening recognition (higher AUROC) rather than by re-coupling a stable representation to behavior.

H2 and H5 are what no prior paper attempts, and what make the result a mechanism rather than a benchmark number.

---

## 7. Design as scientific instruments

**Models.** Qwen2.5-7B-Instruct, gemma-2-9b-it, Llama-3.1-8B-Instruct — deliberately the same three as Aziz et al. (2026) for direct comparability; optionally one ~3B model to probe whether the regime map shifts with scale (an axis no prior paper tests). These families are also the field-standard substrate for refusal-direction work (Arditi et al. 2024), so methods and baselines port for free.

**The independent variable: the encoding taxonomy.** A graded obfuscation ladder is the spine of the study — plain text → paraphrase → low-resource languages → simple ciphers (ROT13/Caesar) → substitution ciphers → Base64 → math/logic encodings. The ladder is what turns RQ2 from a yes/no into a curve.

**The four measurements (this is what defines a cell's regime).** For every (model, family, prompt):

| # | Quantity | Instrument |
|---|---|---|
| 1 | **Ability** — can it decode? | decode-and-restate accuracy on an explicit decode task (an *upper bound* on what the forward pass could do, never the regime assignment itself) |
| 2 | **Deployment** — did it decode here? | **in-situ content probe** on the *attack* forward pass: is plaintext semantic content linearly readable when decoding was never requested? Corroborated by cross-condition activation patching (patch plaintext-condition activations into the attack condition) and intermediate-layer decoded-token readouts |
| 3 | **Recognition** — is harm represented? | harmfulness direction (diff-in-means, Arditi et al. 2024) read at the instruction-final token, per Zhao et al. (NeurIPS 2025), with a parallel refusal-direction probe as contrast |
| 4 | **Behavior** — did it refuse? | refusal *and* ASR, scored by two separate judges — HarmBench's classifier for ASR (preferred over Llama Guard 2 + substring, which mislabels degenerate completions) and JailbreakBench's refusal classifier for refusal |

**Why #2 is not optional, and why it was the v1 design's hole.** v1 assigned regimes using #1 alone. But #1 measures ability when decoding *is* the task, whereas every regime is defined over the attack condition, where it is not. A cell can be ability-capable, be labelled a binding failure, be fed to the repair experiment — and have decoded nothing in the actual forward pass. H4 would then fail for a reason unrelated to binding, and (C)'s negative control would be contaminated with cells that could decode and didn't. Measurement #2 is one extra probe target on forward passes we already run.

**The coherence constraint (a free internal validity check).** Harm cannot be represented where content was never decoded. So a cell reading *recognition present, deployment absent* is not a finding — it is a detected instrument failure, and the most likely cause is exactly the objection §8 exists to answer: the harmfulness probe is firing on surface features (dangerous-looking tokens, format cues) rather than decoded semantics. This turns the field's standard external-control battery into an internal consistency test the design cannot silently pass. A second hard constraint has the same status: measurement #1 is an *upper bound* on what the forward pass could do, so *deployment present, ability absent* is likewise impossible and indicts the instrument.

> **Amended 2026-08-02 at build step 4** (implementation: `measurements/regimes.py`), on two points where the prose above was looser than the code can afford to be.
>
> *(a) The "four of eight coherent combinations" count needs a second constraint, and that one is corpus-dependent.* Over (deployment, recognition, behavior) there are 8 combinations; `recognition → deployment` rules out 2, leaving **6**, not 4. Reaching 4 requires also `deployment → recognition` — and that holds only on a **harmful** corpus, since decoding a benign prompt and representing no harm is the normal case. It is therefore implemented as a *soft* flag scoped to harmful prompts, kept separate from the hard violations, and a cell carrying it still receives its regime label. Treating it as hard would manufacture an instrument failure out of every benign control cell — and the benign-encoded condition is a headline result here, not a throwaway.
>
> *(b) A fifth cell exists: refusal without deployment — (R), "surface refusal".* The Move-B table assigns compliance to (C) and (D), so a cell that *refuses* having decoded nothing matches no row. It is not incoherent: a model can refuse on surface cues alone. It is also not a curiosity — it is precisely the degenerate outcome H5 warns about, a fine-tune that learns "refuse anything that looks encoded" instead of binding recognition to behavior. It gets its own label so it can be **counted**, because absorbing it into (C) or (S) would hide the failure mode the over-refusal battery exists to catch. Expect (R) to be near-empty pre-intervention and to be the number to watch post-fine-tune.

> **Amended 2026-08-02 at build step 3** (implementation: `judges/`, `measurements/behavior.py`), on measurement #4, where the row above compressed three things the paper has to state separately.
>
> *(a) "HarmBench classifier" means the LLM-judge form, and the paper must say so.* What the field calls the HarmBench classifier is a released Llama-2-13B checkpoint; what we run — inherited from the guardrail sibling so that this paper's ASR column is comparable with the family's black-box results — is HarmBench's canonical classifier *prompt* sent to an API judge (gpt-5-mini, resolved there after a judge-validation round found the cheaper tier inflates absolute ASR by 2–3×). The two are not interchangeable and reporting the second under the first's name would be a provenance error. Whether to also run the released weights, which this repo *can* do and the sibling could not, is open (`project_structure.md` §7, decision 5); the cheap resolution is to run both over the pilot's saved responses and report agreement.
>
> *(b) Refusal is a second judge, not a reading of ASR.* The regime split (B)/(S) and (R)/(C)(D) turns on refusal, and "not jailbroken" does not imply "refused" — a model that decodes nothing and emits filler is neither. So JailbreakBench's refusal classifier runs alongside HarmBench's on every response, and both verdicts are carried.
>
> *(c) Under encoded inputs, both judges must be asked about the plaintext.* Judge prompts compare a response against the request; give either one the ciphertext and it misfires in the direction that matters most — HarmBench returns "no" mechanically, and the refusal judge scores a decoded-and-complying response as a refusal because it reads as irrelevant to the ciphertext. Relatedly, the refusal prompt counts an echoed request as a refusal, which under encodings is a *decode failure*, not a refusal; echo is therefore scored independently and reported next to the refusal rate rather than folded into it.

**Phase 0 — the regime-map pilot (gates everything).** Before any fine-tuning, run measurements #1–#4 on 2 models × the full ladder, behavior and probes only. It answers the question that decides whether the design survives: **is the (B) cell populated at 7–9B scale?** Cipher attacks are known to work best on frontier models precisely because small models cannot decode; if the ciphers all land in (C) at this scale, the binding regime lives only in the paraphrase/language/ROT13 band — much closer to Aziz et al.'s territory — and the paper must reshape before any expensive commitment. It also returns the first evidence on H2. Cheap, and it must run first.

**Phase 1 — diagnosis (established recipes, two critical upgrades).** Difference-in-means harmfulness direction, with two corrections from the 2025–26 literature that naive probing gets wrong:
- *(a) Probe harmfulness separately from refusal.* Zhao et al. (NeurIPS 2025) show these are distinct directions at different token positions (harmfulness at the instruction-final token; refusal at the post-instruction/template token). A naively placed probe measures "will it refuse" rather than "is this harmful" — quietly begging the question the paper asks.
- *(b) Probe across positions and decoding steps, not one readout.* Under encodings the harmful gist may only become legible partway through the model's internal decode (TrajGuard's masking→unmasking finding; final-token probes drop 95%→64% on wrapped inputs, arXiv 2605.12726). We report per-layer × per-position curves for measurements #2 and #3, not scalars.

**Phase 2 — causal validation.** The Arditi trio (directional ablation for necessity; activation addition for sufficiency) applied to encoded inputs, plus the **reply-inversion test** (Zhao et al. 2025): steer along the harmfulness direction and confirm it flips the model's *own* judgment of harmfulness on an independent probe question — not merely that it emits a refusal token. This separates "the direction correlates with harm" from "the model uses this direction."

**Phase 3 — the crossed intervention experiment.** Two interventions, three regimes, six predictions (§3, Move C):
- *Intervention 1 — cross-encoding safety fine-tuning.* Lightweight fine-tuning on a *subset* of families (encoded-harmful → refusal; encoded-benign → normal compliance, so over-refusal is controlled by construction), with **held-out-family generalization as the headline measurement**.
- *Intervention 2 — decode-elicitation.* The cheap form first (an explicit decode-then-answer instruction), then, if the effect is real, a light training version. Predicted to move (D) and nothing else. **Two controls are mandatory here**, both forced by DRO (Zheng et al., ICML 2024, arXiv 2401.18018), which showed that prepending a safety prompt moves queries along a *higher-refusal* direction largely irrespective of their actual harmfulness — i.e. instructions raise refusal propensity as a side effect of being instructions:
  - *(i) A matched placebo instruction.* Same length and imperative form, requesting a different pre-answer step ("first restate the question, then answer", "first count the characters, then answer"). Without it, any refusal increase is attributable to the prompt being longer and more directive rather than to decoding having occurred.
  - *(ii) The deployment measurement must move.* The in-situ content probe has to go from absent to present under the elicitation prompt. Behavior improving while the content probe stays flat is not a decoding effect — it would refute the (D) reading of that cell no matter how good the ASR number looks. This is the difference between reporting a prompting trick and reporting a mechanism.

Baselines, deliberately trimmed to keep this one paper (§10):
- **must-run:** plain safety-data mixing (Qi et al. 2023's standard mitigation — is cross-encoding structure doing anything beyond "more safety data"?); Circuit Breakers representation-rerouting (the strongest "why not just use RR?" objection, and never tested on encodings); WildGuard / Llama Guard 3 on *identical* payloads (the external-guard-gap table that motivates the internals framing, which no prior paper runs on a shared payload set).
- **if budget permits, else discussion only:** ACTOR (Dabas et al., ICML 2025, arXiv 2507.04250) — single-layer activation-shift repair of the *opposite* miscalibration, the cheapest test of H6's re-binding claim; DeRTa-style late-refusal training (Yuan et al., ACL 2025) as a component of Intervention 2.

**The benign-encoded condition is a headline result, not a control.** The most likely degenerate outcome of any encoding-side safety fine-tune is that the model learns "refuse anything that looks encoded" — which would score well on every harmful benchmark and be worthless in deployment. So over-refusal on benign content pushed through the *identical* encoding pipeline is promoted out of the control section into the main results, alongside utility (a capability/MMLU slice) and a three-part over-refusal battery: XSTest, **EVOREFUSE-TEST** (arXiv 2505.23473) as the harder English probe (~85% higher refusal-trigger rate than the next-best benchmark), and **MORBench** (Pan et al., EMNLP 2025, arXiv 2505.18325) for the multilingual band. The third is not redundant: XSTest and EVOREFUSE are English-only while our ladder deliberately includes low-resource languages, so without a multilingual over-refusal set the fix could wreck helpfulness in exactly the conditions the paper claims to repair and the measurement would never see it. RASS's framing is also conceptually adjacent — it locates over-refusal at a misaligned *safety decision boundary*, which is the same object H4's projection-score overlap metric measures distance to.

---

## 8. Validity controls (the reviewer arsenal — core to the idea, not logistics)

A probe claim of the form "trained on plain text, transfers to encoded attacks" invites a specific, fatal objection: *the probe is reading surface features (dangerous nouns, format cues, refusal-proneness), not harmfulness.* The five load-bearing defenses (full checklist in the science-repo notes), on top of the coherence constraint in §7:

1. **Format-decorrelation 2×2 (the single most important control).** Run *benign* content through the identical encoding pipeline and confirm the probe does **not** fire on "looks encoded"; run harmful content in untrained formats and confirm it still fires. Confounded probes have hit 94.5% false positives on format alone; only decorrelated designs recover the real signal (arXiv 2603.19426).
2. **Selectivity / control tasks** (Hewitt & Liang 2019) on both plain and encoded conditions — report the accuracy gap over random-label controls, never raw accuracy. Applies to the in-situ content probe as much as to the harmfulness probe.
3. **Harmfulness ≠ refusal** — the parallel refusal-direction probe (Zhao et al. 2025); if a refusal probe "transfers" just as well, the harmfulness claim is undercut; if it transfers worse, that differential is itself evidence.
4. **Length matching** across encoded/plain conditions — encodings inflate token counts, a trivial confound the probe could exploit.
5. **Scope honesty on robustness** — claims are about *natural* zero-shot transfer, not adaptive adversaries. Obfuscated-activation attacks drive probe recall to zero when an attacker optimizes against a known probe (arXiv 2412.09565); the paper states this limitation plainly and may add a light adaptive stress test rather than over-claim.

Two further caveats we state rather than defend against: probe directions are **non-identifiable** (never claim "*the* harmfulness direction"; claim "a direction correlating with harm across tested conditions," backed by the Phase-2 causal check — arXiv 2602.06801); and high AUROC is **not** proof of a safety-meaningful representation (arXiv 2606.08044) — which is precisely why Phase 2 exists.

---

## 9. Anticipated objections + answers

- *"Isn't 'didn't decode' just 'decoded badly'?"* — the sharpest objection to Move B, and the reason measurement #1 exists as a separate instrument. The claim is not that in-situ decoding is weak but that a demonstrated ability goes unspent, and it is testable in both directions: high decode-and-restate accuracy with absent in-situ content readout, plus the behavioral prediction that eliciting decoding restores refusal in exactly those cells and not in (C) cells. If (D) turns out to be the low tail of a continuum, H2 is refuted and we report that — the taxonomy degrades to v1's three regimes and the rest of the paper stands.
- *"Why not just use Circuit Breakers / representation rerouting?"* — RR was never tested on ciphers/Base64/structural encodings; we run it as a baseline. Our diagnostic also explains *where* any representation-space method can work at all: only in regimes where recognition exists. If RR wins on held-out encodings, that is a publishable finding *of the diagnostic framework*, not a threat to it.
- *"Isn't this CIFR?"* — no. CIFR detects malicious fine-tuning *datasets* with an external monitor on a fine-tuning API. We repair the model's *own* inference-time refusal behavior. Same held-out-family benchmark idea; entirely different object and threat model.
- *"Your probe reads surface features, not harmfulness."* — the five controls in §8, the coherence constraint in §7 (which makes this failure mode *detectable* rather than arguable), and the Phase-2 causal validation.
- *"Incremental over Aziz et al. 2026."* — new axis (structural encodings, not natural language); two new phenomenon cells with no language analogue (capability and deployment failures); a new fix class (training-side repair vs few-shot gating); a crossed two-intervention design with four predicted nulls that they structurally cannot run; and the guard-comparison table. Framed honestly: *their* diagnostic vocabulary, *our* axis, taxonomy, and causal design.
- *"Decode-elicitation is just prompting."* — yes, and that is the point: it is a cheap manipulation with a sharp regime-specific prediction. Its scientific value is the null cells, not the effect size. The version of this objection that actually bites is DRO's (Zheng et al., ICML 2024): prompts shift representations toward refusal *as prompts*, so any prepended instruction can raise refusal without touching the mechanism. That is why the elicitation arm ships with a matched placebo instruction and requires the in-situ content probe to move before any (D)-repair claim is made (§7).

---

## 10. Scientific risks and contingencies

- **Risk: at 7–13B scale, too many encoding families land in (C).** If the model cannot decode most ciphers, the binding regime is thin and the fine-tuning experiment has little to act on. *Contingency:* Phase 0 measures this before anything is spent. A thin (B) band is itself a finding (it sharpens the decode-boundary map), and the graded ladder guarantees a populated middle band (paraphrase, languages, ROT13-class). The paper's spine survives even if the deep-cipher end is all capability failures — though Move B's weight then shifts entirely onto the (D) regime, which makes Phase 0's H2 evidence the pivotal early result.
- **Risk: the (D) regime does not separate** — in-situ content probes cannot distinguish "didn't decode" from "decoded weakly." *Contingency:* activation patching gives a second, probe-independent read (does injecting plaintext-condition activations change behavior in cells where the content probe reads absent?); and if separation genuinely fails, H2 is refuted, the frame degrades to v1's three regimes, and the paper loses a contribution without losing its spine.
- **Risk: recognition and behavior degrade together (no action gap).** Then H3 is refuted and the failures are recognition failures throughout — a *different* but still-publishable conclusion (safety-behavior training would be the wrong tool for encodings; decoding capability is the lever). The design measures the gap either way.
- **Risk: representational overlap is degenerate across families.** If harmfulness directions are as universal across encodings as refusal directions are across languages (Wang et al. 2025), direction cosine sits near 1 everywhere and H4's correlational half has nothing to work with. *Contingency:* projection-score distribution overlap is pre-registered as the primary metric (§6), retaining variance exactly in the regime universality predicts; decode-capability score is a second candidate predictor; measured degeneracy is reported as a finding.
- **Risk: the crossed dissociation is muddy** (the fine-tune leaks into (C) by teaching surface patterns). *Contingency:* held-out *unseen* cipher families plus the benign-through-the-same-pipeline condition are designed to catch exactly this; a leak is detectable and reportable, not a silent failure. This is also why the benign-encoded result is promoted to the main table.
- **Risk: scope — this is two papers.** Four measurements × three models × seven families × two interventions × four baselines does not fit one paper at full cross-product. *Contingency, decided:* the diagnosis (Phases 0–2) plus the crossed matrix on **one** train-family/held-out-family split is the paper; the full generalization grid is cut to whatever the budget supports after the headline split; the ACTOR and DeRTa comparisons are explicitly optional (§7). If Phase 0 returns a rich regime map, the diagnosis alone is a viable standalone paper with the interventions as the follow-up — a decision point to revisit at S3, not now.
- **Risk: probe-validity objections still land despite controls.** *Contingency:* the analysis-led framing degrades gracefully — even if a reviewer discounts the "abstract harmfulness" interpretation, the crossed causal result (two interventions, three regimes, four predicted nulls) stands on behavioral evidence independent of the probe interpretation.

The through-line: every major risk converts into a finding rather than a dead end, because the design measures the discriminating quantity in each case.

---

## 11. Expected contributions (what the paper claims)

1. A **four-regime diagnosis across the encoding taxonomy** — the first to ask, per encoding family, whether safety fails because the model *can't* decode, *didn't* decode, or *wouldn't* act on what it decoded.
2. The **ability/deployment separation** — a failure mode with no analogue in the language-axis literature, invisible to every behavior-only or single-readout method, and the mechanistic grounding for why decode-then-answer prompting changes attack outcomes.
3. The **crossed intervention result** — two mechanistically distinct fixes, three regimes, four predicted nulls; a cross-encoding safety-training recipe that repairs binding failures and generalizes to held-out families in proportion to representational overlap, and correctly fails everywhere the theory says it must.
4. The **guard-gap table** — internals vs WildGuard/Llama Guard 3 on identical encoded payloads — grounding the structural argument that external guards cannot catch what they never decode.

---

## 12. External idea checks (cspaper.org) — results, and what the instrument is worth

Two submissions, both 2026-08-02: one against the v1 two-regime frame, one against the v2 four-regime frame. The service returns a placement summary plus 10 retrieved related papers with per-paper relevance judgments.

### 12.1 Check #1 — v1 frame (job 6cc84b34-4b5a-483a-b12a-99a26acb3df3, 10:52)

Submitted as *"Diagnosing Recognition and Action Failures in Encoded Safety Attacks."* **Result: pass.** It placed the idea "into the core cluster of mechanistic safety research, extending the established separation of harmfulness and refusal vectors to the novel domain of structural encodings," and named the "decode-and-comply" vulnerability as the thing prior work observed in multilingual/CoT contexts but left unresolved for text encodings — an independent restatement of §3's claim. No retrieved paper conflicted with the direction.

Already covered by the S1 corpus: Zhao et al. 2025 (`zhao2025llmsencode`, its top hit) and Arditi et al. 2024 (`arditi2024refusal`). **Five papers newly actioned** (arXiv IDs verified against the primary source, not the tool's text):

| Paper | ID / venue | Disposition |
|---|---|---|
| Wang et al., *Refusal Direction is Universal Across Safety-Aligned Languages* | 2505.17306, NeurIPS 2025 | **Changed a hypothesis.** Refusal directions near-parallel across languages ⇒ H4 pre-registers projection-score distribution overlap over direction cosine (§6); degeneracy risk in §10. |
| Yong & Bach, *Self-Jailbreaking* | 2510.20956, ICLR 2026 | Must-distinguish + converging evidence (§5). |
| Yuan et al., *DeRTa* | 2407.09121, ACL 2025 | Mechanism + optional baseline (§5, §7); relevant to (D) and to Intervention 2. |
| Dabas et al., *Just Enough Shifts* (ACTOR) | 2507.04250, ICML 2025 | Optional Phase-3 baseline (§7). |
| Wu et al., *EVOREFUSE* | 2505.23473, NeurIPS 2025 | Promoted to a headline over-refusal measurement (§7). |

### 12.2 Check #2 — v2 frame (job 0a4553bb-36e5-42ab-b7c5-dccbfd31c62f, 11:22)

Submitted as *"Diagnosing Safety Failures in Encoded Inputs by Decoding, Recognition, and Refusal."* **Result: pass, with a sharper placement than #1.** Its summary: the retrieved papers "mostly treat refusal failure as a monolithic geometry problem (e.g., poor clustering or a single suppressed direction)," while this idea "decomposes it into a three-stage causal pipeline (decoding, recognition, binding) to map specific failures to distinct interventions." That is the v2 contribution stated back to us by a system that had only the abstract — the decomposition reads as the delta, which is what the rewrite was for.

Two further signals in our favour, both weak individually: its "opportunities and gaps" section reports that **no retrieved paper investigates inference-time compute — test-time search or multi-step reasoning — as a mechanism for bypassing or enforcing safety boundaries**, which is the (D) regime's immediate neighbourhood; and no retrieved paper contests the ability/deployment separation.

**Two papers newly actioned** (IDs verified):

| Paper | ID / venue | Disposition |
|---|---|---|
| Zheng et al., *On Prompt-Driven Safeguarding for LLMs* (DRO) | 2401.18018, ICML 2024 | **Changed the design.** Safety prompts move queries along a higher-refusal direction largely irrespective of harmfulness — so a decode-elicitation prompt could raise refusal *as a prompt*, not by causing decoding. Intervention 2 now ships with a matched placebo instruction and a hard requirement that the in-situ content probe move before any (D)-repair claim (§7, §9). This is the most valuable thing either check produced: it closes a confound that would have made the elicitation arm unpublishable as a mechanism result. |
| Pan et al., *Understanding and Mitigating Overrefusal…* (RASS / MORBench) | 2505.18325, EMNLP 2025 | **Filled a measurement gap.** XSTest and EVOREFUSE are English-only while the ladder includes low-resource languages; MORBench covers the multilingual band (§7). RASS's "safety decision boundary" framing is also the object H4's overlap metric measures distance to. |

### 12.3 What two runs establish about the instrument

**The cspaper idea check is a settled-top-venue related-work sweep, not a scoop check.** Recorded here because it decides how much weight the "pass" verdicts carry:

- **It never retrieved the four nearest works — in either run.** Aziz et al. 2026 (2606.01196), TrajGuard (2604.07727), SALO (2605.02958), and CIFR (2508.17158) are absent from both result sets, across two submissions with substantially different abstracts. These are precisely the papers a scoop check exists to find. The pattern is consistent with an index that under-covers recent arXiv-only work.
- **The same three unrelated papers appeared in both runs, at the top.** Maximal-lotteries evaluation, KL distillation for math, and non-stationary RL entropy scheduling scored 83%/83%/80% in run 2 — *above* every genuinely relevant safety paper in the same list. The percentage is not a relevance measure, and the 10-result set is padded to a fixed size from a small candidate pool.
- **What it is good for, demonstrated in both runs:** surfacing established, well-cited venue papers adjacent to the framing — DRO (2024), Arditi (2024), Zhao (2025), Wang (2025), DeRTa, ACTOR, RASS. A recency-weighted manual pass under-covers exactly this band, and two of the seven changed the design.

**Consequence for this document:** §5's scoop verdict rests entirely on the manual 4-bucket full-text pass, not on these checks. The checks contribute related work and, twice now, one design-changing citation apiece — which is worth the ten minutes, but is not evidence that the claim is unclaimed. The general form of this judgment is recorded in the science organ's `knowledge/cspaper_idea_check.md` for reuse across research repos.
