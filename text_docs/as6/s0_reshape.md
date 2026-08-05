# AS-6 — S0 reshape (opened 2026-08-05, owner order "yes, go")

**Workflow stage: S0 refine and re-screen — IN PROGRESS.** Entered by looping back from S1 after the adversarial separation pass returned *reject as scoped* (record: `s1_idea_check.md`). Owner funded the reshape 2026-08-05.

**This document is a DRAFT under active screening.** Three checks are running against it and may still kill or reshape it again: an alias-channel prior-art sweep, a hostile review of the reshape itself, and a feasibility/cost check. Nothing here is settled until those return and are recorded in §5.

---

## 1. What failed, in one paragraph

The object cut (AS-5 measures the chain inside the *target*, AS-6 inside the *defense*) survived review untouched. The **scoping** failed: we planned to *observe* whether a guard decodes an encoded payload, across AS-5's ladder. At 8B that ladder has no usable middle. LlamaGuard-3-8B is a fine-tune of Llama-3.1-8B, which AS-5's own pilot measured at 0.03–0.30 decode-and-restate similarity across all twelve cipher rungs — it cannot read them, so every failure there is trivially "can't". The surface band (`zero_width`, `homoglyph`, `fullwidth`, `combining_marks`, `tag_block`) is plaintext modulo the tokenizer, so nothing needs decoding and "did it decode?" is answerable without a forward pass. Undefined on one side, trivial on the other. Compounding it, a correlational probe cannot separate *never decoded* from *decoded but unused* — that gap is the textbook description of distribution shift (Wei et al., NeurIPS 2023, "mismatched generalization").

## 2. The reshape — make decode ability a controlled variable

**The fatal objection becomes the method.** If the middle cell does not exist at this scale, *create it*, and create it causally.

### 2.1 The capability implant (the core experiment)

1. Verify a guard **cannot** read encoding E (decode-and-restate, the AS-5 instrument, reused unchanged).
2. **LoRA-finetune the guard on the DECODING task for E only** — pairs of (encoded text → plaintext). Never on safety labels for encoded inputs; never on harmful content rendered in E.
   **Refusal responses stay IN the corpus (added 2026-08-05, §5.1).** Both published points on this question — Halawi et al.'s 25.8% and Wan et al.'s asserted "remains aligned" — filtered refusals *out* of their encoding-learning data, which is exactly why neither number settles anything. Keeping them is the differentiator, not a detail.
3. Confirm with an **independent ability probe** that the capability was actually acquired — a decode-only fine-tune that failed to teach decoding produces the same null as a conversion failure, and the two must never be confusable.
4. The result is, by construction, **a guard that can read E but was never taught to flag E** — the previously empty middle cell, manufactured rather than hunted.
5. **Run the matched control arm** — an identically-sized LoRA on generic benign data, no decoding content. Mandatory, not optional: benign-only QLoRA is *published* to collapse guard safety geometry outright (2605.02914; Granite Guardian refusal 85% → 0%). Without this arm every result is attributable to "we fine-tuned a guard."
6. Measure the safety verdict on harmful prompts in E, before versus after, **against the control arm rather than against the base model**:
   - verdict does **not** improve → a **conversion** failure, *causally created* rather than inferred from a probe;
   - verdict **does** improve → decoding was the binding constraint, and the repair is decode-side.

**Why this answers the rejection.** It defeats the empty middle by manufacturing the middle. It defeats the collapse objection because the claim no longer rests on a probe's readout at a layer index we chose — it rests on an intervention we performed and a behaviour change we measured. "Decoded but unused" stops being an inference and becomes an experimental condition.

**And it changes the contribution from a diagnosis to a mechanism.** The old contribution was "here is an attribution schema". The new one is a causal claim about how safety behaviour relates to input comprehension in a classifier: *does teaching a guard to read a format make it safe on that format?* If the answer is no, that is a real and slightly alarming result — comprehension does not imply enforcement — and it is the guard-side analogue of the multilingual safety-transfer failure, which is a recognised and well-cited problem.

**Sharpened 2026-08-05 after the alias sweep (§5.1) — the framing is now stronger than "manufacture the middle".** Two published papers give *opposite* answers to this exact question and neither measured it cleanly: Covert Malicious Finetuning (ICML 2024) ablates its phase II and reports 25.8% harmful after cipher-training alone, while Invisible Safety Threat (ICLR 2026) asserts with no measurement that such a model "remains aligned" — **and both filtered refusal responses out of their encoding-learning corpora.** So the contribution is not "we built a new diagnostic"; it is **we resolve a live contradiction that neither side was positioned to settle** — on a guard rather than a generator, with refusals retained, with the acquired capability independently verified, and against a matched benign-LoRA control. That is a much harder claim to wave away, and it hands the paper a genuine opening paragraph.

### 2.2 Predicted nulls (the evidence, and the part that needs power)

- The implant does **not** change verdicts on rungs the guard could already read.
- The implant does **not** change verdicts on plaintext.
- The implant does **not** change clean-input DSR (this one is also the sanity gate — see §4).

Null claims require **TOST equivalence bounds with a pre-registered margin**, not `p > 0.05`. This was a named defect of the rejected version and must not survive into this one.

### 2.3 Supporting causal tests — REVISED 2026-08-05 after the alias sweep (§5.1); both were previously written as contributions, and neither is one

- **Inject / ablate — confirmatory secondary only, NOT a contribution.** Fit a harm *subspace* on plaintext (single directions are the wrong instrument: 2508.17158 finds cipher-induced harm survives out to ~100 orthogonal directions); inject where absent under encoding, ablate where present. The double dissociation has been published three times independently — Zhao et al. NeurIPS 2025 (2507.11878), 2603.05773, MBZUAI 2606.01196 — all three cited on first use. It corroborates the implant result; it cannot carry the paper.
- **A continuous dial — a cited instrument, zero contribution weight.** In-context bijection encodings with tunable dispersion place rungs at the edge of a *specific* guard's ability. This is Bijection Learning's published method (ICLR 2025, 2410.01294), which explicitly claims the quantitative-complexity-knob novelty. We use it; we never present it as ours.

### 2.4 Controls carried forward from the review

| Control | Why it is mandatory |
|---|---|
| Probes fit on **plaintext only**, tested on encoded | a probe fit on encoded data can learn the encoding; this is the difference between a measurement and an artefact |
| **Beat the tokenizer null model** per rung | characters-per-token alone separates encoded from natural text near-perfectly (Broken-Token, 2510.26847); if a 2-parameter statistic matches our probes, the paper has no reason to exist |
| Per-rung **selectivity** over shuffled-label controls | Hewitt & Liang 2019; report the gap, never raw accuracy |
| **Fourth cell: blocked WITHOUT decoding** | the guard as format detector rather than harm detector — the most policy-relevant cell, and it predicts brittleness |
| Prompt length bounded **below each guard's inspection window** | transmit is not free in text (Prompt Overflow, 2605.23196) |
| **Tri-state everything** | unlicensed probe → `None`, never `False`. Inherited AS-5 law; scoring an unlicensed probe as "never decoded" would manufacture the paper's own headline |
| **Matched generic-benign LoRA control arm** *(added 2026-08-05, §5.1)* | benign-only QLoRA is published to destroy guard safety geometry by itself (2605.02914). Every before/after comparison runs against this arm, never against the base model |
| **Independent ability probe after the implant** *(added 2026-08-05)* | a failed implant and a conversion failure produce the identical null; without this the paper's headline is unfalsifiable |
| **Refusals retained in the decode corpus** *(added 2026-08-05)* | both prior data points de-refused their training data; this is the design choice that makes our number mean something theirs did not |

## 3. Prior art this must be argued against (not assumed open)

- **Zhou et al., Findings of EMNLP 2024 (2406.05644)** — recognised-but-not-converted on chat models, shown causally via Logit Grafting, 7B–70B. The *concept* is theirs. What is left: the object (a guard, not a chat model), the input axis (encodings, not jailbreak templates), and the decode link, which has no counterpart in their work.
- **Our own AS-3** (2607.26574) — selected the pre-decoder repair from black-box observation alone. Any contribution phrased as "attribution is needed to pick the repair" is dead on arrival; the surviving phrasing is that attribution predicts *when the pre-decoder is NOT enough*, which the implant tests directly.
- **SIREN (2604.18519)** — probes guard internals to build a detector; its App. C result (LlamaGuard-3-8B 77.0 → 87.1 F1 from its own activations) is motivation, never a finding of ours.
- **Wei et al., NeurIPS 2023** — "mismatched generalization"; must be cited and explicitly differentiated.
- **The multilingual safety-transfer literature** is the closest conceptual analogue and the biggest live threat: if someone has already run *teach the model the language, does safety follow?*, the implant is that experiment with an encoding in place of a language. **This was the alias-channel sweep's top target — and the answer is YES, someone has.** *Tongue-Tied* (Upadhayay & Behzadan, CALCS 2025) LoRA-finetunes on Newari and fabricated languages using non-harmful data and shows the model jailbreaks; logit-lens analysis of late-layer pivots. Serious but not fatal: generator not guard, no verified-cannot-read precondition, no dissociation, and safety labels were never the withheld variable. Must be positioned against explicitly and early — a reviewer who knows it will reach for it first.

**Added by the alias sweep, 2026-08-05 (§5.1) — each must be argued against by name:**

- **Covert Malicious Finetuning** (ICML 2024, 2406.20053) — **the closest experiment in existence.** Its Fig. 5 phase-1-only ablation *is* the implant, on GPT-4, yielding 25.8% harmful. Our differentiation is threefold and must be stated in the intro, not the related-work section: a **guard** rather than a generator, **refusals retained** in the decode corpus rather than filtered out, and the verdict change as the *object of study* rather than an attack-component ablation.
- **Invisible Safety Threat** (ICLR 2026, 2603.08104) — asserts the *opposite* of CMFT ("remains aligned") with no measurement. The contradiction between these two papers is the paper's opening.
- **When Safety Geometry Collapses** (2605.02914) — the methodological threat; forces the control arm. Also a citation *for* us: it establishes that guard fine-tuning is fragile, which is why nobody has run our experiment cleanly.
- **Cipher fine-tuning at 70B** (2508.17158, Anthropic-affiliated) — harm is a subspace, not a direction; re-specifies our probe instrument.
- **Pruthi, Dhingra & Lipton** (ACL 2019, P19-1561) — the pre-LLM ancestor: reconstruction-only training in front of a frozen classifier. Cite as lineage; it never asked our question.
- **Zhao et al.** (NeurIPS 2025, 2507.11878) — harmfulness and refusal directions are separate, shown via reply inversion; also builds a latent guard matching LlamaGuard-3-8B. The nearest owner of the internal decode/decide split.
- **Wei et al.** (NeurIPS 2023) — "mismatched generalization." Already listed; the sweep confirms it is *verbatim* our hypothesis and cannot be treated as background.

## 4. Feasibility risks, named before they are checked

1. **Can the capability even be implanted?** If an 8B model cannot learn base64/Caesar decoding via LoRA, the design dies. BPE tokenisation makes character-level transforms genuinely hard. Candidate ordering by plausibility must come from evidence, not taste.
2. **Format collapse — NO LONGER A HYPOTHESIS. Measured and published (upgraded 2026-08-05, §5.1).** 2605.02914 QLoRA'd LlamaGuard-3-8B, WildGuard and Granite Guardian on 2,000 *benign* Alpaca examples: Granite Guardian refusal **85% → 0%**, CKA → 0.00, 77% of drift inside the safety subspace; WildGuard 35% → 5%. Guard fine-tuning destroys guard behaviour *by default*. **Clean-input DSR must reproduce published numbers after the implant, or the comparison is void** — and the matched control arm (§2.1 step 5, §2.4) exists precisely because this effect would otherwise masquerade as our result. Choice of guard now matters empirically: LlamaGuard-3-8B degraded least in that study and is the right first target.
3. **Implant confounds.** Decode training may teach "this format is suspicious", may leak harmful distribution, or may act as generic robustness. Verdict improvements could then have nothing to do with decoding. The control arm addresses the third; the first two need corpus design.
3b. **Failed implant masquerading as a conversion failure.** If LoRA simply does not teach decoding, the headline null appears anyway. The independent post-implant ability probe (§2.1 step 3) is the only thing standing between us and an unfalsifiable claim.
4. **The object-of-study objection.** A LoRA'd guard is not the deployed guard. The finding must be framed as a claim about the *relationship* between comprehension and enforcement, not about Llama Guard's shipped behaviour.
5. **Budget.** LoRA runs plus before/after evaluation, on one GPU with an 8h wall. The approval gate needs GPU/dollars/wall-clock before anything launches, and `conf/cost.yaml` is still tuned to pre-pilot assumptions that proved 2–4× optimistic.

## 5. Screening results

### 5.1 Alias-channel prior-art sweep — RETURNED 2026-08-05. Verdict: **survives, heavily re-scoped.**

This is the channel the original scoop check never ran (signature terms over a recent window only — the recorded defect). Running it cost the reshape two of its four components and re-sited the contribution. ~60 queries, 2005–2026, across sandbagging / capability-elicitation, multilingual safety transfer, causal mediation, benign-finetuning drift, pre-LLM adversarial-text and deobfuscation, guard robustness, cipher fine-tuning.

**Not found anywhere — the surviving novelty, and it is narrow:** no paper fine-tunes a *guard/classifier* solely on a decoding task for an encoding it provably cannot read, withholding all safety labels for encoded content, and measures the change in its safety *verdict*. Nor does any paper run the harm-direction inject/ablate double dissociation *inside a guard* on encoded inputs. The intersection **guard target × decode-only capability injection × verdict change as outcome** is open. Everything surrounding it is taken.

**DEAD as novelty claims — demote to cited instruments, never contributions:**

| Component | Owner | What survives |
|---|---|---|
| §2.3 continuous dial via in-context bijection encodings with tunable dispersion | **Bijection Learning, ICLR 2025** (2410.01294) — explicitly "the first… to use quantitative hyperparameters to scale encoding complexity" | Use as an instrument, cite on first use. Zero contribution weight. |
| §2.3 inject/ablate double dissociation on a harm direction | Three independent papers: **Zhao et al., NeurIPS 2025** (2507.11878, harmfulness vs refusal directions + a "Latent Guard" matching LlamaGuard-3-8B); *Knowing without Acting* (2603.05773); **MBZUAI** (2606.01196, fits the harm direction on high-resource activations, projects onto low-resource, adds and ablates — conceptually our activation half, one channel over) | Keep only as a confirmatory secondary with all three cited. It cannot carry the paper. |
| The decode-vs-decide framing itself | **Wei et al., NeurIPS 2023** — "mismatched generalization" is verbatim our hypothesis; SI-Attack (ICCV 2025) names the "comprehension ability vs safety ability" gap; CipherBench (2402.10601) measures decode accuracy separately and finds more-capable decoders are more vulnerable; **and our own AS-3 (2607.26574) already coins "the decode gap"** | Inherited framing, explicitly attributed. Never introduced as ours. |

**SERIOUS — the core move is bracketed on both sides:**

- **Covert Malicious Finetuning** (Halawi, Wei, Wallace, Wang, Haghtalab, Steinhardt — **ICML 2024**, 2406.20053). Phase I teaches GPT-4 the novel Walnut53 cipher on **harmless Alpaca data with no safety labels**; **Figure 5 ablates Phase II** and reports **25.8% harmful output on ciphertext vs 0.6% baseline**. That is the implant experiment already run — on a *generator*, as one ablation number inside an attack paper, never as a decode-vs-decide diagnosis, never on a guard's verdict.
- **When Safety Geometry Collapses** (Hossain et al., 2605.02914) — **the single most actionable finding of the sweep.** QLoRA on **LlamaGuard-3-8B, WildGuard and Granite Guardian** using 2,000 *benign* Alpaca examples: Granite Guardian refusal **85% → 0%**, CKA → 0.00, 77% of activation drift inside the safety subspace; WildGuard refusal 35% → 5%, ASR 9.2% → 17.1%. **Any LoRA on a guard wrecks its safety geometry regardless of what you taught it.** §4 risk #2 is therefore not a hypothesis — it is a measured, published outcome, and a decode-only fine-tune that flips the verdict proves *nothing* without a matched control arm.
- **Tongue-Tied** (Upadhayay & Behzadan, CALCS 2025) — LoRA on Newari and *fabricated* languages with non-harmful data alone jailbreaks the model; transfers to GPT-4 and Claude-3.5-Sonnet; logit-lens analysis. The fullest published instance of the whole move, in the language channel, on a generator, without the dissociation. This was §3's named top threat; it is real but not fatal.
- **Anthropic-affiliated cipher fine-tuning at 70B** (2508.17158) — reproduces CMFT across five cipher families and finds **cipher-induced harm is a subspace, not a direction: probes hold accuracy out to ~100 orthogonal directions.** A single harm direction is the wrong instrument; §2.3 must be re-specified as subspace-valued.
- **Pre-LLM ancestor** — Pruthi, Dhingra & Lipton (**ACL 2019**, P19-1561): a word-recognition model trained **only on reconstruction, never on classification labels**, in front of a **held-fixed** BERT classifier, 45.8% → 75%. Structurally our move, framed as defense engineering, with no verified-cannot-decode precondition and no can't/won't decomposition.

**The opening the sweep found — and it is stronger than what we walked in with.**

The literature holds a live, unresolved contradiction on exactly our question, and **both sides are confounded identically**:

- Halawi et al. measure phase-1-only at **25.8% harmful**.
- *Invisible Safety Threat* (Wan et al., **ICLR 2026**, 2603.08104) asserts the opposite **with no measurement at all**: a well-aligned model taught only the encoding "will merely acquire the steganographic technique without producing harmful content… remains aligned."
- **Both papers filtered refusal responses out of their encoding-learning corpus** (Halawi: "to avoid reinforcing refusals, we filter this dataset"; Wan: "filtered to exclude examples with refusal responses"). Neither number is clean — the training data was actively de-refused on both sides.

So the strongest available framing is no longer "we manufacture the middle cell." It is: **resolve a published contradiction that neither side measured cleanly** — decode-only fine-tuning that *keeps* refusal behaviour in the training distribution, on a **guard** rather than a generator, with an independent ability probe confirming the capability was actually acquired, and a matched control arm against generic-benign fine-tuning (mandatory, per 2605.02914). That is a resolution, not a re-run.

**Derived edits applied to this document:** §2.3 rewritten (dial → instrument; double dissociation → confirmatory secondary, subspace-valued); §3 extended; §4 risk #2 upgraded from hypothesis to measured fact; §2.4 gains the matched-control row.

**Bib debt:** ~20 new entries to stage in the science repo's `literature/model-internals/references.bib` as CANDIDATE. Deferred to one pass after all sweeps land, to avoid duplicate entries — the remaining checks are also literature-bearing. One item in the sweep is flagged UNVERIFIED by the agent (Wu, Kambhatla & Sarkar, ALW2 2018 — PDF extraction failed); it stays UNVERIFIED.

### 5.2 Hostile review of this reshape — *in flight*

### 5.3 Feasibility and cost — *in flight*

*Any FATAL finding routes back to S0 again or to kill; the owner's paper-count decision governs a second reshape.*
