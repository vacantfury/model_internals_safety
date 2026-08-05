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
3. The result is, by construction, **a guard that can read E but was never taught to flag E** — the previously empty middle cell, manufactured rather than hunted.
4. Measure the safety verdict on harmful prompts in E, before versus after:
   - verdict does **not** improve → a **conversion** failure, *causally created* rather than inferred from a probe;
   - verdict **does** improve → decoding was the binding constraint, and the repair is decode-side.

**Why this answers the rejection.** It defeats the empty middle by manufacturing the middle. It defeats the collapse objection because the claim no longer rests on a probe's readout at a layer index we chose — it rests on an intervention we performed and a behaviour change we measured. "Decoded but unused" stops being an inference and becomes an experimental condition.

**And it changes the contribution from a diagnosis to a mechanism.** The old contribution was "here is an attribution schema". The new one is a causal claim about how safety behaviour relates to input comprehension in a classifier: *does teaching a guard to read a format make it safe on that format?* If the answer is no, that is a real and slightly alarming result — comprehension does not imply enforcement — and it is the guard-side analogue of the multilingual safety-transfer failure, which is a recognised and well-cited problem.

### 2.2 Predicted nulls (the evidence, and the part that needs power)

- The implant does **not** change verdicts on rungs the guard could already read.
- The implant does **not** change verdicts on plaintext.
- The implant does **not** change clean-input DSR (this one is also the sanity gate — see §4).

Null claims require **TOST equivalence bounds with a pre-registered margin**, not `p > 0.05`. This was a named defect of the rejected version and must not survive into this one.

### 2.3 Supporting causal tests

- **Inject / ablate.** Fit a harm direction on plaintext; inject where absent under encoding, ablate where present. A double dissociation defines the two links causally and is immune to the re-description charge.
- **A continuous dial.** In-context bijection encodings with tunable dispersion (Bijection Learning, ICLR 2025) let rungs be placed at the edge of a *specific* guard's ability — decode difficulty as a knob, not a fixed property.

### 2.4 Controls carried forward from the review

| Control | Why it is mandatory |
|---|---|
| Probes fit on **plaintext only**, tested on encoded | a probe fit on encoded data can learn the encoding; this is the difference between a measurement and an artefact |
| **Beat the tokenizer null model** per rung | characters-per-token alone separates encoded from natural text near-perfectly (Broken-Token, 2510.26847); if a 2-parameter statistic matches our probes, the paper has no reason to exist |
| Per-rung **selectivity** over shuffled-label controls | Hewitt & Liang 2019; report the gap, never raw accuracy |
| **Fourth cell: blocked WITHOUT decoding** | the guard as format detector rather than harm detector — the most policy-relevant cell, and it predicts brittleness |
| Prompt length bounded **below each guard's inspection window** | transmit is not free in text (Prompt Overflow, 2605.23196) |
| **Tri-state everything** | unlicensed probe → `None`, never `False`. Inherited AS-5 law; scoring an unlicensed probe as "never decoded" would manufacture the paper's own headline |

## 3. Prior art this must be argued against (not assumed open)

- **Zhou et al., Findings of EMNLP 2024 (2406.05644)** — recognised-but-not-converted on chat models, shown causally via Logit Grafting, 7B–70B. The *concept* is theirs. What is left: the object (a guard, not a chat model), the input axis (encodings, not jailbreak templates), and the decode link, which has no counterpart in their work.
- **Our own AS-3** (2607.26574) — selected the pre-decoder repair from black-box observation alone. Any contribution phrased as "attribution is needed to pick the repair" is dead on arrival; the surviving phrasing is that attribution predicts *when the pre-decoder is NOT enough*, which the implant tests directly.
- **SIREN (2604.18519)** — probes guard internals to build a detector; its App. C result (LlamaGuard-3-8B 77.0 → 87.1 F1 from its own activations) is motivation, never a finding of ours.
- **Wei et al., NeurIPS 2023** — "mismatched generalization"; must be cited and explicitly differentiated.
- **The multilingual safety-transfer literature** is the closest conceptual analogue and the biggest live threat: if someone has already run *teach the model the language, does safety follow?*, the implant is that experiment with an encoding in place of a language. **This is the alias-channel sweep's top target.**

## 4. Feasibility risks, named before they are checked

1. **Can the capability even be implanted?** If an 8B model cannot learn base64/Caesar decoding via LoRA, the design dies. BPE tokenisation makes character-level transforms genuinely hard. Candidate ordering by plausibility must come from evidence, not taste.
2. **Format collapse.** Llama Guard emits a rigid `safe` / `unsafe\n<category>`. Free-form decode training plausibly wrecks it. **Clean-input DSR must reproduce published numbers after the implant, or the comparison is void.**
3. **Implant confounds.** Decode training may teach "this format is suspicious", may leak harmful distribution, or may act as generic robustness. Verdict improvements could then have nothing to do with decoding.
4. **The object-of-study objection.** A LoRA'd guard is not the deployed guard. The finding must be framed as a claim about the *relationship* between comprehension and enforcement, not about Llama Guard's shipped behaviour.
5. **Budget.** LoRA runs plus before/after evaluation, on one GPU with an 8h wall. The approval gate needs GPU/dollars/wall-clock before anything launches, and `conf/cost.yaml` is still tuned to pre-pilot assumptions that proved 2–4× optimistic.

## 5. Screening results

*(pending — three checks in flight, launched 2026-08-05: alias-channel prior-art sweep · hostile review of this reshape · feasibility and cost. Record each verdict here. Any FATAL finding routes back to S0 again or to kill; the owner's paper-count decision governs a second reshape.)*
