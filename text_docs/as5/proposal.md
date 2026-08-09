# model_internals_safety — proposal (stub)

Workflow stage: S1 idea check COMPLETE — direction committed (as of 2026-08-02), frame revised to v2 same day. **The Phase-0 pilot that gated S2 has ANSWERED (2026-08-03, re-baselined 08-05): the (B) cell is populated, so the direction is viable.** What the pilot and the two runs after it also established is that the *ladder* cannot carry the committed frame — see `evidence_and_story.md`, which is the paper-level synthesis of record and supersedes the scope written into the committed-direction paragraph below.

⚠️ **Read `evidence_and_story.md` before treating anything below as the plan.** The committed direction below is the frame as of 2026-08-02. Since then the benign judge arm (2026-08-07) demoted two of three sound rungs to controls and made every ASR number in this repo unreportable, and the echo crosstab the same day showed ~70% of the (S) cell is a behavioural state the four-regime taxonomy has no cell for. The paragraph below is kept as the S1 record, not as current scope.

**Committed first-paper direction (owner-confirmed 2026-08-02; v2 frame 2026-08-02): the merge of candidates 1 + 4** — "Can't, didn't, or wouldn't? Diagnosing and repairing safety failures under encoded inputs": a four-regime internals diagnostic (capability / deployment / binding failure per encoding family) + causal validation + a crossed two-intervention test (cross-encoding safety fine-tuning × decode-elicitation) whose four predicted null cells are the causal evidence. *(v1 was the two-regime "Recognition or action?" framing; v2 separates decoding ability from decoding deployment — see `s1_idea_check.md` §12.)* Full delta statement, design skeleton, predictions, and controls: `text_docs/s1_idea_check.md`. **This is paper AS-5.** Candidates below kept as the S0 record; unmerged remainders (2, 3) stay future-work seeds; **candidates 5 + 6 resolved into AS-6 — see below.**

## AS-6 — moved

AS-6's proposal now lives in its own namespace: **`text_docs/as6/proposal.md`**.
It sat here from its registration on 2026-08-05 until 2026-08-07 because the
as5 tree was another session's live area; the move was made in the structure
review (`text_docs/shared/pipeline_architecture.md` §5).

**This file is AS-5's, and its top `Workflow stage:` line is AS-5's state of
record.** AS-6's stage line is in its own proposal.

**Namespace convention — SETTLED 2026-08-07, and settled by READING the siblings
rather than by argument.** Neither `llm_guardrail_security` (which hosts FOUR
papers) nor `llm_agent_security` namespaces `src/` by paper: both cut code by
ROLE and cut `text_docs/` by paper. So the family convention is
**role-cut code, paper-cut docs**, and this repo already matched it — `probes/`,
`measurements/`, `models/` are shared instrument serving both papers, and only
the object-of-study layer (`guards/`) is paper-specific. No `src/` change was
made or is owed. Full record and the other two questions: §5 of the
architecture doc.


## Line scope

Safety research inside model parameters — two halves, one pipeline:

- **Training-time:** safety fine-tuning, unlearning, tamper-resistance-style interventions.
- **Non-training internals:** activation steering, representation engineering, weight editing, internals-level analysis of safety behavior.

Model-size-agnostic: the pipeline is identical across scales; small open models are the default experimental substrate (cheaper runs, more seeds, same conclusions).

## First-paper candidates (S0 — none committed; scoop checks pending)

1. **Training-side coverage generalization** *(recommended at founding — our proposal, scoop spot-check PASSED 2026-08-02, full S1 check pending)*: does safety fine-tuning generalize across input representations (math/logic encodings, ciphers, low-resource languages, rendered text), and can training on one encoding family close the gap on unseen ones? Established anchors (to verify at S4): CipherChat (Yuan et al., ICLR 2024) and low-resource-language jailbreaks (Yong et al., 2023) show the transfer-gap phenomenon; Qi et al. (2023) established that fine-tuning erodes safety alignment. The candidate delta = a systematic cross-representation generalization study plus a training-side fix. Reuses the family's encoders as the measurement layer. *Scoop spot-check (2026-08-02, alphaXiv recency pass): no direct hit on the encodings axis; the LANGUAGE axis of the same question is active (multilingual safety alignment via self-distillation, arXiv 2605.02971; cross-lingual shared safety neurons, arXiv 2602.01283; low-resource "action failures, not representation failures", arXiv 2606.01196) — S1 must differentiate the encodings-axis claims from the multilingual line. Enrichment option (from candidate 4's scoop check): use activation probes as the paper's internals analysis layer — where harmful intent surfaces per encoding, before/after the training fix — folding candidate 4's still-open slice into this paper.*
2. **Agent-side safety training** — extends the `llm_agent_security` line at training time; ranked lower for now (agent harness cost is high and that line is itself still founding).
3. **Resampling-robust training** — training-side robustness to best-of-N/variance attacks (extends the guardrail sibling's best-of-N axis); high variance.
4. **Internals-side detection across input representations** *(our proposal, added 2026-08-02 — reshaped from an external-guard idea; scoop-checked same day: PARTIALLY TAKEN)*: text-level guard models (WildGuard, Han et al. 2024; LlamaGuard family, Inan et al. 2023 — established) structurally miss encoded attacks because the external classifier never decodes them, while the target model must — so probe the target model's own activations instead. Scoop status (2026-08-02, alphaXiv + full-text check): the motivating hypothesis is already demonstrated for ciphers by TrajGuard (arXiv 2604.07727 — shows Llama Guard 3/Qwen3Guard fail on cipher prompts while internal-state monitoring succeeds), and the exact plain-text-probe → zero-shot-transfer + layer-emergence methodology exists for the LANGUAGE axis (arXiv 2606.01196). Still-open slice: one unified study of zero-shot probe transfer across the FULL encoding taxonomy (ciphers, Base64, math/logic, rendered text, low-resource languages), per-encoding layer-emergence sweep, and a head-to-head vs text-level guards on identical payloads (no existing paper runs that comparison; SALO, arXiv 2605.02958, explicitly flags encoded inputs as its untested limitation). Must-cites: 2606.01196 · 2604.07727 · 2605.02958; secondary: GUARD-SLM 2603.28817, 2606.10487, SIREN 2604.18519, HiddenDetect (ACL 2025). Verdict: too narrow to carry a first paper alone at a top venue; strongest as the analysis layer folded into candidate 1 (see its enrichment note). Cheapest candidate computationally (forward passes + linear probes, no training runs).

5. **[RESOLVED 2026-08-05 → AS-6's *convert* link.** Not a paper on its own, and no longer an axis bolted onto AS-5's guard-gap table — it is one of AS-6's two links, where the guard is the object of study rather than a baseline. AS-5 keeps its guard-gap table unchanged as contribution #4.**]** **Variation-side internals: why a gate classifier collapses under payload variation** *(our proposal, added 2026-08-04 from the guardrail sibling's AS-4 Round 8 — NOT scoop-checked; do not treat as open until it is)*: the sibling measured a clean double dissociation between the two defense TYPES, and only one half of it has an internals account. On Llama-3.1-8B (T=1.0, gpt-5-mini judge, ensemble any-of-100, campaign `bestofn_attack_r8_factorial`), holding one factor fixed while moving the other:
   - a **transform** defense (SAGE, which routes the request back through the target for a self-check) is broken by **encoding**: 10 → 67 behaviors, while payload variation moves it only 67 → 74 — inside the 67–84 campaign spread, so not a real effect;
   - a **gate** defense (LlamaGuard-3, a classifier on the prompt) is broken by **variation**: 13 → 57, while encoding moves it only 52 → 57.

   **The encoding half is already ours and is NOT open** — it is this repo's (D) deployment regime (the target can decode but does not deploy decoding during the self-check), whose black-box fix is the sibling's AS-3 *Decode Gap* (`modality_complete`'s recover→decode step). Writing it separately would take contribution from AS-3 and from this repo's committed direction at once.

   **The open slice is the variation half.** The committed ladder (§`s1_idea_check.md`) is entirely *obfuscation surface form* — ciphers, Base64, zero-width, homoglyph, fullwidth, combining marks. Nothing in it asks why a classifier's decision boundary fails under *semantically equivalent paraphrase at fixed encoding*, because that is not a decoding question about the target at all — it is a boundary question about the guard. Candidate 4's head-to-head (internals probe vs text-level guard on identical payloads) is the natural place it would attach: the sibling's result predicts the guard should degrade sharply along the variation axis where an internals probe should not, which would be a second, independent axis of evidence for candidate 4's thesis rather than a new paper.

   **Status and cautions.** (a) **No scoop check has been run** — classifier robustness under paraphrase is adjacent to a large adversarial-robustness/NLP-attack literature, so the owner's standing rule applies: a named-open gap is a race signal, and this must be checked before it is called open. (b) The sibling's numbers are behavioral only; nothing internal has been measured on the variation axis. (c) Scope note: this attaches to candidate 4 / the committed direction as an axis, and is recorded here as a candidate strictly so the paper-count decision is made in this repo, where the line is owned — the owner will settle how many papers this line carries (2026-08-04).

6. **[RESOLVED 2026-08-05 → AS-6's declared EXTENSION, not a paper.** The vision-*target* framing below was dropped: it needed a capture stack this repo does not have (vision tower + projector), and HiddenDetect sits closest to it. What survives is the vision *channel* as AS-6's **transmit** link — a guard-side question, where AS-2's OCR-fidelity control is the dose-response — held out of v1. The AS-2 decoy seed below is now DEAD, not on hold: ≈80% of it was the oracle leak.**]** **Vision-channel internals: where harmful content becomes readable when it arrives as pixels** *(our proposal, added 2026-08-04 from the guardrail sibling's AS-2/AS-3 image work — NOT scoop-checked; do not treat as open until it is)*: the committed ladder is **text-only by construction** — "exactly invertible" is a token-level property, and `conf/models/` holds only text LLMs (Qwen2.5-7B, Llama-3.1-8B, Qwen-0.5B). Nothing in the design touches the case where the payload arrives through the vision encoder. The sibling owns that surface outright: a renderer suite (`ir_plain`, FigStep, typography, flowchart, decoy renders), per-render ASR on open-weight VLM targets (Qwen2.5-VL-7B, InternVL3), and — the piece nobody outside has — **per-render OCR-fidelity measurements**.

   **The question.** When harmful content is rendered rather than tokenised, does the harm representation form *before* the layer where the refusal decision is committed, or does late vision-token fusion push semantic recovery past that point? This is the same four-regime frame (can't / didn't / wouldn't) asked of a channel the ladder cannot express, and it predicts a *depth* signature per render type rather than a per-rung binary.

   **Why OCR fidelity is load-bearing, not a detail.** Without it a null is uninterpretable — a flat probe curve means either "the model represented nothing" or "the attack never transmitted". The sibling already hit this exactly: InternVL3 `code_attack ir_plain` measured OCR ≈0.10 and its 24% ASR cell was ruled uninterpretable and dropped from the suite. An outside group running this without fidelity controls would publish that confound. That control is the strongest reason this slice is ours.

   **Nearest prior — the delta must be argued against it, not assumed.** **HiddenDetect (ACL 2025)** already does hidden-state monitoring for jailbreak detection on VLMs and is *already* in candidate 4's secondary must-cites, so the vision axis is **not** virgin territory; the taken work is detection-framed. The candidate delta is the *layer-depth-of-recovery* question conditioned on render legibility — timing and its OCR-controlled dose-response, not "can hidden states detect this". Candidate 4's scooping papers do not reach here (TrajGuard 2604.07727 = ciphers, text; 2606.01196 = language axis, text), but HiddenDetect does, and a scoop check must clear the depth/timing framing specifically before this is called open.

   **Substrate cost (be honest):** a real extension, not a free add-on — VLM targets are not in the model registry, and the vision tower + projector + LM stack is a different capture surface than the residual-stream code assumes.

   **Related seed, ON HOLD (do not build on it yet).** The sibling's AS-2 decoy finding — a content-*unrelated* image shifting whether a defense engages — is unexplained and entirely ours, but its magnitude is collapsing: an oracle leak found 2026-08-03 accounted for ≈80% of the published effect, and the threshold-model-faithful re-measurement is in flight as of 2026-08-04. Revisit only if the deployable arm shows a surviving effect worth explaining.

   **Sequencing.** Gated on phase-0 returning a validated design (TODO item 2). Opening a second front before the gating pilot has spoken is the widen-before-you-validate error; recorded here so the paper-count decision for this line is made in this repo, per candidate 5's note.

## Feasibility items (open, S2)

- **Compute home:** LoRA-class fine-tunes of 7–13B open models fit the free NEU cluster today; verify cluster-access longevity before committing the cost model.

## Publication strategy

Deferred until the first-paper direction settles (S0 → S1 idea check → S2/S3). The venue scan will use the family's canonical conference timeline (`llm_guardrail_security/text_docs/shared/conference_timeline.md`) with a LIVE deadline check at that point.

⚠️ **The nearest-window assessment that stood here has been MOVED OUT OF THIS REPO
(2026-08-08), and the reason is a policy, not tidiness.** It named a specific
venue, its special track and its two deadline dates, in this paper's own
venue-strategy section, in a repo that is **public and name-linked**.

A public repo is *non-anonymous online material*. Checked against the primary
source 2026-08-08 (the venue's own instructions, quoted in the science organ's
venue records): under double-blind review, preprints and public material are
permitted on two conditions — the anonymous PDF must carry no citation or
pointer to the non-anonymous material, **and the non-anonymous material must not
reference the fact that the work was sent anywhere**. Breaking either is grounds
for summary rejection.

So the answer to the open question the paper skeleton carried is **yes, a public
name-linked repo is fine** — provided this repo never names where the work goes.
Window assessments, dates and track choices live in the venue records (private)
and the family's canonical timeline; the pointer above is all that belongs here.

⚠️ **This paragraph is itself written to the rule** — it does not name the venue,
because a public file explaining our own routing would be the same violation one
level up. `tests/test_public_repo_hygiene.py` enforces it, and it caught two
drafts of this very section.
