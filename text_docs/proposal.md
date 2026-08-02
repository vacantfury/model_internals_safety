# model_internals_safety — proposal (stub)

Workflow stage: S0 Topic sourcing (as of 2026-08-02)

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

## Feasibility items (open, S2)

- **Compute home:** LoRA-class fine-tunes of 7–13B open models fit the free NEU cluster today; verify cluster-access longevity before committing the cost model.

## Publication strategy

Deferred until the first-paper direction settles (S0 → S1 idea check → S2/S3). The venue scan will use the family's canonical conference timeline (`llm_guardrail_security/text_docs/shared/conference_timeline.md`) with a LIVE deadline check at that point.

Nearest-window fact (from the canonical timeline, captured 2026-08-02): AAAI-27 AI Alignment (AIA) special track — abstract 2026-08-14, full paper 2026-08-21, AoE; CFP text/submission site still pending as of 07-29. Assessed 2026-08-02: too tight for a training-side first paper from a pre-code repo; a probe-only paper would fit the window but its open delta is narrow (candidate 4's scoop status). Next major windows to scan at S1 settle: ICLR 2027 (CFP pending), USENIX Security 27 cycle 1 (08-18/08-25).
