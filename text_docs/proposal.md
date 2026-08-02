# model_internals_safety — proposal (stub)

Workflow stage: S0 Topic sourcing (as of 2026-08-02)

## Line scope

Safety research inside model parameters — two halves, one pipeline:

- **Training-time:** safety fine-tuning, unlearning, tamper-resistance-style interventions.
- **Non-training internals:** activation steering, representation engineering, weight editing, internals-level analysis of safety behavior.

Model-size-agnostic: the pipeline is identical across scales; small open models are the default experimental substrate (cheaper runs, more seeds, same conclusions).

## First-paper candidates (S0 — none committed; scoop checks pending)

1. **Training-side coverage generalization** *(recommended at founding — our proposal, scoop-check pending)*: does safety fine-tuning generalize across input representations (math/logic encodings, ciphers, low-resource languages, rendered text), and can training on one encoding family close the gap on unseen ones? Established anchors (to verify at S4): CipherChat (Yuan et al., ICLR 2024) and low-resource-language jailbreaks (Yong et al., 2023) show the transfer-gap phenomenon; Qi et al. (2023) established that fine-tuning erodes safety alignment. The candidate delta = a systematic cross-representation generalization study plus a training-side fix. Reuses the family's encoders as the measurement layer.
2. **Agent-side safety training** — extends the `llm_agent_security` line at training time; ranked lower for now (agent harness cost is high and that line is itself still founding).
3. **Resampling-robust training** — training-side robustness to best-of-N/variance attacks (extends the guardrail sibling's best-of-N axis); high variance.

## Feasibility items (open, S2)

- **Compute home:** LoRA-class fine-tunes of 7–13B open models fit the free NEU cluster today; verify cluster-access longevity before committing the cost model.

## Publication strategy

Deferred until the first-paper direction settles (S0 → S1 idea check → S2/S3). The venue scan will use the family's canonical conference timeline (`llm_guardrail_security/text_docs/shared/conference_timeline.md`) with a LIVE deadline check at that point.
