# S1 idea check — first paper (direction committed 2026-08-02)

Status: S1 COMPLETE (literature pass run 2026-08-02: 4-bucket deep read, ~20 papers full-text via alphaXiv; per-paper notes + BibTeX in the science repo's `literature/model-internals/`). Verdict: **proceed to S2.**

## The committed direction

**Recognition or action? Diagnosing and repairing safety failures under encoded inputs.** *(our proposal — the merge of proposal.md candidates 1 + 4)*

When a safety-aligned LLM complies with a harmful request delivered as a cipher, Base64, math/logic encoding, or low-resource language, is the failure:

- **(R) recognition failure** — harmful intent never becomes represented internally (the model can't decode the input), or
- **(A) action/calibration failure** — intent IS represented (linearly detectable, causally implicated) but never converts into refusal?

Phase 1 diagnoses R-vs-A per encoding family via internals probes; Phase 2 validates causally; Phase 3 tests the falsifiable consequence: **cross-encoding safety fine-tuning should repair A-failures (including on held-out families, in proportion to representational overlap) and should NOT create recognition where decode capability is absent** — a causal double-dissociation.

## Why this is open (scoop check, 2026-08-02, full-text verified)

The three nearest lines each hold one piece; none does the combination:

| Prior line | What it established | What it did NOT do |
|---|---|---|
| arXiv 2606.01196 (Aziz et al. 2026 — *the* closest) | Recognition-vs-action diagnostic (diff-in-means direction + causal ablation): low-resource-language failures are calibration failures, AUROC >0.85 while refusal collapses to 44% | ONLY natural-language translation; no ciphers/Base64/structural encodings; fix is few-shot threshold gating, explicitly disclaims adversarial robustness; no text-guard comparison |
| TrajGuard (arXiv 2604.07727) | Behavioral proof that internal decode-time monitoring catches cipher attacks text guards miss (Llama Guard 3 ASR 0.41 vs 0.14) | Detection only — never asks recognition-vs-action; no causal validation; no training-side fix |
| SALO (arXiv 2605.02958, ICML 2026) | Zero-shot detector generalization from plain safety data to unseen attack families; refusal is a sparse upstream trajectory | **Explicitly flags encoded/Base64 inputs as its untested boundary case** ("may not trigger the refusal trajectory") — a direct research-gap invitation |

Training axis (scoop sub-check): **CIFR** (Youstra et al. 2025, arXiv 2508.17158) is the nearest work — held-out cipher families, but it trains a *probe to detect malicious fine-tuning data* on a fine-tuning API; it does NOT train the model's own refusal behavior to generalize across encodings at inference time. Different training target, different threat model — must be distinguished on first mention. Circuit Breakers (Zou et al. 2024) already shows multilingual (incl. low-resource) robustness but tests **no cipher/Base64/structural encodings**; LAT (Sheshadri et al., TMLR 2025) tests none either. **No paper trains safety behavior on some encoding families and measures the model's own refusal generalization to held-out families.**

## Design skeleton (S2/S3 to refine; all choices below have named sources)

**Models:** Qwen2.5-7B-Instruct, gemma-2-9b-it, Llama-3.1-8B-Instruct — deliberately identical to 2606.01196 for comparability; optionally one ~3B for a scale point. All fit LoRA-class runs on the NEU cluster.

**Encoding taxonomy (the measurement layer — family encoders COPIED from llm_guardrail_security per charter):** graded obfuscation dial from plain text → paraphrase → low-resource languages → simple ciphers (ROT13/Caesar) → substitution ciphers → Base64 → math/logic encodings. Per family × model, first measure **decode capability** (can the model recover the plaintext at all) — this splits the space into can't-decode (no risk, no signal expected) / decode-and-refuse (safe) / **decode-and-comply (the gap the paper targets)**.

**Phase 1 — diagnostic (established recipes, extended):** difference-in-means harmfulness direction (Arditi et al. 2024 recipe) with two critical upgrades from the 2025–26 literature: (a) probe **harmfulness at the instruction-final token, separately from refusal at the post-instruction token** — they are distinct directions (Zhao et al., NeurIPS 2025); (b) probe **across token positions and decoding steps, not just a single readout** — under encodings, intent may only become legible during decode-time processing (TrajGuard's masking→unmasking finding; final-token probes drop 95%→64% on wrapped inputs, arXiv 2605.12726). Output: per-family, per-layer, per-position recognition curves.

**Phase 2 — causal validation:** directional ablation + activation addition (Arditi trio) on encoded inputs, plus the **reply-inversion test** (Zhao et al. 2025) to confirm the direction tracks the model's own harmfulness judgment, not refusal surface behavior.

**Phase 3 — training fix:** LoRA safety fine-tuning on a subset of encoding families (encoded harmful prompt → refusal; encoded benign → normal compliance, to control over-refusal), held-out-family generalization as the headline measurement. Baselines: plain safety-data mixing (Qi et al. 2023's mitigation), Circuit Breakers RR (cheap: ~20 min on one A100), optionally MSD-style self-distillation (arXiv 2605.02971) adapted from languages to encodings. Side-by-side: WildGuard / Llama Guard 3 on identical payloads (the guard-gap motivation table). Utility/over-refusal: MMLU-slice + XSTest.

**Falsifiable predictions:** P1 — decode-capable families show A-failures (recognition present, behavior fails); decode-incapable families show R (no signal). P2 — the fine-tune closes A-failures on held-out families, with generalization tracking representational overlap (probe-direction cosine across families). P3 — the same fine-tune does NOT create recognition in decode-incapable families (double dissociation; 2606.01196 never tests this). P4 — post-fix, the recognition signal is unchanged while the action/calibration gap closes (the fix re-binds behavior to existing representation rather than re-learning recognition).

## Controls the reviewers will demand (full checklist in science repo notes; the load-bearing five)

1. **Format-decorrelation 2×2** — benign content through the SAME encoders; probe must not fire on "looks encoded" (the fatal confound; template: arXiv 2603.19426).
2. **Selectivity/control tasks** (Hewitt & Liang 2019) on both plain and encoded conditions — not raw accuracy.
3. **Harmfulness ≠ refusal** — run the refusal-direction probe in parallel; differential transfer is itself evidence (Zhao et al. 2025).
4. **Length matching** across encoded/plain conditions (encodings inflate token counts).
5. **Scope honesty:** claims are about NATURAL zero-shot transfer, not adaptive adversaries — obfuscated-activation attacks defeat probes when the attacker optimizes against them (Bailey et al., arXiv 2412.09565); state this limitation explicitly, optionally add a light adaptive stress test.

## Anticipated objections + answers

- *"Why not just Circuit Breakers?"* — RR was never tested on ciphers/Base64/structural encodings; we run it as a baseline, and our diagnostic explains WHERE representation-space methods can work at all (only where recognition exists). If RR wins on held-out encodings, that is itself a publishable finding of the diagnostic framework.
- *"Isn't this CIFR?"* — no: CIFR detects malicious fine-tuning datasets with a monitor; we repair the model's own inference-time refusal behavior. Same held-out-family benchmark idea, different object entirely.
- *"Your probe reads surface features, not harmfulness"* — the five controls above; plus causal Phase 2.
- *"Incremental over 2606.01196"* — new axis (structural encodings vs natural language), new phenomenon cell (decode-capability split has no language analogue — every language in their study is 'decodable'), new fix class (training vs few-shot gating), the double-dissociation test, and the guard comparison. Framed as: their diagnostic vocabulary, our axis + causal repair test.

## Open items → S2

- Compute home: verify NEU cluster access longevity + queue reality for LoRA-class runs (proposal.md feasibility item; experiment-run approval gate applies before ANY run).
- Encoder copy list: which family encoders port from llm_guardrail_security, which encodings need new implementations (Base64/ROT13 are trivial; math/logic encoders exist in the family).
- Dataset assembly: AdvBench/HarmBench/MaliciousInstruct + Alpaca (the standard sets in every must-cite — comparability for free); refusal judging via HarmBench classifier (preferred over Llama Guard 2 + substring per Prakash et al. 2026).
- Venue scan at S2 settle (canonical timeline; AIA-27 assessed 2026-08-02 as skipped — see proposal.md).
