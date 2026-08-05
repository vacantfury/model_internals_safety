# S1 — external idea check, AS-6 (guard internals)

**Workflow stage: S1 external idea check — PACKAGE READY, `[waiting: haoyu]` (as of 2026-08-05).**

S1 is an owner-hands gate: cspaper.org has no API and automation was declined (recorded decision), so the owner runs it. This file is the execute-ready package and the place the return lands.

**Prerequisites, both done before this package was built:**
- Scoop check — DONE 2026-08-05, verdict **Level 3 (Medium Overlap), FINAL**, all candidates deep-dived. Record: `text_docs/as5/proposal.md` §"AS-6 — the second paper"; log in gitignored `outputs/scoop_check/2026-08-05/`.
- Guard-capture feasibility spike — DONE at config level (verified from live HF configs), weight-loading half owed.

**Route on return:** pass → S2 (doability + home decision) · critiques → refine and re-screen (S0) · kill → record per the workflow's kill rules.

---

## Where to go

<https://cspaper.org/idea-check>

## What to paste

The text between the markers below, verbatim. It is deliberately self-contained — an outside reader needs no access to this repo.

START

Title: Which link broke? Attributing safety-guard failures to decoding versus conversion

PROBLEM

LLM safety guard models (Llama Guard, WildGuard, ShieldGemma and similar) are the dominant deployed defense against harmful prompts, and essentially every published measurement of their failure is an end-to-end rate: defense success rate, bypass rate, or attack success rate. That single number structurally confounds three distinct events, which call for three different repairs:

  (a) the guard never received the evidence;
  (b) the guard received it but never decoded it;
  (c) the guard decoded it and never converted that into a block.

Nothing in the literature tells a practitioner which of the three they are facing, so guard hardening proceeds by trial. A pre-decoder is wasted effort on a guard that already reads the payload; boundary work is wasted on a guard that never read it at all.

NOVELTY CLAIM

We open the guard and measure which link broke, using the guard's own internal activations rather than its output. Concretely, on identical harmful payloads presented across a graded ladder of text encodings — from cipher-class rungs (base64, Caesar, morse) through surface-preserving transforms that leave the text humanly readable (zero-width joiners, homoglyphs, fullwidth characters, combining marks) — we probe the guard's residual stream during its own classification forward pass to ask two questions:

  (1) is the plaintext content recoverable inside the guard?
  (2) is harm represented?

and compare both against the safe/unsafe verdict the guard actually emits. This converts an end-to-end failure rate into a per-link attribution.

POSITIONING AGAINST THE CLOSEST PRIOR WORK

- SIREN (arXiv 2604.18519) probes guard-model internals, but to BUILD a better detector, on standard unencoded safety benchmarks. It reports that a probe on Llama-Guard-3-8B's internals beats that guard's own output by roughly 10 F1 points. That is evidence for our premise -- a guard's activations carry more than its verdict emits -- but it never decomposes a failure, and never touches encoded input.

- DecipherGuard (arXiv 2509.16870) establishes behaviorally that guards lose 24-37% defense success rate under obfuscation, and that inserting a decipher layer recovers much of it. Black-box throughout: no internals, no attribution.

- Gamma-Guard (EMNLP 2025) does look inside a guard under attack, but qualitatively -- PCA scatter at two layers, per-layer safe/unsafe probability curves, one attention heatmap, on single illustrative samples. Its twelve attacks are standard adversarial-NLP (character perturbation, gradient word substitution, sentence paraphrase) with no encoding rung, and its causal story (embedding noise, attention dilution, boundary drift) never distinguishes did-not-decode from decoded-and-did-not-flag.

- The represent-versus-act distinction is established for TARGET models (e.g. "Knowing without Acting"; "LLMs Encode Harmfulness and Refusal Separately"). We transport it to the defense, and add the decode link, which is largely irrelevant for a target model that must decode to answer, and critical for a guard that need not decode to classify.

INTENDED CONTRIBUTION

  1. A per-link attribution method for guard failures, measured inside the guard.
  2. The empirical claim that two guards with matching end-to-end defense success rates fail by different links -- which would demonstrate that the field's standard metric is the wrong instrument for choosing a repair.
  3. A prediction that the attribution selects the repair: rungs attributed to decode-failure are fixed by a pre-decoder and not by boundary work, and rungs attributed to conversion-failure are the reverse, with the predicted null cells carrying the evidence.

WHAT WE MOST WANT CRITIQUED

  (a) Is per-link attribution a large enough contribution for a top-tier venue, or does it read as a diagnostic note attached to someone else's method?

  (b) Can a reviewer collapse our decomposition into ordinary distribution shift -- i.e. is "the guard did not decode it" meaningfully different from "the guard's classifier is out of distribution on this input"? We think it is, because the two predict different repairs, but this is the objection we are least sure of.

  (c) Is there prior work we have missed under different vocabulary? Adversarial-NLP and interpretability may name this problem differently than the LLM-safety literature does.

VENUE CLASS: top-tier ML / NLP / security main conference (ICLR, ACL, USENIX Security class). An AI-safety track is acceptable. Not a workshop.

END

## What to bring back

Two things, nothing more:

1. **The verdict** — whatever pass/refine/kill signal the check returns.
2. **The main critiques**, especially any answer to (b) — that is the objection this design is least defended against, and a good critique there is worth more than a pass.

---

## Return — results of the check

*(empty; filled in when the owner returns the check)*
