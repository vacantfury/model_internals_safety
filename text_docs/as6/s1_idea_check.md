# S1 — external idea check, AS-6 (guard internals)

**Workflow stage: S1 external idea check — RUN AND PASSED 2026-08-05. Next: S2 (doability + home decision), pending the adversarial separation pass on objection (b).**

S1 is an owner-hands gate: cspaper.org has no API and automation was declined (recorded decision), so the owner runs it. This file holds the package that was submitted and the return it produced (§"Return", below).

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

**Run by the owner 2026-08-05 08:14. Job `6c4bd960-b729-45f1-8a9e-04bfcffb195f`. Submitted title: "Attributing Guard Failures to Decoding and Conversion in LLM Safety Filters". Raw report pasted verbatim below the horizontal rule at the end of this file.**

### Verdict: PASS → route to S2 (doability + home decision)

cspaper's idea check returns positioning and a gap list rather than a pass/fail token; on its own terms the result is unambiguous. The two lines that carry it, quoted:

> "your idea **uniquely offers a diagnostic tool to decompose these exact failure rates** via internal probing. It uniquely addresses the operational gap of attributing whether a text-obfuscated prompt bypasses a guard due to a decoding failure or a representation conversion failure."

and, under **Opportunities & gaps**, two statements that map one-to-one onto AS-6's two load-bearing claims:

> "**No papers address the operational distinction between input decoding failures and representation conversion failures within safety classifiers.**"
>
> "**None of the empirical studies investigate the impact of surface-preserving text obfuscations, like homoglyphs or zero-width characters, on internal model activations.**"

The second is worth pausing on: it is an independent statement that the **comprehension band** — the exact rungs AS-5's pilot handed us (`zero_width`, `homoglyph`, `fullwidth`, `combining_marks`, `tag_block`) — is unexamined against internal activations. That band is where AS-6's convert cell lives, so the gap statement lands precisely on the money cell rather than adjacent to it.

### What this result does NOT establish — read before treating it as strong confirmation

1. **The retrieval pool missed every one of our known nearest priors.** None of SIREN (2604.18519), DecipherGuard (2509.16870), Gamma-Guard (EMNLP 2025), or Prompt Overflow (2605.23196) appears in the ten retrieved papers. So the "no papers address…" gap claim is drawn from a pool that did not contain the closest work, and it must be reported as *consistent with* our scoop check, never as independent confirmation at full strength. Our own scoop check outperformed this retrieval.
2. **The pool is off-target at the top.** The #1 match at 81% is an RL paper on entropy scheduling under environment drift, which cspaper itself annotates "The findings on dynamic regret in RL do not transfer." Scores cluster in a narrow 78–81% band — nothing retrieved is genuinely close, which is weak evidence of an empty neighbourhood and equally consistent with a query that under-retrieved.
3. **Critique (b) went unanswered.** The report does not engage the objection we flagged as the one we were least sure of — whether the decode/convert decomposition collapses into ordinary distribution shift. cspaper's idea check is a retrieval-and-positioning instrument, not an adversarial reviewer, and should not be expected to do that job. **The workflow's own rigor checklist covers it: adversarial separation in a fresh context. Run that before S2 closes** — its result belongs in this section.

*(Instrument note, for the running record AS-5 keeps in its §12.3: this is the third cspaper run in the family. It is useful for positioning and for the gap statements, and demonstrably weak at surfacing nearest prior art — on this run it retrieved zero of four known-nearest papers. Treat it as a positioning instrument, not a scoop check.)*

### Papers worth adding to the corpus from the retrieval

Genuinely useful despite the weak pool — stage and verify via `lit-review-loop` Phase 2:
- **#8 LARF** (EMNLP 2025) — safety-sensitive *layers* identified from representations; the closest methodological neighbour to AS-6's layer localisation, and cspaper's own note says it "directly support[s] and transfer[s] to your idea's conversion attribution methodology".
- **#9 Interpretation Meets Safety** (EMNLP 2025) — survey at exactly our intersection; its stated finding that "most current circuit analysis lacks application to complex safety scenarios" is a citable framing anchor.
- **#6 PRP** (ACL 2024) — universal adversarial prefixes against guard models; prior art on *attacking* guards, useful for threat-model positioning.
- **#4 DeRTa** (ACL 2025) — refusal position bias; already known to AS-5, and the token-position dynamics bear on where the convert step reads.

### Adversarial separation pass (2026-08-05) — verdict: REJECT AS SCOPED

Run in a fresh hostile-reviewer context per the workflow's idea-check rigor checklist, precisely because cspaper does not do this job. **It overturns the routing.** Its three fatal-as-scoped objections, with the two load-bearing factual claims verified before being accepted:

**① The empty middle — fatal, and it is built from our own pilot data.** LlamaGuard-3-8B is a fine-tune of Llama-3.1-8B. AS-5's phase-0 pilot measured that base at decode-and-restate similarity **0.03–0.30 across all twelve cipher rungs** — it cannot read them. WildGuard is Mistral-7B-based, older and weaker. So:
- **Cipher band: the decode link is empty by construction.** Every failure is trivially "can't". No (B) cell to attribute.
- **Surface band: no decoding is required.** Zero-width, homoglyph, fullwidth, combining marks, tag block *are* plaintext modulo the tokenizer. "Did the guard decode it" degenerates into "did the tokenizer preserve it" — answerable in milliseconds with no forward pass, no probe, no GPU.

The regime where a guard genuinely decodes a non-trivially transformed input — the only regime where "which link broke" is a real question — is **essentially unpopulated at 8B**.

> **This retracts my own reasoning, and the retraction matters more than the objection.** `design_notes_PREMATURE.md` §3 argued that "AS-5's dead band is AS-6's populated cell" — that the cipher band's inertness, useless to AS-5, becomes a positive result for AS-6 because "never decoded" *is* the answer to RQ1 there. **That is wrong.** An attribution predictable from the base model's decode ability, without opening the guard, is not an attribution — it is a restatement of a known capability limit wearing interpretability vocabulary. I inverted a null result into an asset instead of confronting it. Treat §3 of the design notes as withdrawn.

**② The collapse objection — fatal as scoped, and cspaper never touched it.** A transformer guard has no module labelled "decoder" and none labelled "converter"; it is one continuous map. "Feature present at layer k, readout wrong" *is the definition* of distribution shift in a deep network — what Wei et al. (NeurIPS 2023) called **mismatched generalization** in one phrase, with no probes and no layers. Worse, token identities of a base64 string *analytically determine* the plaintext, so a sufficiently expressive probe achieves near-perfect "recoverability" by learning base64 itself — reading the guard's **input**, not its computation. And the natural defence (a guard may represent decoded content without being able to emit it) removes the behavioural anchor that made the AS-5 instrument credible, leaving the probe as sole arbiter of a quantity with no ground truth. **①  and ② are coupled: escaping either lands in the other.**

**③ Prior work is closer than the scoop check found — a defect in my check, not in the idea.** Two verified misses:
- **Zhou et al., "How Alignment and Jailbreak Work", Findings of EMNLP 2024 (arXiv 2406.05644)** — **read and verified 2026-08-05.** Weak classifiers on intermediate hidden states separate malicious from normal inputs at >95% in the *early* layers; alignment associates those early concepts with emotion tokens in the middle layers (16–24); and verbatim, *"Jailbreak disturbs the transformation of early unethical classification into negative emotions."* Table 2 shows classifiers separate jailbreak inputs early too — the model **recognises and fails to convert**. They then prove it *causally* via **Logit Grafting**. That is the convert-link finding, published 2024, across 7B–70B and five model families. Differences that survive: target chat models not guards, jailbreak templates not encodings, and no decode link. But the conceptual novelty of "represented but not converted" is **established, not ours**.
- **Our own AS-3.** "Recover, Decode, Reguard" (arXiv 2607.26574, 29 Jul 2026) is the guardrail sibling's Decode Gap paper — confirmed present in the science `llm-security` bib and matching the portfolio's AS-3 = C entry. It **selected and validated the pre-decoder repair from black-box observation alone**. If attribution machinery were necessary to choose the repair, that paper could not exist — which guts contribution (3) as written.

**Why the check missed these:** I ran signature terms over a recent window only. The workflow's rigor checklist specifies a **dual-channel** check — signature terms recent *plus* alias terms an adjacent subfield would use over a longer window. Zhou et al. is 2024 and speaks of "emotion association", not decode/convert. Re-run the alias channel before any re-screen.

Serious-but-addressable objections worth carrying regardless: probe validity on an encoding ladder is maximal-risk because every rung has a unique surface signature (*False Sense of Security*, arXiv 2509.03888, verified); a **tokenizer null model must be beaten explicitly** (Broken-Token, arXiv 2510.26847 — characters-per-token alone separates encoded from natural text near-perfectly); link (a) is un-instrumented yet may dominate real deployments (tokenizers dropping payloads, arXiv 2504.11168, Emoji Attack ICML 2025); a fourth cell is missing (**blocked without decoding** — the guard as format detector, the most policy-relevant cell); B=5–13 at n=100 cannot support "the nulls carry the evidence" without **TOST equivalence bounds**; and "matching DSR" across two guards is **threshold-manufacturable** and confounded on six axes.

**The constructive path it names — all cheap and inside existing infrastructure except the last:**
1. **Causal, not correlational:** fit the harm direction on plaintext, then **inject** where absent and **ablate** where present. Decode-failure ≡ direction absent *and* injecting flips the verdict; convert-failure ≡ direction present at plaintext strength, injection does nothing, late-layer intervention flips it. A double dissociation is immune to the re-description charge. Hooks and forward passes; the capture cache already supports it.
2. **Dial the decode link instead of observing it** — bijection encodings with graded, verified per-model dispersion (Bijection Learning, ICLR 2025), and/or LoRA-finetune a guard on a cipher and show the attribution *moves*. Manipulating the link is what turns a taxonomy into science, and it is what fills the empty middle.
3. **Probes fit on plaintext only, tested on encoded**, plus per-rung selectivity controls.
4. **Beat the tokenizer null model** per rung.
5. (Expensive, and what separates a paper from a workshop note) the **repair 2×2 actually run**, with TOST bounds and ≥4 guards spanning architectures.

### Route taken — REVISED

**NOT S2. Route: critiques → refine and re-screen (S0).**

cspaper passed the idea; the adversarial pass says reject as scoped, and it is the better instrument here — it argued from our own pilot numbers and surfaced two prior works our scoop check missed. Under the workflow's three outcomes this is "loop back", and the named stage is S0.

**This is a refine, not a kill.** The object cut (target internals vs defense internals) survives untouched; what fails is the *scoping* — an observational decode/convert split on two 8B guards over a ladder whose two bands are respectively impossible and trivial for them. Items 1 and 2 above are a different and better paper from the same materials: causal manipulation of a dialed decode link inside a guard. That reshape is the S0 work.

**Owner decision this now needs** — recorded rather than guessed: whether to spend the reshape effort at all, given AS-5 is mid-flight and this line's paper count is his call.

---

## Raw report (verbatim, as returned)

Attributing Guard Failures to Decoding and Conversion in LLM Safety Filters
Job ID: 6c4bd960-b729-45f1-8a9e-04bfcffb195f

Completed on Aug 05, 2026 08:14

View submitted idea
Show abstract
Related papers

10

From last 2 years

10 / 10

Your idea in context
Your idea fits squarely into the LLM safety evaluation landscape, bridging the gap between adversarial stress-testing and mechanistic interpretability. While the retrieved papers mostly propose new guard architectures or training techniques to lower aggregate failure rates, your idea uniquely offers a diagnostic tool to decompose these exact failure rates via internal probing. It uniquely addresses the operational gap of attributing whether a text-obfuscated prompt bypasses a guard due to a decoding failure or a representation conversion failure.

NeurIPS
×3
ACL
×2
EMNLP
×2
ICML
×1
NAACL
×1
ICLR
×1
What the field looks like
The dominant technical theme is improving, evaluating, or attacking LLM safety alignment frameworks and secondary safeguard models. This likely caused co-retrieval because the your idea focuses on diagnosing the exact failure points of these safety filters using adversarial inputs and internal representations.

high confidence
Methodological spectrum: Mostly empirical systems building and evaluation (proposing new guard architectures, training methods, or attacks), with one theoretical reinforcement learning outlier and one comprehensive literature survey.

◆
Also touches on
Use of internal states or explicit reasoning capabilities to improve or interpret safety mechanics#2
#3
#8
#9
Focus on specific structural vulnerabilities in safety guard models or refusal mechanisms#4
#10
#6
✦
Opportunities & gaps
No papers address the operational distinction between input decoding failures and representation conversion failures within safety classifiers.
None of the empirical studies investigate the impact of surface-preserving text obfuscations, like homoglyphs or zero-width characters, on internal model activations.
Related work (10)
1
Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning
Tongxi Wang, Zhuoyang Xia, Xinran Chen, Shan Liu
ICML 2026

🔍 Worth a look
81% match

Real-world reinforcement learning often faces environment drift, but most existing methods rely on static entropy coefficients/target entropy, causing over-exploration during stable periods and under-exploration after drift (thus slow recovery), and leaving unanswered the principled question of how exploration intensity should scale with drift magnitude. We show that, under standard assumptions, entropy scheduling in non-stationary maximum-entropy RL can be cast as the dynamic-regret trade-off between tracking a drifting comparator and stabilizing updates, yielding a square-root scaling rule for the entropy weight in terms of a (possibly conservative) online non-stationarity proxy. Building on this, we propose AES (Adaptive Entropy Scheduling), which adaptively adjusts the entropy coefficient/temperature online using observable drift proxies during training, requiring almost no structural changes and incurring minimal overhead. Across 4 algorithm variants, 12 tasks, and 4 drift modes, AES significantly reduces the fraction of performance degradation caused by drift and accelerates recovery after abrupt changes.

Show more
What sets it apart

Establishes a dynamic-regret theoretical framework and a square-root scaling rule for entropy scheduling in non-stationary RL. It introduces the Adaptive Entropy Scheduling plug-in that uses TD-error quantiles as drift proxies, setting a new recovery time baseline for SAC on MuJoCo tasks.

Relevance to your idea

Both address the failure of static systems under changing or adversarial inputs, but in entirely different domains (non-stationary RL versus LLM safety guards). This paper uses TD-error quantiles for entropy scheduling rather than internal activation probing for safety classification. The findings on dynamic regret in RL do not transfer to understanding LLM guard decoding failures.

2
GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning
Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, Jiaheng Zhang, Bryan Hooi
NeurIPS 2025

🔍 Worth a look
80% match

To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model's reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average, as shown in Figure 1. We release data, code, and models (3B/7B) of GuardReasoner-VL: https://github.com/yueliu1999/GuardReasoner-VL.

Show more
What sets it apart

Sets a new performance bar for VLM moderation, achieving a 79.07% F1 on prompt harmfulness detection via a 7B model. It introduces a reason-then-moderate framework using online RL to generate intermediate logical justifications, foreclosing claims that only offline models are viable for multimodal safety.

Relevance to your idea

Both focus on improving LLM and VLM safety guards against complex harmful inputs. While this paper enforces explicit textual reasoning chains to improve classification, your idea probes latent internal activations to diagnose failures. Its findings on the value of intermediate processing steps for safety conceptually transfer, though its multimodal focus diverges from the query's text-obfuscation scope.

3
RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards
Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua
NeurIPS 2025

🔍 Worth a look
80% match

Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models—designed to monitor LLM inputs and outputs and block potentially harmful content—has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: (1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and (2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. Experiments demonstrate that RSafe matches state-of-the-art guard models using limited amount of public data in both prompt- and response-level harmfulness detection, while achieving superior out-of-distribution generalization on both emerging harmful category and jailbreak attacks. Furthermore, RSafe provides human-readable explanations for its safety judgments for better interpretability. RSafe offers a robust, adaptive, and interpretable solution for LLM safety moderation, advancing the development of reliable safeguards in dynamic real-world environments. Our code is available at https://anonymous.4open.science/r/RSafe-996D.

Show more
What sets it apart

Establishes that training-free, dynamic policy adaptation for LLM safeguards can match state-of-the-art static models by achieving 90.4% accuracy across diverse benchmarks. It introduces a two-stage method combining guided reasoning elicitation with GRPO reinforced alignment, requiring future adaptive safeguards to compare against its reasoning-based paradigm.

Relevance to your idea

Both target the vulnerability of LLM safeguards to adversarial or out-of-distribution inputs. This paper relies on generating explicit reasoning chains to improve robustness, contrasting with your idea's focus on analyzing internal hidden states to diagnose existing guard failures. The results on how well reasoning models generalize to new attacks indirectly inform your idea's baseline comparisons, but the methods are disjoint.

4
Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training
Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Jiahao Xu, Tian Liang, Pinjia He, Zhaopeng Tu
ACL 2025

🔍 Worth a look
79% match

This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models’ ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses baseline methods in defending against attacks.

Show more
What sets it apart

Identifies and resolves refusal position bias in LLMs, establishing that models often only associate refusal with initial output tokens. It introduces Reinforced Transition Optimization to train mid-sequence pivoting, a mechanism future jailbreak defenses must consider to prevent delayed harmful generation.

Relevance to your idea

Both investigate the mechanical reasons why LLM safety filters or refusal mechanisms fail under adversarial pressure. This paper alters training objectives to fix delayed generation failures, whereas your idea operates purely diagnostically by analyzing activations under text obfuscation. This paper's insights into token-position refusal dynamics directly transfer to your idea's evaluation of representation conversion success.

5
SELF-GUARD: Empower the LLM to Safeguard Itself
Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong
NAACL 2024

🔍 Worth a look
79% match

With the increasing risk posed by jailbreak attacks, recent studies have investigated various methods to improve the safety of large language models (LLMs), mainly falling into two strategies: safety training and safeguards. Safety training involves fine-tuning the LLM with adversarial samples, which activate the LLM’s capabilities against jailbreak. However, it is not always effective in countering new attacks and often leads to potential performance degradation. Safeguards, on the other hand, are methods using additional models to filter harmful content from the LLM’s response. Nevertheless, they can only reduce a limited amount of harmful output and introduce extra computational costs. Given the distinct strengths and weaknesses of both, we combine them to balance out their flaws and propose a more effective method called Self-Guard.Specifically, we train the LLM to review its responses for any harmful content and append a [harmful] or [harmless] tag to the end of the response. In this way, Self-Guard possesses the advantages of safety training, leveraging the powerful capabilities of the LLMs themselves to detect harmfulness. Besides that, it gains flexibility like safeguards, making the safety check target the output side, which makes the system less vulnerable to attack updates. Experimental results indicate that our Self-Guard can effectively defend against jailbreak attacks and will not cause LLMs’ performance degradation.

Show more
What sets it apart

Validates output-side self-safeguarding by showing that parsing fully generated text is easier than analyzing adversarial prompts, reducing attack success rate to 5.20% on Vicuna-v1.1. It introduces a two-stage Tag Learning method coupled with a regex filter, proving that decoupling safety from generation preserves general task performance.

Relevance to your idea

Both aim to secure LLM pipelines against jailbreaks but focus on different intervention points. This paper proposes an output-generation tagging mechanism, whereas your idea diagnoses input-filtering guards using internal probing. The findings on the comparative ease of evaluating outputs versus obfuscated prompts directly support your idea's premise that input decoding is a major point of failure.

6
PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails
Neal Mangaokar, Ashish Hooda, Jihye Choi, Shreyas Chandrashekaran, Kassem Fawaz, Somesh Jha, Atul Prakash
ACL 2024

🔍 Worth a look
79% match

Large language models (LLMs) are typically aligned to be harmless to humans. Unfortunately, recent work has shown that such models are susceptible to automated jailbreak attacks that induce them to generate harmful content. More recent LLMs often incorporate an additional layer of defense, a Guard Model, which is a second LLM that is designed to check and moderate the output response of the primary LLM. Our key contribution is to show a novel attack strategy, PRP, that is successful against several open-source (e.g., Llama 2) and closed-source (e.g., GPT 3.5) implementations of Guard Models. PRP leverages a two step prefix-based attack that operates by (a) constructing a universal adversarial prefix for the Guard Model, and (b) propagating this prefix to the response. We find that this procedure is effective across multiple threat models, including ones in which the adversary has no access to the Guard Model at all. Our work suggests that further advances are required on defenses and Guard Models before they can be considered effective. Code at https://github.com/AshishHoodaIITD/prp-llm-guard-rail-attack.

Show more
What sets it apart

Demonstrates that secondary LLM Guard Models are highly vulnerable to universal adversarial prefixes propagated by the base model. It establishes an 80% attack success rate against GPT-3.5 guards using a discrete optimization and few-shot propagation method, setting a benchmark for evaluating response-based defense mechanisms.

Relevance to your idea

Both study the vulnerabilities of LLM safety filters to adversarial text. This paper optimizes universal strings to exploit the guard's decision boundary, whereas your idea uses fixed encodings and obfuscations to diagnose where the guard's internal processing pipeline breaks down. This paper's empirical attack success rates provide direct context for the types of end-to-end failure rates your idea seeks to decompose.

7
CARE: Decoding-Time Safety Alignment via Rollback and Introspection Intervention
Xiaomeng Hu, Fei Huang, Chenhan Yuan, Junyang Lin, Tsung-Yi Ho
NeurIPS 2025

🔍 Worth a look
79% match

As large language models (LLMs) are increasingly deployed in real-world applications, ensuring the safety of their outputs during decoding has become a critical challenge. However, existing decoding-time interventions, such as Contrastive Decoding, often force a severe trade-off between safety and response quality. In this work, we propose **CARE**, a novel framework for decoding-time safety alignment that integrates three key components: (1) a guard model for real-time safety monitoring, enabling detection of potentially unsafe content; (2) a rollback mechanism with a token buffer to correct unsafe outputs efficiently at an earlier stage without disrupting the user experience; and (3) a novel introspection-based intervention strategy, where the model generates self-reflective critiques of its previous outputs and incorporates these reflections into the context to guide subsequent decoding steps. The framework achieves a superior safety-quality trade-off by using its guard model for precise interventions, its rollback mechanism for timely corrections, and our novel introspection method for effective self-correction. Experimental results demonstrate that our framework achieves a superior balance of safety, quality, and efficiency, attaining a **low harmful response rate** and **minimal disruption to the user experience** while **maintaining high response quality**.

Show more
What sets it apart

Introduces a targeted decoding-time intervention mechanism that uses hidden token buffers and internal state rollbacks to correct safety violations. It sets a new benchmark for minimizing benign quality degradation compared to Contrastive Decoding by utilizing the model's own self-critique via an Introspection step.

Relevance to your idea

Both tackle the failure of LLMs to safely process malicious prompts while preserving benign performance. This paper intervenes during streaming generation using hidden buffers and rollbacks, whereas your idea passively probes internal activations in classification guards. The findings on localized safety trajectories transfer well to your idea's hypothesis that failure occurs at specific, isolatable links in the processing pipeline.

8
Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment
Hao Li, Lijun Li, Zhenghao Lu, Xianyi Wei, Rui Li, Jing Shao, Lei Sha
EMNLP 2025

🔍 Worth a look
78% match

With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a Layer-Aware Representation Filtering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated.

Show more
What sets it apart

Establishes that LLM safety degradation from benign-seeming finetuning data can be predicted using internal representations at specific safety-sensitive layers. It introduces LARF, a gradient-free filtering method based on representation proximity, foreclosing the need for expensive secondary ranker models to preserve alignment.

Relevance to your idea

Both investigate LLM safety by directly analyzing internal model representations rather than just final outputs. This paper uses layer-wise activation proximity to filter harmful finetuning data, while your idea uses similar residual stream probing to diagnose guard failure modes under obfuscation. The findings on localizing safety features to specific layers directly support and transfer to your idea's conversion attribution methodology.

9
Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety
Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau
EMNLP 2025

🔍 Worth a look
78% match

As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs.

Show more
What sets it apart

Synthesizes nearly 70 works into a unified taxonomy mapping interpretation methods to specific safety enhancements. It establishes a structural link between internal latent probing and safety interventions across the LLM workflow, highlighting that most current circuit analysis lacks application to complex safety scenarios.

Relevance to your idea

Both operate at the intersection of mechanistic interpretability and LLM safety. This survey categorizes methods like latent probing and representation steering, which perfectly aligns with your idea's approach of using residual stream activations to evaluate decoding and conversion. The survey's findings on the lack of complex safety circuit analysis directly validate your idea's contribution to the interpretability toolkit.

10
HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models
Seanie Lee, Haebin Seong, Dong Bok Lee, Minki Kang, Xiaoyin Chen, Dominik Wagner, Yoshua Bengio, Juho Lee, Sung Ju Hwang
ICLR 2025

🔍 Worth a look
78% match

Safety guard models that detect malicious queries aimed at large language models (LLMs) are essential for ensuring the secure and responsible deployment of LLMs in real-world applications. However, deploying existing safety guard models with billions of parameters alongside LLMs on mobile devices is impractical due to substantial memory requirements and latency. To reduce this cost, we distill a large teacher safety guard model into a smaller one using a labeled dataset of instruction-response pairs with binary harmfulness labels. Due to the limited diversity of harmful instructions in the existing labeled dataset, naively distilled models tend to underperform compared to larger models. To bridge the gap between small and large models, we propose **HarmAug**, a simple yet effective data augmentation method that involves jailbreaking an LLM and prompting it to generate harmful instructions. Given a prompt such as, "Make a single harmful instruction prompt that would elicit offensive content", we add an affirmative prefix (e.g., "I have an idea for a prompt:") to the LLM's response. This encourages the LLM to continue generating the rest of the response, leading to sampling harmful instructions. Another LLM generates a response to the harmful instruction, and the teacher model labels the instruction-response pair. We empirically show that our HarmAug outperforms other relevant baselines. Moreover, a 435-million-parameter safety guard model trained with HarmAug achieves an F1 score comparable to larger models with over 7 billion parameters, and even outperforms them in AUPRC, while operating at less than 25\% of their computational cost. Our [code](https://anonymous.4open.science/r/HarmAug/), [safety guard model](https://huggingface.co/AnonHB/HarmAug_Guard_Model_deberta_v3_large_finetuned), and [synthetic dataset](https://huggingface.co/datasets/AnonHB/HarmAug_generated_dataset) are publicly available.

Show more
What sets it apart

Sets a baseline for resource-constrained safety guards by achieving 0.836 AUPRC with a 435M-parameter DeBERTa model. It introduces a synthetic data generation mechanism exploiting affirmative prefix pre-filling, proving that simple completion forcing can replace complex sampling for distillation.

Relevance to your idea

Both deal with the evaluation and improvement of LLM safety guards. This paper focuses on data augmentation and knowledge distillation to build smaller guards, while your idea provides an interpretability framework for diagnosing why existing guards fail. The findings on distillation efficiency do not directly transfer, though the synthesized adversarial datasets could be used to stress-test your idea's diagnostic ladder.

