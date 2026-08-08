# The model slate — which models, why, and what paper describes them

*Fourth canonical sibling, founded 2026-08-08. `instrument_layer.md` = what the
instruments were FOUND to do · `instrument_build_plan.md` = what WILL be built ·
`pipeline_architecture.md` = how the code is ARRANGED · **this = the objects of
study**. Both papers CITE it; neither copies it. Guards live here too — a guard
is a model, and AS-6's slate has the same provenance duty as AS-5's.*

## 0. The rule (owner instruction 2026-08-08)

> *"if we use new model, we better find its original paper if there exist"*

**Binding, before any run.** A model enters either paper's slate only with its
original paper LOCATED and three things recorded: the paper, its evidence tier,
and — the part that has already cost us — **whether the paper describes the exact
checkpoint we run**. A vendor's model family paper is not automatically a
description of the weights on disk.

This is the method-provenance rule (global law: *"every named attack/defense/
method states on FIRST mention whether it is established literature (with
source) or our proposal"*) extended from methods to targets. Tiers follow the
repo convention (`instrument_build_plan.md` §0): **(A)** peer-reviewed at a named
venue · **(B)** preprint with substantial independent uptake · **(C)** recent
unrefereed preprint · **(D)** unverified beyond an abstract.

⚠️ **Semantic Scholar's `venue` field is STALE for at least two papers here.**
It reports `arXiv.org` for both Tülu 3 and OLMo 2, and both are **published at
COLM 2025** — the OpenReview PDFs carry the "Published as a conference paper at
COLM 2025" header. A metadata-only lookup would have tiered the strongest
candidate in the slate two tiers too low. Check the venue against OpenReview or
the proceedings, not only the API. (Same failure class as the coverage sweep's
lesson: a failed lookup is not evidence of absence.)

## 1. The slate, verified 2026-08-08

| model | original paper | tier | covers the checkpoint we run? |
|---|---|---|---|
| Llama-3.1-8B-Instruct | *The Llama 3 Herd of Models*, 2407.21783 | (B) — 17,556 cites, no venue found | ✅ 8B Instruct is in the paper |
| Qwen2.5-7B-Instruct | *Qwen2.5 Technical Report*, 2412.15115 | (B) — 4,636 cites, arXiv | ✅ |
| Mistral-7B-Instruct-**v0.3** | *Mistral 7B*, 2310.06825 | (B) — 3,748 cites, arXiv | ❌ **NO — the paper is v0.1** |
| Llama-3.1-Tulu-3-8B-{SFT, DPO, ""} | *TÜLU 3*, 2411.15124 | **(A) — COLM 2025**, 807 cites | ✅ **all three are the paper's own released artifacts** |
| Gemma-2-9b-it | *Gemma 2*, 2408.00118 | (B) — 2,268 cites, arXiv | ✅ |
| OLMo-2-1124-7B-Instruct | *2 OLMo 2 Furious*, 2501.00656 | **(A) — COLM 2025**, 240 cites | ✅ (not currently in the slate) |
| WildGuard *(AS-6)* | *WildGuard*, 2406.18495 | **(A) — NeurIPS 2024 D&B**, 483 cites | ✅ |
| Llama Guard 3 8B *(AS-6)* | *Llama Guard*, 2312.06674 (v1) + the Llama 3 Herd paper | (B) — 1,215 cites, arXiv | ⚠️ **v1 paper; Guard 3 is documented in the Herd paper + model card** |

Corpora, for completeness — both peer-reviewed, both already cited:
JailbreakBench 2404.01318 (NeurIPS 2024) · XSTest 2308.01263 (NAACL 2024).

## 2. What following the rule actually found

### 2.1 ⚠️ Mistral-7B-Instruct-v0.3 has no paper, and its safety is a PROMPT

Job `9010897` is queued against this model as the third family. Two facts from
2310.06825 that its preset does not know:

1. **The paper describes v0.1** (Table 1: `vocab_size 32000`). We run v0.3
   (vocab 32768, v3 tokenizer, function calling). v0.2 and v0.3 shipped as
   weights with model cards, no paper. So the *post-training of the checkpoint
   we run is undocumented*, and the paper's own description of v0.1's instruct
   tuning is one paragraph: *"we fine-tuned it on instruction datasets publicly
   available on the Hugging Face repository. No proprietary data or training
   tricks were utilized… a simple and preliminary demonstration."*
2. **Mistral's guardrails are a SYSTEM PROMPT, not training** (§5). The paper
   gives the exact prompt and reports that with it *"the model properly declines
   to answer 100% of the harmful questions"* on 175 unsafe prompts. Our config
   runs `system_prompt: null`, i.e. deliberately without the mechanism the paper
   says carries Mistral's safety.

**Consequence for the preset's gate.** `dissociation_mistral.yaml` says a low
benign-refusal + high binding-failure result would REFUTE the §4a axis. It would
not. Mistral is not a weakly-safety-trained model on the same axis as Llama and
Qwen — it is a model whose safety training is undocumented and whose documented
safety mechanism is switched off in our harness. It anchors a **different**
point: *what the four regimes look like with essentially no safety training*.
That is worth having and the run is free, so it stands — but its gate language
must be corrected, and it cannot carry the axis test.

*Filed, not done:* a second Mistral arm WITH the paper's published system prompt
would turn this into a within-model contrast of safety-by-prompt vs
safety-by-training. Scope creep for now.

### 2.2 ✅ Tülu 3 is the only arm whose exact checkpoints a peer-reviewed paper describes

And it is stronger than that. From 2411.15124:

- **Table 1** names the three checkpoints as released artifacts of the paper —
  base `meta-llama/Llama-3.1-8B` → SFT → DPO → final (RLVR). **The same base our
  Llama-3.1-8B-Instruct arm is post-trained from**, so the ladder holds
  pretraining, tokenizer, vocabulary and layer count fixed and varies only
  post-training.
- **Table 7 names the safety data and its per-stage prompt counts**: Tülu 3
  CoCoNot (10,983), Tülu 3 WildJailbreak (50,000), Tülu 3 WildGuardMix (50,000),
  each used in both SFT and DPO. No other model in the slate lets a result be
  attributed to identifiable training data.
- **§4.2 makes a prediction our benign arm directly tests**: *"adding contrastive
  prompts, such as those in CoCoNot, were helpful for preventing our models from
  over-refusing safe prompts."* AI2 explicitly trained against over-refusal on
  benign content — so Tülu 3 should show a LOW benign refusal rate, which is the
  §3.6 measurement. A prediction from the model's own authors, checkable by an
  instrument built before we knew it existed.
- **Table 6 publishes a staged safety curve for the 8B ladder**: safety (6-task
  average) **SFT 93.1 → DPO 87.2 → final 85.5**. Aggregate safety *falls* through
  preference tuning and RLVR.

**That last row is why this arm is the experiment.** The paper's thesis is that
conditioning on comprehension inverts safety rankings that aggregate metrics
produce. Here is a published aggregate safety curve, computed by the model's own
authors over six standard tasks, across three checkpoints that differ only in
post-training. Measuring comprehension-conditioned binding failure across the
same three checkpoints asks whether our conditioning tracks or inverts *their*
number — not one we invented. No other available model provides that.

### 2.3 ⚠️ AS-6's guard and the proposed AS-5 ladder SHARE TRAINING DATA

WildGuard (2406.18495) and Tülu 3 (2411.15124) come from the same AI2 group
(Lambert, Dziri, Choi on both), and **Tülu 3's safety mix contains
`Tülu 3 WildGuardMix` — 50,000 prompts drawn from WildGuard's own training
corpus** (Table 7, citing Han et al. 2024), plus WildJailbreak (Jiang et al.
2024).

Two consequences, both to state rather than to fix:

- **Never evaluate a Tülu-3 checkpoint against WildGuard in AS-6** without
  declaring the overlap. Target and guard would share safety training data.
- **Tülu 3 has been trained on adversarial jailbreak prompts** (WildJailbreak).
  Whether that corpus contains *encoding-based* attacks is an open check, and it
  bears directly on our ladder: if it does, Tülu's behaviour on our rungs is
  partly in-distribution. **Check before reporting any Tülu result.**

### 2.4 The two Chinese-lab and vendor tech reports are tier (B), not (A)

Llama 3 Herd, Qwen2.5 TR and Gemma 2 are all vendor technical reports with no
peer-reviewed venue found. High uptake puts them at (B). This is the normal
state for open-weight model releases and is not a defect — but it means **the
only tier-(A) descriptions of any target in this repo are Tülu 3, OLMo 2 and the
two AS-6 corpora**, and any claim resting on *what a model's safety training did*
is tier-(B) sourced everywhere except the Tülu ladder.

## 3. Tokenisation facts, verified against the real tokenizers

⚠️ **`models/loader.tokenize_batch` hardcodes `add_special_tokens=False`,
justified by "the chat template already emits BOS". That premise is FALSE for
Tülu 3** — verified 2026-08-08:

```
template   : '<|user|>\nHELLO\n<|assistant|>\n'        <- no BOS anywhere
add_special=False -> ['<','|','user','|','>Ċ', ...]    <- NO BOS
add_special=True  -> ['<|begin_of_text|>', ...]        <- adds 128000
'<|user|>'        -> [27, 91, 882, 91, 29]             <- plain text, 5 tokens
```

The paper's own Figure 27 confirms the template is plain-text role markers with
no BOS. So a Tülu arm run under the repo default gets **no BOS at all**, while
the Llama-3.1-8B-Instruct arm it is meant to be compared against gets one from
its template string — a silent distribution shift, not an error, in exactly the
comparison the arm exists to make.

Fourth entry in this repo's tokenizer-surprise ledger, after WildGuard's missing
template, Llama Guard 3's stray leading space, and Mistral v0.3's literal `<s>`
(double-BOS under `add_special_tokens=True`). **The fix follows the repo's own
rule — make the omission inexpressible**: the loader asserts that a rendered
template emits BOS when the tokenizer declares one, and a model config must
declare the exception explicitly with a verified reason. Threading a flag into
callers is what failed three times already.

Gemma-2-9b-it emits `<bos>` in its template string (`'<bos><start_of_turn>user\n…'`),
so it matches the Llama/Mistral pattern and needs no exception.
