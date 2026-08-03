# Code structure — first paper ("Can't, didn't, or wouldn't?")

*Written 2026-08-02 pre-code; §§1–7 are the original design, §§8–12 record what was actually built the same day. Companion to `s1_idea_check.md` (the science) and TODO item 1(b) (encoder inventory).*

**State: build steps 1–4 of §6 are done, `scripts/phase0_regime_map.py` is written, and the cost model behind the approval gate is built (§12) — the pilot is code-complete, costed and unlaunched.** What remains before it runs is not code and not an estimate: the owner's explicit go, and a cluster account whose longevity past graduation is settled (S2a). 182 hermetic tests + 3 real-weights tests, green.

## 1. Answer first — how much comes from the guardrail sibling

**The front end and the back end port; the entire middle is new.** Concretely, from `llm_guardrail_security` we copy the **encoder layer** (the paper's independent variable) and the **judge layer** (behavior scoring + the guard-gap table). Everything between them — activation capture, probes, interventions, training — has no analogue there and is built from scratch.

The evidence for "no analogue," from a read of the sibling's tracked source (139 Python files):

| Sibling module | Size | Verdict |
|---|---|---|
| `prompt_transformations/text/` | ~3.3k lines | **copy, selectively** — 17 encoders, ~7 of them are our ladder rungs |
| `evaluation/` (HarmBench, WildGuard, JBB-refusal, ORBench judges) | ~1.3k lines | **copy, near-verbatim** — depends only on `llm_utils` + a logger + pandas |
| `data/*.jsonl` (HarmBench, JBB harmful/benign, ORBench) | 9 files | **copy** |
| `utils/logger.py`, `paths.py` | ~440 lines | **copy** (trivial) |
| `experiment/` (config loader, cluster dispatch, task/stage machinery, model discovery) | ~4.1k lines | **do not copy** — built to fan API calls across clusters; ours is single-node GPU forward passes |
| `defense/` (black-box defenses: CIDER, SelfDefend, ECSO, …) | ~3.4k lines | **do not copy** — the sibling's research object, not ours |
| `analysis/paper_c_*`, `paper_d_*` | ~7.5k lines | **do not copy** — paper-specific |

**The decisive fact: the sibling imports `torch` in exactly one file** (`defense/cider.py`). It is an API-orchestration repo — it never opens a model. This project's spine is a forward-pass instrumentation harness, so the sibling contributes inputs and scoring, not machinery.

Two inherited conventions worth keeping, both already family-standard: `llm_utils` pinned as a git dep by tag (the sibling is on `v5.0.0`) for every judge/LLM-encoder call, and a gitignored `run` wrapper injecting API keys from the secret manager at launch so no plaintext secret touches disk. Two conventions we do **not** inherit: the `conf/experiment/**` preset tree (hundreds of YAMLs for one dispatcher — the sibling uses no Hydra, just its own loader, and our run shapes are different), and MLflow (used in 3 files there; not adopted here without a separate decision).

## 2. The load-bearing interface change

The sibling's encoder contract is "given a harmful prompt, produce an attack string." Ours needs three things it does not have, because the four measurements are defined over them:

1. **the ground-truth plaintext** — measurements #1 (ability) and #2 (deployment) score *recovery*, so they need a reference;
2. **an invertibility class** — see §3, this is the important one;
3. **a recovery scorer** — exact-match for deterministic encodings, judge/embedding similarity for semantic ones.

So `BaseEncoder` is **re-authored, not copied**: the sibling's version is shaped around a `Prompt` schema and a `Modality` enum for its image pipeline, neither of which we have. The individual encoder *bodies* (the cipher tables, the LLM prompt templates, the CipherChat-faithful Caesar port) copy over fine and carry their provenance comments with them — those comments are a real asset, since they record which parts are faithful ports of published attacks and which are our own variants.

## 3. The ladder splits in two, and this is a science decision

Not every rung supports the same measurement quality:

- **Exactly invertible** — Base64, Caesar/ROT13, keyed substitution, homoglyph. A unique plaintext exists; decode-and-restate is scored by string match; the in-situ content probe has a clean target. The four-regime claims are crisp here.
- **Semantically invertible** — paraphrase, low-resource languages, formal logic, set theory. There is no unique inverse. "Did the model decode it?" has to be scored by an LLM judge or embedding similarity, which puts judge noise directly inside the deployment measurement — the newest and most contestable claim in the paper.

**SETTLED (owner, 2026-08-02):** the primary (C)/(D)/(B) claims are made on the **exactly-invertible band**; the semantic band is carried as an extended-ladder generalization check, labelled as such. Reason: the (D) regime is the paper's novel object and must be established where the measurement is unimpeachable. The semantic band still earns its place — it connects the result to Aziz et al.'s language axis and gives RQ2 its full curve — but a reviewer attacking "you can't tell *didn't decode* from *decoded loosely*" is met with string-match evidence, not a judge score.

**Consequence — the deterministic band must be widened, and that is now the encoder priority** (owner note 2026-08-02: many more encoding attacks exist than the sibling implemented). With the headline claims resting on exactly-invertible encodings, every additional deterministic rung is worth more than any semantic one, and they are individually cheap. Candidates, all exactly invertible:

- **Deliberately unported by the sibling, from CipherChat** (Yuan et al., ICLR 2024): Morse, ASCII-decimal, Atbash (its published implementation has an uppercase-reflection bug — port with the fix and say so), Unicode/UTF escapes, GBK.
- **Standard encodings not yet anywhere in the family:** hex, binary, Base32, ROT-*n* for *n* ≠ 13, keyed Vigenère, Unicode tag-block smuggling, zero-width-character insertion.
- **Orthographic transforms:** reversed text, word-order reversal, character interleaving/spacing, leetspeak, Pig Latin, NATO-alphabet spelling.

Two payoffs beyond coverage. First, resolution: RQ2's graded curve currently has ~4 deterministic points and would have 15+. Second, and better — **with a wide deterministic band the ladder can be ordered by a measured quantity rather than by intuition.** Decode-and-restate accuracy (measurement #1) *is* a difficulty scale, so "obfuscation depth" stops being a hand-ordered list and becomes an empirical axis. That materially strengthens RQ2: the claim becomes "behavior fails at a measured decode-difficulty threshold," not "we sorted these by how hard they look."

The S2 encoder inventory (TODO item 1b) therefore includes a literature sweep for encoding families used in published attacks, filtered by one test: **exactly invertible → primary-band candidate.**

Consequence for the copy manifest: the sibling's `llm_classical_language_encoder` covers Classical Chinese, Latin, Sanskrit, Old English — *classical*, not the modern low-resource languages (Swahili, Zulu, …) that Aziz et al. and PolyRefuse use. The machinery ports; the language list does not. The language rung takes PolyRefuse's set for comparability, and the classical languages become an optional bonus rung (they are genuinely low-resource in the training-data sense).

## 4. Proposed layout

```
model_internals_safety/
  pyproject.toml              # uv; deps: torch, transformers, peft, scikit-learn,
                              #   pandas, pyyaml, pydantic, llm_utils @ git tag
  run                         # gitignored; op run wrapper (judge API keys)
  conf/                       # every tunable (global law: no magic numbers)
    models/*.yaml             #   per-model: layer sweep range, chat template, dtype
    encodings/*.yaml          #   per-family: params, invertibility class, decode prompt
    probes.yaml               #   position/layer grids, probe type, control-task settings
    training.yaml             #   LoRA rank/lr/steps, train-family splits, held-out sets
    judges.yaml               #   judge model ids, thresholds
  data/                       # copied prompt sets + PolyRefuse subset
  src/internals_safety/
    encodings/                # COPIED bodies, NEW base
      base.py                 #   Encoder protocol: encode() -> EncodedPrompt
                              #   (ciphertext, plaintext, invertibility, decode_instruction)
      registry.py             #   family name -> encoder, built from conf/encodings/
      deterministic/          #   base64, caesar/rot13, substitution (new), homoglyph
      semantic/               #   paraphrase, language, formal_logic, set_theory
      recovery.py             #   scorers: exact-match | judge | embedding similarity
    models/                   # NEW — the spine
      loader.py               #   HF model+tokenizer, dtype/device, chat template
      capture.py              #   residual-stream hooks; (layer x position) tensors to disk
      generate.py             #   batched generation for behavior measurement
    measurements/             # NEW — one module per quantity, mirrors s1 §7's table
      ability.py              #   #1 decode-and-restate
      deployment.py           #   #2 in-situ content probe (+ patching corroboration)
      recognition.py          #   #3 harmfulness / refusal directions
      behavior.py             #   #4 refusal & ASR via judges
      regimes.py              #   the four -> a regime label, incl. the coherence check
    probes/                   # NEW
      directions.py           #   diff-in-means, layer/position selection sweep
      linear.py               #   logistic probes, AUROC, selectivity vs control tasks
      overlap.py              #   projection-score distribution overlap (H4's metric)
    interventions/            # NEW
      ablation.py             #   directional ablation (necessity)
      addition.py             #   activation addition (sufficiency)
      patching.py             #   cross-condition activation patching
      elicitation.py          #   decode-then-answer prompts + matched placebo
    training/                 # NEW
      sft.py                  #   LoRA cross-encoding safety fine-tuning
      baselines.py            #   safety-data mixing; Circuit Breakers RR hook
    judges/                   # COPIED near-verbatim from sibling evaluation/
      base.py, parsing.py, harmbench.py, wildguard.py, llamaguard.py, orbench.py
    analysis/                 # NEW — tables and figures for the paper
  scripts/                    # thin entry points, one per phase
    phase0_regime_map.py      #   the pilot that gates everything
    phase1_diagnose.py
    phase2_causal.py
    phase3_repair.py
  tests/                      # pytest; smoke tests at first code (global testing law)
```

Two notes on the layout. `measurements/` deliberately mirrors the four-measurement table in `s1_idea_check.md` §7 one-module-per-row, so the code and the paper's instrument list stay legible against each other. `scripts/` stays thin — plain `argparse`, config from `conf/`; **no orchestration/CLI/tracking framework is assumed** (global law), and the question of whether we ever need one is deferred until the phase-0 pilot shows what the run shapes actually are.

## 5. Copy manifest (concrete)

**Copy and adapt** (from `llm_guardrail_security`):
- `src/prompt_transformations/text/encoders/non_llm_cipher_encoder.py` — Base64 + Caesar, a CipherChat-faithful port; ROT13 falls out as shift 13. Add keyed substitution (new, ~20 lines).
- `.../non_llm_homoglyph_encoder.py` — exactly invertible via the `homoglyphs` lib's `to_ascii`; optional rung.
- `.../non_llm_addition_equation_split_reassemble_encoder.py`, `.../non_llm_conditional_probability_encoder.py` — the deterministic math rungs.
- `.../llm_formal_logic_encoder.py`, `.../llm_set_theory_encoder.py`, `.../llm_paraphrase_encoder.py` — semantic band; prompt templates come with them.
- `.../encoders/constants.py` — shared strings.
- `src/evaluation/{base_evaluator,judge_parsing}.py` + `harmbench_evaluation/`, `wildguard_evaluation/`, `jailbreakbench_refusal_evaluation/`, `orbench_evaluation/`.
- `src/utils/logger.py`, `src/paths.py`.
- `data/{harmbench_prompts,jbb_prompts,jbb_benign_prompts,orbench_benign_1k_prompts}.jsonl`. **Copied 2026-08-02** from sibling commit `845cccd`, schema `{id, category, source, prompt}` kept byte-identical — payload identity is what the external-guard-gap table (§7 baselines) needs later, so diverging the schema would cost a comparison. `data/` is gitignored, so a fresh clone re-copies.

**Do not copy:** all of `image/`, `artprompt`, `best_of_n`, `symbol_injection`, `code_attack`, `deep_inception`, `semantic_camo`, `decode_evasion`, `ecso_evade`, `variance_channel`, `quantum_mechanics` (sibling-specific attack research); all of `experiment/`, `defense/`, `analysis/`; the `conf/experiment/**` preset tree.

**New data needed:** PolyRefuse (or Aziz et al.'s extension) for the language rung; AdvBench + MaliciousInstruct + Alpaca for contrast-set construction per Arditi's recipe; XSTest, EVOREFUSE-TEST, MORBench for the over-refusal battery.

Copying is per the CLAUDE.md scope rule — **copied, never imported**; the oikos charter bars research-bet → research-bet dependencies. Each copied file keeps its provenance comment and gains a one-line header naming the source repo and commit.

## 6. Build order

1. ~~`models/loader.py` + `models/capture.py` + a smoke test~~ — **DONE 2026-08-02** (commit `96de5a2`).
2. ~~`encodings/` (copy + re-authored base) + `measurements/ability.py`~~ — **DONE 2026-08-02.** 15 exactly-invertible rungs (up from the sibling's ~4), each round-trip-verified; `models/generate.py` landed here too, since ability needs it.
3. ~~`measurements/behavior.py` + `judges/` (copied)~~ — **DONE 2026-08-02.** Two judges (ASR + refusal), the judge LLM reached through `llm_utils`, keys injected at launch. See §10.
4. ~~`probes/` + `measurements/{deployment,recognition}.py` + `regimes.py`~~ — **DONE 2026-08-02.** Note the pilot *script* waited on step 3: a regime label needs measurement #4, so `scripts/phase0_regime_map.py` is written after the judge layer, not here.
5. Everything in `interventions/` and `training/` — only after the pilot returns a populated (B) cell.

Steps 1–4 are the phase-0 pilot's full dependency set and are complete, and `scripts/phase0_regime_map.py` (§11) is written on top of them. Step 5 is gated on the pilot's result; the pilot is gated on the approval gate, not on code — and the gate's own input, the cost model, is now built too (§12).

## 7. Open decisions

1. ~~Primary claim band~~ — **settled 2026-08-02 (owner): exactly-invertible band carries the primary claims; semantic band is the labelled extended check.** See §3, including the consequent priority on widening the deterministic band.
2. **Codename** for this paper. The letter is **E** (owner ruling 2026-08-02, reassigned from the retired "Smuggled Actions"; registry of record = science `portfolio.md`). No codename settled yet — optional, and cheap to defer to S2.
3. Deferred until the pilot reveals actual run shapes: whether any experiment-orchestration layer is needed at all, and whether runs get tracked (MLflow or otherwise). Not decided now, deliberately.
4. **NEW (opened at build step 2): does the ladder share one canonical corpus?** Some rungs are exactly invertible only over a restricted alphabet — Morse carries no case, so its reference is the uppercased, table-filtered prompt. Composing every rung's projection would impose that on all fifteen rungs to satisfy one, and case is exactly what the cipher rungs act on; so projections are currently **per rung**, and each cell is scored against its own reference. That is sound per cell but means rungs no longer see byte-identical prompts. `registry.canonicalization_report` quantifies the drift per rung. The alternative — restrict the shared corpus up front — is a *corpus* decision and belongs in S3, not in the encoder layer.
5. **NEW (opened at build step 3): LLM-judge HarmBench, or the released classifier weights?** The sibling's HarmBench evaluator — copied here — is the **LLM-judge form**: HarmBench's canonical `LLAMA2_CLS_PROMPT` sent to an API model (gpt-5-mini), not the released HarmBench-Llama-2-13b-cls checkpoint. That was forced in the sibling, which never opens a model. It is *not* forced here: this repo already puts 7–9B models on a GPU, so running the released classifier locally is a real option — free per judged sample, and immune to judge-model drift and deprecation (the current judge model has an announced shutdown date). Against it: every recorded ASR number in the family comes from the LLM-judge form, so switching costs cross-paper comparability, and a 13B classifier competes for the same GPU as the experiment. **Not decided.** The cheap resolution is to run both over the phase-0 pilot's saved responses and report their agreement — a judge-agreement number belongs in the paper regardless, and it turns this from a choice into a measurement. Deferred to S3.

6. **NEW (opened writing the pilot): is the licensing-plus-reading construction the right way to make a population probe label one prompt?** §7's table defines all four measurements "for every (model, family, prompt)", but two of them are probes, and an AUROC is a property of a population. The pilot resolves this with a two-step reading (§11): a (layer, position) cell must first read a signal above its shuffled-label control — otherwise every prompt in that family reads negative — and only then is an individual prompt's decision value compared against the *same-condition negative class*. Both halves are defensible on their own (the first is just refusing to label from noise; the second is the only threshold the encoding-plus-template shift cannot move), but the combination is **our construction, not a method carried over from the literature** — Arditi et al. and Zhao et al. both report population-level separation and never need a per-prompt label. Two consequences to settle at S3: the reading percentile is an operating point rather than an estimate (at 50 the benign control reads positive half the time by construction, so only the harmful-minus-benign *gap* carries information), and the alternative — reporting regimes as per-family *rates* rather than per-prompt labels — would avoid the construction entirely at the cost of the coherence check, which only bites cell by cell. Not decided; the pilot reports both the per-prompt distribution and the underlying curves, so either framing can be written up from the same run.

## 8. Build-step-2 record (2026-08-02)

What landed, and the decisions inside it that are not obvious from the layout.

**Two prompt forms per rung, not one.** `EncodedPrompt` renders both an `attack_prompt` (the encoded request as an attacker sends it — names the encoding, matching CipherChat's threat model, but *never* asks the model to decode) and a `restate_prompt` (measurement #1: "write out the decoded text"). This is the interface consequence of the v2 reframe: the attack condition must not request decoding, or the measurement asks the very question it is testing. A test asserts the word "decode" appears in one and not the other, because a single careless template edit would silently destroy the (D) regime.

**Invertibility is verified, not declared.** `Invertibility.EXACT` is the promise the paper makes to a reviewer who challenges "you cannot distinguish *didn't decode* from *decoded loosely*". A class an author merely asserts would put that promise on the honour system, so `tests/test_encodings.py` round-trips every EXACT rung over an awkward corpus (mixed case, punctuation, digits, repeated spaces, non-ASCII) and fails if the inverse is not exact. An encoder that cannot pass is SEMANTIC by definition.

**The deterministic band went from ~4 rungs to 15.** base64, base32, hex, binary, ascii-decimal, unicode-escape, ROT13, Caesar-3, Caesar-7, Atbash, Vigenère, Morse, character-reversal, word-reversal, zero-width insertion. Three notes. Atbash is ported **with CipherChat's uppercase-reflection bug fixed** (its `N - ord(s)` constant only reflects lowercase; uppercase lands in the control range — a faithful port would corrupt every capitalized prompt), and the divergence is pinned by a test. The three Caesar shifts are deliberate: same mechanism, different familiarity (ROT13 is memorised, shift 7 is not), which lets RQ2 separate *decode difficulty* from *mechanism*. GBK was dropped — it is a CJK byte encoding that degenerates to ASCII passthrough on English prompts, i.e. a rung that does nothing.

**Three named candidates did not make the band, with reasons.** *Leetspeak*: the digit substitutions (a→4, e→3, o→0…) are not injective on any plaintext containing those digits, so the inverse is exact only over a digit-free corpus — a corpus restriction, not an encoder property, so it waits on open decision 4. *NATO spelling*: works, but needs the same uppercase-and-filter projection as Morse for strictly less novelty; low priority. *Homoglyph*: the sibling's rung depends on the `homoglyphs` package, and adding a dependency for an optional rung was not worth it at step 2.

**Config shape deviates from §4:** one `conf/encodings.yaml` keyed by family rather than `conf/encodings/*.yaml`. With 15 rungs the per-family files would be four lines each, and the attack/restate templates only make sense read side by side.

**Harness check (2026-08-02, CPU, $0):** the full ladder × 2 prompts on Qwen2.5-0.5B-Instruct. Recovery is 0.00 on all 15 rungs, which is the expected result at that scale and is itself a useful pilot input — measurement #1 is degenerate below the 7B class, so the difficulty axis only exists at the pilot slate's size. The graded `similarity` signal is *not* degenerate even at 0.5B (zero-width 0.64 → base64 0.00), which is why the scorer reports it separately from the binary.

## 9. Build-step-4 record (2026-08-02)

`probes/{directions,linear,overlap}.py`, `measurements/{recognition,deployment,regimes}.py`. 40 new tests; 90 hermetic total, green in 3s.

**The probe layer reports selectivity, never a bare AUROC.** A linear classifier on a few hundred high-dimensional examples can fit a great deal, so every probe is refit on shuffled labels using the *same* split and the gap is what gets reported. 0.95 with a 0.90 control is memorisation; 0.95 with a 0.52 control is signal. The threshold above which a probe "reads signal" is in `conf/measurements.yaml` with its tuning path named — it is set from the pilot's control-task distribution, not from taste.

**Measurement #2 is fit-here-evaluate-there, and never refits.** The content probe is trained on the plain-text condition, where content is present by construction, and evaluated on the attack condition without refitting. Refitting on the encoded condition would ask whether encoded activations are separable *at all* — which they may well be on surface form alone — and that is a different question. Two tests pin the instrument: it must find planted content, and it must return chance on structureless activations. Without the second, every (D) cell in the paper would still report a number.

**Recognition reads at `instruction_final`, with `last` as the contrast.** Zhao et al. (NeurIPS 2025) put harmfulness at the instruction-final token and refusal at the post-instruction/template token; our two captured positions map onto exactly that pair. A probe placed at the later position measures "will it refuse", which would beg the question (B) is defined by — harm represented *while* the model complies. Both curves are carried; only the first bears the regime.

**Everything is a per-layer × per-position curve, and `recognized`/`deployed` are any-cell predicates.** §7(b): under encodings the harmful gist may only become legible partway through the internal decode, so a fixed readout layer would systematically under-detect in exactly the deep-obfuscation rungs the paper cares most about. The layer at which a signal appears is a finding, not a constant.

**Two amendments to §7's prose, made in `regimes.py` and back-propagated to `s1_idea_check.md` §7.** (a) The "four of eight coherent combinations" count needs a second constraint (`deployment → recognition`) which only holds on a harmful corpus, so it is implemented as a soft, harmful-scoped flag that does not void the cell's label — otherwise every benign control cell would be recorded as an instrument failure. (b) A fifth cell exists — **(R) surface refusal**, refusal without deployment — which matches no row of the Move-B table, is not incoherent, and is exactly the degenerate outcome H5 warns about. It gets counted rather than absorbed.

**`seed` entered the project here**, as the run-logging skill required at the first stochastic step: probe train/test splits. It lives in `conf/measurements.yaml` and belongs in every run record.

**Still owed for the pilot:** step 3 (measurement #4 — behaviour/ASR via the copied judges), then `scripts/phase0_regime_map.py`. A regime label needs all four measurements, so the pilot script cannot precede the judge layer. *(Step 3 landed the same day — §10. The pilot script is what remains.)*

## 10. Build-step-3 record (2026-08-02)

`judges/{parsing,base,harmbench,refusal}.py`, `measurements/behavior.py`, `conf/judges.yaml`. 25 new tests; 115 hermetic total, green in 0.4s. The four measurements now all exist.

**The copied HarmBench evaluator is an API judge, not the released classifier weights** — checked before copying, because it decides whether this layer costs money or GPU. The sibling sends HarmBench's canonical `LLAMA2_CLS_PROMPT` (plus a JSON wrapper) to a caller-injected judge LLM, resolved project-wide to gpt-5-mini after its own judge-validation round found gpt-5-nano inflates absolute ASR by 2–3×. Its only weights-based judge is WildGuard, and even that is served over HTTP rather than loaded. The consequence for us is that measurement #4 is the first thing in this repo that needs an API key and spends money per sample — and that running the real 13B classifier is now an *option* rather than an impossibility, which is open decision 5.

**Prompts copied verbatim, container re-authored** — the same split §2 made for `BaseEncoder`, for the same reason. The classifier templates and the verdict parser are what make our ASR column comparable with the family's recorded numbers, so they are byte-identical (including one inherited cosmetic defect: the refusal template's JSON example carries doubled braces, never collapsed because the sibling fills it with `str.replace`; harmless, and left alone because every recorded family number was produced with it). The container is not: the sibling's `evaluate()` takes prompt dicts plus a parallel processed-prompt list plus an id-keyed response dict and returns a pandas DataFrame — a shape built for its experiment dispatcher. Ours takes and returns dataclasses like every other measurement module here.

**The judges never see the ciphertext, and a test pins it.** Both classifiers are asked about the *plaintext* request even though the model was attacked with an encoded one. HarmBench asks whether a generation exhibits a behaviour: hand it a base64 blob and every verdict is mechanically "no". Worse, the refusal judge counts a response "entirely irrelevant to the request" as a refusal — so against a ciphertext request, a model that decoded and complied (the (B) cell, the paper's whole object) reads as irrelevant and would be scored as refusing. One careless edit passing `attack_prompt` would leave the pipeline green and silently empty (B). This is the step-3 analogue of step 2's "the word *decode* appears in one prompt and not the other" test.

**Refusal is measured, not read off ASR.** Two judges run on every response. "Not jailbroken" and "refused" are different claims, and the regime taxonomy splits on the second: a model that decodes nothing and produces filler is neither, and collapsing the two would move every such cell out of (C)/(D) into (S)/(R) — erasing the distinction the paper exists to draw.

**Echo is scored independently of the refusal verdict.** The commonest non-answer under an encoded attack is the model parroting the ciphertext back, and the refusal judge's prompt explicitly counts an echo as a refusal. But an echo is a decode failure — (C) — and scoring it as a refusal would inflate (R), the metric H5 watches for the "refuse anything that looks encoded" degenerate outcome. So `BehaviorRecord` carries `echoed_ciphertext` from the same string scorer measurement #1 uses, next to the verdict rather than derived from it. Which reading is right in a given cell is a pilot finding, not something to hard-code now.

**Judge-parse fallbacks ride on every record.** An unparseable verdict resolves to the safe word (copied fail direction). On the ASR judge that under-counts attack success — conservative. On the refusal judge it resolves to *not refused*, which pushes cells toward (B), the headline regime. That asymmetry is real, so `judge_fallback` is per-record and `FamilyBehavior.fallback_rate` is reported per rung; a non-trivial rate is a result, not a rounding error.

**Two of the sibling's four evaluators were deliberately left behind, with triggers named.** WildGuard belongs to the external-guard-gap table (§7 baselines) and additionally needs a served checkpoint plus a helper from `defense/`, which the copy manifest excludes; OR-Bench belongs to the over-refusal battery. Both are phase-3 work; copy each with its experiment. The manifest's `utils/logger.py` was also skipped — this repo has no logging layer, and pulling in a colour-logging dependency for two modules is the wrong trade until something actually needs run-time logs.

**Secrets wiring.** `llm_utils` pinned at `v5.0.0` (the sibling's tag) as a git dep; keys are injected into the process at launch by a gitignored bootstrap in the repo root, so no plaintext key touches disk and nothing vault-shaped is committed (public-repo rule). Verified end to end, not assumed: two live judge calls returned parsed verdicts with no fallback. The launcher is also what will work on the cluster, where desktop secret tooling does not exist.

## 11. Phase-0 pilot record (2026-08-02)

`scripts/phase0_regime_map.py`, plus the four supports it needed: `data.py` (corpus loading), `provenance.py` (the run record and the dirty-tree guard), `models/capture.capture_or_load` (the activation cache), and the per-prompt reading layer in `probes/linear.py` + `measurements/{deployment,recognition}.py`. `conf/pilot.yaml` is new. 51 new tests; 166 hermetic total, green in ~2s.

**A regime label is per prompt; two of the four measurements are not.** This is the design problem the pilot had to solve and it was not visible from the module boundaries. Ability and behaviour are measured on one prompt each. Deployment and recognition are probes, and a probe needs a population — `measure_deployment` returns AUROC curves, not verdicts. Broadcasting a family-level boolean to every prompt in the family looked adequate until the coherence checks were traced through it: `deployment ✓ / ability ✗` and `recognition ✓ / deployment ✗` are declared *hard* instrument failures, so a family with 60% ability and family-level deployment would have flagged 40% of its cells as (X) — an incoherence storm manufactured entirely by mixing granularities, and `hard_incoherence_rate` is the number that invalidates a map rather than qualifying it. So the probe layer grew a per-prompt reading instead (open decision 6).

**Both per-prompt readings are out-of-sample, and they get there differently.** The deployment probe is fit on the plain condition and read on the encoded one, so every encoded example is held out by construction — no split needed. The recognition probe is fit *inside* the condition it reads, so it has no free held-out set and its per-example scores are cross-validated; without that it would be reporting its own training set back. Same reading rule, different route to it, and conflating them would have made recognition look better than it is at exactly the rungs where it matters.

**The reading threshold comes from the concurrent negatives, not from the decision boundary.** Encoding a prompt and wrapping it in an attack template moves *both* classes together. A common shift leaves ranking — and therefore the AUROC that licensed the cell — untouched, but it moves every raw decision value, so thresholding at zero would let the wrapper decide the label. Thresholding against the same-condition negative class cannot, because the confound is common-mode. A test plants a shift of 25 and asserts the reading does not move.

**The 2×2 makes §8's first control free rather than an extra arm.** The "single most important" validity control asks that benign content go through the *identical* encoding pipeline so a probe firing on "looks encoded" is caught. The benign set is already the probe's negative class in both encoded conditions, so a surface-form probe cannot open a gap between the two classes it separates — the pilot reports that gap per rung as `deployment_gap`. What it cannot do for free is length matching (§8 control 4): encodings inflate token counts, so mean plaintext and ciphertext lengths are recorded per rung and left as a stated confound, since the fix is a matched corpus and that is an S3 corpus decision.

**The contrast pair is JailbreakBench's, not HarmBench's, and that was a correction.** The first plan paired HarmBench's 240 harmful prompts with JBB's 100 benign ones. But JBB ships its benign set as a *theme-matched* counterpart to its own harmful set — `jbb_defamation` ↔ `benign_defamation`, down all 100 — and an unmatched negative class would let the content probe separate on topic or provenance and report that as "the content is present". Matched pairs cost 140 harmful prompts and buy the thing the whole measurement rests on. HarmBench's set stays on disk for the broader ASR reference.

**The plain conditions are captured once per model, not once per rung.** They do not depend on the encoding, so across a 15-rung sweep the activation cache turns 32 capture passes into 2 + 30. The cache key covers the model, the exact prompt strings and the capture geometry — deliberately *not* the batch size, which must not change activations and is pinned not to by the mask-aware `position_ids` in `tokenize_batch`. Every run record names the caches it read, because "re-run it" is not a well-defined instruction otherwise.

**`--dry-run` exists to feed the approval gate.** It prints capture passes, generations, judge calls and a token ceiling without loading a model or needing a key, so the GPU/$/wall-clock estimate the family rule requires is derived from the script rather than guessed. At the configured defaults, one model over the full ladder is 3,200 capture forward passes, 3,000 generations (≤1.15M new tokens) and 3,000 judge calls.

**The dirty-tree guard refuses on GPU and warns on a laptop.** Per the run-logging skill, but the reason is sharper here than the general one: the paper's central object is a regime *label* built by combining four measurements on one prompt, so a silent code change between measurements corrupts the label rather than merely perturbing a number. Local CPU/MPS debugging still records the diff inline — inline rather than a path, because results get rsynced down from the cluster without the tree.

**Verification.** `main()` is covered end to end by a hermetic test — real corpus, two rungs, six prompts, the tiny in-process model and stub judge services — that asserts the run record carries the skill's full schema and that the captured condition is the *attack* prompt, never the restate one. Nothing about the pilot has been run against real weights or a real judge; the first such run is the gated one.

## 12. Cost-model record (2026-08-02)

Built to close S2(c) — the family's experiment-run approval gate (owner 2026-07-22) requires GPU count and type, money, and wall-clock before any launch, and none of those existed as numbers. `scripts/cost_model.py` + `src/internals_safety/cost.py` + `conf/cost.yaml` produce them. It loads a tokenizer but no weights, so it runs on a laptop with no GPU, no key and no cluster.

**Tokens, not passes, because the ladder makes the difference an order of magnitude.** `Plan` in the pilot counts forward passes and generations, which answers "what work happens" but not "how long does it take": wall-clock is driven by tokens, split between compute-bound prefill and bandwidth-bound decode. The encodings make that split load-bearing rather than pedantic — measured against Qwen2.5's tokenizer over the real 100-prompt corpus, the rungs inflate prompt length from **1.2× (reverse_words, 55 tokens) to 13.9× (binary, 758 tokens)** relative to the shortest rung. Any single assumed inflation factor is wrong by an order of magnitude somewhere on the ladder, so the census tokenises every rung for real instead of assuming one.

**What that census returns, per model over the full ladder:** 1,232,922 prefill tokens (measured), ≤1,152,000 decode tokens (the configured budget ceiling), 3,000 judge calls. The longest single prompt is **1,446 tokens against a 131,072-token context** — worth stating because a rung that overflowed context would be silently truncated and every regime in it would be garbage, so the census reports the maximum as a check rather than leaving it implicit.

**Everything is a range, and the ranges cross.** Hardware throughput cannot be known before running on the actual node, so `conf/cost.yaml` carries ranges and every output is a range; the low end of GPU-hours comes from the *high* end of tokens/second, which a test pins because getting it backwards would report the optimistic case as the worst case. The gate is served better by "0.7–2.4 hours against an 8-hour wall" than by a confident single number that is wrong.

**Gather-and-cover.** Those throughput ranges span ~4×, and they have exactly one tuning path: a real run measuring them. So the instrumentation ships with the knob rather than after it — the pilot now times its sweep and writes `throughput` into `results.json`, labelled an upper bound (decode counts are budgets, not realised lengths, and the elapsed time includes probe fitting and judge calls). Phase 1's estimate is then a measurement instead of a second guess.

**Prices are read from `llm_utils`, not configured here.** `LLMModel` already carries the per-model price table the whole family bills against (gpt-5-mini at $0.25/M input, $2.00/M output — verified against OpenAI's pricing page 2026-08-02), so a copy in our YAML would be a second home for a number that changes upstream. What *is* configured is the assumption that number multiplies: gpt-5-mini is a reasoning model whose hidden reasoning tokens bill as output even though they never appear in the verdict, so the judge bill is charged at 150–700 output tokens per verdict rather than the ~60 the visible JSON suggests.

**One defect found and fixed while costing the judge path.** `llm_utils` auto-routes a job to the provider's native batch API when its estimated cost clears $1.00, and it estimates output as `max_tokens` per request. Ours is a deliberately generous 16,384, so the estimate lands at $3.30 per family and **every judge call would have gone to the batch queue on a ~40× overestimate of a short JSON verdict**. The judges are called synchronously inside `run_family` with the model resident on the GPU, so batch latency (poll interval 30s, timeout 3600s, per call, 30 calls per run) would be charged to the job's wall-clock allocation and could exhaust the 8-hour partition limit while the GPU sat idle polling. `conf/judges.yaml` now pins `use_batch_api: false`. This cannot move a verdict — batch and realtime are the same model at the same temperature on the same prompts, a delivery channel rather than an instrument — and the whole judge bill is single-digit dollars, so the 50% batch discount does not buy back the risk. The knob's tuning path is named: a large offline re-judge that holds no GPU is exactly when to flip it.

**Phases 1–3 are projections, and say so.** Phase 0 is costed from measured tokens; the later phases are the phase-0 census times a per-phase multiple declared in `conf/cost.yaml`, plus a fine-tune count for phase 3. They are planning figures for the S3 design doc, each re-derived from real numbers once the phase before it has run.
