# Code structure — first paper ("Can't, didn't, or wouldn't?")

*Written 2026-08-02 pre-code; §§1–7 are the original design, §§8–9 record what was actually built the same day. Companion to `s1_idea_check.md` (the science) and TODO item 1(b) (encoder inventory).*

**State: build steps 1, 2 and 4 of §6 are done; step 3 (the judge layer, measurement #4) is next and is the first step needing API keys.** 90 hermetic tests + 3 real-weights tests, green.

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

Two inherited conventions worth keeping, both already family-standard: `llm_utils` pinned as a git dep by tag (the sibling is on `v5.0.0`) for every judge/LLM-encoder call, and a gitignored `run` wrapper injecting API keys from 1Password via `op run` so no plaintext secret touches disk. Two conventions we do **not** inherit: the `conf/experiment/**` preset tree (hundreds of YAMLs for one dispatcher — the sibling uses no Hydra, just its own loader, and our run shapes are different), and MLflow (used in 3 files there; not adopted here without a separate decision).

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
- `data/{harmbench_prompts,jbb_prompts,jbb_benign_prompts,orbench_benign_1k_prompts}.jsonl`.

**Do not copy:** all of `image/`, `artprompt`, `best_of_n`, `symbol_injection`, `code_attack`, `deep_inception`, `semantic_camo`, `decode_evasion`, `ecso_evade`, `variance_channel`, `quantum_mechanics` (sibling-specific attack research); all of `experiment/`, `defense/`, `analysis/`; the `conf/experiment/**` preset tree.

**New data needed:** PolyRefuse (or Aziz et al.'s extension) for the language rung; AdvBench + MaliciousInstruct + Alpaca for contrast-set construction per Arditi's recipe; XSTest, EVOREFUSE-TEST, MORBench for the over-refusal battery.

Copying is per the CLAUDE.md scope rule — **copied, never imported**; the oikos charter bars research-bet → research-bet dependencies. Each copied file keeps its provenance comment and gains a one-line header naming the source repo and commit.

## 6. Build order

1. ~~`models/loader.py` + `models/capture.py` + a smoke test~~ — **DONE 2026-08-02** (commit `96de5a2`).
2. ~~`encodings/` (copy + re-authored base) + `measurements/ability.py`~~ — **DONE 2026-08-02.** 15 exactly-invertible rungs (up from the sibling's ~4), each round-trip-verified; `models/generate.py` landed here too, since ability needs it.
3. `measurements/behavior.py` + `judges/` (copied) — gives ASR, which makes the pilot runnable.
4. ~~`probes/` + `measurements/{deployment,recognition}.py` + `regimes.py`~~ — **DONE 2026-08-02.** Note the pilot *script* still waits on step 3: a regime label needs measurement #4, so `scripts/phase0_regime_map.py` is written after the judge layer, not here.
5. Everything in `interventions/` and `training/` — only after the pilot returns a populated (B) cell.

Steps 1–4 are the phase-0 pilot's full dependency set. Step 5 is gated on its result.

## 7. Open decisions

1. ~~Primary claim band~~ — **settled 2026-08-02 (owner): exactly-invertible band carries the primary claims; semantic band is the labelled extended check.** See §3, including the consequent priority on widening the deterministic band.
2. **Codename** for this paper. The letter is **E** (owner ruling 2026-08-02, reassigned from the retired "Smuggled Actions"; registry of record = science `portfolio.md`). No codename settled yet — optional, and cheap to defer to S2.
3. Deferred until the pilot reveals actual run shapes: whether any experiment-orchestration layer is needed at all, and whether runs get tracked (MLflow or otherwise). Not decided now, deliberately.
4. **NEW (opened at build step 2): does the ladder share one canonical corpus?** Some rungs are exactly invertible only over a restricted alphabet — Morse carries no case, so its reference is the uppercased, table-filtered prompt. Composing every rung's projection would impose that on all fifteen rungs to satisfy one, and case is exactly what the cipher rungs act on; so projections are currently **per rung**, and each cell is scored against its own reference. That is sound per cell but means rungs no longer see byte-identical prompts. `registry.canonicalization_report` quantifies the drift per rung. The alternative — restrict the shared corpus up front — is a *corpus* decision and belongs in S3, not in the encoder layer.

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

**Still owed for the pilot:** step 3 (measurement #4 — behaviour/ASR via the copied judges), then `scripts/phase0_regime_map.py`. A regime label needs all four measurements, so the pilot script cannot precede the judge layer.
