# Code structure — first paper ("Can't, didn't, or wouldn't?")

*Written 2026-08-02, pre-code. Companion to `s1_idea_check.md` (the science) and TODO item 1(b) (encoder inventory). Design only — no code exists yet.*

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

**Recommendation on record:** make the primary (C)/(D)/(B) claims on the exactly-invertible band and carry the semantic band as an extended-ladder generalization check, labelled as such. Reason: the (D) regime is the paper's novel object and it should be established where the measurement is unimpeachable. The semantic band still earns its place — it is what connects the result to Aziz et al.'s language axis and gives RQ2 its full curve — but a reviewer attacking "you can't tell 'didn't decode' from 'decoded loosely'" must be met with the deterministic band, not a judge score. **Pending owner decision (§7).**

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

1. `models/loader.py` + `models/capture.py` + a smoke test that captures activations from one model on one prompt. Nothing else can be validated until this works.
2. `encodings/` (copy + re-authored base) + `measurements/ability.py` — the cheapest real measurement.
3. `measurements/behavior.py` + `judges/` (copied) — gives ASR, which makes the pilot runnable.
4. `probes/` + `measurements/{deployment,recognition}.py` + `regimes.py` — completes the phase-0 pilot.
5. Everything in `interventions/` and `training/` — only after the pilot returns a populated (B) cell.

Steps 1–4 are the phase-0 pilot's full dependency set. Step 5 is gated on its result.

## 7. Open decisions

1. **Primary claim band** (§3) — exactly-invertible rungs only, or the full ladder treated equally? Recommendation on record: exactly-invertible for the primary (C)/(D)/(B) claims, semantic band as an extended generalization check. **Owner's call — it changes what the paper claims, not just how the code is arranged.**
2. Deferred until the pilot reveals actual run shapes: whether any experiment-orchestration layer is needed at all, and whether runs get tracked (MLflow or otherwise). Not decided now, deliberately.
