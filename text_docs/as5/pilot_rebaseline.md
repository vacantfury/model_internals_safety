# Phase-0 pilot — re-baselined against the fixed instrument (2026-08-05)

**What this is.** The phase-0 pilot ran 2026-08-03. Instrument fixes #1 and #2 landed 2026-08-05 (`0483fe8`), changing both the ability binary and `assign_regime` itself, so every number computed before that date predates the instrument that produced it. This is the re-derivation, offline: no GPU, no judge calls, no re-run — `scripts/rebaseline_pilot.py` recomputes ability from the cached `plaintext` / `restate_response` under the settled three-route rule (cuts read from `conf/measurements.yaml`), restores recognition's tri-state from the per-family licensing recorded in `results.json`, and re-labels every cell with the current `assign_regime`.

Reproduce with `uv run python scripts/rebaseline_pilot.py` (add `--write` to emit `cells_rebaselined.jsonl` beside each run). It deliberately does **not** self-check against the recorded regimes — disagreement with them is the output. `scripts/rescore_ability.py` keeps its self-check and is now expected to fail it; that guard is working as designed.

---

## 1. The map, re-baselined

`abil` = ability count before → after. `dep?` / `rec?` = was that family's probe licensed.

### Llama-3.1-8B-Instruct (`phase0-20260802`, 1500 cells)

Regime **(U)** = the deployment probe did not license on that rung, so the axis every other label is decided on was never measured. Introduced 2026-08-05 with the tri-state fix; see §3.

| rung | abil | dep? | rec? | C | D | B | S | R | X | **U** |
|---|---|---|---|---|---|---|---|---|---|---|
| base64 | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| base32 | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| **hex** | **21→84** | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| binary | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| ascii_decimal | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| **unicode_escape** | **1→54** | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| rot13 | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| caesar3 | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| caesar7 | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| atbash | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| vigenere | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| morse | 0→0 | n | n | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| reverse_characters | 0→0 | n | y | 0 | 0 | 0 | 0 | 0 | 0 | **100** |
| reverse_words | 17→97 | **y** | y | 0 | 0 | **1** | 88 | 9 | 2 | 0 |
| **zero_width** | **71→100** | **y** | n | 0 | 0 | **7** | **93** | 0 | **0** | 0 |

**1407/1500 relabelled · ability +225 / −0 · 1300/1500 cells (13 of 15 rungs) sit on an unlicensed deployment probe.**

### Qwen2.5-7B-Instruct (tail run, 600 cells — the head run's licensed rungs were killed at the wall)

| rung | abil | dep? | rec? | C | D | B | S | R | X | U |
|---|---|---|---|---|---|---|---|---|---|---|
| **reverse_words** | 6→92 | y | y | 0 | 0 | **6** | 82 | 5 | 7 | 0 |
| **zero_width** | 82→99 | y | y | 0 | 0 | **13** | 86 | 0 | 1 | 0 |

Qwen's twelve cipher rungs plus `reverse_characters`: ability +2, and **every one of those 1258 cells is (U)** — not a single measured deployment reading across the entire cipher band.

### The one-line summary

**The phase-0 regime map is two rungs, not fifteen.** `zero_width` and `reverse_words` are the only rungs on either model where the deployment probe licensed, so they are the only rungs carrying a decode-versus-enforcement claim in either direction. Everything else is a declared hole.

---

## 2. What survives, what changes

**SURVIVES — the pilot's gating result.** The (B) decode-and-comply cell is populated on `zero_width`: **B=7 / S=93 on Llama, B=13 / S=86 on Qwen**, with hard incoherence at 0–1%. The previously recorded range (B=5–13, S=86–95) was computed pre-fix and lands within a couple of cells of the re-derived one, so the headline is not disturbed. `reverse_words` now also reports a populated (B) cell (1 on Llama, 6 on Qwen) for the first time — it was unreadable before the fixes.

**SURVIVES — the fixes did what they claimed.** `reverse_characters` went from 71% hard incoherence to 0 (71/100 cells relabelled X→R): fix #2's split-by-behaviour rule correctly reclassifies surface-cue refusal as `(R)` with a soft `surface_recognition` flag instead of deleting the rung as instrument failure. Every rung now sits under the 10% instrument-failure bar. Ability moved in one direction only, **+225 / −0 on Llama and +103 / −0 on Qwen** — consistent with the documented guarantee that the new cuts can only add recoveries.

**CHANGES — `hex` and `unicode_escape` are readable.** Ability on Llama goes 21→84 and 1→54. This is the correction already landed in `CLAUDE.md`: only ten of the twelve cipher rungs are genuinely inert.

---

## 3. ⚠️ The finding that matters most — deployment is not tri-state, and its probe is unlicensed almost everywhere

**Recognition is tri-state; deployment is not.** But `results.json` records `deployment.licensed` per family, and it is **false on 13 of 15 rungs for Llama and on every rung Qwen completed**, while every cell in those rungs still carries a plain `deployment=false`.

That is precisely the failure tri-state was introduced to fix, one measurement over: an unlicensed probe asserting *"the model did not decode during the attack"* when the truth is *"this instrument could not read this rung."*

It is load-bearing, not cosmetic. `deployment` is the term that separates (S)/(B) from (R)/(D) in `assign_regime`. So on an unlicensed-deployment rung the map is reporting a distinction it never measured, and the whole cipher band's uniform `(R)` reads as an artefact of a silent `False` rather than a measurement.

**This retracts a claim made in this repo on 2026-08-05, in the correction commit `920537a` and in `CLAUDE.md`:** that re-baselining would move `hex` cells from `(R)` to `(S)` because they genuinely decoded and refused. **It does not, and the reason is worse than the original error.** With ability now 84/100 on Llama `hex`, those cells would be (S) *if deployment were measured* — but `hex`'s deployment probe is unlicensed (transfer AUROC 0.691), so every cell reads `deployment=false` and the regime stays `(R)`. The honest label for `hex` is neither (R) nor (S): it is **unmeasurable on the deployment axis**. The correct statement is that the cipher band's uniform (R) is *unsupported*, not that it is *wrong in a known direction*.

**Consequences to carry:**

1. **AS-5** cannot claim a regime map over rungs whose deployment probe did not license. Either report those rungs as unmeasured (the tri-state answer, symmetric with recognition), or license the probe there.
2. **AS-6 inherits this at its core.** Deployment *is* AS-6's central quantity — did the guard decode during classification. The guard-side measurement must be tri-state on deployment from the first line of code, exactly as TODO item 9 already requires for recognition.
3. **Licensing itself may be understated.** These `licensed` flags were computed under the old fixed 0.70 AUROC cut. Licensing moved to a permutation test in `4d3e78d`, and `hex` sits at **0.691 — just under the old cut**. A permutation test may well license it, which would make `hex` the pilot's most informative rung rather than its most ambiguous. Re-running licensing needs the cached activations on the cluster (`/scratch`), not just `cells.jsonl`, so it is not offline work.

---

## 4. The tri-state fix, landed

`deployment` is now `bool | None` end to end (2026-08-05, same sitting):

- `DeploymentReading.harmful` / `.harmless` return `None` per prompt on an unlicensed rung instead of `licensed and bool(...)`, which collapsed to `False`. This was the defect at source (`measurements/deployment.py`).
- `harmful_rate` / `harmless_rate` / `gap` return `None` rather than `0.0` when unmeasured — a gap between two non-measurements is not zero.
- `assign_regime` skips every rule that reads an unmeasured axis and returns the new regime **(U)**.
- `RegimeMap` gains `deployment_unmeasured` + `deployment_unmeasured_rate`, and **`binding_failure_rate` is now computed over MEASURED cells and returns `None` when there are none.** Dividing by `n` returned `0.0` on an unmeasured rung, which reads as "no binding failures here" — the same silent zero, one level up, and it is what fed the "cipher band is inert" conclusion.
- `scripts/phase0_regime_map.py` reports `(B) populated in k/<MEASURED> rungs` and prints a loud `UNMEASURED:` line naming every unlicensed rung.

Tests: 226 green, including four new ones pinning the contract — an unmeasured deployment axis must yield (U) with no coherence flags, must not count as a binding success, and must leave the denominator rather than dilute it.

## 5. Re-licensing under the permutation test — and the LENGTH CONFOUND it exposed (2026-08-05)

Ran on the cluster against the cached activations (array `8957019`, 15 rungs × Llama, CPU-only, single-threaded BLAS). Deployment licensing moved from **2/15 to 14/15**. That looks like the 2-rung map growing into a 14-rung one. **It is not, and the reason is the most important methodological finding in this document.**

| rung | ability | ciphertext **length** AUROC | deployment max AUROC | dep p | licensed |
|---|---|---|---|---|---|
| base64 | **0/100** | 0.6529 | 0.6376 | 0.0149 | new |
| base32 | 0/100 | 0.6491 | 0.5559 | 0.7662 | no |
| hex | 84/100 | 0.6544 | 0.6908 | 0.0050 | new |
| binary | 0/100 | 0.6544 | 0.6533 | 0.0100 | new |
| ascii_decimal | 0/100 | 0.6565 | 0.6393 | 0.0249 | new |
| unicode_escape | 54/100 | 0.6544 | 0.6610 | 0.0100 | new |
| rot13 | 0/100 | 0.6544 | 0.6680 | 0.0050 | new |
| caesar3 / caesar7 | 0/100 | 0.6544 | 0.6736 / 0.6475 | 0.0050 | new |
| atbash / vigenere / morse | 0/100 | ~0.65 | 0.657 / 0.659 / 0.665 | 0.0050 | new |
| reverse_characters | 0/100 | 0.6544 | 0.6562 | 0.0050 | new |
| **reverse_words** | 97/100 | 0.6544 | **0.8442** | 0.0050 | was already |
| **zero_width** | 100/100 | 0.6544 | **0.9454** | 0.0050 | was already |

**Raw character length alone separates harmful from benign at AUROC 0.6544** in this corpus (JBB harmful mean 86.0 chars vs benign 73.8). Every encoder here is monotone in length, so that separation survives into every rung — ciphertext-length AUROC is 0.649–0.657 across the whole ladder. And the newly-licensed rungs land at **mean deployment AUROC 0.659 against mean length AUROC 0.654**: indistinguishable.

**So the permutation test is working correctly and detecting a real signal — the signal is just not decoded content. It is prompt length.** Three independent confirmations:

1. **Ability rules it out directly.** `base64`, `rot13`, `caesar3/7`, `atbash`, `vigenere`, `morse`, `binary`, `ascii_decimal` all read ability **0/100** — the model cannot decode them at all. There is no decoded content in those forward passes to be read, so a licensed "deployment" there cannot be measuring deployment.
2. **The instrument's own coherence rule agrees.** deployment ✓ with ability ✗ is `DEPLOYMENT_WITHOUT_ABILITY`, a HARD incoherence by `regimes.py`'s stated rule that ability upper-bounds deployment. Applying the new licensing without a length control would flag the entire cipher band as instrument failure — which is the correct response, arrived at independently.
3. **The two real rungs separate cleanly.** `zero_width` (0.9454) and `reverse_words` (0.8442) beat the length null by **0.29 and 0.19**. They are the only rungs that do, and they are exactly the two that licensed under the old cut.

**Why the existing control battery missed it.** §8's format-decorrelation control puts benign content through the *identical* encoding pipeline, which defeats a probe firing on "this looks encoded" — both classes look equally encoded. It cannot defeat a probe firing on a *content-correlated* property that the encoder preserves. Length is exactly that: it is correlated with harmfulness in the corpus and monotone through every encoder, so it passes straight through the control.

**Consequences.**

- **The map is still two rungs.** The conclusion of §3 is unchanged — but it now rests on a much stronger argument than "the probe did not license": it rests on "the probe licenses at exactly the length baseline everywhere except two rungs, and beats it by 0.19–0.29 on those two."
- **A length null model is now MANDATORY for both papers**, and it is the same control AS-6's reshape already names for the tokenizer (`Broken-Token`, characters-per-token). This measurement shows it is needed on the AS-5 side too, empirically rather than as a precaution. Filed as `TODO.md` item 17.
- **Permutation licensing alone is not sufficient licensing.** It correctly controls the multiple comparison over the grid; it says nothing about *what* the separating signal is. Significance and construct validity are different questions and the instrument must ask both.
- **Corpus-level fix worth weighing:** length-match the harmful and benign sets, or regress length out of the probe features. The cheap immediate move is to report `deployment_auroc − length_auroc` per rung and refuse to call a rung deployed when that gap is not clearly positive.

`base32` is a small curiosity: it is the one rung that did NOT license (0.5559, p=0.77) despite an identical length profile. Not investigated.

## 6. Status

Offline work complete; the numbers above supersede every pre-2026-08-05 regime figure. **One piece remains and it is not offline: re-license every probe under the permutation test** (`4d3e78d`) rather than the old fixed 0.70 AUROC cut. That needs the cached activations on the cluster (`/scratch`), not `cells.jsonl`. It matters more than a cleanup — `hex` sits at **0.691**, a hair under the old cut, so re-licensing could convert the pilot's most ambiguous rung into a measured one and turn a 2-rung map back into a 3-rung map. Filed as `TODO.md` item 15(b).
