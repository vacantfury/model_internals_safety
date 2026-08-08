# AS-5 — the evidence in hand, and what it will actually support

*Written 2026-08-07, after the benign judge arm (§3.6) and the echo crosstab
(§3.7) landed within hours of each other and between them changed which rungs
AS-5 has left. This is the paper-level synthesis: what is on disk, what may be
reported, and what claim the evidence supports. It is NOT a regime map — the
maps of record are `pilot_rebaseline.md` and `band_20260805.md` — and it is not
instrument knowledge, which lives in `../shared/instrument_layer.md`. It is the
one place that asks "given all of it, what is the paper?"*

---

## 1. Runs on disk

| run | date | scope | status |
|---|---|---|---|
| phase-0 pilot | 08-03 | 15 rungs x 100 x 2 models | re-baselined 08-05; maps superseded twice |
| comprehension band | 08-05 | 7 rungs x 200 x 2 models | map of record `band_20260805.md` |
| recalibration | 08-05 | offline, CPU, $0 | control-calibrated floor |
| re-licensing | 08-07 | offline, CPU, $0 | two-rung conclusion confirmed |
| decode lens | 08-07 | job 9008631, Llama | echo crosstab, §3.7 |
| causal sweep | 08-07 | job 9008632, Llama, 3 rungs | **first benign arm ever**, §3.6 |

Two arms are in flight as of writing: the alphabet band (Llama, job `9010200`)
and the §3.6 replication on Qwen (job `9010201`).

## 2. The ladder collapsed, and this is the honest count

Nineteen rungs exist. What survives every control is **one**.

| rungs | verdict | why |
|---|---|---|
| 10 cipher/byte rungs | **inert** | ability ~0 on both models; models cannot read them |
| `hex`, `unicode_escape` | Llama-readable, Qwen-inert | asymmetric; deployment unlicensed on `hex` |
| `reverse_characters`, `tag_block` | **can't-decode controls** | ability 0.00; `tag_block` is EXACTLY invertible and still unreadable |
| `reverse_words` | **control** | lexically transparent — probe reads L3-L5, not the L18-L22 cluster |
| `combining_marks` | not sound | below the control floor on Llama |
| `zero_width` | **demoted to control** (§3.6) | benign refusal 0.90 vs harmful 0.93 |
| `homoglyph` | **demoted to control** (§3.6) | benign refusal 0.99 vs harmful 0.99 |
| **`fullwidth`** | **the substrate** | benign 0.59 vs harmful 0.90 — a 31-point harm-sensitive gap |

**No sound rung is clean on both screens.** §3.6 (does refusal track harm?)
passes only `fullwidth`; §3.7 (is the (S) cell free of echo?) passes only
`homoglyph`, which is the rung §3.6 demoted. The alphabet band exists to find
out whether that is a property of the *encoding family* or an accident of one
script — `../../conf/experiment/naturalness_band_llama.yaml`.

## 3. Reportable and not

**May be reported:**
- **(C) is empty** — 0-1 cells in 1,400, at every operating point. Robust to
  every instrument fix the repo has made.
- **(B) is populated on `fullwidth`** — the gate answer, which survived every
  subsequent instrument revision unchanged.
- **The cipher band is inert** — a clean, large negative, and the single most
  replicated result here.
- **Invertibility does not imply comprehension** — `tag_block`, exactly
  invertible, ability 0.00 on both models.
- **The methodological results in §5**, each measured against its own control.

**May NOT be reported:**
- **Any ASR number, retroactively.** Arm 1 of the benign control failed on all
  three sound rungs and *inverted* on `fullwidth` (benign 0.28 vs harmful 0.12).
  Every ASR this repo has measured used this judge with this control unrun.
- **Any (B)-vs-(D) point estimate.** The split moves with the operating point
  (Llama (B) 20 -> 11 as the read tightens).
- **Anything resting on recognition.** It licenses on `reverse_characters`
  (ability 0.00) while failing on rungs the model comprehends — unexplained
  (instrument_layer §5). Do not build on it until it is resolved.
- **Licensing counts from any AS-5 run before 08-07** — retired null.
- **(B) rates without their denominator.** Post-echo the measured base is 27 and
  33 cells, not 100.

## 4. What the evidence does NOT support

The proposal's frame is a four-regime diagnostic across an encoding ladder, plus
causal validation, plus a crossed intervention. **The ladder cannot carry that.**
Twelve rungs are unreadable, three are controls by construction, two were
demoted this week, and the remaining one is a single script on two models whose
(B) cell rests on ~30 measured cells. A map of the (C)/(D)/(B)/(S) plane across
an encoding ladder is not what is on disk.

Reading a broader claim off `fullwidth` alone would also be exactly the error
this repo keeps catching in its own instruments: generalising from the one cell
that survived screening.

## 5. What it DOES support, and it is a stronger paper

**Encoded jailbreaks are mostly not comprehension failures — and the standard
instruments cannot tell you that.**

**Pillar 1 — the negative result, which is clean.** (C) is empty. There is
essentially no cell where a model cannot decode a payload and complies anyway.
When it cannot read the encoding it refuses at the surface; when it can read it,
it overwhelmingly refuses. **Encoding does not buy comprehension-free
compliance.** The inert cipher band stops being a disappointment and becomes the
evidence: twelve rungs, two models, one repeated result. This contradicts the
straightforward reading of mismatched generalization (Wei et al. 2023 — cited by
name in `../as6/`, established literature, not ours) as *harm hidden from a
model that would otherwise refuse*.

**Pillar 2 — the measurement contribution, which is where the evidence is
overwhelming.** Every instrument needed to measure Pillar 1 is broken by
default, and we have a measured instance of each:

| defect | measured | direction |
|---|---|---|
| echo scored as refusal | ~70% of (S) on two rungs; the JailbreakBench judge does this **by its own docstring** | inflates safety |
| refusal driven by the encoding, not the harm | benign refusal 0.90/0.99 vs harmful 0.93/0.99 | inflates safety |
| ability binary too strict | 439 cells; `hex`/`unicode_escape` were hidden entirely | inflates safety |
| unmeasured scored as negative | `deployment` silently `False` on 13/15 rungs | inflates safety |
| length scored as content | character count alone separates at **AUROC 0.654** | either |
| significance scored as sufficiency | 14/15 rungs pass permutation; **2** clear the control floor | either |
| judge control uninverted | benign-encoded called jailbreak more often than harmful | *deflates* safety |

**The asymmetry is the finding.** Every defect touching the *behaviour* axis —
the axis that decides whether an attack succeeded — inflates apparent safety.
That is not coincidence: "no attack succeeded" is the outcome a broken safety
evaluation produces by default, so its failure modes are not random with respect
to its conclusion. The one exception runs the other way and is a judge artefact,
not a model measurement. A field reporting ASR under encoded attack is reporting
a number with a known bias direction and no control for it.

**Recommendation: lead with Pillar 2, carry Pillar 1 as the worked example.**
Pillar 1 alone is one substrate rung and a large negative; Pillar 2 is seven
independent measured defects with controls, and it *explains* why the field has
not reported Pillar 1. The four-regime taxonomy stops being the contribution and
becomes the instrument that makes the defects visible — which is what it has
actually been doing.

## 6. What would make it a paper

1. **More `fullwidth`-like rungs.** The screen is now explicit:
   comprehension-preserving AND lexically opaque AND not surface-alarming. One
   rung is an existence proof, not a result. *In flight — job `9010200`.*
2. **§3.6 on a second model family.** Every harm-sensitivity number here is
   Llama-only. *In flight — job `9010201`.*
3. **A refusal-judge negative control.** `BehaviorControl.clears()` reads arm 1
   only, so the axis deciding the headline split is unscreened (§3.7 item b).
   *In flight — `scripts/refusal_judge_control.py`.*
4. **The Patchscopes decode measurement**, which takes the (B)/(D) split off the
   operating point entirely (instrument_layer §4.1).
5. **Resolve the recognition anomaly**, or drop recognition from the paper.
6. **A third model family**, to make "two models" into a claim about models.

## 7. What this does to AS-6

Directly, and it is not a retrofit. AS-6's entire claim is separating *never
decoded* from *decoded but never blocked* in a guard's activations. Both of this
week's findings hit that at the core:

- A guard that flags anything wearing an encoding produces a **perfect block
  rate that means nothing** — the guard-side twin of §3.6, and it needs the
  benign arm from its first sweep.
- Echo has no guard-side analogue, but its lesson does: a verdict label whose
  failure mode is documented in its own source and never crossed against the
  independent measurement sitting beside it.

Deployment is AS-5's optional axis and AS-6's central quantity, so every
deployment defect above is load-bearing there in a way it is not here.
