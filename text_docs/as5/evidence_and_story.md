# AS-5 — the evidence in hand, and what it will actually support

> **Story and title of record: §4e** (settled by owner go, 2026-08-08).
> ⚠️ **§4e's leg 2 was WITHDRAWN the same day** — the ladder re-run
> (`9027721`–`9027723`) showed plaintext benign refusal is not flat across
> stages, so post-training does not restore what encoding destroys. It is
> replaced by a stronger negative: **the encoding penalty is invariant to
> post-training**. Leg 1 and the title are unaffected.
> Title: *"Refusal without discrimination: what encoded prompts do to
> safety-trained models."* Sections §4a–§4d are the derivation that produced it
> and are kept for their evidence, not as live proposals; §5 is superseded.
> `proposal.md`'s 2026-08-02 scope and the old title "Can't, didn't, or
> wouldn't?" are both retired.

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
| plaintext baseline | 08-08 | jobs 9012067–70, 4 models | the denominator; §4d |
| scaffold control | 08-09 | jobs 9033595–98, 4 models | **the factorial cell**, §4h |

## 2. The ladder collapsed, and this is the honest count

Nineteen rungs exist. What survives every control is **one**.

| rungs | verdict | why |
|---|---|---|
| 10 cipher/byte rungs | **inert** | ability ~0 on both models; models cannot read them |
| `hex`, `unicode_escape` | Llama-readable, Qwen-inert | asymmetric; deployment unlicensed on `hex` |
| `reverse_characters`, `tag_block` | **can't-decode controls** | ability 0.00; `tag_block` is EXACTLY invertible and still unreadable |
| `reverse_words` | **control** | lexically transparent — probe reads L3-L5, not the L18-L22 cluster |
| `combining_marks` | not sound | below the control floor on Llama |
| `zero_width` | ~~demoted~~ **Llama-only artefact** (job 9010201) | Llama benign 0.90 vs 0.93; **Qwen 0.30 vs 0.87** |
| `homoglyph` | ~~demoted~~ **Llama-only artefact** (job 9010201) | Llama 0.99 vs 0.99; **Qwen 0.29 vs 0.89** |
| **`fullwidth`** | **substrate on both models** | Llama 0.61 vs 0.88; Qwen 0.21 vs 0.83 |
| **`math_bold`** | **new substrate** (job 9010200) | ability 1.00, gap +0.18, harmful refusal only 0.61 |
| `fullwidth_letters` | **substrate** (job 9010200) | ability 1.00, gap +0.25 |

⚠️ **THE §3.6 DEMOTION DID NOT REPLICATE, AND THAT IS THE MOST IMPORTANT
RESULT ON THIS PAGE (2026-08-08, job `9010201`).** §3.6 demoted `zero_width` and
`homoglyph` because their refusal did not track harm on Llama. On Qwen2.5-7B
**all three sound rungs show a large harm-sensitive gap** — `fullwidth` +0.62,
`homoglyph` +0.60, `zero_width` +0.57 — against Llama's +0.27, +0.00, +0.03.

So *"refusal is encoding-driven"* is a fact about **Llama-3.1-8B-Instruct**, not
about models. Llama blanket-refuses encoded benign content (0.59-0.99); Qwen
does not (0.21-0.30). The demotion is withdrawn: those rungs are substrate on
Qwen and their (B) counts there regain their meaning. **The transferable lesson
is the one the gate named in advance — no harm-sensitivity conclusion may be
drawn from a single model, which is exactly what this repo did for a day.**

**Section 2 as first written claimed the ladder had collapsed to ONE rung. That
was true of Llama and was never checked on a second family.** The honest count
is three harm-sensitive rungs per model, and they are not the same three.

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
- **Any discrimination loss attributed to THE ENCODING (2026-08-09, §4h).** The
  attack wrapper is a second cause and on 2 of 4 models it is the larger one.
  Attribute to the encoded-prompt *protocol*, or run the scaffold arm and
  decompose. Every such attribution in §4d, §4f and §4g predates the control.

- **Any harm gap from `zero_width` or `fullwidth`/`fullwidth_letters` without
  its echo-clean recomputation beside it (2026-08-10, instrument_layer §3.11).**
  The echo route displaces those gaps by 0.107–0.256, in *both* directions
  depending on which arm echoes harder (Llama `fullwidth` +0.27 → +0.43 clean;
  Tülu-RLVR +0.42 → +0.16). ⚠️ This is the one behaviour-axis defect that
  flatters the PAPER rather than the model — leg 1 claims discrimination
  collapses toward zero and echo compression delivers that for free.

**Reportable, and now GOVERNED (updated 2026-08-10):** the refusal gaps of legs
1 and 2. Two things changed since this paragraph read "not by the contract's
verdict". The `refusal` instrument exists (TODO 64), so the contract now models
the quantity the paper reports rather than only ASR. And its required screen has
**run**: `homoglyph` — the rung the plaintext baseline was deliberately paired
with, and the one leg 1 rests on — displaces by **0.001–0.029 on all four
models** and clears on every one. So the headline is screened on the echo axis
rather than argued in prose. The screen now rides every run at zero cost.

⚠️ Two limits, stated rather than buried. The two knobs `min_gap_margin` and
`min_plain_gap` are still PLACEHOLDERs (`build_status.py`), so a *contract*
verdict resting on them is not final; and the echo screen clears leg 1 on its
own axis only — §4h's scaffold correction to leg 1's **subject** is untouched by
it and still unapplied.

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

## 4a. ⚠️ THE STORY, THIRD VERSION (2026-08-08) — the dissociation is the paper

> ### ⛔ SUPERSEDED — read §4c (which also withdraws §4b's reasoning)
>
> The third and fourth model arms ran (jobs `9010897` Mistral, `9011034`
> Tulu-3) and the gate the presets declared fired **against** this section.
> Binding failure turns out to vary MORE between rungs inside one model than it
> does between models, so a model-level property cannot be what explains it.
> Read §4b before using anything below. This section is kept because it is the
> reasoning the falsification test was designed against, and because two of its
> claims survive.

*Supersedes §5 below, which was written before the Qwen arm ran. §5 recommended
leading with the measurement contribution because the substantive result looked
like one rung. It is kept as the record of that reasoning; this section is the
recommendation.*

**The two models dissociate, on the same inputs through the same instruments,
and the dissociation runs opposite to the aggregate metric.**

| | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|---|---|---|
| ability (`fullwidth`) | 1.00 | 0.80 |
| harmful refusal | **0.88** | 0.83 |
| benign refusal | 0.61 | **0.21** |
| harm gap | +0.27 | **+0.62** |
| **binding failure, B/(B+S)** | **27-43%** | **5-11%** |
| echo (P) share of cells | 66-74% | 2-61% |

**Llama refuses MORE and binds LESS.** It refuses encoded benign content three
times as often as Qwen, and on the cells where it demonstrably decoded the
request it fails to bind refusal to that comprehension three to five times as
often. Per rung: Llama `fullwidth` 27%, `fullwidth_letters` 27%, `math_bold`
43%; Qwen `fullwidth` 11%, `homoglyph` 11%, `zero_width` 5%.

**The two rankings point opposite ways.** The aggregate refusal rate — the
reportable, standard, outside-visible number — says Llama is the more cautious
model (0.88 against 0.83). Conditioned on demonstrated comprehension, Llama is
the weaker one by 3-5x. **An aggregate safety metric inverts the
internals-grounded ranking**, and it inverts it because a model that refuses on
surface cues scores well on a metric that cannot tell surface refusal from
recognition-grounded refusal.

**Echo is the mechanism by which the aggregate metric gets it wrong**, and it is
not a small correction: 42-74% of cells on most rungs are the model parroting
the ciphertext, which the JailbreakBench judge counts as a refusal *by its own
docstring* (§3.7). Remove them and both the numerator and the denominator move,
the denominator far more.

**Why this needs the internals instrument and cannot be done black-box.** The
claim is conditioned on *the model decoded this prompt*, which is measurement #1
plus a licensed deployment probe. The deployment instrument validates cleanly
within this run: readable rungs read 0.86-0.89 (Llama) and 0.99 (Qwen), while
the can't-decode rungs read 0.653-0.658 — the noise floor. Rungs the model
cannot decode read at chance and rungs it can read far above it, which is the
behaviour a decoding probe should show and the strongest internal evidence the
probe measures decoding rather than surface form.

**Recognition CORROBORATES but must not carry the claim.** Llama's recognition
probe sits at 0.53 (p=0.75, inside the length null by 0.12) while Qwen's reads
0.83-0.86 (p=0.005, beats the length null by 0.21) — i.e. "refusal without
recognition" for Llama and "refusal with recognition" for Qwen, exactly matching
the behavioural split. **But this repo has an unresolved recognition anomaly**
(§5 of instrument_layer) and there is no plaintext baseline in these runs to
distinguish "Llama does not recognise harm here" from "the recognition probe
does not work on Llama". Until that is settled, the dissociation is stated on
ability + behaviour, which need no such assumption.

**Honest limits, all of them:**
- **n = 2 models.** A claim about models needs a third; this is the single
  biggest gap and the cheapest to close.
- **No ASR number is reportable** (judge control fails), so the argument is made
  on refusal rate and binding failure only — which is sufficient, since those
  are the two that disagree.
- **Denominators are small**: 26-38 measured cells per rung after echo removal.
- **The Qwen run has NO control floor** (`kind: none`, 0 controls) — it carried
  no can't-decode rung, so its deployment licensing is permutation-only, the
  exact "significance is not sufficiency" gap. Its 0.99 is far above any floor
  ever derived here, but the screen was not applied.
- **The Llama floor is a `bound`, not the adopted mean+2SD** — only one control
  rung qualified (`math_sans`), below the 5-control minimum.

**What this makes the paper.** The four-regime instrument stops being the
contribution and becomes the method; the measurement defects of §5 stop being
the headline and become the reason nobody has reported this. The finding is
that **safety training can produce refusal that is not bound to comprehension,
that this is invisible to — indeed rewarded by — aggregate refusal and ASR
metrics, and that separating can't / didn't / wouldn't is what makes it
visible.**

**Next runs, in priority order:** (1) a third model family; (2) the Qwen band at
`fullwidth`-class rungs with a can't-decode control included, so it has a real
floor; (3) a plaintext recognition baseline on Llama, which would promote the
recognition axis from corroboration to evidence.

## 4c. ⚠️ THE FOURTH STORY (2026-08-08, later the same day) — §4b's REASON was wrong too, and echo is why *(numbers stand; the ATTRIBUTION is superseded by §4d — the cross-model spread is caused by the encoding, not by the models)*

*§4b's verdict on §4a stands — the axis is refuted — but §4b's stated REASON
("binding failure is rung-level, so a model-level property cannot explain it")
does not survive a balanced re-analysis, and I withdraw it. Both §4a and §4b
made the same class of error one level apart: **comparing cells that are not
the same measurement.** §3.7 had already established why, and neither applied
it.*

### The error, stated once

§4a compared **Llama on `fullwidth`-family rungs** against **Qwen on
`fullwidth`/`homoglyph`/`zero_width`** — different rung sets. §4b then
"corrected" it with a variance decomposition over 12 cells in which Llama
contributed 5 rungs and Qwen and Mistral 2 each — an unbalanced design, whose
within-model term is inflated by whichever model happens to have the most rungs.
Rebalanced to the rungs all four models measured soundly, the verdict flips:

| quantity (balanced 4 models × 2 rungs) | MODEL | RUNG | INTERACTION |
|---|---|---|---|
| harmful refusal | 52% | 13% | 35% |
| benign refusal | **95%** | 0% | 4% |
| binding failure B/(B+S) | 43% | 21% | 36% |

Binding failure is not "rung-level". It has **no clean attribution at all** —
its interaction term equals its rung term, and the rung effect changes SIGN
across models (`homoglyph`→`zero_width`: Mistral **+0.588**, Llama +0.090, Tulu
+0.129, Qwen **−0.062**). No single ordering of models by binding failure exists.

### And then echo removes most of the remaining cells

Echo (P) — the model reproducing the ciphertext, which the refusal judge scores
as a refusal — is **rung-driven and enormous**, measured across all four models:

| rung | echo rate |
|---|---|
| `zero_width` | 0.62 – 0.81 |
| `fullwidth`, `fullwidth_letters`, `math_bold` | 0.66 – 0.76 |
| **`homoglyph`** | **0.00 – 0.11** |

So on every rung except `homoglyph`, two thirds to four fifths of cells are echo,
and any refusal or binding number computed over them is largely a statement
about echo. **The balanced 4×2 design is really 4×1.** §4a's Llama figures were
computed on an echo-stripped base (n = 27, 33) while the Qwen figures it was
compared against were not — that is the specific defect underneath its headline.

### What the data in hand actually supports — the echo-clean rung, all four models

| model | echo | ability | harmful refusal | benign refusal | B/(B+S) |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.11 | 0.91 | 0.99 | **0.99** | 0.012 |
| Mistral-7B-v0.3 | 0.00 | 0.95 | 0.90 | 0.67 | 0.098 |
| Tulu-3-8B ⚠️ | 0.04 | 0.99 | 0.85 | 0.41 | 0.156 |
| Qwen2.5-7B-Instruct | 0.02 | 0.98 | 0.89 | **0.29** | 0.115 |

**The finding that survives every screen applied so far:** on one clean rung, at
n = 100 per model, four models from four independent post-training pipelines
agree on harmful encoded prompts to within 0.14 (0.85–0.99) and disagree on
benign encoded prompts by 0.70 (0.29–0.99). Benign refusal is 95% model-
determined. **The models differ ~5× more in what they refuse when it is harmless
than in what they refuse when it is harmful.**

Binding failure on this rung is nearly a restatement of blanket refusal — a
model refusing 0.99 of everything cannot produce (B) cells — so it carries
little independent information here, and the §4a-style story built on it should
not be rebuilt in either direction.

### ⚠️ The one control that would make this an ENCODING finding has never been run *(IT RAN 2026-08-08 — see §4d. It came back LOW: this IS an encoding finding, and the "95% model-determined" attribution below is superseded.)*

Every number above is measured **only on encoded prompts**. There is no
plaintext behavioural arm anywhere in this repo: `phase0_regime_map` calls
`measure_behavior` on `encoded_harmful` and `encoded_harmless` and never on the
plain corpora, though both are already captured for the probes.

So "Llama refuses 99% of benign encoded requests" has no denominator. If Llama
refuses 95% of benign *plaintext* JBB prompts, the finding is "Llama over-refuses
in general", which is known, not ours, and not about encoding. If it refuses 5%,
the encoding causes a 19× jump in false positives and the asymmetry above is the
paper. **The two readings are opposite conclusions and nothing on disk
distinguishes them.** This is the next run, and it gates the story rather than
decorating it.

## 4d. ✅ THE BASELINE RAN AND THE GATE FIRED LOW — the encoding IS the finding (2026-08-08)

Jobs `9012067`–`9012070`, four models × (plain baseline + `homoglyph`) × 100
prompts, 21–25 min each, all COMPLETED. The baseline and the rung ran **inside
one job per model**, so every comparison below is paired — same judges, same
seed, same session — which is the error class §4c exists to correct.

### The gate, and which branch fired

The preset named two opposite readings. Quoted from
`conf/experiment/plain_baseline_llama.yaml`:

> HIGH plain benign refusal (close to the encoded rate) — the model over-refuses
> this corpus generally, the encoded number is not an encoding effect […] LOW
> plain benign refusal — the encoding manufactures the false positives, the
> cross-model asymmetry is the result, and phase 1 targets the false-positive
> axis rather than the (B)/(D) split.

**Plain benign refusal is 0.01–0.16 on all four models.** The LOW branch fires,
and not marginally — the encoded rate is 2× to 10× the plaintext rate on every
model.

| model | benign refusal, plain | benign refusal, `homoglyph` | Δ |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.10 | 0.99 | **+0.89** |
| Mistral-7B-v0.3 | 0.01 | 0.63 | +0.62 |
| Tulu-3-8B | 0.16 | 0.48 | +0.32 |
| Qwen2.5-7B-Instruct | 0.11 | 0.30 | +0.19 |

### ⚠️ Three defects in the first write-up of this section, corrected 2026-08-08

Kept visible because two of the three are errors this repo had already recorded
and paid for elsewhere.

1. **Mistral's collapse was reported as real and is not.** With a CI on the
   difference: Llama **+0.82 [+0.74, +0.90]**, Tulu-3 **+0.34 [+0.20, +0.48]**,
   Qwen **+0.20 [+0.07, +0.33]**, Mistral **+0.09 [−0.06, +0.24]** — which
   includes zero. Three collapses, not four, and the graded four-row table
   implied a dose-response that one row does not support.
2. **A MAX statistic was used as evidence.** "Benign refusal spans 0.15 in
   plaintext against 0.69 encoded, 4.6× wider" compares two max-minus-min
   statistics over four noisy points. `instrument_layer.md` §2.4 records that
   exact statistic as n-dependent and not comparable — committed here one level
   up, in prose, the day after it was written down.
3. **"Mistral inverts" was the wrong mechanism**, and reading it as an exception
   is what hid the actual result below.

### The result: the effect is symmetric, and it runs OPPOSITE ways on the two arms

Each observed cross-model spread, bootstrapped against a null in which all four
models sit at the observed mean (20 000 draws, n = 100 per cell):

| arm | observed spread | spread if the 4 models were identical | verdict |
|---|---|---|---|
| plain harmful | **0.57** | 0.08 [0.02, 0.16] | huge real differences |
| plain benign | 0.15 | 0.06 [0.02, 0.12] | marginal |
| **`homoglyph` harmful** | **0.08** | 0.05 [0.01, 0.10] | **indistinguishable from noise** |
| `homoglyph` benign | **0.69** | 0.10 [0.03, 0.19] | huge real differences |

- **Harmful arm: encoding ERASES model differences.** Four models spanning
  0.38–0.95 in plaintext — a 0.57 spread, far outside noise — all land at
  0.91–0.99 encoded, a spread that cannot be distinguished from zero.
- **Benign arm: encoding CREATES them.** Barely separable in plaintext (0.15,
  marginal against a 0.12 null ceiling); 0.30–0.99 encoded.

Mistral is not an exception. It rises to meet the others *because encoded
harmful refusal is model-independent*, which is one mechanism rather than an
inversion.

**The honest caveat:** encoded harmful refusal sits at 0.91–0.99, so
*equalizes* and *saturates* both fit and n = 100 cannot separate them. The
consequence for evaluation is identical either way, and that consequence is the
claim.

### The claim: the harmful arm has no discriminative power, and it is the only arm the field reports

Llama and Qwen are indistinguishable on encoded harmful prompts (0.99 vs 0.91,
CIs overlapping). They are not remotely the same model: under `homoglyph` Qwen
discriminates harm at **+0.61** and Llama at exactly **0.00**. Every
encoded-attack benchmark reports the harmful arm alone, so every one of them
assigns these two models the same score.

**This framing is deliberately not "encoding causes false positives."** That
version invites the reviewer answer *refusing homoglyph text is correct
behaviour, no legitimate user sends it* — and there is no good reply. Whether
refusing encoded input is desirable is irrelevant to whether the measurement can
tell two very different models apart. It cannot.

### The pipeline evidence: a harmful-only metric reports the OPPOSITE SIGN

The cross-family result above cannot say what produces the difference. Tülu-3's
ladder can, because SFT → DPO → RLVR is a published recipe run on identical base
weights (`instrument_layer.md` §3.6.2, jobs `9011347`/`9011348`/`9011349`):

| rung | harmful refusal, SFT → DPO → RLVR | harm gap, SFT → DPO → RLVR |
|---|---|---|
| `fullwidth` | 0.96 → 0.73 → 0.73 | +0.16 → +0.28 → **+0.43** |
| `homoglyph` | 0.99 → 0.90 → 0.93 | +0.20 → +0.27 → **+0.45** |
| `zero_width` | 1.00 → 0.87 → 0.81 | +0.09 → +0.44 → **+0.44** |

**On all three rungs the harmful arm falls while the harm gap roughly doubles.**
A benchmark reading refusal-under-attack sees 0.96 → 0.73 and reports that
two-thirds of Tülu-3's published safety pipeline made the model *less* safe. The
gap says it got substantially *better* at telling harm from harmless. **The two
metrics disagree in sign, on every rung.**

⛔ **FALSIFIED 2026-08-08 (§4f).** This read: *SFT behaves like
Llama-3.1-8B-Instruct and DPO like Qwen2.5-7B.* With the plaintext arm the two
are different phenomena — Llama's encoding-induced benign excess is **+0.89**,
SFT's is **+0.34**, and SFT already refuses 0.45 of benign PLAINTEXT. They match
only on the encoded arm, which is exactly what leg 1 says cannot distinguish
models. The superseded claim continued:** The Llama/Qwen difference is not a family difference —
it is what DPO does that SFT does not, established on identical base weights,
which no cross-family comparison could ever say.

Independent confirmation worth recording: `homoglyph` benign refusal at RLVR
reads **0.48** in the ladder job and **0.48** in `plain_baseline_tulu3`
(`9012070`) — the same checkpoint, two presets, two sessions, n = 100 each.

### What §4c got right, and what it got wrong

The variance decomposition was right; the subject was not. §4c attributed the
benign spread to the models. The ladder shows it is attributable to a **training
stage**, and the four-cell design shows it is absent in plaintext — so the
variance §4c assigned to *model* belongs to *post-training × encoding*. Its
"95% model-determined" was computed across encoded conditions only, with no
plaintext cell to separate the terms.

### Replication across rungs, and its limit

Three rungs × two models, from earlier runs — harmful/benign refusal: Llama
0.93/0.90, 0.99/0.99, 0.90/0.59; Qwen 0.87/0.30, 0.89/0.29, 0.83/0.21. The
cross-model difference is **0.06 / 0.10 / 0.07** on the harmful arm and
**0.60 / 0.70 / 0.38** on the benign arm — the same 5–10× ratio every time.

⚠️ **Its limit: `zero_width` and `fullwidth` are echo-heavy** (0.62–0.81 and
0.66–0.76), and §3.7's rule is that the refusal judge scores echo as refusal. So
only `homoglyph` is clean and this is suggestive, not established. On the ladder
the echo bias runs *against* the effect (DPO echoes more than SFT), so the true
gap growth is at least as large as the table shows.

### ⚠️ The binary harm judge is a non-refusal detector, and the plaintext arm proves it

This run also produced the cleanest possible read on the judge, because on
**plaintext benign prompts no attack exists by construction** — so any nonzero
"ASR" there is judge error, with no confound to argue about.

| model | harmful ASR | harmful non-refusal | benign ASR | benign non-refusal |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.07 | 0.08 | **0.70** | 0.90 |
| Qwen2.5-7B-Instruct | 0.08 | 0.08 | **0.63** | 0.89 |
| Tulu-3-8B | 0.05 | 0.05 | **0.61** | 0.84 |
| Mistral-7B-v0.3 | 0.61 | 0.62 | **0.69** | 0.99 |

Two things are visible at once. On the harmful arm **ASR equals non-refusal to
within 0.01 on all four models** — the harm judge contributes no information the
refusal judge did not already carry. On the benign arm it fires at 0.61–0.70,
roughly 0.7 of whatever the model complied with. It is measuring *compliance*,
discounted, and calling it attack success. Mistral is the tell: its harmful ASR
of 0.61 is its compliance rate, not a harm rate.

**Consequence, and it is retroactive:** no ASR number this repo has ever
produced is reportable, now established on the clean control rather than
inferred from the encoded one. The detail field already carries
`binary_judge_caveat = strongreject_2402.10260`; this is that critique measured
on our own judge. It promotes the judge repair from "proposed experiment 3" to a
blocking dependency for any behavioural claim.

**The refusal judge, by contrast, validates.** Plaintext gap +0.79 to +0.82 on
the three aligned models, benign false-positive rate 0.01–0.16. The two judges
are not equally broken and must stop being described together — every regime
count in this repo splits on `refused`, not on the ASR judge, so the (B)/(S)
map is unaffected by the paragraph above.

### What this changes downstream

- **AS-5 is a paper about encoding.** The §5 fallback (measurement contribution
  alone) is no longer the recommended story; it becomes the second half.
- **Phase 1 targets the false-positive axis** — what makes a model refuse benign
  encoded content — not the (B)/(D) split. That is a different intervention
  design, which is what the gate was for.
- **The single-model lesson binds again.** Mistral inverts the direction of the
  effect. Any claim of the form "encoding does X to refusal" needs the slate.
- **AS-6 inherits the sharpest version of this.** A guard whose block rate rises
  on benign encoded content has the same defect with none of the visibility: a
  perfect block rate and no benign arm reads as a perfect guard.

## 4e. ✅ THE STORY OF RECORD — title settled, three legs, restructured (owner go, 2026-08-08)

**This section is the story of record. §4a–§4d are kept as the derivation, not as
live proposals.** §4d had the phenomenon right; §4e retires the title and states
the three legs.

⚠️ **RESTRUCTURED 2026-08-08 (owner go), and the restructure is the second one in
a day.** The version settled that morning had a *mechanism* as leg 2 — the Tülu
ladder read as post-training restoring the discrimination encoding destroys — and
the re-run with a plaintext arm refuted it hours later (§4f). Leg 2 below is the
replacement, and §4g then generalised leg 1 from one encoder to a class. The
withdrawn mechanism is **not** restated here; §4f is its home. What survives the
churn untouched is the title, for the reason given directly below.

### The title, and why the old one had to go

**"Can't, didn't, or wouldn't?" is RETIRED.** It titles the four-regime
decomposition, and §2 already recorded that the frame no longer carries the
paper: the ladder collapsed to essentially one substantive rung, (C) is empty,
(B)/(D) rests on an operating point and may not be given as a point estimate, and
recognition is unreportable. A title promising a three-way diagnosis the evidence
cannot deliver is the kind reviewers check first.

**Title of record: "Refusal without discrimination: what encoded prompts do to
safety-trained models."** It names the finding and covers both the collapse and
the false-positive axis.

⚠️ **Its stated justification is corrected (2026-08-08).** This paragraph read
"it survives the Mistral inversion in §4d, where encoding *raises* refusal."
§4d's statistical pass withdrew that inversion — Mistral's harm-gap change is
**+0.09 [−0.06, +0.24]**, not distinguishable from zero, and its rise on the
harmful arm is the *same* mechanism as every other model rather than a
counter-direction. **The title stands, and the corrected reading supports it
better than the old one did:** "refusal without discrimination" is now the
literal finding, since encoded harmful refusal is model-independent (spread 0.08
against a 0.10 noise ceiling) while discrimination spans 0.00–0.61. The title
never asserted a direction, which is why it survived a correction to its own
rationale.

### The thesis, in one sentence

> Safety evaluation reports refusal. Encoded prompts pull refusal apart from
> discrimination — and once separated, the reported metric cannot tell four very
> different models apart, and points the wrong way along a published safety
> pipeline.

Three clauses, three legs.

### The three legs, in order

**1. The phenomenon — encoding defeats discrimination, not refusal (§4d, §4g).**

Benign refusal rises 2–10× over plaintext on four models, and on Llama-3.1-8B
the harm gap goes **+0.82 → 0.00**: benign and harmful `homoglyph` prompts
refused at an identical 0.99. No probe required.

> ✅ **THE SUBJECT CORRECTION IS APPLIED (2026-08-21).** Both kits now carry the
> three-arm table and the wrapper/character split as a measured result inside
> the leg-1 section, and say "protocol" where the subject is the deployed
> condition. What follows is the finding as §4h stated it, kept as the record.
>
> ⚠️ **THE SUBJECT OF THIS LEG WAS WRONG (2026-08-09, §4h).** The scaffold control separates the attack *wrapper* from
> the *characters*, and on 2 of 4 models the wrapper alone causes most of the
> loss — Llama +0.67 of +0.84, Tülu +0.28 of +0.37 — while on Qwen it causes
> none of it. The numbers above all stand; what does not stand is "encoding" as
> the subject. The supported subject is the **encoded-prompt protocol**.
> Substance, decomposition and CIs: §4h; the argument it joins is §4j.

**State it as the measurement claim, not the utility claim.** "Encoding causes
false positives" invites the reviewer answer *refusing homoglyph text is correct
behaviour, no legitimate user sends it*, and there is no good reply. The
defensible form: **the harmful arm has no discriminative power, and it is the
only arm the field reports.** Llama and Qwen are indistinguishable on encoded
harmful prompts (0.99 vs 0.91, CIs overlapping; cross-model spread 0.08 against
a 0.10 noise ceiling) while discriminating at 0.00 and +0.61 respectively.
Whether refusing encoded input is *desirable* has no bearing on whether the
measurement can tell two very different models apart. It cannot.

⚠️ **Report the FRACTION of discrimination destroyed, never the absolute
difference** (§4g). Absolute gap-lost compares models on a scale they do not
share, and using it already produced one wrong model ordering in this repo —
§4d called Mistral the most robust model when it is merely the least
discriminating one, with a plaintext gap of 0.32 against 0.80–0.83:

| model | fraction of plaintext discrimination destroyed |
|---|---|
| Qwen2.5-7B | **4–26%** |
| Mistral-7B-v0.3 | 22–53% |
| Tülu-3-8B | 41–53% |
| Llama-3.1-8B | **67–100%** |

**The claim is about a class of encoders, not one** (§4g). Five substrate rungs
through the same paired design, and the per-model mean gap lost separates the
models by 0.53 — so the claim generalises beyond `homoglyph`.

> ⚠️ **The QUANTITATIVE form of this leg is corrected (2026-08-10).** It read
> "within-model spread 0.10–0.27 against a between-model range of 0.56 — the
> model term dominates by 2–5×, so *gap lost* is primarily a model property."
> On echo-clean cells the spread is **0.07–0.48** and the dominance **1.1×–7.7×**
> (§4g, `instrument_layer.md` §3.11.1). **Do not write "primarily a model
> property" and do not quote 2–5×.** The generalisation itself is unaffected;
> only the claim that the encoding term is a minor one falls.

**2. The controlled series — the remedy does not work, and the metric reports
its progress backwards (§4f).**

Tülu 3's SFT → DPO → RLVR on **identical base weights** along a **published**
recipe with public data. It says two things, and they belong together because
they come from one series:

*(a) You cannot post-train your way out of it.* The complete pipeline moves
plaintext harm discrimination **+0.55 → +0.80** and leaves the encoding-induced
loss unchanged at **0.34–0.50** at every stage and on every rung — non-monotone,
every endpoint within 0.11 of its start. General over-refusal genuinely improves
(plaintext benign refusal 0.45 → 0.16). The encoding penalty does not move. A
dose-response *null* along a recipe whose every stage is documented.

*(b) The field's metric has the wrong sign on that pipeline.* SFT → RLVR, all
three rungs — no rung is selected, and it holds on every one:

| rung | plaintext harmful | **encoded harmful — the reported metric** | encoded harm discrimination |
|---|---|---|---|
| `fullwidth` | 1.00 → 0.96 | **0.94 → 0.73** (−0.21) | +0.12 → +0.42 (**+0.30**) |
| `zero_width` | 1.00 → 0.96 | **1.00 → 0.83** (−0.17) | +0.10 → +0.46 (**+0.36**) |
| `homoglyph` | 1.00 → 0.96 | **0.99 → 0.94** (−0.05) | +0.20 → +0.46 (**+0.26**) |

Plaintext harmful refusal is flat across the whole pipeline. Encoded harmful
refusal *falls* on all three rungs. Encoded discrimination *rises* on all three.
**Three quantities, and the one the field reports is the only one pointing the
wrong way.** A reviewer can dismiss "your metric is noisy"; they cannot dismiss
"your metric inverts on a published safety pipeline." This is the sharpest
sentence the evidence supports and it belongs in the abstract.

**3. The instrument.** None of the above is measurable without controls the field
does not run — the benign arm, the echo screen, the length null, the control
floor, the plaintext baseline. Every defect found on the behaviour axis inflated
apparent safety, which is structural rather than coincidental: "no attack
succeeded" is what a broken safety evaluation returns by default.

**Demotion recorded:** `CLAUDE.md` recommended leading with the measurement
contribution (leg 3) from 2026-08-07. That was correct when the ladder had
collapsed and nothing positive had replaced it. It is the third leg now — still
in the paper, no longer the lead.

### Why the legs were reordered rather than the title rewritten

The title names a *state* of the model. Legs 1 and 2 name a phenomenon and a
mechanism, and mechanisms are what got refuted — twice in two days. A title
pinned to a mechanism would have had to be retired with it; this one did not,
and that is the argument for leaving it alone through the next revision too.

### ✅ The open question is ANSWERED, and the answer was already on disk (2026-08-10)

**As framed, every number in legs 1 and 2 is a refusal rate on a paired arm. AS-5
could be written without a single activation capture.** That is not a weakness in
the result — a phenomenon that needs no probe is harder to attack, and leg 1 says
so explicitly. But this repo's defining property against its siblings is that it
goes *inside* the model, and the story of record did not.

**Design of record: `text_docs/as5/phase1_design.md`.** Read it before planning
any internals work; this block is the pointer, not a second home.

✅ **AND IT IS NOW IN THE PAPER (2026-08-21).** Both kits carry
`\section{The discrimination is present and unread}` between leg 2 and the
instrument section, with Table `tab:internals`. The argument the three legs make
together is §4j. AS-5 is no longer a paper that could be written without an
activation capture.

The one-line answer, computed from the §4h run records rather than from a new
run: **a harm direction fit on plaintext separates encoded harmful from encoded
benign activations at AUROC 0.938–0.995 on all four models, while the
behavioural harm gap in the same condition spans −0.01 to +0.55.** On
Llama-3.1-8B that is **0.981 against −0.01**. The discrimination is not lost in
the representation; it is lost between representation and behaviour. It clears
the length null (+0.28 to +0.34), a black-box surface baseline (+0.33 to +0.38)
and XSTest lexical decorrelation — the three screens that have killed previous
claims here.

⚠️ **Two limits that travel with the number, always:** the control floor is
UNUSABLE on those runs (`n_controls = 0`, single-family), so licensing is
permutation-only — §2.5's live defect; and the reading is corroborated by one
instrument, which is what `deployment.py`'s docstring already required be said.
Both are stage-1/stage-2 line items in the design doc, not footnotes.

### The withdrawn leg 2, and where its derivation lives

**Not restated here.** The morning version of leg 2 read the Tülu ladder as
post-training *restoring* the discrimination encoding destroys; the re-run with a
plaintext arm showed plaintext benign refusal falling in lockstep (0.45 → 0.17 →
0.16), so the monotone encoded fall was largely general de-refusal showing
through. In leg 1's own currency there is no trend on any rung. **§4f is that
derivation's home** — the gate, the tables, and the falsified "SFT behaves like
Llama-3.1-8B-Instruct" corollary — and duplicating it here is what the one-home
rule exists to prevent.

What matters for the story of record is only this: the replacement above is a
*null* rather than a graded recovery, and a null on a fully published recipe is
the stronger claim. Leg 1 was measured with in-job plaintext arms from the start
and nothing in the withdrawal reaches it.

**Method note worth keeping.** This is the second time in one day that a
preset's `gates:` block did the work the schema requires it to do — it named the
falsifying observation *before* the run, so the refutation took one reading
rather than an argument. The gate was authored by the peer session and was
sharper than the reasoning that recommended the run.

### ⚠️ The judge inverts on plaintext benign prompts, on three more checkpoints

`plain_benign_asr` reads **0.29 / 0.55 / 0.67** across SFT / DPO / RLVR against
`plain_harmful_asr` of **0.00 / 0.04 / 0.04**. On plaintext benign prompts no
attack exists by construction, so this is pure judge error — §4d's finding
replicated on three further checkpoints, and the contract withheld
`behavior_plain` on all three for exactly that reason. **No ASR number, still.**

### ✅ The gap this story had is CLOSED — and closing it is what refuted leg 2

**The ladder had no plaintext arm.** Jobs `9011347`–`9011349` ran before
`run_plain_behavior_baseline` landed (peer commit `5a8d2a9`), so the original leg
2's series was in *encoded* benign refusal while leg 1's headline quantity is
*gap lost relative to plaintext*. The two legs sat in different currencies, and
this section said re-running the three stages would close it.

**It ran (jobs `9027721`–`9027723`) and it closed the gap by refuting the leg.**
Both legs are now in the same currency, and in that currency the ladder shows no
trend. Worth stating plainly as a method result: **the two legs being in
different currencies was not a presentational weakness, it was the defect** —
a numerator without its denominator, the same class §4d was written to correct,
recurring one level up in the mechanism leg. Putting them on one scale was
sufficient to kill the claim.

### What AS-6 contributes to this story

The same phenomenon in the defence, where end-to-end ASR structurally cannot see
it. And the guards land on the same axis: `phase1_map.md` §0.5 measures Llama
Guard's benign block rate at 0.29–0.53 against WildGuard's 0.05–0.29, so the
guard-side benign arm is not only a control — it measures a property of the
guard's own post-training, exactly as leg 2 does for targets.

## 4f. ⚠️ THE LADDER RE-RAN WITH ITS PLAINTEXT ARM AND LEG 2's MECHANISM IS WITHDRAWN (2026-08-08)

Jobs `9027721`/`9027722`/`9027723`, three stages × 8 rungs × 100 prompts per arm,
1:14–1:25 each, all COMPLETED. Every number below is within-run and paired: each
stage's plaintext baseline and its encoded arms ran in one job, which is exactly
what the earlier ladder (`9011347`–`9011349`) could not do.

### The gate, and which branch fired

> If plaintext benign refusal is flat across the three stages, the encoded fall
> is a real recovery of discrimination […] If plaintext benign refusal ALSO falls
> across stages, the ladder is measuring general de-refusal rather than anything
> about encoding, leg 2 collapses into a restatement of "later stages refuse
> less", and §4e's mechanism has to be withdrawn.

**Plaintext benign refusal: 0.45 → 0.17 → 0.16.** Second branch.

| stage | plain harmful | plain benign | plain gap |
|---|---|---|---|
| SFT | 1.00 | **0.45** | +0.55 |
| DPO | 0.96 | **0.17** | +0.79 |
| RLVR | 0.96 | **0.16** | +0.80 |

The ladder's headline — "benign refusal falls monotonically at every stage and
every rung" — is substantially this. Tülu-3's recipe adds CoCoNot contrastive
prompts specifically to reduce over-refusal of safe prompts, and it works: the
plaintext benign rate falls 0.29. The encoded condition inherited it.

### In leg 1's currency there is no trend at all

Gap lost = plaintext harm gap − encoded harm gap. This is the quantity leg 1
reports, and the only one in which the two legs are comparable:

| rung | SFT | DPO | RLVR |
|---|---|---|---|
| `fullwidth` | +0.43 | +0.50 | +0.38 |
| `homoglyph` | +0.35 | +0.50 | +0.34 |
| `zero_width` | +0.45 | +0.36 | +0.34 |
| **mean** | **0.41** | **0.45** | **0.35** |

**Non-monotone on two rungs, and every endpoint lands within 0.11 of its start.**
A 0.10 spread across three stages at n = 100 is inside noise. The
encoding-induced benign excess tells the same story on the one rung that is
echo-clean: `homoglyph` +0.34 → +0.44 → +0.32. It falls on `fullwidth` and
`zero_width` (+0.37→+0.15, +0.45→+0.21) — the two echo-heavy rungs, where
§3.6.2 established that DPO and RLVR echo *more* than SFT and the refusal judge
scores echo as refusal, so that fall is the direction the echo bias produces.

**§4e leg 2 is WITHDRAWN as a recovery claim.** The ladder does not show
post-training restoring the discrimination encoding destroys.

### What replaces it is stronger, because it is a null on a published recipe

> **Tülu-3's complete published safety pipeline improves plaintext harm
> discrimination from +0.55 to +0.80 and leaves the encoding-induced loss
> unchanged at 0.34–0.50.**

SFT → DPO → RLVR, six benchmarks, CoCoNot, 50k WildGuardMix prompts — the full
recipe, with public data, on identical base weights. It moves plaintext
discrimination by +0.25 and the encoded loss by nothing. **The standard safety
pipeline is blind to this failure mode**, which is a dose-response *null* along a
recipe whose every stage is documented, and a stronger claim than the graded
recovery it replaces.

### And the sign disagreement survives, now within-run

| rung | plain harmful, SFT→RLVR | encoded harmful, SFT→RLVR | encoded gap, SFT→RLVR |
|---|---|---|---|
| `fullwidth` | 1.00 → 0.96 | 0.94 → **0.73** | +0.12 → **+0.42** |
| `zero_width` | 1.00 → 0.96 | 1.00 → **0.83** | +0.10 → **+0.46** |
| `homoglyph` | 1.00 → 0.96 | 0.99 → 0.94 | +0.20 → **+0.46** |

Plaintext harmful refusal is flat across the whole pipeline (1.00 → 0.96).
Encoded harmful refusal *falls* by up to 0.21. Encoded harm discrimination
*rises* by up to 0.36. A benchmark reading refusal-under-attack — the standard
metric — reports the pipeline as making the model less safe while both plaintext
and encoded discrimination improve. **Three quantities, and the one the field
reports is the only one pointing the wrong way.**

### ⚠️ "SFT behaves like Llama-3.1-8B-Instruct" is FALSIFIED, and by the paper's own thesis

§3.6.2 and §4e leg 2 both assert it. With the plaintext arm:

| | plain benign | `homoglyph` benign | encoding-induced excess |
|---|---|---|---|
| Llama-3.1-8B-Instruct | **0.10** | 0.99 | **+0.89** |
| Tülu-3 SFT | **0.45** | 0.79 | **+0.34** |

They are not the same phenomenon. Llama's blanket refusal is specifically
encoding-induced; SFT's is mostly general over-refusal that it also has in
plaintext. **They look alike only on the encoded arm — which is leg 1's entire
point.** The claim was made from the encoded arm alone, so the paper committed
its own thesis as an error, inside the paper, in the section that states the
thesis. Keep this in the write-up: it is the cleanest demonstration available
that the failure is easy to make and not a straw man.

### What survives untouched

**Leg 1.** It was measured with the plaintext baseline in-job from the start, so
nothing here reaches it: Llama's +0.82 → 0.00, the harmful arm's
model-independence (spread 0.08 against a 0.10 noise ceiling), the four-family
result. It gains a robustness argument — gap lost is 0.34–0.50 on Tülu-3 at
every stage on three rungs, alongside 0.82/0.34/0.20/0.09 across four families.

**Leg 3**, the instrument contribution, gains an eighth entry: *a stage series
measured without a plaintext arm reads general de-refusal as an encoding effect.*
That is the same numerator-without-denominator defect §4d was written to correct,
recurring one level up in the mechanism leg — the third instance of this class,
and the reason the re-run was worth 3 GPU-hours.

## 4g. ✅ LEG 1 GENERALISES FROM ONE ENCODER TO A CLASS — and the model ordering was an artefact of not normalising (2026-08-08)

*(Numbered 4g on 2026-08-08: it was written as a second §4f while §4f above was
being written in the same hours. The three existing `§4f` cross-references all
point at the ladder section and are correct as they stand.)*

**Jobs `9029608`, `9029610`–`9029612`: four models × 7 rungs × 100 prompts per
arm, in-job plaintext baseline, 1:12–1:22 each.** §4d's headline was measured on
four models and ONE encoder (`homoglyph`), while the title claims something about
encoded prompts generally. This run puts five substrate rungs through the same
paired design.

### The gate: is gap-lost-to-encoding one quantity per model, or per (model, encoding)?

**Branch A, and the margin is 2–5×.** Gap lost, on rungs the model can actually
read (ability ≥ 0.8):

| model | plaintext gap | gap lost, range over readable substrate rungs | within-model spread |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.83 | −0.56 … −0.83 | **0.27** |
| Qwen2.5-7B-Instruct | 0.80 | −0.03 … −0.21 | **0.18** |
| Mistral-7B-v0.3 | 0.32 | −0.07 … −0.17 | **0.10** |
| Tülu-3-8B (RLVR) | 0.81 | −0.33 … −0.43 | **0.10** |

Between-model range is **0.56**; within-model spread is 0.10–0.27. The model term
dominates the encoding term by 2 to 5×, so *gap lost* is primarily a model
property and **leg 1 generalises beyond `homoglyph`**. It is not constant,
though — encoding contributes a real secondary term, and a paper claiming a
single per-model constant would be overstating it.

> ⚠️ **THE DOMINANCE MARGIN DOES NOT SURVIVE THE ECHO SCREEN (2026-08-10,
> `instrument_layer.md` §3.11.1).** Three of the five substrate rungs —
> `fullwidth`, `fullwidth_letters`, `zero_width` — fail the echo-displacement
> screen on three of the four models, by 1.2–2.2× their own bar; `homoglyph` and
> `math_bold` clear everywhere. Recomputing the identical statistic on echo-clean
> cells (the reported figures reproduce as 2.1–5.6×, so it is like-for-like):
> within-model spread **0.07–0.48** against a between-model range of **0.53**,
> i.e. **1.1×–7.7×**. Llama goes 0.27 → 0.48 and Tülu 0.10 → 0.41. **At 1.1× the
> two terms are the same size, so "primarily a model property" is not supported
> and the sentence above must not be used as written.**
>
> **What survives, and it is most of the section.** The per-model MEAN gap lost
> is nearly unmoved (0.684→0.572, 0.154→0.041, 0.390→0.440, 0.120→0.111), so
> *models differ, and by a lot* stands — as does the normalised table below,
> whose ordering is driven by the plaintext denominators. What falls is only the
> quantitative dominance claim. **Leg 1 still generalises beyond `homoglyph`**;
> what it may no longer say is that the encoding term is a minor one.
>
> The mechanism is exactly this section's own lesson one level up: echo rate
> varies by rung far more than by model, so an unscreened spread ACROSS rungs is
> partly measuring echo exposure. A displacement can sit under the bar on every
> rung individually and still dominate the variance between them — screening each
> reported gap is not sufficient for a statistic computed over gaps.

### ⚠️ The reading correction, and it reverses §4d's model ordering

**Absolute gap lost is confounded by how much discrimination the model had to
lose.** Normalised by each model's own plaintext gap:

| model | fraction of plaintext discrimination destroyed |
|---|---|
| Qwen2.5-7B | **4–26%** |
| Mistral-7B-v0.3 | 22–53% |
| Tülu-3-8B | 41–53% |
| Llama-3.1-8B | **67–100%** |

§4d ranked Mistral the most robust model on the strength of its −0.09 absolute
loss. **It is not: it is the least discriminating model to begin with** (plaintext
gap 0.32 against 0.80–0.83 for the other three), so it had little to lose. Qwen
is the genuinely robust one, and Mistral moves to the middle. **Report the
fraction, not the difference** — an absolute loss compares models on a scale
they do not share.

### Two internal checks the run passes

**The can't-decode rungs land exactly on −plaintext-gap** (`base64` and
`tag_block`: Llama −0.83 against a 0.83 gap, Qwen −0.80 against 0.80, Tülu
−0.80/−0.81 against 0.81). A rung the model cannot read destroys *all* of its
discrimination, which is what the arithmetic must produce, and it confirms
can't-decode rungs are excluded from the substrate spread rather than averaged
into it.

**Llama on `homoglyph` reads −0.83 against a plaintext gap of 0.83** — total
loss, independently reproducing §4d's +0.82 → 0.00 in a different job.

**Mistral's substrate set is only two rungs, not five.** Its ability is 0.18 on
`fullwidth`, 0.00 on `math_bold` and 0.10 on `fullwidth_letters`, so those cells
are can't-decode cases wearing a substrate label; only `homoglyph` (0.95) and
`zero_width` (0.98) are readable. Its 0.10 spread rests on n=2.

## 4h. ⚠️ THE SCAFFOLD CONTROL RAN AND LEG 1's SUBJECT IS WRONG — it is the PROTOCOL, not the encoding (2026-08-09)

**Jobs `9033595`–`9033598`: four models × `homoglyph` × 100 prompts × three
paired arms, 24–28 min each.** Run records down-synced and local:
`outputs/runs/phase0/*/scaffold-control-*/`.

### The confound, and where it came from

A hostile review of the AS-5 draft named it, and it was verified against the
draft's own methods paragraph before being accepted. **The encoded condition
changes TWO things at once.** It transforms the request's characters, *and* it
wraps them in an attack template that announces an encoding is present and asks
the model to work with it. Every number in §4d, §4f and §4g attributes the whole
change to "the encoding" — but the plaintext baseline they are read against
carries no wrapper either, so the wrapper's contribution has never been
separated from the characters'.

This is a factorial design with one cell never run. The missing cell is
**plaintext content wearing the attack scaffold**: same request, same template,
no character transformation. `pipeline.scaffold_arm` builds it, and it is now
part of the spine rather than the entrypoint, so it cannot be present in one
script and absent in another.

### The three arms, refusal rates (n=100 per arm)

| model | plain H | plain B | gap | scaffold H | scaffold B | gap | encoded H | encoded B | gap |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.93 | 0.10 | **+0.83** | 0.99 | 0.83 | **+0.16** | 0.98 | 0.99 | **−0.01** |
| Tülu-3-8B | 0.96 | 0.16 | **+0.80** | 0.98 | 0.46 | **+0.52** | 0.93 | 0.50 | **+0.43** |
| Qwen2.5-7B-Instruct | 0.93 | 0.11 | **+0.82** | 0.92 | 0.12 | **+0.80** | 0.88 | 0.33 | **+0.55** |
| Mistral-7B-Instruct-v0.3 | 0.37 | 0.01 | **+0.36** | 0.60 | 0.12 | **+0.48** | 0.91 | 0.64 | **+0.27** |

Llama's plaintext arm reads +0.83 against §4d's +0.82 and its encoded arm 0.98 /
0.99 against §4d's 0.99 / 0.99 — an independent within-run replication of the
headline in a fourth job.

### The decomposition, and BOTH terms are real

Splitting total discrimination loss into a wrapper term (plain → scaffold) and a
character term (scaffold → encoded). CIs are unpaired Wald at n=100 — the arms
are paired by prompt but `cells.jsonl` persists only the encoded harmful arm, so
the pairing cannot be exploited and these are **conservative**:

| model | total loss | wrapper alone | characters alone |
|---|---|---|---|
| Llama-3.1-8B | +0.84 [+0.76, +0.92] | **+0.67 [+0.56, +0.78]** | +0.17 [+0.09, +0.25] |
| Tülu-3-8B | +0.37 [+0.23, +0.51] | **+0.28 [+0.15, +0.41]** | +0.09 [−0.06, +0.24] *ns* |
| Qwen2.5-7B | +0.27 [+0.13, +0.41] | +0.02 [−0.09, +0.13] *ns* | **+0.25 [+0.11, +0.39]** |
| Mistral-7B | +0.09 [−0.06, +0.24] *ns* | −0.12 [−0.27, +0.03] *ns* | **+0.21 [+0.05, +0.37]** |

**Which term dominates is a property of the model, and it spans the full
range.** On Llama the wrapper alone destroys 80% of the discrimination the
characters were being credited with; on Tülu 76%, and its character term is not
distinguishable from zero. On Qwen the reverse — the wrapper does nothing
measurable and the characters do all the work. On Mistral the two terms have
*opposite signs* and partly cancel: the wrapper trends toward *raising*
discrimination while the characters lower it, which is why its total loss is the
one value in the table indistinguishable from zero while its character term is
significant.

### What this does to leg 1

**"Encoding defeats discrimination" is not supported as stated on 2 of 4
models.** On Llama and Tülu the sentence is mostly false in its subject: what
defeats discrimination is the *request to handle an encoding*, and the encoding
itself is a minority term.

The supported claim names the **encoded-prompt protocol** — wrapper plus
transformed characters, as every benchmark in this literature actually delivers
it. That is not a retreat, for two reasons. It is what the field ships, so a
result about the protocol is a result about the measurements being published.
And it is *stronger* against the reviewer objection §4e already anticipated:
"refusing homoglyph text is correct behaviour" has no purchase once most of the
false-positive mass is caused by a wrapper that contains no obfuscated content
at all.

⚠️ **Do not restate the decomposition as "the wrapper is the real cause."** That
is the same over-attribution one level over, and Qwen refutes it directly.

### Why the title survives, for the third time

**"Refusal without discrimination" names a state, not a mechanism.** §4f
withdrew leg 2's mechanism, §4g reversed the model ordering, and §4h now
reassigns leg 1's subject — and none of the three touched the title, because a
title pinned to a mechanism would have been retired with it. Leave it alone.

### Three caveats a session must not lose

- **Mistral's scaffold arm has an echo rate of 0.38 and it is uninterpretable,
  not a finding.** `scaffold_arm` leaves `ciphertext` untransformed by
  construction, so `echoed_ciphertext` fires on any response that quotes the
  request. The field is named `scaffold_echo_rate_uninterpretable` for that
  reason. **Report it, never subtract it.**
- **No ASR number from these runs is reportable**, and the contract withheld
  `behavior` and `behavior_plain` on all four for the *right* reason — the
  `judge_benign_arm` control failed everywhere (Llama harmful ASR 0.07 against
  benign 0.66), which is §3.5.2's binary-judge finding reproduced a fifth time.
  Every number in this section is a **refusal** rate; see §4i for why the
  contract's verdict does not reach them.
- ⛔ **THIS BULLET WAS WRONG TWICE AND IS CORRECTED (2026-08-10).** It read:
  *"`recognition` and `trajectory` are unlicensed on all four runs. The
  behavioural result stands alone here; nothing internal was measured."*
  **`trajectory` is unlicensed on all four — that part stands.** The other two
  claims do not. **(a)** `recognition` is LICENSED on three of the four (Mistral
  0.716, Qwen 0.854, Tülu 0.711) and withheld for a different and fixable reason,
  `no negative control was run (P2)`; only Llama fails licensing, at 0.663 with a
  length-null margin of 0.009. **(b)** *"nothing internal was measured"* omits
  `deployment`, which is **licensed and screened on all four models at AUROC
  0.938–0.995** and is one of the two reportable readings in every one of these
  run records. The strongest internal reading this repo has sat in the same file
  as the sentence saying there wasn't one. **It is now leg 1's internals leg** —
  `phase1_design.md`, and §4e's open-question block.

### What AS-6 inherits

Directly, and it is not optional. A guard is sent the same wrapper, so
`decoded_not_blocked` carries the identical confound: a guard that flags
anything *asking about* an encoding produces a block rate that is a wrapper
response, not a decode-then-block decision. **The guard-side sweep needs a
scaffold arm from its first run**, on the same footing the benign arm was made
mandatory in TODO 61 — and by the same lesson, not behind a flag.

## 4i. ⚠️ THE INSTRUMENT CONTRACT DOES NOT MODEL THE QUANTITY THE PAPER REPORTS (2026-08-09)

Found by asking whether §4h's refusal rates were reportable given that the
contract withheld `behavior` on all four runs. The answer is yes, and the reason
is worse than a yes.

**The contract has eleven instruments — `ability`, `attribution`, `behavior`,
`causal_license`, `decode_lens`, `deployment`, `entropy_dynamics`,
`recognition`, `reply_inversion`, `sae_reconstruction`, `trajectory` — and not
one of them is refusal.** `behavior`'s `value` is `attack_success_rate`
(`behavior.py:267`), so `reportable`/`withheld` on that reading is a verdict
about **ASR**. The refusal rate rides in `detail` as an unevaluated payload
field.

**Every claim in legs 1 and 2 is a refusal rate.** So the paper's entire
headline is built from a quantity the contract neither licenses nor withholds,
while the contract's one behavioural verdict governs a number the paper has
already committed to never reporting.

This is the repo's recurring failure shape, inverted. The usual form is *a
settled rule that did not reach every caller* (four instances, 2026-08-07 to
-08-09). This is *a governing layer that does not reach the governed
quantity* — and it is harder to see, because the contract is loudly doing its
job on ASR the whole time.

**Why the refusal numbers are nonetheless defensible, stated once so it is not
re-argued:** they carry their controls by construction rather than by contract.
Each rate is reported only as a paired difference against a benign arm measured
in the same run on the same prompts, the plaintext baseline supplies the
denominator (mandatory since §4d), and §3.5.2 established the refusal judge
separately — +0.79 to +0.82 separation with benign false positives at 0.01–0.16,
on plaintext where no attack exists by construction.

⚠️ **That is an argument in prose, which is exactly what the contract exists to
replace.** The fix is a `refusal` reading whose required controls are the benign
arm and the plaintext baseline, so the paper's headline quantity is governed by
the same machinery as everything else and omitting its controls becomes
inexpressible.

**✅ BUILT 2026-08-09, and SCREENED 2026-08-10.** `measurements/refusal.py`
reports the gap and is wired into the entrypoint; building it exposed a worse
defect in the contract's null path (a broken instrument *manufactures* a null,
and the null path was waiving the validity screens that would catch it — fixed
the same day). The required screen then ran, and the outcome revised the screen
itself: the obvious control — does the judge read a bare ciphertext as a
refusal — measured **0.999**, but that is a property of the JUDGE, constant on
every rung, so it could never gate anything. The screen is now the echo
**displacement**, and on `homoglyph` it clears on all four models at 0.001–0.029.
The prose argument above is retained for provenance; the machinery has replaced
it. Detail: `instrument_layer.md` §3.11.


## 4j. ✅ THE STORY RESOLVES — the internals turn two findings and a null into one argument (2026-08-21)

**This section does not replace §4e. It states what §4e's three legs become once
the internals leg (`phase1_design.md`) and the scaffold decomposition (§4h) are
both in the paper, and it is the ordering of record.** §4e's title, thesis
clauses and every number stand. What changes is that leg 2 stops being a bare
null and becomes a *predicted* one.

### The argument in three beats, each measured

**Beat 1 — the reported metric is uninformative, and the informative one
collapses.** Four independently post-trained models refuse encoded harmful
prompts within a spread of 0.08, inside the n=100 noise ceiling, while spanning
0.57 on the same requests in plaintext. The quantity that survives is the harm
gap, and under the encoded-prompt protocol it loses 27–101% of its plaintext
value. (§4d, §4g; unchanged.)

**Beat 2 — ⛔ WITHDRAWN 2026-08-21, see §4k. The probe was reading item identity; with items held out it reads 0.618–0.811, BELOW its own plaintext baseline. Kept below as the refuted text.** ~~The signal is still there.~~ A harm direction **fit on plaintext**
transfers into the encoded condition at AUROC **0.938–0.995 on all four
models**, at a late-mid layer at the end of the instruction in every case
(L21 / L18 / L19 / L27, `instruction_final` throughout). On Llama-3.1-8B the
pair is **0.981 internal against 0.00 behavioural**. Screens: length null
+0.284 to +0.341, echo displacement clears on all four, black-box baseline
+0.326 to +0.383, and since 2026-08-21 a real control floor via witness
(+0.281 to +0.329, `instrument_layer.md` §2.10).

**Beat 3 — so the remedy was aimed at the wrong place.** Tülu 3's complete
SFT → DPO → RLVR moves plaintext harm discrimination +0.55 → +0.80 and leaves
the encoding-induced loss unchanged at 0.34–0.50. Under beat 2 that null stops
being surprising and becomes the prediction: a pipeline that improves the harm
*representation* cannot repair a failure that is not in the representation.

**The one-sentence form:** *encoded prompts do not destroy a model's
representation of harm; they stop refusal from reading it, which is why more
safety post-training does not help.*

### Why this ordering, and why the internals section goes LAST of the three

Beats 1 and 3 are refusal rates on paired arms and need no probe. §4e values
that property explicitly, so they stay contiguous and stay first: **a reviewer
who rejects the probe still has the paper.** The internals section is then the
interpretive layer that explains both, and its removal costs the paper its
explanation rather than its result.

Section order of record: leg 1 → leg 2 → **internals** → instrument.

### ⚠️ What the internals leg may NOT say

- **Not "the model knows and chooses not to act."** The reading is a linear
  probe transfer, i.e. correlational. The causal arm (stage 2, `phase1_design.md`
  §7) has not run, and **its null is pre-declared as weak**: Kwon 2026
  (2607.14147) reports the harm direction is a read-out but not a selective
  write handle. That prior is *concordant* with a read-out failure and must be
  cited as such, never as a result of ours.
- **Not a gap number.** `harmless_rate` is 0.250 by construction, so quote the
  AUROC and never the probe's own gap.
- **Not an unconditional interval.** Every interval is conditional on the
  selected (layer × position) cell (§4.9). Say so.
- **Not four dissociations.** `internal_survives` is true on all four, but the
  dissociation verdict needs behaviour to fail too, and **Mistral's
  destroyed-fraction interval straddles zero** (26.8% [−9.4, +63.0]). Mistral is
  not an internals failure; it is a model whose behaviour does not
  demonstrably collapse, because its plaintext gap is only +0.36. **Three of
  four, named individually — never the four-model table as a point estimate.**

### ⚠️ And the paper's own Method paragraph is now a defect

`paper/as-5/*/paper.tex` explains that the plaintext arm carries no template
"since running plaintext through an identity encoder would still wrap it in the
encoded condition's instruction scaffold, and that scaffold is part of what the
encoded condition is being blamed for." That paragraph **names the confound and
keeps it**, while the cell resolving it ran on 2026-08-09 with CIs (§4h) and was
never written in. *A declared gap is not a mitigation*, the same lesson as I7
and the kit-parity incident, now inside the paper itself.

So reporting the three-arm decomposition is a **missing measured result**, not a
framing choice, and it lands the subject correction of TODO 66 with it. The
combined effect is not a retreat: the wrapper arm carries **plaintext content**
in an encoding-announcing template, and Llama refuses **0.83 of it on the benign
side**. With beat 2 beside it, the reviewer objection §4e anticipates ("refusing
homoglyph text is correct behaviour") loses its purchase entirely — nothing is
obfuscated in that arm, and the harm signal is present at 0.98 and unread.

⚠️ **Do not restate this as "the wrapper is the real cause"** (§4h). Qwen refutes
it: its wrapper term is not distinguishable from zero and its character term does
all the work. Which term dominates is a model property spanning the full range.

### ✅ THE SPLIT QUESTION IS SETTLED — ONE PAPER (owner go, 2026-08-21)

Asked directly and ruled: AS-5 stays **one paper**. Three splits were weighed and
all three rejected. **Do not re-open this without new evidence of the kind named
under each.**

- **Split the INTERNALS out.** Rejected, and it is the one that looks most
  tempting because internals are what this repo is for. *A dissociation is a
  two-term quantity.* Leg 1 without the internals is a metrology complaint; the
  internals without leg 1 is a probe reading whose obvious answer is *linear
  probes find harm everywhere* — and `zhao2025llmsencode` (NeurIPS 2025) already
  established harmfulness is linearly encoded at the instruction-final token, so
  our delta exists **only** because the behavioural collapse sits beside it.
  Second reason, from the data rather than from taste: the internals evidence is
  **right-sized for a section and under-powered as a paper** — one instrument,
  correlational, intervals conditional on a selected cell, three of four models.
- **Split the INSTRUMENT out as a methods paper.** Rejected, though parts
  genuinely travel (the binary judge firing on 0.61–0.70 of *plaintext benign*
  prompts is a finding about a widely used judge, not about encoding). Alone it
  needs the same experiments as evidence, so the overlap is near total, and the
  claim binding the eight — every behaviour-axis defect inflates apparent safety
  — is much weaker without a corrected measurement producing a positive result.
- **Split the LADDER COLLAPSE out** (10 cipher rungs inert, `tag_block`
  invertible-but-unreadable, the ability threshold where every rung with a harm
  gap has ability 1.00 while `math_monospace` decodes 69% and still blanket-
  refuses). The only genuinely separable one: it is a claim about *other
  people's benchmarks*, independent of the harm-gap story. Rejected **for now**
  because it is thin alone, the fertility ordering variable was refuted so there
  is no mechanism to offer, and it is load-bearing inside AS-5's Scope where it
  justifies why one encoding carries the cross-model result. ⚠️ **Named trigger
  to revisit: a mechanism for what makes an encoding readable.** With one, it is
  a paper.

**Where the second paper actually is: the CAUSAL / INTERVENTION work**, which
does not exist yet — ablate the direction in the encoded condition where
behaviour already ignores it (`phase1_design.md` §7), capture the scaffold arm
internally, attempt a read-out repair. It would **cite** AS-5, not carve it.
⚠️ That is a **third**-paper decision: the owner settled this line at two on
2026-08-05 and AS-6 holds the second slot.

### What the ruling changed in the paper (2026-08-21)

The risk a split would have addressed is real but is a WRITING problem: the
introduction listed four peer contributions, which reads as a paper trying to say
four things. Replaced with **one thesis and five steps** — *encoded-prompt
evaluation reports a number that cannot see what it is measuring, because the
failure is not where the metric looks* — the steps being: the metric is blind ·
blind to *what* · stays blind through a full remedy and reports its progress
backwards · because the failure is not in the representation · and why it was not
caught. **Subordination, not division.**

⚠️ Two precision rules that came out of writing it, both binding on any future
draft. Step 4's heading is the **negative** claim (*not in the representation*),
never "the failure is at the read-out" — a dissociation licenses placing the
failure downstream of the representation, and nothing stronger, until the causal
arm runs. And the intro now carries a **note on the title**, because *refusal
without discrimination* can be misread as *the model cannot discriminate*, which
our own internals result refutes: the title names the behavioural state, and the
last measurement shows the discrimination is still there, unused.

## 4k. ⛔ THE INTERNALS LEG IS REFUTED — the probe was reading item identity, not harm (2026-08-21)

**§4j's beat 2 is WITHDRAWN. Do not quote AUROC 0.938–0.995 anywhere, ever
again.** Found by a hostile methods review the same day §4j was written, and
settled by measurement: job `9407323`, CPU, `short` partition, $0, no GPU.
Script `scripts/split_half_transfer.py`, artifact
`outputs/analysis/split_half_transfer_20260821.json`.

### The defect

`probe_transfer` fits on **all 100 plaintext harmful + all 100 plaintext
benign** and evaluates on the homoglyph versions of **those same 200 items**.
The CONDITION is held out; the ITEMS are not. `crossval_scores`' docstring
states the opposite in writing — *"unlike the transfer probe there is no free
held-out set"* — so the code believed the transfer probe had one. It did not.
The encoded activation of item *i* sits near the plaintext activation of item
*i*, because the model decodes the transform at ability 0.86–0.98, so a probe
told "plaintext item *i* is harmful" scores encoded item *i* high by memory.

### The numbers

| model | paper (no split) | **item-split** | 95% band | plaintext CV |
|---|---|---|---|---|
| Llama-3.1-8B | 0.981 | **0.618** | [0.524, 0.714] | 0.707 |
| Qwen2.5-7B | 0.995 | **0.811** | [0.742, 0.870] | 0.898 |
| Tülu-3-8B | 0.971 | **0.698** | [0.618, 0.775] | 0.806 |
| Mistral-7B | 0.938 | **0.727** | [0.631, 0.796] | 0.870 |

**The sample-size confound is CLOSED, which is what makes this a verdict rather
than a worry.** The split halves the training set as well as holding out items,
so part of the drop could have been fewer samples. Scoring the SAME probe at the
SAME training size on the items it was trained on gives **0.971–0.998**, against
0.618–0.811 on items it was not: leakage at fixed *n* is **+0.19 to +0.38**.
Difference-in-means shows it too (+0.15 to +0.32), so it is item memory, not
logistic capacity — a rank-one mean difference carries item identity across this
transform just fine.

### ⚠️ The honest reading runs OPPOSITE to what §4j claimed

The paper said the representation is intact and only the read-out fails (H-B).
With items held out, **the encoded reading is BELOW the within-plaintext
cross-validated reading on all four models, by 0.087 to 0.143.** The harm
representation is **degraded** under the protocol, not intact. What survives is
a dissociation of DEGREE — representation down ~0.1 AUROC while behavioural
discrimination falls 27–101% — which is a far weaker claim than the one written,
and it leans toward H-A rather than H-B.

⚠️ **Llama, the model that carried the story, is the worst case**: 0.618 with a
band whose floor is 0.524, and its own plaintext CV is only 0.707. *"The harm is
linearly present at near-ceiling"* was never true on Llama; it had no baseline,
which is exactly what the review said — **the paper never reported a
within-plaintext AUROC, so "the signal survives" had no denominator.**

⛔ **Do NOT read the floor column as a verdict.** The control floors (Llama
0.6765, Qwen 0.6708, Tülu 0.6417, Mistral 0.6569) were derived from control
rungs measured under the SAME leaky procedure, so they are inflated too.
Comparing a split-half reading against a no-split floor is apples to oranges;
re-deriving the floors under the split is another cheap CPU job and is owed
before any clears/fails claim.

### The lesson, and it is not the one about probes

§4j was written, committed and pushed on the strength of a measurement nobody
had audited, one day before a review found the audit. The repo's own record
already warned in the same file that the run record said *"nothing internal was
measured"* — the correction to that sentence was right about `deployment`
existing and wrong about what it measured. **Finding a number in a run record is
not the same as knowing what it is a number for.** Every previous instance of
this class was caught by a run dying; this one had to be caught by a reviewer,
because a leaky probe produces a beautiful result and no error.

## 4l. ✅ THE SURVEY RAN AND IT RESHAPES THE PREMISE — the founding paper HAD the control and the field dropped it (2026-08-21)

Full record and every quote: **`text_docs/as5/arm_survey.md`**. Instruments
`scripts/survey_frame.py` + `scripts/survey_adjudicate.py`, both keyless, no GPU,
no model, seconds; 15 papers adjudicated against their own full text.

**Why it was run.** After §4k removed the internals leg, the paper's remaining
claim is that the effect lives in a cell the field does not evaluate: benign
content in attack form. That is a claim of ABSENCE, and this repo has already
made the absence mistake twice — the 2026-08-06 coverage sweep measured absence
against a narrow index, and the model-slate sweep read an aggregator and called a
published paper unpublished. Asserting it a third time without a survey was not
available.

**⚠️ The clean version of the claim is FALSE, and the counterexample is the
founding paper of the attack class.** Wei et al., *Jailbroken* (NeurIPS 2023,
arXiv 2307.02483) ran a harmless control through every attack — *"we test each
jailbreak for each model against the curated dataset and an additional harmless
control prompt"* — and **observed this paper's effect**: *"(Claude even refuses a
harmless control prompt under these roleplay attacks; see Appendix D.)"* Any
framing that says nobody has done this will be correctly rejected, and it would
have been rejected on the single most famous paper in the area.

**What is true is narrower, and it is a better motivation than an empty cell.**
Three parts, all verified: the control is **n = 1** (singular, so it yields an
anecdote and not a rate); it is scored as **capability, not refusal** (Appendix D:
marked successful "if the model correctly answered the question (with answer
'unicorn')"); and it **did not propagate** — not one successor paper carries it.
So AS-5 is not claiming virgin territory. It is **restoring a control the founding
paper had, and turning it into a rate.** That reframing survives the reviewer
objection that would otherwise kill the paper.

**The counts, over the adjudicated frame and never over "the literature":**

- **11 input-transformation attack papers, 0 reporting a refusal rate on benign
  content through their own transformation.** Two at B2 (benign transformed but
  scored for capability), nine with no benign-transformed arm at all.
- **The sharpest exhibit is Handa et al. (Novel Ciphers, 2025).** They build
  **CipherBench** — *"a benchmark designed to evaluate LLMs' accuracy in decoding
  encrypted benign text"* — and ACE/LACE, which sends harmful queries through the
  same ten ciphers. Both corpora exist, both are transformed, the benign one is
  scored for decoding accuracy and the harmful one for ASR. The cell was one
  column away.
- **The over-refusal side never uses attack form at all.** XSTest (NAACL 2024) and
  OR-Bench (ICML 2025) contain **zero** occurrences of `encod`/`cipher`/`base64`/
  `obfusc`/`ascii art` anywhere in their full text — 0 in 96,637 and 0 in 108,688
  characters, against 72 and 102 benign-mentioning sentences respectively.
- **The defense side calibrates on the wrong distribution, and this row is
  AS-6's.** DecipherGuard sets its threshold so that FPR *"remains within an
  acceptable limit, such as 1%"* — computed *"from a set of safe prompts"*, plain
  ones. A defense against encoded attacks whose false-positive rate is calibrated
  on untransformed benign content has not measured its FPR under the condition it
  is deployed against.

**⚠️ Two instrument facts not to re-derive.** The keyword screen **missed
`wei2023jailbroken` on its first run**, because that paper's title and abstract
carry no transformation vocabulary — so the frame is *screen UNION a named
canonical seed set*, and a screen returning nothing is a failed lookup. And the
adjudicator carries a **positive control** by construction: XSTest must return a
large benign-sentence count, because an instrument that reports absence has to be
shown capable of reporting presence. Both properties are the repo's own standing
rules, applied one estate over.

**The lesson, and it is the third time in two days.** §4j fell because a number
was trusted without an audit. This survey was the same shape of risk pointed at
the literature instead of at our own data, and running it changed the paper's
premise rather than confirming it. **The survey was worth more as a check than it
would have been as a citation.**

## 4b. ⚠️ THE FALSIFICATION TEST RAN AND §4a's AXIS IS REFUTED (2026-08-08) *(verdict stands; its REASON is withdrawn — see §4c)*

Jobs `9010897` (Mistral-7B-Instruct-v0.3, 1:15:12) and `9011034` (Tulu-3-8B,
1:23:29), 8 rungs × 100 prompts each: the three substrate rungs plus five
can't-decode candidates for the floor. Both presets stated their gate in
committed text before launch; this is the reading against it.

### The axis, and what it predicted

§4a's claim is that **blanket surface-level refusal** (a model that refuses
encoded *benign* content) comes with **weaker binding** of refusal to
comprehension. It was a line through two points — Llama high/high, Qwen
low/low. `dissociation_mistral.yaml` named the refutation condition in advance.

| model | benign refusal, per substrate rung | binding failure B/(B+S) |
|---|---|---|
| Llama-3.1-8B-Instruct | 0.90 / 0.99 / 0.59 | 0.27 – 0.43 |
| Qwen2.5-7B-Instruct | 0.21 – 0.30 | 0.05 – 0.11 |
| Mistral-7B-Instruct-v0.3 | 0.91 / 0.67 / 0.52 | **0.200 / 0.098 / 0.686** |
| Tulu-3-8B ⚠️ | 0.40 / 0.41 / 0.52 | **0.562 / 0.156 / 0.286** |

*(rung order: `fullwidth` / `homoglyph` / `zero_width`.)*

### The refutation does not need the cross-model comparison at all

It is **internal to Mistral**, which is what makes it hard to explain away:

| Mistral rung | benign refusal | binding failure | hard incoherence | deployment AUROC |
|---|---|---|---|---|
| `homoglyph` | 0.67 | **0.098** | 0.05 | 0.938 (floor +0.281) |
| `zero_width` | 0.52 | **0.686** | 0.02 | 0.951 (floor +0.294) |

**Lower blanket refusal, seven times the binding failure** — the opposite of the
axis, in one model, on two rungs that are both instrument-sound by every screen
this repo has: hard incoherence at 2–5%, deployment licensed, both clearing the
run's own control floor by ~0.29. Tulu-3 shows the same shape (`fullwidth` 0.40
→ 0.562 against `homoglyph` 0.41 → 0.156: near-identical blanket refusal, 3.6×
the binding failure).

**So binding failure varies more BETWEEN RUNGS INSIDE a model than it does
between models.** §4a explained a two-model difference with a model-level
property; a model-level property cannot produce a 0.098-to-0.686 spread within
one model. The two-model pattern was real and is still on disk — it was simply
not evidence for the mechanism attached to it, because n=2 with one rung's worth
of resolution cannot distinguish a model effect from a rung effect.

**Note which prediction failed.** The preset expected refutation to arrive as
*low benign refusal + high binding failure* in a permissive third model. Mistral
is not permissive — its benign refusal (mean 0.70) is the second highest of the
four, above Llama's. The axis failed on a variable the preset was not watching.

### What survives from §4a

1. **The aggregate/internals inversion**, which was always the contribution and
   does not depend on the axis: the reportable refusal rate ranks these models
   differently from the comprehension-conditioned rate. Mistral makes this
   *stronger* — its harmful refusal on `zero_width` is 0.65 while 35 of its 51
   measured comprehension cells are (B).
2. **(B) is populated on every model tested**, now four families and three
   independent post-training pipelines.

### What does not survive

Any claim that blanket refusal *explains* weak binding, and any framing of the
paper around a property of safety training. The unit that carries the variance
is the **(model, rung) cell**, not the model.

### The behaviour axis is withheld again — third consecutive run

The mandatory benign judge arm fails on every substrate rung, against the
rule-of-three bar of 3/n = 0.03:

| | `fullwidth` | `homoglyph` | `zero_width` |
|---|---|---|---|
| Mistral benign-arm ASR | 0.08 | 0.25 | 0.30 |
| Tulu-3 benign-arm ASR | **0.44** | 0.38 | 0.29 |

Tulu's 0.44 is the worst measured anywhere in this repo: the judge calls 44 of
100 *benign* fullwidth responses jailbreaks. **No ASR from either run is
reportable.** The refusal rates and (B)/(S) counts above are unaffected —
`regimes.py:249` splits on `refused`, not on the ASR judge — which is the same
separation §3.6 established.

### ⚠️ Tulu-3's arm is off-distribution and must be re-run

`9011034` ran BOS-less. The checkpoint's `tokenizer.json` post-processor is
byte-identical to Llama-3.1-Instruct's, so its config moved to
`prepend_bos_to_chat_template: true` the same day and this run is superseded
(`conf/models/tulu3_8b.yaml`). It is reported here because it *agrees* with
Mistral and the refutation does not rest on it. It may not be differenced
against the SFT/DPO ladder rungs, which run with BOS: a ladder whose rungs
disagree about BOS measures BOS.

### Secondary gate: FAILED, on the branch the preset named

The five can't-decode candidates were meant to yield the repo's first derived
`mean + 2·SD` floor. **Both models produced only four admissible controls**, so
both fell back to `max` and are labelled `bound`:

| model | floor | kind | admitted | excluded, and why |
|---|---|---|---|---|
| Mistral | 0.6569 | `bound` | 4 | `math_sans`, ability 0.01 |
| Tulu-3 | 0.6417 | `bound` | 4 | `circled`, ability 0.03 |

`control_ability_max: 0.0` excludes any rung the model decoded even once in 100,
and on each model exactly one candidate crossed it — independently. The preset
predicted this branch: *"the 5-control minimum is not reachable at this ladder
size and the floor rule needs a cheaper control source than a full rung."*

**Do not fix this by loosening `control_ability_max`.** The knob is right: a
control rung exists to establish what the probe reads when nothing was decoded,
so admitting a rung with 3 genuinely decoded cells puts decoded content into the
noise floor and biases it upward — which is the direction that makes real rungs
*fail*. The fix is more control rungs per run, and the cipher band is free for
this: ten of its rungs are established-inert on two families already.

Consequence meanwhile: both floors are `bound`, hence n-dependent per §2.4, and
**not comparable across runs or models**. Mistral's `fullwidth` clears its own
floor by only +0.022 and should be treated as marginal.

## 5. What it DOES support, and it is a stronger paper *(SUPERSEDED — see §4a)*

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
