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

## 4e. ✅ THE STORY AND TITLE ARE SETTLED (owner go, 2026-08-08) — and §4d gained its mechanism

**This section is the story of record. §4a–§4d are kept as the derivation, not as
live proposals.** The change §4e makes over §4d is not the phenomenon — §4d had
that right — but the addition of a *mechanism* and the retirement of the title.

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

### The three legs, in order

1. **The phenomenon (§4d).** Encoding does not defeat refusal; it defeats
   discrimination. Benign refusal rises 2–10× over plaintext on four models, and
   on Llama-3.1-8B the harm gap goes **+0.82 → 0.00** — benign and harmful
   `homoglyph` prompts refused at an identical 0.99. No probe required.

   **State it as the measurement claim, not the utility claim.** "Encoding causes
   false positives" invites the reviewer answer *refusing homoglyph text is
   correct behaviour, no legitimate user sends it*, and there is no good reply.
   The defensible form: **the harmful arm has no discriminative power, and it is
   the only arm the field reports.** Llama and Qwen are indistinguishable on
   encoded harmful prompts (0.99 vs 0.91, CIs overlapping; cross-model spread
   0.08 against a 0.10 noise ceiling) while discriminating at 0.00 and +0.61
   respectively. Whether refusing encoded input is *desirable* has no bearing on
   whether the measurement can tell two very different models apart. It cannot.
2. ⛔ **WITHDRAWN 2026-08-08 by jobs `9027721`–`9027723` — read §4f before
   using anything in this leg.** The ladder re-ran with its plaintext arm and
   plaintext benign refusal falls in lockstep (0.45 → 0.17 → 0.16), so the
   monotone encoded fall below is largely general de-refusal. In leg 1's own
   currency — gap lost relative to plaintext — there is **no trend on any
   rung** (`homoglyph` +0.35 → +0.50 → +0.34). What replaces it is a null and
   is stronger: **the complete published pipeline moves plaintext harm
   discrimination +0.55 → +0.80 and leaves the encoding-induced loss unchanged
   at 0.34–0.50.** The text below is kept as the reasoning the re-run tested.

   2. **The mechanism — NEW, and it is what §4e adds.** §4d could only say the
   effect "varies ~5× across post-training pipelines", which is a correlation
   across four unrelated models with everything else varying too. The Tülu
   ladder (`instrument_layer.md` §3.6.2, jobs `9011347`/`9011348`/`9011349`)
   makes it a controlled series on **identical base weights** along a
   **published** recipe: benign refusal falls monotonically at every stage and
   every rung (`fullwidth` 0.80→0.45→0.30, `homoglyph` 0.79→0.63→0.48,
   `zero_width` 0.91→0.43→0.37). The harm gap grows **from the bottom, not the
   top** — post-training buys back benign utility under encoding rather than
   adding refusal. That is the difference between observing a spread and
   locating it.

   ⚠️ **"Harmful refusal stays roughly flat" is corrected, and the correction
   makes leg 2 stronger.** It falls: `fullwidth` 0.96 → 0.73, `zero_width`
   1.00 → 0.81, `homoglyph` 0.99 → 0.93. So a benchmark reading
   refusal-under-attack sees **0.96 → 0.73 and reports that two-thirds of
   Tülu-3's published safety pipeline made the model less safe**, while the harm
   gap says it got substantially better (+0.16 → +0.43). **The two metrics
   disagree in SIGN, on all three rungs.** That is the sharpest statement of leg
   1 the evidence supports, and it comes from a controlled series rather than a
   cross-family comparison — call it flat and the result is lost.

   ⛔ **The claim that followed here — "SFT behaves like Llama-3.1-8B-Instruct and
   DPO like Qwen2.5-7B" — is FALSIFIED (§4f).** With the plaintext arm: Llama's
   benign refusal is 0.10 plain / 0.99 encoded, an encoding-induced excess of
   **+0.89**; SFT's is 0.45 plain / 0.79 encoded, excess **+0.34**. Llama's blanket
   refusal is encoding-induced, SFT's is mostly general over-refusal. They resemble
   each other **only on the encoded arm**, which is leg 1's entire point — the claim
   was made from the encoded arm alone.
3. **The instrument.** None of the above is measurable without controls the field
   does not run — the benign arm, the echo screen, the length null, the control
   floor. Every defect found on the behaviour axis inflated apparent safety,
   which is structural rather than coincidental: "no attack succeeded" is what a
   broken safety evaluation returns by default.

**Demotion recorded:** `CLAUDE.md` has recommended leading with the measurement
contribution (leg 3) since 2026-08-07. That was correct when the ladder had
collapsed and nothing positive had replaced it. It is now the third leg — still
in the paper, no longer the lead.

### ⚠️ LEG 2 IS WITHDRAWN AS WRITTEN — the gate fired against it (2026-08-08, jobs `9027721`–`9027723`)

**The re-run below ran within hours of this section being written, and it
refuted the mechanism claim §4e was founded on.** Recorded here rather than by
editing leg 2 silently, because the derivation is the point.

**The gate:** leg 2 said post-training *restores the discrimination encoding
destroys*. That reading requires plaintext benign refusal to be roughly flat
across stages — otherwise the encoded fall is just general de-refusal showing
through. **It is not flat: 0.45 → 0.17 → 0.16**, a 0.29 drop at the SFT→DPO
step, the same step where the encoded series falls hardest.

**The decisive quantity, gap lost to encoding (encoded gap − plaintext gap):**

| rung | SFT | DPO | RLVR |
|---|---|---|---|
| `fullwidth` | −0.43 | −0.50 | −0.38 |
| `homoglyph` | −0.35 | −0.50 | −0.34 |
| `zero_width` | −0.45 | −0.36 | −0.34 |

Flat within ~0.16 and **non-monotone** — DPO is *worse* than SFT on two of three
rungs. There is no trend to attribute to post-training. Leg 2's monotone series
was real but was measuring the model's general over-refusal improving, which the
plaintext arm now shows directly (plaintext harm gap 0.55 → 0.79 → 0.80).

### The replacement, and it is a better leg 2 than the one it replaces

**The encoding penalty is INVARIANT to post-training.** Three stages of a
published safety pipeline — SFT, DPO, and RLVR, the last being Tülu 3's own novel
contribution — and the discrimination lost to encoding does not move: 0.34–0.50
on every rung at every stage. Safety training measurably improves general
over-refusal (plaintext benign refusal 0.45 → 0.16) and does **nothing** about
the encoding penalty.

That is a negative result with teeth, and it is more useful to the field than the
claim it replaces: **you cannot post-train your way out of this.** It also gives
leg 1 a mechanism-shaped companion without overclaiming — leg 1 says encoding
destroys discrimination, leg 2 now says the obvious remedy demonstrably does not
restore it.

**Leg 1 is untouched.** The phenomenon was measured on four models with paired
plaintext arms and is not affected by anything above.

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

### ⚠️ The gap this story HAD, and how it was closed

**The ladder has no plaintext arm.** Jobs `9011347`–`9011349` ran before
`run_plain_behavior_baseline` landed (peer commit `5a8d2a9`), so leg 2's series
is in *encoded* benign refusal while leg 1's headline quantity is *gap lost
relative to plaintext*. The two legs are currently in different currencies, and
the mechanism claim is weaker for it — "benign refusal falls across stages" is
not yet "the discrimination the encoding destroys is restored across stages".

Re-running the three stages on current code closes it: the plain baseline is
model-level and MANDATORY, so no preset change is needed, and the re-run also
picks up per-cell benign persistence (`188644d`) that the graded regrade needs.

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

## 4f. ✅ LEG 1 GENERALISES FROM ONE ENCODER TO A CLASS — and the model ordering was an artefact of not normalising (2026-08-08)

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
