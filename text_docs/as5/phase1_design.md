# AS-5 phase 1 — where the internals enter

*Founded 2026-08-10. Settles the open question `evidence_and_story.md` §4e names
and deliberately leaves open ("⚠️ The open question this story does not answer:
where the internals enter"). That section becomes a pointer to this file; this
file is the design of record for AS-5's internals leg.*

*Namespace: AS-5-specific, because the object of study is the target model.
Anything that turns out to be a property of the MEASUREMENT belongs in
`shared/instrument_layer.md` and is cited from here, never copied.*

---

## 0. The headline, and it cost nothing

**The internals leg is already measured on all four models, it clears three
controls, and the run record that carries it says in the same breath that
"nothing internal was measured."**

On the four scaffold-control runs of 2026-08-09 (`9033595`–`9033598`,
`homoglyph`, n=100 per arm), a harm direction **fit on plaintext** separates
**encoded harmful from encoded benign** activations at transfer AUROC
**0.938–0.995** — while the behavioural harm gap in the same condition spans
**−0.01 to +0.55**.

On Llama-3.1-8B the two numbers are **0.981 and −0.01**. The harm is linearly
present in the residual stream at near-ceiling, and the refusal decision reads
none of it.

That is leg 1 with a location. "Refusal without discrimination" stops being a
behavioural state and becomes a *dissociation between two measurable quantities
in the same forward pass*.

---

## 1. The question, and the two hypotheses it separates

Leg 1 establishes **that** discrimination is destroyed. It says nothing about
**where** in the forward pass it is lost. Two stories fit the behaviour equally
well and they have opposite implications:

| | claim | what would fix it |
|---|---|---|
| **H-A — representation failure** | the protocol prevents the harm from being represented at all | teach the model to decode; better parsing |
| **H-B — decoupling** | the harm *is* represented and refusal stops reading it | nothing about decoding helps; the failure is downstream |

These are not a philosophical distinction. They are the difference between a
defence that preprocesses inputs and a defence that repairs the link between
representation and behaviour — and the field currently assumes H-A without
testing it, because ASR cannot see the difference.

---

## 2. The evidence already on disk

Computed from `outputs/runs/phase0/*/scaffold-control-*/results.json`, not
transcribed. `deployment` is measurement #2: fit a content probe on the plain
condition, transfer **without refitting** to the encoded condition
(`measurements/deployment.py`).

| model | layer | AUROC | length-null margin | black-box margin | XSTest lexical | behavioural gap (§4h) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 21 | **0.981** | +0.326 | +0.369 | 0.978 vs floor 0.619 | **−0.01** |
| Qwen2.5-7B-Instruct | 19 | **0.995** | +0.341 | +0.383 | 0.982 vs floor 0.619 | +0.55 |
| Tülu-3-8B | 27 | **0.971** | +0.317 | +0.359 | 0.967 vs floor 0.619 | +0.43 |
| Mistral-7B-Instruct-v0.3 | 18 | **0.938** | +0.284 | +0.326 | 0.988 vs floor 0.619 | +0.27 |

All four read at `instruction_final`, the position Zhao et al. (NeurIPS 2025,
2507.11878 — tier (A)) established clusters by intrinsic **harmfulness** rather
than by behaviour. Our layers (18–27 of 32) sit in the late-mid band §3.1 already
identified as *content recovered by computation* rather than lexical presence.

**The shape of the result is a dissociation, and both halves matter.** Internal
separation is nearly model-independent (spread **0.057** across four very
different post-training pipelines); the behavioural use of it spans **0.56**.
That is the same structure leg 1 reports one level up — encoded harmful refusal
is model-independent while discrimination is not — and finding it again in the
activations is convergent rather than circular, because the two measurements
share no estimator, no corpus arm and no judge.

**`homoglyph` neutralises the length confound by construction:**
`mean_ciphertext_chars` = `mean_plaintext_chars` = 86.0. The rung leg 1 is built
on is the one rung where the repo's most persistent confound cannot operate.

---

## 3. The screens it passes, and the one it does not

Passing, on all four models:

1. **Length null** (`measurements/length_null.py`) — margin 0.284–0.341 against
   a 0.05 bar. The confound that has cost this repo three claims reads 0.654;
   the probe reads 0.938–0.995.
2. **Black-box surface baseline** (P4) — a surface classifier gets 0.612 on the
   same encoded inputs. Margin 0.326–0.383.
3. **XSTest lexical decorrelation** — 0.967–0.988 against floor 0.619 + 0.10.
   This is the control that defeats *a probe reading harm-adjacent vocabulary
   rather than harm*, and it is the one most likely to have killed the claim.

⚠️ **Not passing, and it must be stated whenever the number is quoted: the
control floor is UNUSABLE on these runs.** `control_floor.usable = false`,
`n_controls = 0` — each run carried a single family, so there were no
can't-decode rungs to build a floor from, and licensing is therefore
**permutation-only**. That is precisely §2.5's open defect ("significance is not
sufficiency"), live in the run that carries this result.

✅ **CLOSED 2026-08-21, offline and with no job — see §5.2.** The limit is true
of those RUNS and does not follow for the MEASUREMENT: the identical reading
sits in runs that carried control rungs, and it clears every one of their
floors. Llama's witness is a `distribution` floor at n=6. The paragraph below
still binds and is what made the closure legitimate rather than a shortcut.

**Do not paper over it by importing a floor from another run** — §2.4 settled
that the floor statistic is n-dependent and cross-run floors are not comparable.
The honest position: the three screens above are *better targeted* than the floor
(they name the specific confounds), but they are not the floor, and closing it is
a stage-1 line item, not a footnote.

---

## 4. What this does NOT support

- **It is not a claim about decoding.** `deployment` asks whether plaintext
  semantic content is linearly readable, not whether the model *decoded*
  anything. §4.1's Patchscopes/lens instrument is what answers the second, and
  §6.3 already says a linear direction cannot distinguish a harm feature from a
  surface-anomaly feature.
- **`gap` is not a measured benign rate.** `harmless_rate` is exactly **0.250**
  on all four models because the per-cell read sits at the 75th percentile by
  construction — the same artefact §2.1 caught at the median. **Quote the AUROC,
  which is operating-point free; never quote the 0.68–0.75 gap as if the benign
  rate had been measured.**
- **One instrument, not two.** `deployment.py`'s own docstring: *"Until the
  patching arm exists, a positive deployment reading is corroborated by one
  instrument, not two — say so when reporting it."* I6 now exists and has not
  been run against this. Stage 2.
- **Correlational only.** §6.3.1 is binding: a permutation test answers *is this
  separation real*, never *is this separation the thing*. Arditi's causal
  selection screens confounds a permutation test structurally cannot see.
- **Tülu's hard incoherence is 0.130**, over the 0.10 bar. Flag it; do not drop
  the model for it and do not quote its cell counts without the flag.
- **`recognition` is withheld on all four**, and for three of them the sole
  reason is `no negative control was run (P2)` — a fixable gap, not a null. On
  Llama it additionally fails licensing at 0.663 with a length-null margin of
  0.009, i.e. it reads the confound and nothing else. **An unlicensed recognition
  probe is `None`, never "harm not represented"** — the tri-state rule, and the
  reason this leg is built on `deployment` rather than on `recognition`.

---

## 5. Stage 0 — offline, no job, $0

Everything here runs against records already local. Nothing may be quoted until
it does.

1. **Reproduce the table in §2 from the records** with a committed script rather
   than the ad-hoc read that produced it, so the numbers have provenance. Home:
   `scripts/` beside `echo_displacement.py`.
2. **Apply the echo displacement screen** (§3.11.1). `homoglyph` cleared on all
   four models at 0.001–0.029, so this is expected to pass — run it anyway,
   because §3.11's rule is that a statistic computed over gaps must be computed
   on echo-clean cells, and the dissociation is a statistic over gaps.
3. **Pair the internal and behavioural quantities per model with CIs.** The
   behavioural arms are unpaired Wald at n=100 (§4h); the AUROC needs its own
   interval. A dissociation claim needs both, or it is two point estimates in a
   sentence.
4. **State the fraction, not the difference** — §4g's rule applies here too. The
   quantity is *how much of the plaintext harm gap survives into behaviour, given
   that the representation survives*.

**Gate:** if the dissociation does not survive the echo screen and the CIs, stop
— there is no internals leg and AS-5 ships behavioural, which §4e already says is
a publishable paper.

### ✅ 5.1 STAGE 0 RAN 2026-08-12 — the gate PASSES, on THREE of four models

**Offline, no job, $0.** `scripts/internals_dissociation.py` over
`measurements/dissociation.py`; artifact
`outputs/analysis/internals_dissociation_20260812.json`. Reproduce with
`uv run python scripts/internals_dissociation.py`. Encoded rates below are
**echo-screened**; the plaintext arm deliberately is not (§5's asymmetry).

| model | internal AUROC [95%] | plain gap | encoded gap | destroyed | verdict |
|---|---|---|---|---|---|
| Llama-3.1-8B | **0.981** [0.961, 1.000] | +0.83 | **−0.01** | **101.1%** [96.2, 105.9] | ✅ |
| Qwen2.5-7B | **0.995** [0.986, 1.000] | +0.82 | +0.57 | 30.5% [15.0, 46.0] | ✅ |
| Tülu-3-8B | **0.971** [0.947, 0.995] | +0.80 | +0.42 | 47.8% [32.6, 62.9] | ✅ |
| Mistral-7B-v0.3 | 0.938 [0.903, 0.973] | +0.36 | +0.26 | 26.8% [**−9.4**, 63.0] | ⛔ |

**The echo screen passes on all four** — displacement 0.001–0.020 against bars
0.034–0.112. §3.11's rule is satisfied and `homoglyph`'s cleanliness (TODO 67)
reaches the internals leg intact.

⛔ **Mistral fails, and the failure is on the BEHAVIOURAL half — a denominator
story, not an internals one.** Its representation is fine (0.938, lower bound
0.903, clear of the floor). What fails is that its plaintext discrimination is
small enough (+0.36 against the others' +0.80 to +0.83) that the encoded gap is
not distinguishable from it: the destroyed-fraction interval straddles zero.
**There is not enough discrimination there to destroy.** That is the same fact
§4d already recorded one level up — Mistral is the *least discriminating* model,
not the most robust one — arriving now in the internals leg, so it is
convergent rather than a new problem.

⚠️ **The point estimate alone would have passed it.** 26.8% reads as a real
loss; the interval says it is not resolvable at n=100. §5 item 3 required CIs
for exactly this, and this is its first bite. **Never quote the four-model
version of this table.**

**What it does to stage 1.** §8's "passes → build stage 1" branch is taken. The
scope narrows to the three models where plaintext discrimination is large enough
to lose, which does not touch §6(a)'s contrast — Llama and Qwen are both in.

⚠️ **Of the two limits recorded here, the FIRST is now closed (§5.2, 2026-08-21)
and the second is not.** As written that day: the control floor is UNUSABLE on
all four runs (permutation-only licensing, §2.5) — closed offline, every model
screened against a witness run's own floor; and the AUROC intervals are conditional on the
selected (layer × position) cell, so they are narrower than the truth by an
amount stage 0 cannot estimate (`instrument_layer.md` §4.9). Neither is closed
by this run and neither may be dropped from a quotation of it.

---

### ✅ 5.2 STAGE 1(b) IS CLOSED — offline, no job, $0 (2026-08-21)

**It was never a GPU job.** §5.1, TODO 71 and the board all carried *the control
floor is UNUSABLE on all four runs, permutation-only licensing, stage 1(b)'s
job*. True of the runs; not true of the measurement. `deployment` is
deterministic given (model, family, corpus, cached activations, probe config),
and the identical reading — verified on twelve fields, not assumed — sits in
runs that carried can't-decode rungs:

| model | witness run | floor | grade | n | leg AUROC | margin |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | `lens-floor` (9010530) | 0.6765 | **distribution** | 6 | 0.9808 | **+0.304** |
| Qwen2.5-7B | `band2-20260805` | 0.6708 | bound | 2 | 0.9952 | +0.324 |
| Tülu-3-8B | `dissociation-tulu3` | 0.6417 | bound | 4 | 0.9711 | +0.329 |
| Mistral-7B-v0.3 | `dissociation-mistral` | 0.6569 | bound | 4 | 0.9383 | +0.281 |

`scripts/internals_floor_screen.py` (keyless, no GPU, seconds); artifact
`outputs/analysis/internals_floor_screen_20260821.json`; rule and full
derivation `instrument_layer.md` §2.10. **Llama's is the adopted
`mean + sigma*SD` floor at n=6 with its sigma window recomputed to
[1.285, 27.817) against the configured 2.0** — the strongest screening this repo
has applied to any number.

⚠️ **A witness never upgrades a grade.** Three floors are `bound`; a bound stays
a bound (§2.4) and must be quoted as one. What the margins say is that the
distinction is not load-bearing here — all four clear by +0.28 to +0.33 against
an n-dependence never observed above 0.018.

**What stage 1 still is.** Only §6(a): the scaffold arm was never captured, so
the wrapper-vs-characters split cannot be done internally. §6(b) shrinks from
"make the floor usable" to the optional "raise three models from `bound` to
`distribution`", which changes a grade and no verdict — fold it into 6(a)'s
preset, never cost a job for it alone.

**The lesson, and it is the reason this section exists rather than a run
record:** a limitation was scoped to a *run* when it belonged to a
*measurement*, and the mis-scoping held an offline check as a filed GPU job for
nine days. Before costing a run to close a gap, ask whether the quantity the gap
is about already exists elsewhere on disk.

---

## 6. Stage 1 — the two capture gaps, and they are the run

Both are forward-pass-only: no generation, no judge calls, so the money line is
$0 and the estimate is GPU-hours alone. **Not costed here** — the approval gate
requires `cost_model.py` against a committed preset, and no preset exists yet.

**(a) The scaffold arm was never captured.** `activations_path` on all four runs
carries `plain_*` and `encoded_*` only; the middle cell of §4h's factorial has
behaviour and no activations. So the wrapper-vs-characters decomposition — the
thing that made leg 1's subject wrong — **cannot currently be done internally**,
and an internals leg that ignores it repeats §4h's error one level down. Llama
and Qwen are the contrast that matters: their behavioural loss runs through
*different terms* (Llama +0.67 wrapper, Qwen +0.02 wrapper / +0.25 characters).
If the internal separation survives both arms on both models, the dissociation is
independent of which term drives the behaviour — which is a stronger claim than
either model alone.

**(b) Control rungs, to make the floor usable.** Adding ability-0 rungs to the
same run turns `control_floor.usable` true and converts permutation-only
licensing into a screened reading, in-run, at the n the floor is computed on.
`reverse_characters` and `tag_block` are the established can't-decode rungs
(§3.3) and they cost one forward pass each.

Both fold into one preset per model. `gates:` must answer *what would I build
differently depending on the result* — draft: a dissociation that survives the
scaffold arm sends stage 2 to the causal test; one that appears only in the
encoded arm makes it a claim about the encoding rather than the protocol and
narrows the leg accordingly.

---

## 7. Stage 2 — causal, and its null is pre-declared

Gated on stage 1. §6.3.1 settled that the causal test is the **licensing layer**,
not the last instrument, and I5/I6 are built (`build_status.py`: `[OK]`).

The test, ported from Arditi et al. (NeurIPS 2024, 2406.11717 —
`other_repos/refusal_direction`, already the source of measurement #3's
estimator):

- **Necessity** — ablate the direction on plaintext harmful. Refusal should
  bypass. This is the positive control for the direction being a refusal handle
  at all.
- **Sufficiency** — add it on plaintext benign. Refusal should be induced.
- **Specificity** — ablating it must leave unrelated behaviour within KL 0.1.
- **The sharp one** — ablate it in the *encoded* condition, where behaviour
  already ignores it. **H-B predicts little or no change**: you cannot remove the
  influence of something that is not being read.

⚠️ **The null is pre-declared as weak, and this must be written before the run,
not after.** Kwon 2026 (2607.14147) reports the harm direction is *"a read-out
but not a selective write-handle"*, with no clean steering window and no single
refusal-decision direction above 0.73. So a null on the necessity arm is a
**predicted** outcome that would say the direction is not a write handle — it
would **not** say the harm is unrepresented. A design that would read its own
predicted null as confirmation is not a test.

---

## 8. The gate structure, in one place

| stage | cost | passes → | fails → |
|---|---|---|---|
| 0 offline | $0, no job | the dissociation is real; build stage 1 | no internals leg; ship behavioural (§4e says this is a paper) |
| 1 capture | $0, GPU-h TBD by preset | leg covers the protocol, floor usable | leg narrows to the encoding; state the restriction |
| 2 causal | $0, GPU-h TBD | necessity + specificity hold → repair claim available | direction is a read-out not a handle (**predicted**); leg stays a dissociation, which is still the result |

**Every branch ends in a writable paper.** That is the property that makes this
worth building — unlike the I4 route, no outcome here leaves AS-5 with nothing.

---

## 9. What this does to I4, and to AS-6

**I4 is off the critical path for AS-5.** The leg above is built from the linear
probe layer, the capture spine and the causal toolkit — all `[OK]`. I4's licence
is thin (Scope 22, min ratio 0.810 against a placeholder 0.800) and the knob
argument it needs is now *conditional*: derive `min_transfer_ratio` if and when a
feature-level claim is wanted, and record in the roster that AS-5 does not
require one. **This does not retire I4** — a feature-level account of *what* the
represented-but-unread content is would strengthen the leg — it removes it from
the blocking path.

**AS-6 inherits the frame directly and the numbers not at all.** Its whole thesis
is *decoded but not blocked*, which is this dissociation with a guard in place of
a target — so the design generalises, while §2.6 and §0.6 have now twice measured
that the target-side calibration does not port to a guard. Take the shape; derive
the constants per guard.
