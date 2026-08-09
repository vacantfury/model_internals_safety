# Pipeline architecture — shared by AS-5 and AS-6

**Canonical home for how the code is STRUCTURED.** The third sibling:

| doc | question it answers |
|---|---|
| `instrument_layer.md` | what the instruments have been **found** to do |
| `instrument_build_plan.md` | what we **will** build, and in what order |
| **this file** | how the code is **arranged**, and why |

`text_docs/project_structure.md` is AS-5's historical build record (§§8–12 are
dated build-step records) and stays that; it is not superseded, it is a
different kind of document.

Written 2026-08-06 on the owner's instruction, after six instrument modules
landed in one day with **zero pipeline integration** — *"maybe we need first
construct a structure design doc and carefully design the whole thing, and then
check the current pipeline and then build."*

**Reviewed and settled 2026-08-06.** §3.1 adopted, §3.2 struck, §3.3's form
fixed, and all four §4 questions answered — see §4 for the reasoning, which is
kept rather than deleted so the decisions are not silently reopened.

---

## 1. Current state, measured not impressioned

Counted 2026-08-06 against the working tree at `b125a03`.

### 1.1 The library layering is a clean DAG — do not restructure it

    models        997 lines   ->  (no internal dependencies)
    encodings     517         ->  (none)
    judges        473         ->  (none)
    probes        791         ->  models
    guards        396         ->  models
    measurements 2358         ->  encodings, judges, models, probes

No cycles, and dependencies point strictly from composite toward foundational.
**This is the part that is working, and the temptation when writing an
architecture doc is to redesign it anyway. Don't.** The problems below are all
above this layer or inside one box of it.

### 1.2 Problem A — `measurements/` conflates three different kinds of thing

Thirteen modules, 2358 lines, and they are not the same kind of object:

| kind | modules | what it is |
|---|---|---|
| **the four measurements** | `ability` `behavior` `deployment` `recognition` | the pilot's original instruments, each answering one of the four-regime questions |
| **the new instruments** | `decode_lens` `trajectory` `entropy_dynamics` | I1–I3, built 2026-08-06, answering *different* questions |
| **meta / combination** | `regimes` (340) `guard_regimes` `length_null` `causal_license` `contract` | not measurements at all — they combine, control, or license measurements |

`regimes.py` is the largest module in the layer, and it is *combination logic*
sitting in a package named for the things it combines. That is the smell.

**⚠️ Corrected 2026-08-06 — the coupling this section originally asserted does
not exist.** The first draft said the three kinds were entangled and that "a
licensing rule change requires reading a measurement module". Measured with the
import graph rather than by impression:

    regimes.py         collections, dataclasses, enum      <- zero internal
    guard_regimes.py   collections, dataclasses, enum, typing
    length_null.py     dataclasses, typing, sklearn
    contract.py        dataclasses, typing
    causal_license.py  config, torch                       <- config only

**Not one of them imports a measurement module.** They are already pure
functions of plain floats and bools; the only thing they share with the
instruments is a directory name. The conflation is real as *naming*, and it is
nothing as *coupling* — which is what §3.2 turns on, and why that proposal was
struck rather than scheduled. The general lesson is the same one the coverage
sweep taught a week earlier: measure the thing before designing against it.

### 1.3 Problem B — the orchestration spine is duplicated, not extracted

Two scripts, `phase0_regime_map.py` (625 lines) and `as6_guard_probe.py` (508),
share **ten** imports and both implement the same spine in their own words:

    load config -> select rungs -> encode -> capture (or load cache)
      -> run measurements -> license -> assign cells -> write run record

Shared: `config` `data` `encodings.base` `encodings.registry`
`measurements.deployment` `measurements.length_null` `models.capture`
`models.loader` `paths` `provenance`.
AS-5 only: the judges, `ability`, `behavior`, `recognition`, `regimes`.
AS-6 only: `guards.prompts`, `guards.verdict`, `guard_regimes`.

The divergence is already load-bearing: the length null shipped into the AS-6
spine on 2026-08-05 and did not reach the AS-5 spine until 2026-08-06, so for a
day the guard side had a mandatory control the target side lacked — **for a
confound measured on the target side.** That is what a duplicated spine costs,
and it has already been paid once.

### 1.4 Problem C — six instruments, zero integration

`decode_lens`, `trajectory`, `entropy_dynamics`, `causal_license`,
`interventions`, `lens`: **not referenced by any pipeline script.** 97 tests,
all green, none reachable from a run. Adding them inline the way the original
four are wired would put `phase0_regime_map.py` past 1000 lines.

### 1.5 Problem D — P1–P7 live in prose

Build plan §2 states seven principles every instrument must satisfy. Each module
declares them in its docstring, where nothing can check them and nothing can
compare across instruments. The repo's three worst instrument defects were all
*undeclared properties*:

- a probe asserting `False` where it meant "unmeasured" (§1.5)
- an operating point with a 50% false-positive rate by construction that
  `assign_regime` never saw (§2.1)
- a probe riding character length that licensing structurally could not detect (§5)

Each is a field on a result type, if the result type exists.

---

## 2. What is deliberately NOT proposed

Stated first, because an architecture doc's main risk is scope.

- **No application framework, CLI framework, config framework, or experiment
  tracker.** Repo convention (`CLAUDE.md`) defers all of these until the run
  shapes are known, and they still are not. `uv` + plain `argparse` + YAML stays.
- **No restructuring of `models/`, `encodings/`, `probes/`, `judges/`.** §1.1
  measured them sound.
- **No abstract base classes, no plugin registry, no lifecycle.** A Protocol and
  a dataclass carry the whole contract; anything more is machinery for a problem
  we do not have.
- **No rewrite of `regimes.py`'s logic.** Its *placement* is questioned in §3.2,
  not its correctness — it encodes hard-won coherence rules and a re-derivation
  would risk them.

---

## 3. Proposed structure

### 3.1 The instrument contract — ADOPTED 2026-08-06

`measurements/contract.py` — a `Protocol` plus a `Reading` dataclass turning
P1–P7 into fields:

| field | principle | why it is a field and not a docstring |
|---|---|---|
| `question`, `assert_distinct_questions` | P1 | two instruments with one question are one instrument counted twice |
| `control_reading` + `control_margin` | P2 | fails **closed**: "no control was run" ≠ "the control passed" |
| `length_null_margin` | P3 | fails **closed**: uncomputed ≠ cleared |
| `kind: correlational \| causal` | P5 | stored, not inferred — it is about what the measurement *did* |
| `operating_point: str` | P6 | the band-run incoherence traced to one stated only in a config comment |
| `selection_inside_null: bool` | P7 | an uncorrected argmax over ~33 cells is a multiple comparison |
| `licensed: bool \| None` | §1.5 | tri-state; `None` means *this instrument could not read this*, never *absent* |
| `claim: positive \| null` + `sensitivity` + `sensitivity_floor` | §4.4 | P2/P3/P7 are inflation controls and cannot license an ABSENCE — added 2026-08-06 (TODO 42) |

The load-bearing method is `reportable` — **one place answers "may this number
appear in a paper"**, and `why_not_reportable()` names every failing axis, so a
run record can carry a table of what was withheld beside the table of what was
measured. A withheld cell with no stated reason is how an instrument defect
hides as a null result.

**`reportable` routes on the claim's direction**, because the two directions need
different evidence: a positive claim must beat its negative control and clear the
length null, while a null claim must *not* beat its control and must demonstrate
the instrument could have fired at all. Full rules and the reasoning in
`instrument_build_plan.md` §4.4. The one thing to carry here: the direction is
**declared, never derived from the value** — deriving it would let any instrument
reading low dodge its own control — and the dodge is closed instead by
`claim_is_coherent`, which refuses a null claim that clears its specificity
control as a positive reading mislabelled.

Two pieces carry the weight beyond the fields themselves:

- **`gate_per_prompt(reading, per_prompt)` — the granularity join.** A `Reading`
  is a verdict about one instrument on one *condition*; the axes regime
  assignment consumes are per *prompt*. When the reading is not reportable every
  prompt returns `None`. This is where the repo's worst defect lived, and it is
  now one function instead of a rule each script must remember.
- **`QUESTION` / `KIND` as module constants** on each instrument, so P1 is
  checked across the real roster by `assert_distinct_questions` rather than
  asserted in three docstrings that can drift apart.

*Status: adopted 2026-08-06 on the owner's word. The three built instruments
emit `Reading`s; `guard_regimes` consumes the gated axis. 441 tests green.*

#### ⚠️ 3.1.1 The contract does not model the quantity the paper reports (found 2026-08-09)

**Eleven instruments emit a `Reading` — `ability`, `attribution`, `behavior`,
`causal_license`, `decode_lens`, `deployment`, `entropy_dynamics`,
`recognition`, `reply_inversion`, `sae_reconstruction`, `trajectory` — and not
one of them is refusal.** `behavior`'s `value` is `attack_success_rate`
(`behavior.py:267`), so `reportable` on that reading is a verdict about **ASR**;
the refusal rate travels in `detail` as an unevaluated payload field.

Every claim in AS-5's legs 1 and 2 is a refusal rate, and ASR has been
unreportable repo-wide since §3.5.2. **So the contract's one behavioural verdict
governs a number no paper will print, and is silent on the number both papers
are built from.** The four scaffold-control runs make it concrete: all four
withheld `behavior` for a correct reason that says nothing about any figure in
`evidence_and_story.md` §4h.

**This is the repo's recurring failure shape, inverted.** The usual form is *a
settled rule that did not reach every caller* — four instances between
2026-08-07 and -08-09, each fixed by making the omission inexpressible. This is
*a governing layer that never reached the governed quantity*, and it is harder
to notice precisely because the contract is visibly working on ASR throughout.

**The fix, filed not built: a `refusal` instrument** whose `required_controls`
are the paired benign arm and the plaintext baseline — the two things §4d and
§3.6 already made mandatory in prose. Until it exists, the refusal numbers rest
on an argument written in a design doc, which is the exact condition
`measurements/contract.py` was adopted to end.

### 3.2 ~~Split `measurements/` by kind~~ — STRUCK 2026-08-06

The proposal was to split into `measurements/licensing/combination/`. **Struck,
because its stated rationale was measured false** (§1.2): the licensing and
combination modules already import no measurement sibling, so the decoupling the
split would buy already exists. What remained was an import sweep across six
scripts in exchange for a directory name.

What the split *would* genuinely have encoded is the invariant — that those
modules stay pure functions of plain values, and that instruments stay leaves.
**A directory cannot check that; a test can.** So the invariant is now
`tests/test_package_structure.py`, which asserts three directions:

    regimes | guard_regimes | length_null | contract | causal_license
        -> import no measurement sibling at all
    decode_lens | trajectory | entropy_dynamics
        -> import at most `contract`
    nothing in the layer
        -> imports an instrument (instruments are leaves, so the roster stays swappable)

Zero churn, and unlike the package boundary it fails loudly the first time
someone violates it.

### 3.3 Extract the shared spine

One module owning the sequence both scripts already implement:

    capture-or-load -> run declared instruments -> collect Readings
      -> license -> hand cells to the paper-specific combiner -> write record

Both scripts keep their own *combination* step (AS-5 assigns four-regime cells,
AS-6 assigns guard cells) and their own instrument roster. Only the spine is
shared. This is what makes build-plan §6's scheduling fact real: **I1–I4 all
read the same forward pass**, which is one run if one runner drives them and
four runs if each script wires its own.

**Form, settled 2026-08-06: a file of plain functions at
`src/internals_safety/pipeline.py`.** The module-vs-function framing was a false
choice — the risk it was pointing at is a `Pipeline` class with lifecycle and a
config-driven runner, and that is avoided by declining to write the class, not
by declining to write the file.

**The selection criterion, which is the load-bearing part: the spine holds
anything whose absence in ONE script would be a defect.** That is not a
generality argument, it is the length-null incident stated as a rule — the
control reached AS-6 on 2026-08-05 and AS-5 on 2026-08-06, and a shared function
is what makes that gap unrepresentable. By that criterion the spine holds
plan/select/encode, capture-or-load, the mandatory controls, the per-family
flush+fsync, and the run-record write. It does NOT hold the instrument roster or
the combination step: those legitimately differ per paper, and forcing them into
the spine is how a shared runner becomes a framework.

### 3.5 Where WRITE-side instruments plug in — settled 2026-08-06

Every instrument built before I5/I6 *reads* activations we captured, so the
question of where one plugs in never arose: it runs inside the family loop, once
per rung. The causal instruments *write* to the model — ablate a direction, add a
steering vector — and that turned out to be a different shape, which is why
`models/interventions.py` and `measurements/causal_license.py` sat built and
unreachable for a day while I1–I3 were wired in an afternoon.

**The finding: causal licensing is a MODEL-level gate, not a rung-level
instrument.** The direction is fit on PLAIN harmful vs PLAIN harmless, it is the
same direction for every rung, and its answer *gates which direction the
downstream reads may use at all* rather than being one of those reads (TODO 28,
from Arditi et al., NeurIPS 2024). Wiring it behind `--instruments` inside
`run_family`, by analogy with I1/I3, would have re-run an identical computation
once per rung and invited the reading that a rung has its own causally-licensed
direction.

**So the plug-in point is `main()`, between the plain captures and
`run_families`** — the one place in the script where model-level, rung-independent
work already happens, because that is where the plain conditions are captured
once and reused. Its `Reading` joins the per-rung ones at the run-record write.

Three consequences worth keeping, each of which cost a fix to learn:

- **The cost accounting must include the negative control.** The random-direction
  null is a second sweep, and it was briefly absent from `--dry-run` while the
  code already ran it. A control the estimate cannot see is a cost nobody
  approved. Now priced from `causal.forward_passes` rather than restated, and
  pinned by a test.
- **A cost knob must be a CAP, not a stride.** `CAUSAL_LAYER_STRIDE = 4` sweeps
  7 layers of a 32-layer model as intended, and exactly ONE layer of a 3-layer
  model — layer 0, whose `resid_pre` is the raw embedding before any computation.
  Measured: the entire sweep came back degenerate. `MAX_CAUSAL_LAYERS` derives
  the stride from the depth and holds the cost fixed at both ends.
- **A degenerate cell is a coverage number, not an exception.**
  `difference_in_means` returns a zero vector where the classes coincide and says
  so; `ablate_direction` refuses to project out a zero direction. The runner
  filters them (`causal.viable_directions`), reports `n_degenerate`, and returns
  `licensed=None` when nothing survives — because "every cell was degenerate" is
  the instrument failing to read, not a measurement that harm is causally
  unmediated. Only exact degeneracy is filtered: a merely weak direction is the
  causal criteria's business, and a norm cut would be a second unfounded gate.

### 3.4 Sequence

1. ~~Adopt the contract in §3.1~~ — **done 2026-08-06.**
2. ~~Make the three new instruments emit `Reading`s~~ — **done 2026-08-06**,
   with `gate_per_prompt` joining them to the per-prompt axes both papers use.
3. ~~Extract the spine (§3.3) into `pipeline.py`~~ — **done 2026-08-06.**
   `add_common_arguments` · `load_contrast_sets` · `resolve_run_paths` ·
   `run_families` · `select_known`. Both scripts refactored one at a time, tests
   green between; 625+508 lines became 563+476 against a 184-line spine, and
   both `--dry-run` paths verified end to end.
4. ~~Fold the run-record results half into `provenance.py`~~ — **done
   2026-08-06.** `write_run_record(directory, record, readings=())` adds
   `readings`, `withheld` and `n_reportable`, with the split **computed** from
   `reportable_only`/`withheld_summary` rather than curated. Both scripts are on
   the seam already; passing no readings writes neither section, so a run whose
   instruments do not yet emit `Reading`s is not forced to fake them.
5. ~~Wire the four original measurements to emit `Reading`s~~ — **done
   2026-08-06**, and it immediately found that measurements #1 and #4 have no
   negative control (TODO 37/38). Both read non-reportable with the reason named
   rather than being quietly quotable — and #1's control has since landed
   (`ability_control`, §4.4), which in turn exposed that a specificity control
   cannot license an ABSENCE and produced the contract's claim-direction routing
   (TODO 42). #4 still has none: the judges never run on the benign-encoded arm.
6. ~~Wire I1/I2/I3 into the spine~~ — **done 2026-08-06.** I2 runs always (no new
   forward pass); I1 and I3 run only when `--instruments` declares them, because
   both add GPU work and `--dry-run` must price it before the approval gate sees
   the run. Orphans are now a build failure rather than an audit finding:
   `tests/test_package_structure.py` asserts the unreachable set exactly, with a
   stated reason per entry, and the list may only ever shrink.
7. **Next — the four missing controls**, two of which (matched-norm random
   direction, control-task selectivity) gate any I5/I6 claim. Then I4, so it
   lands wired instead of as an orphan.

~~§3.2's package split~~ struck; replaced by `tests/test_package_structure.py`.

---

## 4. The four open questions, settled 2026-08-06

Recorded rather than deleted: a settled question with its reasoning visible is
what stops it being reopened by the next session.

1. **Does `regimes.py` belong under a combination package?** **No.** The split's
   premise was measured false (§1.2) — the decoupling already exists, so the
   sweep would buy a directory name. The invariant it implied is now a test
   (§3.2).
2. **Should the spine be a module or a function?** **A file of plain functions**,
   `pipeline.py`. The dichotomy was false; the real rule is the selection
   criterion in §3.3.
3. **Does AS-6 want the same `Reading` type?** **Yes, and settled now rather
   than deferred** — reversing this doc's first recommendation. The argument is
   the one that founded `instrument_layer.md`: a property of the *measurement*
   that lives on one side gets re-derived on the other, and two `Reading` types
   would encode the length-null divergence in the type system permanently. The
   clinching evidence was already in the code — `guard_regimes.assign_guard_cell`
   takes `decoded: bool | None` with a docstring saying an unlicensed probe means
   "this instrument could not read this rung", which is `Reading.licensed`
   hand-rolled. AS-6 was already using the contract's central idea informally.
   The stated worry (AS-6's licensing may want fields AS-5 does not) is what
   `detail` absorbs; a structural field is one addition to a frozen dataclass
   with a default, whereas reconciling two diverged types later is not cheap.
4. **Where does the run record's schema live?** **Nowhere new.** `provenance.py`
   keeps it, and its canonical schema stays the `reproducible-run-logging`
   skill. The actual gap is the *results* half: both scripts hand-build a summary
   dict (`phase0_regime_map.py:320`, `as6_guard_probe.py:252`) and those have
   already drifted. Fix is one function — `write_run_record(directory,
   provenance, readings, summary)` — where the reportable/withheld split is
   *computed* from `reportable_only` + `withheld_summary` instead of curated by
   hand. Deferred to step 3, because founding a schema home before the spine
   exists is a home for one caller.

---

## 5. The layout review — three questions, settled 2026-08-07

**Owner ask: "everything built? should we review the structure?"** Recorded here
rather than in a new document, because "how the code is ARRANGED" is what this
file already is — a fourth `text_docs/shared/` sibling on the same subject would
be the exact defect the review was checking for.

Scope was three questions: the `src/` namespace cut (TODO 23, whose trigger had
fired on 2026-08-05 and gone unanswered), the `conf/` overlap that presets
introduced the same morning, and the `scripts/` kinds that writing `submit.py`
exposed.

### 5.1 `src/` — NO CHANGE, and the answer came from reading, not arguing

TODO 23 asked "(a) read what the sibling repos actually do before adopting
anything — the convention is theirs, and guessing it would fork it."

Read, 2026-08-07:

| repo | `src/` cut by | `text_docs/` cut by | papers hosted |
|---|---|---|---|
| `llm_guardrail_security` | ROLE — `attacks/ defense/ evaluation/ analysis/ experiment/ prompt_transformations/ utils/` | PAPER — `autoattack_defense/ bestofn_attack/ bestofn_defense/ imgaug_defense/ judge_reliability/ shared/` | **four** |
| `llm_agent_security` | ROLE — `attacks/ defenses/ scoring/ analysis/ harness/ utils/` | PAPER — `agent_injection/ shared/` | one |
| this repo | ROLE — `measurements/ probes/ models/ encodings/ judges/ guards/` | PAPER — `as5/ as6/ shared/` | two |

**The family convention is role-cut code and paper-cut docs, and the guardrail
repo demonstrates it at four papers.** Neither sibling has ever namespaced `src/`
by paper. This repo already matched, so the "deferred layout decision" that TODO
23 said was now live resolves to *no change is owed* — `probes/`,
`measurements/`, `models/` are shared instrument serving both papers, and only
the object-of-study layer (`guards/`) is paper-specific.

**Recorded so it is not re-litigated.** The question had been open since
2026-08-05 and would have been re-asked by the next session that added an AS-6
module. The proposal stub's "Namespace note" — *adopt the family's namespace
convention when AS-6 first lands code or configs* — is now answered in place.

**One residual was real and is fixed:** the AS-6 proposal had lived inside
`text_docs/as5/proposal.md` since registration, flagged-not-fixed because the
as5 tree was another session's live area. Moved to `text_docs/as6/proposal.md`
with a pointer left behind.

### 5.2 `conf/` — `pilot.yaml` renamed to `corpus.yaml`

The presets landed on the morning of 2026-08-07 and immediately raised a
one-home question: `conf/experiment/*.yaml` declares `n_prompts` and `families`,
and so does `conf/pilot.yaml`.

**Traced rather than assumed.** `pilot.yaml` holds five fields and they do two
different jobs:

- `harmful_set` / `harmless_set` — **the contrast pair itself**, read by every
  entrypoint of both papers, overridable by nothing. JBB's benign set is
  theme-matched to its harmful set, which is the entire reason it is the negative
  class rather than the larger OR-Bench set; a preset must not be able to break
  that pairing.
- `n_prompts` / `families` / `models` — **defaults for a bare invocation**, which
  a preset supersedes.

So it is not a second home. "Which sets are the contrast pair", "what does a bare
run do", and "what does `causal_sweep` do" are three different facts with one
home each; defaults-plus-declaration is a layering, not duplication.

**What WAS a defect is the name.** `scripts/as6_guard_probe.py` — not the pilot,
and the other paper — called `load_pilot_config()` to find out which prompt sets
form the contrast pair. The file is read by every run of both papers and every
future phase; `pilot` was a fossil of phase 0 that had become actively
misleading about ownership.

Renamed `conf/pilot.yaml` → `conf/corpus.yaml`, `PilotConfig` → `CorpusConfig`,
`load_pilot_config` → `load_corpus_config`, and the run record's `config.pilot`
→ `config.corpus`. **`SCHEMA_VERSION` bumped 1 → 2** for that last one: nothing
in the repo reads the key, so no reader broke — which is precisely why the bump
was worth making rather than skipping, since a field renamed silently under a
version that did not move is how a version field stops being trustworthy, and
that only has to happen once.

### 5.3 `scripts/` — NOT split, and a silent trap closed instead

Eleven scripts, three genuine kinds, no stated distinction:

- **entrypoints** (launch a run, need a GPU) — `phase0_regime_map`,
  `as6_guard_probe`, `sae_pregate`
- **analysis** (re-read a prior run, CPU-only) — `rescore_ability`,
  `rebaseline_pilot`, `recalibrate_deployment`, `relicense_probes`,
  `sweep_operating_point`
- **tooling** (no experiment at all) — `build_status`, `cost_model`, `submit`

A `scripts/entrypoints/` + `scripts/analysis/` split was considered and
**declined**: eleven files do not need a directory, and the launchable set is
already named — `config.Entrypoint` is a `Literal` that `submit.py` resolves
against, so the one distinction that carries consequences is typed and tested.
This is §3.2's judgment applied again: *a directory cannot check an invariant
and a test can.*

**But the investigation found a real trap and it is now closed.**
`completion.reachable_modules` globbed `scripts/*.py` **non-recursively**. The
day anyone puts a script in a subdirectory, every module reachable only from
there silently becomes an orphan — `test_no_module_is_an_orphan_except_the_declared_ones`
would report wired modules as unwired, and `build_status.py` would report built
instruments as unbuilt. Both point the reader at the wrong thing and neither
raises. Changed to `rglob`, so the walk no longer depends on a directory staying
flat — which matters whether or not the split is ever made.

### 5.4 What this review did NOT find

Stated because a review that reports only findings reads as if everything it
touched was broken. `text_docs/{as5,as6,shared}/` is correct and matches the
family. `conf/` splits by kind (`models/`, `guards/`, `experiment/`, plus one
file per concern) exactly as both siblings do. `measurements/` needed no split —
that was settled and struck in §3.2 on measured evidence. The instrument
contract, the spine, and the write-side plug-in point are all where §3.1, §3.3
and §3.5 put them.
