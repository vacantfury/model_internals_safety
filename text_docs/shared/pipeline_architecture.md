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

The load-bearing method is `reportable` — **one place answers "may this number
appear in a paper"**, and `why_not_reportable()` names every failing axis, so a
run record can carry a table of what was withheld beside the table of what was
measured. A withheld cell with no stated reason is how an instrument defect
hides as a null result.

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
5. **Next — wire the four original measurements to emit `Reading`s**, which is
   what turns the seam from available into load-bearing. Only then I4, so it
   lands wired instead of as a seventh orphan.

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
