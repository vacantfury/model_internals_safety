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
check the current pipeline and then build."* Nothing in §3 is adopted until this
doc is reviewed.

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

### 3.1 The instrument contract (prototyped, 18 tests, NOT adopted)

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

*Status: written and tested; wired to nothing. Adoption is this doc's decision.*

### 3.2 Split `measurements/` by kind

    measurements/     the four regime measurements + the new instruments
    licensing/        length_null, causal_license, contract  (controls and gates)
    combination/      regimes, guard_regimes                 (cells and maps)

Rationale: §1.2's three kinds have different consumers and different change
rates. A licensing rule change must not require reading a measurement module,
and today it does. **Cost: an import sweep across ~15 files and both scripts.
This is the only proposal here that touches working code, and it is the one
most safely deferred** — the contract delivers most of the benefit without it.

### 3.3 Extract the shared spine

One module owning the sequence both scripts already implement:

    capture-or-load -> run declared instruments -> collect Readings
      -> license -> hand cells to the paper-specific combiner -> write record

Both scripts keep their own *combination* step (AS-5 assigns four-regime cells,
AS-6 assigns guard cells) and their own instrument roster. Only the spine is
shared. This is what makes build-plan §6's scheduling fact real: **I1–I4 all
read the same forward pass**, which is one run if one runner drives them and
four runs if each script wires its own.

### 3.4 Sequence

1. Adopt (or reject) the contract in §3.1 — no code changes beyond it.
2. Make the three new instruments emit `Reading`s. Offline, mechanical.
3. Extract the spine (§3.3), one script at a time, tests green between.
4. Then and only then, I4 — so it lands wired instead of as a seventh orphan.
5. §3.2's package split last, or never, judged on whether §3.1 already fixed it.

---

## 4. Open questions this doc does not settle

1. **Does `regimes.py` belong under a combination package, or is placement
   churn not worth it?** §3.2 argues yes; the counter-argument is that the split
   costs an import sweep and buys tidiness rather than correctness.
2. **Should the spine be a module or a function both scripts call?** A module
   invites a framework; a function might not carry enough.
3. **Does AS-6 want the same `Reading` type?** Deployment is its central
   quantity and its licensing may need fields AS-5 does not — decide when the
   guard-side instruments are built, not now.
4. **Where does the run record's schema live** once readings are typed? Today
   `provenance.py` owns it and each script hand-builds its summary dict.
