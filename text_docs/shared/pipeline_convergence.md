# Pipeline convergence with `llm_guardrail_security` — the divergence table

**Owner order, 2026-08-06:** *"after everything built, we need review and try to
be similar to the pattern of llm guardrail pipeline (I know structure very
different, I just mean experiment management, json files, raw output folders,
yamls, which can be similar and not weaken the pipeline can be considered)."*

Scope is his enumeration — **the plumbing, explicitly not the science layer.**
He states up front that the structures differ and is not asking this repo to
imitate the sibling's measurement design.

**The test he named cuts both ways: converge only where it does not WEAKEN this
pipeline.** Several divergences here are load-bearing and must survive any
alignment — the tri-state `licensed`, `Reading.reportable`/`withheld`, the
dirty-tree guard, per-family flush+fsync, and absolute activation paths pointing
at cluster scratch. Where the sibling is BETTER, adopt it; the direction is not
assumed.

Read from the sibling's **actual pipeline**, not its docs: `dispatch.py`,
`main.py`, `conf/`, and a real `results.json` from
`outputs/bestofn_defense/defense+evaluate/jailbreakbench/`.

---

## (a) Experiment management

| | this repo | sibling | verdict |
|---|---|---|---|
| invocation | argparse flags: `--model --families --n-prompts --instruments` | `python main.py <preset>` reading `conf/experiment/<campaign>/<preset>.yaml` | **contested — see below** |
| multi-cluster | none (single-GPU by NURC's QOS) | `dispatch.py` splits one preset across a cluster pool, dry-run by default | **not applicable yet** |
| tracker | none, deliberately deferred | MLflow (`mlflow_run_id` in every record) | **do not adopt** |
| cost gate | `--dry-run` + `scripts/cost_model.py` | dispatcher prints the split plan, submits nothing without `--submit` | **already aligned in spirit** |

**The preset question was the one genuinely contested item. SETTLED 2026-08-07
(owner: "go"), and the deciding evidence came from the cluster, not the argument.**

The original case was the **experiment-run approval gate**: a preset YAML is a
reviewable artifact that can be committed, diffed, and pointed at in the
approval request; a shell command in a chat message is not. The counter-case was
that their presets exist because their unit of work is a *matrix* — model ×
defense × encoding × benchmark — where a command line becomes unreadable, while
ours is one model × a rung list, which argparse expresses fine.

**What settled it: the flag strings were never in chat either.** The cluster
carried THREE launchers — `phase0.sbatch`, `as6_phase1.sbatch`,
`relicense.sbatch` — and two of them existed only there: authored on the
cluster, never brought back, absent from the laptop and absent from git.
`relicense.sbatch` hardcodes a fifteen-element family array, two absolute
`results.json` paths, and array-index arithmetic over two bash arrays, in a file
nobody can review, diff, or reproduce from the repo. Every new experiment was
spawning another one. The run declaration was already being written down — just
somewhere unversioned, on one machine.

**Shipped:** `conf/experiment/*.yaml` (six presets) + one generic
`ops/run.sbatch` + `scripts/submit.py`, dry-run by default. Three
experiment-specific launchers collapse to one. Four properties are load-bearing:

- **The schema is CLOSED.** `tests/test_presets.py` asserts no preset field
  shares a name with any `measurements.yaml` knob, and `StrictModel` forbids
  unknown keys. A preset declares WHICH RUN, never HOW an instrument reads —
  otherwise a run ships a number with no registered tuning path, which is the
  magic-number problem re-entering through the launcher.
- **`gates:` is a REQUIRED field.** The owner's 2026-08-06 rule — a run must be
  a GATE, not a measurement — was prose, and prose is enforced by memory. Now
  the loader refuses to build a job until "what would I build differently
  depending on the result?" is answered in writing, in a committed file, where
  it can be disagreed with before the GPU is allocated.
- **Command construction is Python, not bash.** `ops/run.sbatch` calls
  `submit.py --resolve <index>` and executes what it is handed. The array-index
  arithmetic that used to live in bash is `PresetConfig.tasks`, which is tested.
- **`cost_model.py --preset <name>`** costs exactly what will be submitted.
  Before this it priced the full 19-rung ladder at the pilot's corpus size, so
  the gate's dollar figure described a different run than the one being approved.

**`ops/` stays gitignored, and that is now the correct line rather than a
compromise.** The family sync standard splits code by git and the ops layer by
rsync; the split this refactor makes is the same one stated in experiment terms
— **committed = WHAT the experiment is (`conf/experiment/`), gitignored = WHERE
it runs (`ops/run.sbatch`: scratch paths, venv, `HF_HOME`, secret sourcing).**
The original defect was never that a launcher was gitignored, it was that the
*declaration* was inside it. `run.sbatch` now carries no experiment at all, so
it is pure environment and belongs on the rsync side. Do not "fix" this by
committing it — a committed launcher is how cluster-specific detail creeps back
into a public repo.

**Three defects surfaced by writing the presets down**, each of which would
otherwise have been discovered after a queue wait: two presets read the
deployment probe without declaring its required `lexical` control (every
deployment reading would have been non-reportable); two asked for `n_prompts:
200` against a 100-per-class corpus, `--n-prompts` being per class while the
band-run write-up's "200 prompts" is the total; and the Base arm could not run
at all, because `render_chat` fails closed on a checkpoint with no chat template.
All three are now tests or config.

**`dispatch.py` becomes relevant the moment the xc-cluster question is settled.**
It already implements exactly the routing that a NURC+xc split would need. If
this project moves to xc, port it rather than re-deriving it.

---

## (b) JSON record schemas

Both repos write `results.json` per run. The overlap is large; each has fields
the other lacks, and the gaps run in both directions.

**What the sibling has that we lack — four of these are real defects here:**

| field | what it does | verdict |
|---|---|---|
| `schema_version: 1` | lets a reader know which shape it is holding | **⚠️ ADOPT — we have none** |
| `status` + `warnings[]` | success/failure, and accumulated warnings in the record | **⚠️ ADOPT — ours go to stderr and are lost** |
| `upstream_ref{source_dir, results_sha256}` | content-pins the stage this run consumed | **⚠️ ADOPT — live need, see below** |
| `primary_metric` | names WHICH metric is the headline | **adopt, cheap** |
| `judge_config_hash` | pins judge behaviour | adopt when judges vary |
| `elapsed_seconds` | top-level wall-clock | we have `throughput`; equivalent |
| `mlflow_run_id` | tracker linkage | do not adopt |

**`upstream_ref` is the one with a live need.** `scripts/rescore_ability.py` and
`scripts/rebaseline_pilot.py` both read a PRIOR run's `cells.jsonl` and emit new
numbers. Today nothing in the output pins which run they consumed by content —
and this repo has already been bitten by exactly that class of drift, when
`rescore_ability`'s self-check went red because `cells.jsonl` predated instrument
fixes #1/#2. A `results_sha256` on the consumed file would have said so
mechanically instead of by inference.

**What we have that the sibling lacks — none of it may be traded away:**

| field | why it must survive |
|---|---|
| `readings[]` + `withheld{}` + `n_reportable` | the whole contract layer; a table of what was measured is honest only beside what was measured and thrown away |
| `env_lock`, `python`, `platform` | full environment capture; theirs records `git_sha` only |
| `activations_path{}` | names the cache the numbers came from — "re-run it" is otherwise not well-defined |
| `seed` at top level | theirs is inside `target_model_config` |
| `corpus{digest}` | content-pins the prompt set |

**Net: their record is stronger on run LIFECYCLE (version, status, lineage);
ours is stronger on run PROVENANCE and claim validity.** The two are
complementary and there is no conflict — adopting all four of their fields costs
nothing and removes nothing.

---

## (c) Raw output folders

| | this repo | sibling |
|---|---|---|
| layout | `outputs/runs/<phase>/<model>/<run_name>/` | `outputs/<campaign>/<mode>/<benchmark>/<model>_<defense>_<encoding>_<ts>_<jobid>/` |
| per-run files | `results.json` + `cells.jsonl` | `results.json` + `raw_results.jsonl` |
| cache | `outputs/activations/` (reusable across analyses) | none — no activation capture |
| invalidated runs | *nothing* | **`outputs/_quarantine/<reason>_<date>/`** |

**Two adoptions, one of them overdue.**

**1. `outputs/_quarantine/` — ADOPT.** The sibling quarantines runs found
invalid, moved rather than deleted, in a directory named for the reason
(`oracle_leak_20260805`, `figstep_incomplete_20260805`). This repo needs it more
than they do: **every quantitative map from both of our runs has been revised at
least once**, and the pilot's `cells.jsonl` is currently superseded-but-in-place
with the fact recorded only in prose. A run that has been invalidated and still
sits at its original path is a trap for the next session.

**2. Collision-proof run directories — ADOPT.** Their directory name carries a
timestamp and the SLURM job id, so two runs cannot collide. Ours uses a
human `--run-name`, and re-running with the same name **silently overwrites**.
That is a data-loss path we have simply not hit yet. Their scheme is the fix;
ours can keep the readable name and append `_<ts>_<jobid>`.

Their deep `<campaign>/<mode>/<benchmark>/` nesting is *not* worth copying — it
encodes their matrix, and our `<phase>/<model>/<run>` already carries ours.

---

## (d) YAML configs

| | this repo | sibling |
|---|---|---|
| per-model | `conf/models/*.yaml`, `conf/guards/*.yaml` | `conf/llm/*.yaml` (30+) |
| tunables | `conf/measurements.yaml` (one file, sectioned) | spread across preset + model files |
| run declaration | argparse | `conf/experiment/<preset>.yaml` |
| cluster | none committed | `conf/cluster_pool.yaml` (+ `.example`) |

**Already aligned, and where we differ we are ahead.** Both split by kind and
keep one file per model. Our `conf/measurements.yaml` centralises every tunable
with a stated tuning path per knob and a config-discipline TEST enforcing it —
the sibling has no equivalent, and the science organ has already ruled our
version the family reference implementation. **Do not converge downward here.**

The one thing to copy: their `conf/cluster_pool.example.yaml` pattern — a
committed example beside a gitignored real file. We will need it whenever the
xc-cluster question is settled.

---

## Actions, in the order they should happen

| # | action | area | cost | why now |
|---|---|---|---|---|
| 1 | `schema_version` in `write_results` | (b) | 15 min | every record already written lacks it |
| 2 | `status` + `warnings[]` in the record | (b) | 30 min | warnings currently go to stderr and are lost |
| 3 | `outputs/_quarantine/` + a `quarantine_run` helper | (c) | 1 h | superseded runs sit in place today |
| 4 | collision-proof run dirs (`_<ts>_<jobid>`) | (c) | 30 min | silent overwrite is a live data-loss path |
| 5 | `upstream_ref` + `results_sha256` | (b) | 1 h | the re-scoring scripts need it now |
| 6 | `primary_metric` | (b) | 15 min | cheap, and the contract makes it meaningful |
| 7 | presets for cluster runs | (a) | ~half a day | ✅ **DONE 2026-08-07** — see §a |
| 8 | port `dispatch.py` | (a) | — | **blocked on the xc-cluster decision** |

Items 1–7 are done. **8 is downstream of a decision he has not made** — and it
matters less now than when this table was written: `submit.py` already owns the
preset→resources→sbatch path, so porting `dispatch.py` would only add the
multi-cluster ROUTING, which is precisely the part the xc decision governs.

**A family convention that emerges from this belongs in the science handbook,
not in one repo's CLAUDE.md** — specifically the run-record schema (items 1, 2,
5, 6) and the quarantine convention (item 3), both of which are repo-neutral.
