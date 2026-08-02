---
name: reproducible-run-logging
description: Make experiment runs in this repo reproducible by capturing a provenance record alongside each run's results. Use proactively whenever building, modifying, or adding an experiment/eval/training entrypoint or its results-writing code — even if the user doesn't ask — and when the user says "make this reproducible", "log the git hash / config / seed", "set up run logging", "record provenance", "what should I save with results", "is this run reproducible", or writes a new run/eval/train script. Defines the canonical run-record schema and the dirty-tree guard. Companion to check-experiment-results (read side); this is the write side.
---

# Reproducible Run Logging

Use this when you're constructing or touching the code path that launches an experiment and writes its results, or when the user asks to make runs reproducible / log provenance.

**Installed 2026-08-02, before the first entrypoint exists.** That is deliberate: the schema below is what `scripts/phase0_regime_map.py` must emit when it is written (build step 4), so it is cheaper to have it now than to retrofit it onto runs that already produced numbers.

## The one rule that carries the weight

**A git hash is only honest if the working tree was clean when the run started.** If there were uncommitted edits at launch, the recorded hash points to code that isn't what actually ran — the record silently lies. So the dirty-tree check is non-negotiable; everything else is bookkeeping.

## Canonical run record

Each run writes `results.json` into its `outputs/runs/<phase>/<model>/<run_name>/` dir, containing at least:

| Field | What | Why |
|---|---|---|
| `git_hash` | `git rev-parse HEAD` at run start | pins the code |
| `git_dirty` | `git status --porcelain` non-empty? (bool) | makes `git_hash` trustworthy — **load-bearing** |
| `git_diff` | the uncommitted diff, if dirty (string or path) | recovers what actually ran when dirty |
| `config` | full resolved config for the run | pins inputs (snapshot the frozen pydantic configs from `internals_safety.config` with `.model_dump()` — do NOT re-read the YAML, record what the run actually used) |
| `seed` | the RNG seed(s) used | replay stochastic evals bit-for-bit |
| `env_lock` | path to / hash of the committed `uv.lock` + Python version | pins the environment, not just the code |
| `metrics` | the headline result(s) | the payload the triage side validates |
| `raw_output_path` | relative path to per-example outputs | lets a result be re-audited |
| `activations_path` | relative path into `outputs/activations/` for any cached capture this run read or wrote | activations are the expensive artifact and are cached across runs, so a run's record must name which cache it used |
| `timestamp` | run start time | ordering / provenance |

No run record exists yet, so there are no legacy key names to preserve — this table is the schema.

**`seed` has no config key yet.** Nothing in the repo is stochastic as of build step 1 (capture is a deterministic forward pass). Add a `seed` knob to `conf/` at the first stochastic step — generation sampling, probe train/test splits, or LoRA init — and wire it here in the same change. A run record with `"seed": null` is only honest while the run genuinely has no RNG.

## Provenance-capture snippet

Inject a single helper at the run entrypoint (`scripts/phase0_regime_map.py` first; `scripts/phase{1,2,3}_*.py` after), called once at run start, merged into `results.json`:

```python
import subprocess, sys, json

def capture_provenance(config: dict, seed=None) -> dict:
    def _git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    dirty = bool(_git("status", "--porcelain"))
    rec = {
        "git_hash": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "config": config,
        "python": sys.version.split()[0],
        # env_lock: record the committed uv.lock path/hash here
    }
    if dirty:
        rec["git_diff"] = _git("diff")           # or write to a file and store the path
    if seed is not None:
        rec["seed"] = seed
    return rec
```

## Dirty-tree posture

**This project: refuse-unless-`--allow-dirty`, for any run on the cluster.** Two reasons specific to here. Cluster runs pass the family's experiment-run approval gate (GPU count/type, $, wall-clock, explicit owner go), so they are few, deliberate, and expensive — none of them should be exploratory. And the paper's central objects are *regime assignments*: a prompt is labelled (C)/(D)/(B)/(S) by combining four measurements, so a silent code change between measurements would corrupt the label rather than merely perturb a number.

Local CPU/MPS debugging runs stay **warn-and-record** — the record is honest either way, and blocking them buys nothing.

## Environment pin

`uv.lock` is the environment pin and it **is committed** (verified 2026-08-02). Record which lockfile hash and Python version each run used. Note the cluster and the laptop resolve different torch builds from the same lockfile (CUDA vs CPU/MPS wheels), so the Python version and platform both belong in the record — the lockfile alone does not identify the environment here.

## When NOT to over-do it

- Don't build a tracking framework (MLflow/W&B) just for provenance — the record file already carries it. Whether this project ever needs one is an open decision deliberately deferred until the phase-0 pilot reveals the real run shapes (`text_docs/project_structure.md` §7.3). Add a tracker only if the user wants run *comparison/UI*, and discuss first (global no-framework-by-default rule).
- Don't capture a pip freeze — `uv.lock` already exists.
- A throwaway/debug run doesn't need the full record; this is for runs whose numbers might end up in the paper.

## Hand-off

Once runs emit this record, the triage side (`check-experiment-results`, not installed yet — it lands when the first cluster runs exist) reads `results.json` to triage and curate. The two skills share the same file — keep their schema in sync.
