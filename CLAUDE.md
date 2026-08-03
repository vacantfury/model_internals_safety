# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Visibility: public *(deliberately public from the start — owner policy 2026-07-09: science projects are public from birth, and the repo doubles as résumé/portfolio evidence. Consequence: public-grade discipline is MANDATORY — never commit personal data, ARR/reviewer text (`text_docs/reviews/` is gitignored), task files, secrets, or secret-manager references.)*

## Project

Research home for the **model-internals safety** line: safety research that operates INSIDE model parameters — training-time interventions (safety fine-tuning, unlearning) and parameter/activation-space work (activation steering, representation-level analysis and edits). **Model-size-agnostic by design:** the pipeline is the same for a 1B and a 70B; small open models are the default experimental substrate (cheaper runs, more seeds). Like its siblings, this is a **shared home for a line of work**, not a single paper.

**Founded 2026-08-02** (owner-ratified, from a science-organ session). The defining property vs the siblings: they are eval-only on frozen models (black-box inputs/outputs); THIS repo goes inside the model — a new cost shape (GPU training/intervention runs rather than API-call evals).

## Scope boundary (load-bearing — read before deciding where work goes)

- **THIS repo owns everything INSIDE the model:** training-time safety interventions, weight/activation-space methods, internals-level analysis of safety behavior, and their evaluation.
- **`llm_guardrail_security`** owns the black-box model-side line (encoding/imaging jailbreak attacks and content-guard defenses on frozen models; papers AS-1…AS-4, aliases A–D). **`llm_agent_security`** owns the agent line. The **science organ** (`personal/science`) owns cross-project research knowledge (venue records, the shared literature corpus).
- **One concrete example each side:** "LoRA-fine-tune a 7B on refusal data and measure generalization to unseen encodings" = **THIS repo**; "test whether a decoy image lowers a frozen VLM's ASR" = the guardrail sibling.
- If this line needs the family's encoders as eval payloads, they get **COPIED**, never imported (oikos charter bars research-bet→research-bet dependencies; rule of two → extract a standalone package).

## Status

The first paper (Paper AS-6, alias E, "Can't, didn't, or wouldn't?") is at **S1 complete — direction committed, S2 feasibility open** — see `text_docs/proposal.md` (its `Workflow stage:` line is the state of record) and `text_docs/s1_idea_check.md`.

Code has started, ahead of S2, because the phase-0 pilot is itself the S2 gate. **Build steps 1–4 of `text_docs/project_structure.md` §6 are done**: the forward-pass spine, the 15-rung encoding ladder, the probe layer, the judge layer — all four measurements of `s1_idea_check.md` §7 — and `scripts/phase0_regime_map.py`, which combines them into a per-(model, family) regime map.

**The pilot RAN 2026-08-03 (Qwen2.5-7B + Llama-3.1-8B, 15 rungs × 100 prompts, 4 SLURM jobs on H200) and its gating question is NOT answered — the instrument failed on the rungs that decide it.** What the run establishes, and what it does not:

- **Established (clean, X=0, reproduced on both models):** across the twelve deterministic cipher rungs (base64 → morse), the regime map is uniformly **(R) surface refusal** — models refuse on the *look* of the ciphertext, with the content probe finding no plaintext in the residual stream. **The decode-and-comply (B) regime does not appear anywhere in the cipher band at 7–8B.**
- **Not established:** the only rung showing any (B) is `zero_width` (B=3–12/100 across three runs), and it is INSTRUMENT-FAILED at 18–29% hard incoherence, as are `reverse_words` and (on Llama only) `reverse_characters`. The design's own rule — "do not report these regimes" — applies. So the paper's gating question stands open, and it is open **exactly at the top of the ladder** where decoding is near-trivial.
- **Two instrument defects, root-caused and filed** (TODO items 6–7): the ability binary is `exact or contains` with the computed `similarity` unused, which false-negatives genuine decodes and produces the `deployment_without_ability` incoherence; and recognition/deployment disagree in both directions, undiagnosed. Both must be fixed before any (B) reading is trustworthy.
- **Reproducibility signal:** re-running the same (model, rung) reproduced the probe measurements *exactly* (identical X counts); only the B/S split moved by ≤2 cells, which is refusal-judge variance, not probe noise.

Run tests with `uv run pytest` (hermetic — no network, no keys); `uv run pytest -m slow` adds the real-weights smoke tests. Anything that calls a judge or downloads gated weights needs keys, so prefix it with the repo-root launcher: `./run python scripts/…`. Two scripts cost a run without launching it, both keyless and GPU-free: `--dry-run` on the pilot prints the work (capture passes, generations, judge calls), and `scripts/cost_model.py` turns that into the GPU-hours, dollars and wall-clock the approval gate requires — tokenising the real corpus under every rung rather than assuming an inflation factor. Assumptions live in `conf/cost.yaml`. **The first real run has now replaced them, and the pre-run estimate was badly optimistic: 0.7–2.4 GPU-hours per model was predicted; measured on H200 was ~4.5 h for Llama-3.1-8B (15 rungs, 4:30:43) and ~8+ h for Qwen2.5-7B (13 rungs in 8:00:26, killed at the wall) — 2–4× above the TOP of the predicted range, i.e. the error is in the throughput assumption, not the token census.** Per-rung: ~18 min Llama, ~32–37 min Qwen. Re-tune `conf/cost.yaml` from these before costing any later phase (TODO item 5); note also that one model exceeding the partition's 8 h wall means phase-0-scale work must be split across jobs by `--families`.

The pilot's corpus is four JSONL prompt sets copied from the guardrail sibling into the gitignored `data/`; a fresh clone re-copies them (`text_docs/project_structure.md` §5).

## Conventions

- **Package manager is `uv`** (global law); `uv.lock` is committed. No application frameworks, CLI structure, or trackers assumed before discussion — whether this project ever needs one is deferred until the phase-0 pilot shows the real run shapes (`text_docs/project_structure.md` §7.3).
- **Artifact layout:** `outputs/activations/` is a capture cache reusable across analyses; `outputs/runs/<phase>/<model>/<run_name>/` holds one dir per run with its `results.json` provenance record. Both gitignored; both defined in `src/internals_safety/paths.py`.
- **LLM provider layer:** the `llm_utils` base package as a pinned git dep by tag (family convention; never vendor a copy) — pinned at `v5.0.0`, the same tag as the guardrail sibling, so judge behaviour matches across the family.
- **API keys** reach the process from the secret manager at launch, via the gitignored repo-root `run` wrapper and its pointer map. Nothing vault-shaped is ever committed; no plaintext key touches disk. The same wrapper is the cluster path.
- **Experiment-run approval gate (family rule, owner 2026-07-22):** training and intervention runs are heavy — BEFORE launching ANY run, report an explicit estimate of (1) GPU count + type, (2) money ($), and (3) wall-clock time, and get the owner's explicit go.
- **Cluster sync (family standard, settled 2026-08-02):** git clone/pull for committed code + rsync/scp for the gitignored ops layer (`*.sbatch`, cluster configs) + rsync-down for results — canonical: science organ `knowledge/cluster_sync_convention.md`. This repo is public, so the code path is plain `git clone` (no auth); wire the down-sync sites + ops-file rsync at S2 when code lands. Never fork a per-repo cluster-sync scheme.
- **Conference deadlines:** the canonical timeline lives in `llm_guardrail_security/text_docs/shared/conference_timeline.md` — consult and update THERE, never fork a per-repo deadline list.
- **Public-grade discipline (mandatory):** no secrets / PII / secret-manager or vault references in any committed file; `TODO.md`, `NOW.md`, `knowledge/`, `outputs/`, `data/`, `text_docs/reviews/` are gitignored.
- **English only** in task files (human names may stay as-is).

## Skills installed

- `reproducible-run-logging` — the write side of reproducibility: the canonical `results.json` run-record schema and the dirty-tree guard. Installed 2026-08-02, deliberately *before* the first entrypoint exists, so `scripts/phase0_regime_map.py` is born emitting it.

Not installed yet, with its trigger named: `check-experiment-results` (the read/triage side) — it needs a job runner, per-run output dirs, and a results doc to triage, none of which exist before the first cluster runs. Install it in the same sitting as the first phase-0 job.

## Task system

Root `TODO.md` (gitignored — task text is personal), psyche task standard (position = priority; finished items move to the central psyche archive). Registered in the psyche oikos map as a research bet. First-paper ideation is tracked in the science repo's TODO (ideation runs from science sessions).
