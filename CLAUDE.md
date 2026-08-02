# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Visibility: public *(deliberately public from the start — owner policy 2026-07-09: science projects are public from birth, and the repo doubles as résumé/portfolio evidence. Consequence: public-grade discipline is MANDATORY — never commit personal data, ARR/reviewer text (`text_docs/reviews/` is gitignored), task files, secrets, or 1Password references.)*

## Project

Research home for the **model-internals safety** line: safety research that operates INSIDE model parameters — training-time interventions (safety fine-tuning, unlearning) and parameter/activation-space work (activation steering, representation-level analysis and edits). **Model-size-agnostic by design:** the pipeline is the same for a 1B and a 70B; small open models are the default experimental substrate (cheaper runs, more seeds). Like its siblings, this is a **shared home for a line of work**, not a single paper.

**Founded 2026-08-02** (owner-ratified, from a science-organ session). The defining property vs the siblings: they are eval-only on frozen models (black-box inputs/outputs); THIS repo goes inside the model — a new cost shape (GPU training/intervention runs rather than API-call evals).

## Scope boundary (load-bearing — read before deciding where work goes)

- **THIS repo owns everything INSIDE the model:** training-time safety interventions, weight/activation-space methods, internals-level analysis of safety behavior, and their evaluation.
- **`llm_guardrail_security`** owns the black-box model-side line (encoding/imaging jailbreak attacks and content-guard defenses on frozen models; Papers A–D). **`llm_agent_security`** owns the agent line. The **science organ** (`personal/science`) owns cross-project research knowledge (venue records, the shared literature corpus).
- **One concrete example each side:** "LoRA-fine-tune a 7B on refusal data and measure generalization to unseen encodings" = **THIS repo**; "test whether a decoy image lowers a frozen VLM's ASR" = the guardrail sibling.
- If this line needs the family's encoders as eval payloads, they get **COPIED**, never imported (oikos charter bars research-bet→research-bet dependencies; rule of two → extract a standalone package).

## Status

Pre-code: the first paper is at **S1 complete — direction committed, next S2 (feasibility)** — see `text_docs/proposal.md` (its `Workflow stage:` line is the state of record) and `text_docs/s1_idea_check.md`. No Python scaffold yet; scaffold `uv` + `tests/` + pytest smoke test when the first code lands (global testing law).

## Conventions

- **Package manager will be `uv`** (global law). No application frameworks, CLI structure, or trackers assumed before discussion.
- **LLM provider layer, when needed:** the `llm_utils` base package as a pinned git dep by tag (family convention; never vendor a copy).
- **Experiment-run approval gate (family rule, owner 2026-07-22):** training and intervention runs are heavy — BEFORE launching ANY run, report an explicit estimate of (1) GPU count + type, (2) money ($), and (3) wall-clock time, and get the owner's explicit go.
- **Cluster sync (family standard, settled 2026-08-02):** git clone/pull for committed code + rsync/scp for the gitignored ops layer (`*.sbatch`, cluster configs) + rsync-down for results — canonical: science organ `knowledge/cluster_sync_convention.md`. This repo is public, so the code path is plain `git clone` (no auth); wire the down-sync sites + ops-file rsync at S2 when code lands. Never fork a per-repo cluster-sync scheme.
- **Conference deadlines:** the canonical timeline lives in `llm_guardrail_security/text_docs/shared/conference_timeline.md` — consult and update THERE, never fork a per-repo deadline list.
- **Public-grade discipline (mandatory):** no secrets / PII / 1Password refs in any committed file; `TODO.md`, `NOW.md`, `knowledge/`, `outputs/`, `data/`, `text_docs/reviews/` are gitignored.
- **English only** in task files (human names may stay as-is).

## Task system

Root `TODO.md` (gitignored — task text is personal), psyche task standard (position = priority; finished items move to the central psyche archive). Registered in the psyche oikos map as a research bet. First-paper ideation is tracked in the science repo's TODO (ideation runs from science sessions).
