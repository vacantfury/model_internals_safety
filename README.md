# model_internals_safety

Safety research **inside model parameters**: training-time interventions (safety fine-tuning, unlearning) and parameter/activation-space work (activation steering, representation-level analysis and edits), evaluated size-agnostically on open models.

This repo is the home of a research line (multiple papers over time), sibling to [`llm_guardrail_security`](https://github.com/vacantfury/llm_guardrail_security) (black-box attacks/defenses on frozen models) and [`llm_agent_security`](https://github.com/vacantfury/llm_agent_security) (LLM-agent security). Where those lines study safety from *outside* a frozen model, this one asks how safety is carried — and can be strengthened — in the weights themselves.

Status (2026-08-02): first paper's direction committed; feasibility open. The measurement code is built — an encoding ladder of 15 exactly-invertible rungs, residual-stream capture, linear probes with control-task selectivity, and LLM-judge behaviour scoring — and the phase-0 pilot that gates the design is the next thing to run. See `text_docs/proposal.md` for the stage of record and `text_docs/project_structure.md` for what exists.

Setup: `uv sync`, then `uv run pytest` (hermetic — no network, no keys, no weights). Judge-scored measurements need `OPENAI_API_KEY`; see `.env.example` for the full variable list.
