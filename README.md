# model_internals_safety

Safety research **inside model parameters**: training-time interventions (safety fine-tuning, unlearning) and parameter/activation-space work (activation steering, representation-level analysis and edits), evaluated size-agnostically on open models.

This repo is the home of a research line (multiple papers over time), sibling to [`llm_guardrail_security`](https://github.com/vacantfury/llm_guardrail_security) (black-box attacks/defenses on frozen models) and [`llm_agent_security`](https://github.com/vacantfury/llm_agent_security) (LLM-agent security). Where those lines study safety from *outside* a frozen model, this one asks how safety is carried — and can be strengthened — in the weights themselves.

Status: founding (2026-08-02); first paper in ideation — see `text_docs/proposal.md`.
