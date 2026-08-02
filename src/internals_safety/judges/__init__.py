"""LLM-judge classifiers behind measurement #4 (behaviour).

Copied from `llm_guardrail_security/src/evaluation` per
`text_docs/project_structure.md` §5 — copied, never imported.

Two judges land at build step 3, exactly the two measurement #4 needs: ASR
(HarmBench) and refusal (JailbreakBench). Two more exist in the sibling and are
deliberately NOT copied yet, each with its trigger named:

* **WildGuard** — the family's response-harm panel judge. It belongs to the
  external-guard-gap table (s1 §7 baselines: "WildGuard / Llama Guard 3 on
  identical payloads"), which is phase-3 work, and it needs both a served
  WildGuard checkpoint and the sibling's `defense/guard_utils.format_wildguard_input`
  — a file the copy manifest excludes. Copy it when the guard-gap table is built.
* **OR-Bench** — a 3-class over-refusal judge for the over-refusal battery
  (XSTest / EVOREFUSE-TEST / MORBench), also phase 3. Copy it with that battery.
"""

from internals_safety.judges.base import Judge, JudgeItem, Verdict
from internals_safety.judges.harmbench import HarmBenchJudge
from internals_safety.judges.parsing import parse_judge_response
from internals_safety.judges.refusal import RefusalJudge

__all__ = [
    "HarmBenchJudge",
    "Judge",
    "JudgeItem",
    "RefusalJudge",
    "Verdict",
    "parse_judge_response",
]
