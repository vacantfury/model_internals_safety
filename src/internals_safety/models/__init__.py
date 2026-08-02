"""Model loading and forward-pass instrumentation — the spine of this project.

The sibling repo (`llm_guardrail_security`) contributes encoders and judges but
no machinery here: it imports `torch` in exactly one file and never opens a
model. Everything in this subpackage is new.
"""
