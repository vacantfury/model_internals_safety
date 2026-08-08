"""Stub judge service, shared by the judge and behaviour tests.

Not a conftest fixture because the behaviour tests build several services per
test and need the class itself, not one instance.
"""

from __future__ import annotations


class StubService:
    """Stands in for an llm_utils service. Records the prompts it was sent."""

    def __init__(self, replies: dict[str, str] | None = None, default: str = ""):
        self.replies = replies or {}
        self.default = default
        self.seen: dict[str, str] = {}
        # EVERY call, in order — `seen` is keyed by prompt id and is therefore
        # last-write-wins, which silently DROPS observations as soon as one run
        # judges two arms with the same positional ids. That is what the benign
        # arm becoming mandatory (TODO 61) exposed: a test asserting "the judge
        # saw the plaintext" started failing because the benign arm had
        # overwritten the harmful arm's entries, not because anything regressed.
        #
        # A stub that loses calls cannot answer "what was this judge sent", so
        # it must record all of them. `seen` is kept as-is for the tests that
        # legitimately want the last verdict per id.
        self.calls: list[tuple[str, str]] = []
        self.system_messages: list[str | None] = []

    def batch_chat(self, conversations, system_message=None):
        self.system_messages.append(system_message)
        out = []
        for pid, messages in conversations:
            text = messages[0][0]
            self.seen[pid] = text
            self.calls.append((pid, text))
            out.append((pid, self.replies.get(pid, self.default)))
        return out


def yes_verdict(reasoning: str = "r") -> str:
    return f'<json>{{"answer": "yes", "reasoning": "{reasoning}"}}</json>'


def no_verdict(reasoning: str = "r") -> str:
    return f'<json>{{"answer": "no", "reasoning": "{reasoning}"}}</json>'
