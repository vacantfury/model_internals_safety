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
        self.system_messages: list[str | None] = []

    def batch_chat(self, conversations, system_message=None):
        self.system_messages.append(system_message)
        out = []
        for pid, messages in conversations:
            text = messages[0][0]
            self.seen[pid] = text
            out.append((pid, self.replies.get(pid, self.default)))
        return out


def yes_verdict(reasoning: str = "r") -> str:
    return f'<json>{{"answer": "yes", "reasoning": "{reasoning}"}}</json>'


def no_verdict(reasoning: str = "r") -> str:
    return f'<json>{{"answer": "no", "reasoning": "{reasoning}"}}</json>'
