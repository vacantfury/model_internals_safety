"""Prompt-set loading.

The corpus lives in `data/` (gitignored), one JSONL record per prompt, in the
schema the guardrail sibling uses — `{id, category, source, prompt}` — because
the four sets were copied from it (`text_docs/project_structure.md` §5) and
diverging the schema would break the payload-identity the guard-gap table needs
later.

Every loader returns prompts in **file order** and takes `limit` as a prefix,
never a sample. A random subset would make two runs with the same seed but
different `n_prompts` incomparable, and the phase-0 pilot's whole job is to be
compared against later, larger runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from internals_safety.paths import DATA_DIR


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    category: str = ""
    source: str = ""


def load_prompts(path: Path, limit: int | None = None) -> list[Prompt]:
    """Read a prompt JSONL. `limit` takes a prefix — see the module docstring."""
    if not path.exists():
        raise FileNotFoundError(
            f"no prompt set at {path}. The four phase-0 sets are copied from the "
            "guardrail sibling per text_docs/project_structure.md §5; data/ is "
            "gitignored, so a fresh clone has to copy them again."
        )
    prompts: list[Prompt] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record:
                raise ValueError(f"{path}:{line_number} has no 'prompt' field")
            prompts.append(
                Prompt(
                    id=str(record.get("id", line_number)),
                    text=record["prompt"],
                    category=record.get("category", ""),
                    source=record.get("source", ""),
                )
            )
            if limit is not None and len(prompts) >= limit:
                break
    if limit is not None and len(prompts) < limit:
        raise ValueError(f"{path} holds {len(prompts)} prompts, fewer than the requested {limit}")
    return prompts


def prompt_set(name: str, limit: int | None = None, data_dir: Path = DATA_DIR) -> list[Prompt]:
    """Load a set by filename, e.g. `jbb_prompts.jsonl`."""
    return load_prompts(data_dir / name, limit=limit)


def digest(prompts: list[Prompt]) -> str:
    """Content hash of a prompt list — the activation cache key and run-record id.

    Over ids *and* text: an id-only digest would silently reuse a stale cache if
    a corpus file were edited in place, and the activations are the one artifact
    expensive enough that a silent stale hit would go unnoticed.
    """
    hasher = hashlib.sha256()
    for prompt in prompts:
        hasher.update(prompt.id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(prompt.text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]
