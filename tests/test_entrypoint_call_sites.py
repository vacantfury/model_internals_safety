"""Every entrypoint's calls into the library must BIND against the live signature.

**Why this exists (2026-08-07).** `scripts/sae_pregate.py` called
`guard_working_tree()` with no arguments against a signature requiring `device`.
It killed all three tasks of job `8995805` twenty seconds in, after a queue wait,
on a run that had passed the approval gate.

Nothing caught it, and the reason is the thing worth fixing:

1. **969 tests were green.** No test invokes this script's `main()`.
2. **`--dry-run` passed and proved nothing.** The dry-run branch `return 0`s
   BEFORE the guard, so it exercises argument parsing and a cost printout and
   not one line of the real path. The approval gate is built on `--dry-run`,
   so a preset can be approved on evidence that touches none of the code.
3. **`build_status.py` reported the instrument "built, tested and reachable".**
   Reachability is an import-graph property; it cannot see a call that is wrong.

A real smoke test needs weights and a GPU, which is exactly the cost this is
avoiding. So this checks statically what the failure actually was — an argument
list that cannot bind — across every script, for the shared library functions an
entrypoint must get right. It is cheap, it needs no model, and it fails at
`pytest` time rather than after a queue wait.

Deliberately NOT a whole-program type check: the aim is the narrow, recurring
failure of a shared function whose signature moved out from under one caller.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from internals_safety import provenance
from internals_safety.measurements import deployment, length_null

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# The shared functions whose call sites an entrypoint must get right, mapped to
# the live object. Grow this when a signature change breaks a caller — that is
# the trigger, and this session hit it twice in one day (`strata` on
# `measure_deployment`, `device` on `guard_working_tree`).
WATCHED = {
    "guard_working_tree": provenance.guard_working_tree,
    "write_run_record": provenance.write_run_record,
    "capture_provenance": provenance.capture_provenance,
    "measure_deployment": deployment.measure_deployment,
    "length_strata": length_null.length_strata,
}


def call_sites(path: Path):
    """Every call to a WATCHED name in `path`, as (name, node)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in WATCHED:
            yield name, node


def script_paths() -> list[Path]:
    # rglob, not glob: `completion.py` was fixed the same way after a subdir of
    # scripts went unseen. A checker that misses files reports false health.
    return sorted(p for p in SCRIPTS.rglob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", script_paths(), ids=lambda p: p.name)
def test_every_watched_call_binds_to_the_live_signature(path: Path):
    for name, node in call_sites(path):
        signature = inspect.signature(WATCHED[name])
        positional = [object()] * len(node.args)
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # *args unpacking: arity is not statically known
        keywords = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
        if any(kw.arg is None for kw in node.keywords):
            continue  # **kwargs unpacking: likewise
        try:
            signature.bind(*positional, **keywords)
        except TypeError as error:
            raise AssertionError(
                f"{path.name}:{node.lineno} calls {name}() in a way that cannot bind "
                f"to {name}{signature}: {error}. This is the class of defect that "
                f"costs a queue cycle — job 8995805 died on exactly it."
            ) from error


def test_the_watchlist_actually_matches_something():
    """A watchlist that matches nothing passes vacuously and looks like health.

    The same failure mode as the coverage sweep measuring absence against a
    narrow index: a green check over an empty set is not evidence.
    """
    found = {name for path in script_paths() for name, _ in call_sites(path)}
    missing = sorted(set(WATCHED) - found)
    assert not missing, f"watched but never called in scripts/: {missing}"
