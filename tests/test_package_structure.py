"""Structural invariants of the measurement layer.

**Why a test rather than a package split.** `pipeline_architecture.md` §3.2
originally proposed splitting `measurements/` into `measurements/licensing/
combination/`, on the stated rationale that "a licensing rule change must not
require reading a measurement module, and today it does". Measured 2026-08-06,
that rationale was FALSE: the combination and licensing modules already import
no measurement sibling. The split would have moved five pure modules across six
scripts' import lines and bought a directory name.

What the split would genuinely have encoded is the *invariant* — that those
modules stay pure functions of plain values. A directory cannot check that and
this file can, which is why the proposal was struck in favour of the test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from internals_safety import measurements

MEASUREMENTS_DIR = Path(measurements.__file__).parent

# Modules that combine, control, or license measurements. They take plain floats
# and bools and return verdicts, so they must not reach back into the modules
# that produced those numbers — that direction is what makes them reusable by
# both papers and testable without a model.
PURE_MODULES = (
    "regimes",
    "guard_regimes",
    "length_null",
    "black_box_baseline",
    "lexical_decorrelation",
    "contract",
    "causal_license",
)

# The instruments. These MAY import `contract` — that direction is the point:
# an instrument declares its verdict in the shared type.
INSTRUMENTS = ("decode_lens", "trajectory", "entropy_dynamics")


def sibling_imports(module: str) -> set[str]:
    """Every `internals_safety.measurements.*` module `module` imports."""
    tree = ast.parse((MEASUREMENTS_DIR / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("internals_safety.measurements."):
                found.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("internals_safety.measurements."):
                    found.add(alias.name.rsplit(".", 1)[-1])
    return found


@pytest.mark.parametrize("module", PURE_MODULES)
def test_combination_and_licensing_modules_import_no_measurement_sibling(module):
    assert sibling_imports(module) == set(), (
        f"{module}.py imports a measurement sibling. It combines or licenses "
        "measurements and must stay a pure function of plain values — otherwise "
        "a licensing change starts requiring a measurement module to be read, "
        "which is the coupling §3.2 was written to prevent."
    )


@pytest.mark.parametrize("module", INSTRUMENTS)
def test_instruments_depend_only_on_the_contract(module):
    """The allowed direction, asserted so it cannot quietly become a web."""
    assert sibling_imports(module) <= {"contract"}


def test_the_contract_is_the_layer_s_sink():
    """Nothing in the layer may import an instrument.

    An instrument is a leaf. If a combination module imported one, the roster
    would stop being swappable and every paper would inherit every instrument.
    """
    for module in PURE_MODULES + INSTRUMENTS:
        assert not sibling_imports(module) & set(INSTRUMENTS)


# Modules not yet reachable from any entrypoint, each with the reason it is not.
# **This list may only ever SHRINK.** `pipeline_architecture.md` §1.4 counted six
# orphans — 97 tests, all green, none reachable from a run — as one of the four
# problems the architecture pass existed to fix. A list is what stops that
# recurring silently: a new orphan fails this test at the moment it is created,
# rather than being noticed in an audit weeks later.
DECLARED_ORPHANS = {
    # I5/I6. Causal write operations; gated on the matched-norm random-direction
    # control, which is not built. Wiring them before that control exists would
    # let "steering worked" mean "perturbing anything worked".
    "measurements.causal_license",
    "models.interventions",
    # Predates the contract; H4 overlap metric, used from the docs not the spine.
    "probes.overlap",
    # The XSTest control. Pure scoring, fully tested — but wiring it needs the
    # probe READ on XSTest activations, which means capturing a third prompt set
    # per rung. That changes what a run costs, so it goes behind --instruments
    # with its own dry-run line rather than in by default (TODO 41).
    "measurements.lexical_decorrelation",
}

# I1 (`measurements.decode_lens` + `models.patching`) and I3
# (`measurements.entropy_dynamics` + `models.lens`) left this list on 2026-08-06
# when `--instruments` landed. They are reachable but OFF by default, because
# both add GPU work and turning them on silently would change what the approval
# gate is approving. `Plan.describe` costs them separately — extra forward passes
# for I1, lens readouts for I3 — so a --dry-run states the delta before anyone
# approves a run.


def reachable_modules() -> set[str]:
    """Every library module reachable by imports from any script."""
    from collections import deque

    package = Path(measurements.__file__).parent.parent

    def deps(path: Path) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("internals_safety"):
                continue
            module = node.module.removeprefix("internals_safety").lstrip(".")
            if module:
                found.add(module)
            for alias in node.names:
                candidate = f"{module}.{alias.name}" if module else alias.name
                if (package / (candidate.replace(".", "/") + ".py")).exists():
                    found.add(candidate)
        return found

    seen: set[str] = set()
    queue: deque[str] = deque()
    for script in (package.parent.parent / "scripts").glob("*.py"):
        queue.extend(deps(script))
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        path = package / (module.replace(".", "/") + ".py")
        if path.exists():
            queue.extend(deps(path))
    return seen


def test_no_module_is_an_orphan_except_the_declared_ones():
    """Every built module is reachable from a run, or declared with its reason.

    An instrument that no entrypoint can reach is not an instrument yet, however
    well tested. This is the check that turns that from a periodic audit finding
    into a build failure.
    """
    package = Path(measurements.__file__).parent.parent
    built = {
        str(path.relative_to(package)).removesuffix(".py").replace("/", ".")
        for path in package.rglob("*.py")
        if path.name != "__init__.py"
    }
    orphans = built - reachable_modules()
    assert orphans == DECLARED_ORPHANS, (
        f"orphan set changed: newly unreachable {sorted(orphans - DECLARED_ORPHANS)}, "
        f"newly wired {sorted(DECLARED_ORPHANS - orphans)}. If a module was wired, "
        "delete it from DECLARED_ORPHANS; if one was added, wire it or declare why."
    )
