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
