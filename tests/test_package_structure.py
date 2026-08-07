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
from internals_safety.completion import reachable_modules

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
    "ability_control",
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
    # Predates the contract; H4 overlap metric, used from the docs not the spine.
    "probes.overlap",
    # I4's Instruct-reconstruction pre-gate. Built and tested against SAE test
    # doubles, but nothing can call it until an SAE LOADER exists — the gate
    # takes a dictionary and we have no way yet to produce a real one (SAELens /
    # Llama Scope, TODO 55). Deliberately built first anyway: the build plan's
    # own sequencing is "pre-gate FIRST, then features", because the pre-gate can
    # REFUSE the whole instrument and doing the features first would risk
    # building on a dictionary that does not transfer to Instruct.
    "measurements.sae_reconstruction",
}

# `measurements.lexical_decorrelation` left this list on 2026-08-06, and its
# stated reason turned out to be measurably wrong rather than merely discharged.
# TODO 41 deferred the wiring because it "means capturing a third prompt set PER
# RUNG — 450 more forward passes per (model, rung)". The probe it controls is
# fitted on the PLAIN contrast sets (`measure_deployment`: "the probe is never
# refitted on the encoded condition"), so it is the same probe for every rung and
# XSTest is plain text: one capture serves the whole ladder. 450 passes, not
# 450 x n_families. The per-rung part is a logistic fit over activations already
# in hand, which costs no GPU at all.
#
# `measurements.causal_license` and `models.interventions` left this list on
# 2026-08-06 when `measurements/causal.py` and `run_causal_gate` landed. Their
# stated reason for being orphans — "gated on the matched-norm random-direction
# control, which is not built" — was discharged: the control is built, drawn from
# the same anchor cell, and priced in `--dry-run` alongside the real sweep.
#
# I1 (`measurements.decode_lens` + `models.patching`) and I3
# (`measurements.entropy_dynamics` + `models.lens`) left this list on 2026-08-06
# when `--instruments` landed. They are reachable but OFF by default, because
# both add GPU work and turning them on silently would change what the approval
# gate is approving. `Plan.describe` costs them separately — extra forward passes
# for I1, lens readouts for I3 — so a --dry-run states the delta before anyone
# approves a run.


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
