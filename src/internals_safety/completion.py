"""Is the instrument build done? — one command instead of a manual sweep.

**Why this exists.** The question "is the build all done?" was asked twice, and
both times answering it meant sweeping code, tests and three canonical docs by
hand. The second sweep got the orphan count wrong. That is formalization debt of
the ordinary kind: a fact that has to be re-derived every time it is needed,
where re-derivation is error-prone.

**The design rule that keeps it from rotting: anything derivable is DERIVED,
never declared.** A hand-maintained status table is a second source of truth that
drifts the moment someone builds something and forgets to update it — which is
exactly the failure being fixed, one level up. So this module declares only the
ROSTER (what we intend to build, a judgment) and reads everything else off the
tree: whether a module exists, whether any entrypoint can reach it, whether a
config knob is still a placeholder.

`tests/test_completion.py` then asserts the declared set against the derived one,
so a new instrument that nobody added to the roster fails the build in the same
way a new orphan does.

**Scope: BUILD state, not claim state.** Whether a given number may appear in a
paper is a per-run property and the contract already answers it — `reportable`
per `Reading`, with `withheld` in every run record. An instrument can be finished
and still produce withheld readings all day; conflating the two would let "the
build is done" mean "the numbers are good", which is a much stronger claim.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from internals_safety.paths import PROJECT_ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONF_DIR = PROJECT_ROOT / "conf"


@dataclass(frozen=True)
class Item:
    """One thing the build plan says we intend to have."""

    key: str
    what: str
    # Every module that must exist and be reachable for this item to count as
    # built. Empty means the item is declared but not yet designed to a module,
    # which reads as NOT BUILT rather than as vacuously satisfied.
    modules: tuple[str, ...]
    note: str = ""
    # Set when an item's modules all exist and are wired but the item is still
    # NOT finished — I6 is the case: ablation, addition and the licensing gate
    # are built and reachable, while patching-based attribution is not written.
    #
    # ⚠️ This field exists because the first version of this module reported I6
    # as "wired" on the strength of module reachability while its own note said
    # otherwise. A completion check that can report done when it is not done is
    # worse than no check, so incompleteness is DECLARED and dominates every
    # derived signal.
    incomplete: str = ""


# The instrument roster, `instrument_build_plan.md` §3. Declared, because what we
# intend to build is a judgment no filesystem can report.
ROSTER: tuple[Item, ...] = (
    Item("I0", "capture spine", ("models.capture",)),
    Item("I1", "decode lens (Patchscopes)", ("measurements.decode_lens", "models.patching")),
    Item("I2", "processing trajectory", ("measurements.trajectory",)),
    Item("I3", "entropy dynamics", ("measurements.entropy_dynamics", "models.lens")),
    Item("I4", "SAE features", (), "not started; needs SAELens + an Instruct-reconstruction pre-gate"),
    Item("I5", "reply-inversion causal test", ("measurements.reply_inversion",)),
    Item(
        "I6",
        "causal toolkit",
        ("models.interventions", "measurements.causal", "measurements.causal_license"),
        incomplete="ablation, addition and the licensing gate are wired; "
        "patching-based attribution is not written",
    ),
)

# Measurement modules that are deliberately NOT roster instruments, each with the
# reason. Declared so that `tests/test_completion.py` can assert the roster plus
# this set covers `measurements/` exactly — which is what turns "someone built an
# instrument and forgot the manifest" into a failing test rather than a silently
# optimistic status report.
NOT_ROSTER: dict[str, str] = {
    "ability": "measurement #1 — the pilot's original four, predating the I-roster",
    "behavior": "measurement #4 — likewise",
    "deployment": "measurement #2 — likewise",
    "recognition": "measurement #3 — likewise",
    "contract": "the instrument contract itself, not an instrument",
    "regimes": "combination logic: per-prompt axes -> a four-regime cell",
    "guard_regimes": "combination logic, AS-6 side",
}

# The mandatory control battery, `instrument_build_plan.md` §4.
CONTROLS: tuple[Item, ...] = (
    Item("length-null", "raw character length", ("measurements.length_null",)),
    Item("black-box", "surface classifier (P4)", ("measurements.black_box_baseline",)),
    Item("lexical", "XSTest vocabulary decorrelation", ("measurements.lexical_decorrelation",)),
    Item("ability", "mismatched-plaintext derangement", ("measurements.ability_control",)),
    Item("random-direction", "matched-norm null", ("measurements.causal_license",)),
    Item("pairing", "derangement primitives", ("pairing",)),
)


def python_modules() -> set[str]:
    """Every library module in the package, dotted and relative to it."""
    return {
        str(path.relative_to(PACKAGE_ROOT)).removesuffix(".py").replace("/", ".")
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    }


def _internal_imports(path: Path) -> set[str]:
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
            if (PACKAGE_ROOT / (candidate.replace(".", "/") + ".py")).exists():
                found.add(candidate)
    return found


def reachable_modules() -> set[str]:
    """Every library module reachable by imports from any script.

    One home for this walk: `tests/test_package_structure.py` had it first and
    now imports it from here, because two copies of a reachability rule is how
    the orphan guard and the completion check start disagreeing about what
    "wired" means.
    """
    seen: set[str] = set()
    queue: deque[str] = deque()
    for script in SCRIPTS_DIR.glob("*.py"):
        queue.extend(_internal_imports(script))
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        path = PACKAGE_ROOT / (module.replace(".", "/") + ".py")
        if path.exists():
            queue.extend(_internal_imports(path))
    return seen


@dataclass(frozen=True)
class ItemStatus:
    item: Item
    missing: tuple[str, ...]
    unwired: tuple[str, ...]

    @property
    def built(self) -> bool:
        """Declared modules all exist. An item with no modules is NOT built."""
        return bool(self.item.modules) and not self.missing

    @property
    def wired(self) -> bool:
        """Built, reachable from an entrypoint, and not declared incomplete.

        A module no run can reach is not an instrument yet, however well tested
        — the argument `pipeline_architecture.md` §1.4 made when it counted six
        of them. And a DECLARED incompleteness dominates reachability, because
        reachability cannot see a piece that was never written.
        """
        return self.built and not self.unwired and not self.item.incomplete

    @property
    def state(self) -> str:
        if self.wired:
            return "wired"
        if self.item.incomplete and self.built and not self.unwired:
            return "partial"
        if self.built:
            return "built-not-wired"
        if self.missing and len(self.missing) < len(self.item.modules):
            return "partial"
        return "not-built"


def status_of(item: Item, present: set[str], reachable: set[str]) -> ItemStatus:
    return ItemStatus(
        item=item,
        missing=tuple(m for m in item.modules if m not in present),
        unwired=tuple(m for m in item.modules if m in present and m not in reachable),
    )


def placeholder_knobs() -> list[str]:
    """Config values still marked PLACEHOLDER, with their file and line.

    Derived by reading the files rather than tracked in a list, so a knob that
    gets tuned stops being reported the moment its marker is removed — and one
    that gets ADDED starts being reported without anyone remembering to.

    **The marker is UPPERCASE and the match is case-sensitive**, which is not
    fussiness: the first version matched case-insensitively and reported twelve
    knobs, five of which were prose about *string* placeholders (`{prompt}`,
    `response_placeholder`). A check that over-reports is as useless as one that
    under-reports, because both end in the reader ignoring it.

    Known limit, stated rather than hidden: this counts MARKED knobs. An untuned
    value nobody marked is invisible here, and the defence against that is the
    tuning-path law at the point a knob is introduced, not this function.
    """
    found: list[str] = []
    marker = re.compile(r"\bPLACEHOLDER\b")
    sources = sorted(CONF_DIR.glob("*.yaml")) + [PACKAGE_ROOT / "config.py"]
    for path in sources:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if marker.search(line):
                found.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    return found


@dataclass(frozen=True)
class BuildStatus:
    roster: tuple[ItemStatus, ...]
    controls: tuple[ItemStatus, ...]
    placeholders: tuple[str, ...]

    @property
    def outstanding(self) -> list[str]:
        """Everything standing between here and a finished instrument layer."""
        reasons: list[str] = []
        for status in self.roster + self.controls:
            if status.state != "wired":
                detail = (
                    status.item.incomplete
                    or status.item.note
                    or ", ".join(status.missing + status.unwired)
                )
                reasons.append(f"{status.item.key} ({status.state}): {detail}")
        if self.placeholders:
            reasons.append(
                f"{len(self.placeholders)} placeholder knob(s) untuned — "
                "no reported number may depend on them"
            )
        return reasons

    @property
    def complete(self) -> bool:
        return not self.outstanding


def build_status() -> BuildStatus:
    present, reachable = python_modules(), reachable_modules()
    return BuildStatus(
        roster=tuple(status_of(item, present, reachable) for item in ROSTER),
        controls=tuple(status_of(item, present, reachable) for item in CONTROLS),
        placeholders=tuple(placeholder_knobs()),
    )
