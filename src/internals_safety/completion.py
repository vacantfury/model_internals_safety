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

# A config key: `name: value` in YAML, `name: type = value` in a dataclass. One
# pattern for both, because at the point that matters they are the same shape.
_KEY = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*:")
# How far below a marker to look for the key it annotates. Comment blocks here
# run long — the ability-0 cut's is 24 lines — but a marker separated from its
# key by more than this is unreadable to a human too.
# plumbing: a scan window in the build-status REPORT; reaches no measurement, and
# a marker it fails to resolve is reported under a `file:line` key rather than
# dropped, so widening or narrowing it cannot silently lose a knob.
_KEY_SEARCH_LINES = 30


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
    Item(
        "I4",
        "SAE features",
        ("measurements.sae_reconstruction", "models.sae_loader"),
        incomplete="the pre-gate is ANSWERED across EIGHT layers (jobs 9049084/9049085, "
        "9065751/9065802, 9066418/9066464; 2026-08-10) and the dictionary reads "
        "chat-templated Instruct activations at exactly ONE of them: Scope 22, "
        "min(variance ratio, KL ratio) = 0.810. Every other probed layer fails one "
        "term or the other — variance falls monotonically with depth while KL does "
        "NOT rise monotonically (Scope 24 dips 0.28 below both neighbours). The "
        "FEATURE instrument is not built yet because the 0.810 clears a bar "
        "(min_transfer_ratio = 0.800) that is still a PLACEHOLDER, so the license "
        "rests on an unearned number; deriving that knob from what I4 actually needs "
        "is the binding precondition (TODO 70). This item stays [PART] because the "
        "roster asks what is BUILT, and a gate naming the one layer it could be built "
        "at does not build it. See instrument_layer.md §4.8.2",
    ),
    Item("I5", "reply-inversion causal test", ("measurements.reply_inversion",)),
    Item(
        "I6",
        "causal toolkit",
        (
            "models.interventions",
            "models.patching",
            "measurements.causal",
            "measurements.causal_license",
            "measurements.attribution",
        ),
    ),
    # Added 2026-08-09. A roster ITEM rather than an I6 module, by I5's precedent:
    # I6 is the causal machinery, I5 and this are specific causal TESTS built on
    # it. Its question is one no other instrument answers — whether encoding
    # DESTROYS harm recognition or merely SUPPRESSES its expression, which no
    # paired-arm refusal rate can separate because both predict the same rates.
    Item(
        "I7",
        "encoding-direction ablation",
        ("measurements.encoding_direction",),
        # Complete as of 2026-08-09. The end-to-end gap declared here is closed by
        # tests/test_encoding_ablation_entrypoint.py, written after job 9032777
        # died on an H200 in 76 seconds — exactly the outcome this entry had
        # predicted in writing and shipped anyway. Worth keeping as a record of
        # what the declaration was FOR: it named the risk correctly and naming it
        # is not mitigating it. Writing the test then found two more defects the
        # crash had been hiding behind it (a None harm judge that would have
        # crashed one line later, and a degenerate direction that took the whole
        # run down instead of marking one cell unmeasured).
    ),
)

# Measurement modules that are deliberately NOT roster instruments, each with the
# reason. Declared so that `tests/test_completion.py` can assert the roster plus
# this set covers `measurements/` exactly — which is what turns "someone built an
# instrument and forgot the manifest" into a failing test rather than a silently
# optimistic status report.
NOT_ROSTER: dict[str, str] = {
    "intervals": (
        "not an instrument — the uncertainty estimators every reported rate and "
        "contrast is quoted with. Landed 2026-08-22 when the external review's "
        "most-repeated objection turned out to be that whole tables carried point "
        "estimates. It measures nothing and licenses nothing: it takes counts and "
        "returns bounds, which is why it sits outside both the roster and the "
        "control battery. It is also the one home of the confidence level, derived "
        "from `measurements.probes.alpha` rather than written as a literal"
    ),
    "ability": "measurement #1 — the pilot's original four, predating the I-roster",
    "behavior": "measurement #4 — likewise",
    "refusal": (
        "measurement #4's DISCRIMINATION half, split out 2026-08-09 (TODO 64) — not an "
        "I-roster instrument but the one the papers actually report. `behavior`'s value is "
        "ASR, unreportable repo-wide since instrument_layer §3.5.2, so the contract governed "
        "a number no paper will print and was silent on the refusal gap legs 1 and 2 are "
        "made of. Separate module because P1 demands a distinct question: `behavior` asks "
        "whether the attack succeeded, this asks whether refusal carries information about "
        "harm — a model can score perfectly on the first and zero on the second"
    ),
    "deployment": "measurement #2 — likewise",
    "recognition": "measurement #3 — likewise",
    "contract": "the instrument contract itself, not an instrument",
    "regimes": "combination logic: per-prompt axes -> a four-regime cell",
    "guard_regimes": "combination logic, AS-6 side",
    "control_floor": "the control-calibrated floor DERIVATION shared by the screens — a calibration rule every instrument is judged against, not an instrument itself",
    "floor_witness": (
        "control_floor's sibling, and the same category: not an instrument but the rule "
        "deciding WHICH run's floor may legitimately judge a reading. A measurement "
        "present in two runs is one measurement, so a run carrying control rungs can "
        "screen a reading taken in a run that carries none — which is how AS-5's "
        "internals leg cleared §2.5's permutation-only gap with no job. It is listed "
        "here rather than as a control because it defeats no confound of its own; it "
        "gates the pairing that lets a real control floor reach a number "
        "(instrument_layer.md §2.10)"
    ),
}

# The mandatory control battery, `instrument_build_plan.md` §4.
CONTROLS: tuple[Item, ...] = (
    Item("length-null", "raw character length", ("measurements.length_null",)),
    Item("black-box", "surface classifier (P4)", ("measurements.black_box_baseline",)),
    Item("lexical", "XSTest vocabulary decorrelation", ("measurements.lexical_decorrelation",)),
    Item("ability", "mismatched-plaintext derangement", ("measurements.ability_control",)),
    Item("judge", "benign-arm judge control (#4)", ("measurements.behavior_control",)),
    # Added 2026-08-07. The battery screened the ASR axis twice and the REFUSAL
    # axis not at all — while refusal is what (B) is split from (S) on, so the
    # paper's contribution rested on the unscreened one.
    Item("refusal-judge", "paired echo-flip control (62b)", ("measurements.refusal_control",)),
    # Added 2026-08-08. The battery screened the TARGET's judges and left the
    # GUARD's own verdict unscreened — while `blocked` is what AS-6's central
    # cell is split on, so the guard paper's contribution rested on the
    # unscreened axis. Same sentence as the line above, one object of study over.
    Item(
        "guard-benign",
        "benign-arm guard control (AS-6)",
        ("measurements.guard_benign_control",),
    ),
    # Added 2026-08-09 (TODO 65). The scaffold ARM below is a corpus
    # construction shared by both papers; this is the guard-side SCORER that
    # reads it, and they are separate battery entries because the target side
    # scores the arm through `refusal`/`behavior` while the guard has no such
    # path — its verdict is its own instrument.
    #
    # Distinct from `guard-benign` and not a stronger version of it: that floor
    # is the ENCODED benign arm and defeats "flags anything wearing the
    # encoding"; this floor is the SCAFFOLD benign arm and defeats "flags
    # anything asking about an encoding". A guard can clear the first and fail
    # this one.
    Item(
        "guard-scaffold",
        "attack-wrapper guard control (AS-6)",
        ("measurements.guard_scaffold_control",),
    ),
    # Not a control against a confound — the combination layer for AS-5's
    # internals leg, listed here because the roster's job is to account for
    # every measurement module and an unlisted one is how a built instrument
    # goes unreachable. Its own screen (`internal_behavioural_dissociation`)
    # guards against claiming "represented but unread" from a probe too weak to
    # tell that apart from "not represented".
    Item(
        "dissociation",
        "internal-vs-behavioural comparison (AS-5 stage 0)",
        ("measurements.dissociation",),
    ),
    Item("random-direction", "matched-norm null", ("measurements.causal_license",)),
    Item("pairing", "derangement primitives", ("pairing",)),
    # Added 2026-08-09 (`instrument_layer.md` §3.9). The battery screened the
    # judges, the probes and the directions, and left the PROMPT unscreened:
    # an encoded condition changes the payload's characters AND wraps them in a
    # template announcing an encoding, so every rate was reading their sum. The
    # factorial cell is plaintext content wearing the same scaffold, and on 2 of
    # 4 models the wrapper alone causes most of the discrimination loss.
    #
    # It lives in the SPINE rather than in `measurements/`, which is why its
    # module is `pipeline`: the arm is a corpus construction, not a scorer, and
    # putting it beside the entrypoints is what let the plain baseline go
    # missing from one script for a day.
    Item("scaffold", "attack-wrapper factorial arm", ("pipeline",)),
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

    **`rglob`, not `glob` — and the difference is a trap, not a preference
    (structure review 2026-08-07, `pipeline_architecture.md` §5).** With a flat
    `glob("*.py")`, the day anyone puts a script in `scripts/<subdir>/` every
    module reachable ONLY from there silently becomes an orphan: the orphan
    guard would report modules as unwired that are wired, and `build_status.py`
    would report instruments as unbuilt that are built. Both failures point the
    reader at the wrong thing, and neither raises.

    The review considered splitting `scripts/` into `entrypoints/` and
    `analysis/` and declined — eleven files do not need a directory, and the
    launchable set is already named by `config.Entrypoint`. But the trap is
    closed either way, because a walk that depends on a directory staying flat
    is a walk that breaks on a reorganisation nobody thinks to check.
    """
    seen: set[str] = set()
    queue: deque[str] = deque()
    for script in SCRIPTS_DIR.rglob("*.py"):
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


def _knob_key_below(lines: list[str], start: int) -> tuple[str, str] | None:
    """The config key a PLACEHOLDER marker at `start` is talking about.

    A marker sits in the comment block above its key, so the key is the next
    assignment line. Both surfaces are scanned with one rule because both are
    `name: value` at the point that matters — YAML mappings and annotated
    dataclass fields alike.

    Returns `(leaf, qualified)`. The caller decides which to use, and it must:
    **the leaf alone is not always an identity, and the qualified form is not
    always mergeable (2026-08-22).** `conf/cost.yaml` holds one
    `decode_tokens_per_s` PER GPU PROFILE, so marking a second card collapsed
    two different knobs into one name and tripped the marked-twice-in-one-file
    guard on a correct pair of markers. But qualifying everything is worse: a
    YAML knob nests under its section while its `config.py` fail-safe nests
    under a class, so the two surfaces would stop merging and the headline would
    double again, which is the exact over-report this function's history is
    about. Qualify only what collides inside one file.
    """
    for offset, line in enumerate(lines[start : start + _KEY_SEARCH_LINES]):
        match = _KEY.match(line)
        if not match:
            continue
        key = match.group(1)
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            return key, key
        # Walk back for the nearest shallower mapping key. Comments are skipped,
        # so the parent of a key buried under a comment block is still found.
        for above in reversed(lines[: start + offset]):
            if not above.strip() or above.lstrip().startswith("#"):
                continue
            parent = _KEY.match(above)
            if parent and (len(above) - len(above.lstrip())) < indent:
                return key, f"{parent.group(1)}.{key}"
        return key, key
    return None


def placeholder_knobs() -> dict[str, list[str]]:
    """Untuned config knobs, keyed by name, each with every place it is marked.

    Derived by reading the files rather than tracked in a list, so a knob that
    gets tuned stops being reported the moment its marker is removed — and one
    that gets ADDED starts being reported without anyone remembering to.

    **The marker is UPPERCASE and the match is case-sensitive**, which is not
    fussiness: the first version matched case-insensitively and reported twelve
    knobs, five of which were prose about *string* placeholders (`{prompt}`,
    `response_placeholder`). A check that over-reports is as useless as one that
    under-reports, because both end in the reader ignoring it.

    **Keyed by KNOB, not by marker, for exactly that reason (fixed 2026-08-06).**
    The second version reported 11 when there were 6: every knob is marked twice
    — once on the live YAML value and once on its fail-safe mirror in `config.py`
    — so counting markers inflated the headline ~2x. That is the same
    over-reporting defect as the case-insensitive match, one level subtler,
    caught by the owner asking whether the build was done. The mirrors are still
    LISTED under their knob, because a knob marked in one surface and not the
    other is a marking inconsistency worth seeing.

    Known limit, stated rather than hidden: this counts MARKED knobs. An untuned
    value nobody marked is invisible here, and the defence against that is the
    tuning-path law at the point a knob is introduced, not this function.
    """
    found: dict[str, list[str]] = {}
    per_file: list[tuple[Path, int, tuple[str | None, str | None]]] = []
    marker = re.compile(r"\bPLACEHOLDER\b")
    sources = sorted(CONF_DIR.glob("*.yaml")) + [PACKAGE_ROOT / "config.py"]
    for path in sources:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not marker.search(line):
                continue
            # `number` is 1-based and the key search starts at the NEXT line,
            # which is `lines[number]` 0-based — the marker's own line included
            # only when it IS the assignment (`trained_on: "PLACEHOLDER ..."`).
            key = _KEY.match(line)
            if key:
                names = (key.group(1), key.group(1))
            else:
                names = _knob_key_below(lines, number) or (None, None)
            per_file.append((path, number, names))

    # Qualify ONLY the leaf names that collide inside one file. Everything else
    # keeps its bare leaf so a YAML knob and its `config.py` fail-safe still
    # merge into one entry — see `_knob_key_below` for why both halves matter.
    colliding: set[tuple[Path, str]] = set()
    seen: dict[tuple[Path, str], int] = {}
    for path, _, (leaf, _qualified) in per_file:
        if leaf is None:
            continue
        seen[(path, leaf)] = seen.get((path, leaf), 0) + 1
        if seen[(path, leaf)] > 1:
            colliding.add((path, leaf))

    for path, number, (leaf, qualified) in per_file:
        name = qualified if (path, leaf) in colliding else leaf
        found.setdefault(name or f"{path.name}:{number}", []).append(
            f"{path.relative_to(PROJECT_ROOT)}:{number}"
        )
    return found


@dataclass(frozen=True)
class BuildStatus:
    roster: tuple[ItemStatus, ...]
    controls: tuple[ItemStatus, ...]
    # knob name -> every file:line marking it. Keyed by KNOB so the count is a
    # count of things needing tuning, not of comments mentioning them.
    placeholders: dict[str, list[str]]

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
        placeholders=placeholder_knobs(),
    )
