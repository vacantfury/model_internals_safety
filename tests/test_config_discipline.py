"""Every number in the code declares why it is not in YAML.

**Science ruling 2026-08-06** (science TODO 17; law: science
`handbook/instrument_building.md`), after the owner caught the YAML-config law
being violated across a whole build session: enforcement is a family standard
and it is a TEST, not more prose. The workflow's S7 already *cited* the law and
the law was violated anyway. This repo builds the reference implementation;
other research repos copy the file (no shared code across research bets).

## What it catches, and both are real history here

1. **An unconfigured tunable.** `CONTROL_ABILITY_MAX = 0.02`, `N_LAYERS_ASSUMED
   = 32`, `DEFAULT_PERCENTILES` — three cuts with no config home, each of which
   turned out to be substantively wrong, not merely misplaced. They carry no
   marker, so they would fail here.
2. **A second copy of a configured value.** `DEFAULT_LENGTH_BINS = 10` was a
   duplicate of `probes.length_strata_bins`, already in YAML — two
   independently-editable copies of one setting, and the moment either moved, a
   "length-matched" claim would silently mean two different things. A
   `# config:` marker does not merely *assert* the mirror: this test resolves
   the named key and **compares the values**, so the copy cannot drift.

Every real defect in this repo's history was a MODULE-LEVEL constant. Signature
defaults are covered because the ruling named them, not because one has bitten
us yet.

## The markers

One per numeric site, in the comment block immediately above it or trailing on
its own line. Signature defaults name their parameter: `# config(batch_size):`.

    # config: <yaml.key>        a fail-safe default mirroring YAML. The key is
                                resolved and the values COMPARED. Several keys,
                                comma-separated, mirror a tuple.
    # derived: <from what>      the live value is computed from config or data at
                                every real call site, so no scalar key exists to
                                mirror; the literal is a standalone fail-safe.
    # constant: <why>           a fact about the world, a spec, a unit conversion
                                or a mathematical definition — 26 letters in the
                                Latin alphabet, a Unicode block, 3600 s/h.
    # definitional: <what>      OUR choice, defining the quantity rather than
                                tuning it, kept in code so the measure stays
                                computable with no config in hand. Must ALSO name
                                a tuning path: this is the marker most able to
                                launder a cut, and the law requires every
                                heuristic number to carry one.
    # plumbing: <why>           cannot change any reported number — a batch size,
                                a display precision, a counter's zero, an
                                off-by-default. The claim is strong on purpose:
                                if it CAN move a result it is not plumbing.

Five, settled against the real 34-site population rather than in the abstract.
The ruling named three; `derived` and `plumbing` were forced by sites that fit
none of them, and `unit` folded into `constant` because nothing turned on the
distinction.

**Why a structured comment and not a naming prefix** (the ruling left the
convention to this build): a prefix cannot carry WHICH key a mirror mirrors, and
that resolution is the only thing that would have caught the length-bins
duplicate. A prefix would also have forced renames on names that already read
well (`SECONDS_PER_HOUR` → `CONST_SECONDS_PER_HOUR`).

## Scope

The whole package plus `scripts/`, which is WIDER than the ruling's "measurement
package + the scripts that write records". Deliberate: `encodings/recovery.py`
holds `_TOKEN_MATCH_RATIO`, a cut the ability measurement reports through, and
it sits in `encodings/`. A scope drawn by directory would have excluded it.
Tests are excluded — fixture numbers are the test's own data.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from internals_safety.config import load_measurements_config, load_model_config
from internals_safety.cost import load_cost_config
from internals_safety.paths import PROJECT_ROOT

PACKAGE_ROOT = PROJECT_ROOT / "src" / "internals_safety"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

MARKER = re.compile(
    r"#\s*(config|derived|constant|definitional|plumbing)"
    r"(?:\((?P<param>[A-Za-z_][A-Za-z0-9_]*)\))?\s*:\s*(?P<detail>.+)"
)
# `definitional` is the marker most able to excuse a genuine cut, so it carries
# the extra duty the tuning-path law already imposes: name how it would be tuned.
TUNING_PATH = re.compile(r"tuning path", re.IGNORECASE)


def source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py")) + sorted(SCRIPTS_DIR.glob("*.py"))


def is_numeric(node: ast.AST) -> bool:
    """A numeric literal, a signed one, or a homogeneous tuple/list of them.

    Booleans are excluded — `True` is an int in Python and a flag is not a knob.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_numeric(node.operand)
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(is_numeric(element) for element in node.elts)
    return False


@dataclass(frozen=True)
class Site:
    path: Path
    line: int
    name: str
    value: ast.AST
    param: str | None = None  # set for signature defaults
    # The enclosing `def`'s line, so a marker above a long signature reaches a
    # default several lines inside it.
    owner_line: int | None = None

    def __str__(self) -> str:
        return f"{self.path.relative_to(PROJECT_ROOT)}:{self.line} {self.name}"


def is_config_schema(node: ast.ClassDef) -> bool:
    """A pydantic config model — its field defaults ARE the schema.

    Exempt because `conf/*.yaml` is their source of truth by construction and
    `StrictModel` already rejects unknown keys. Requiring a `# config:` marker on
    `ProbeConfig.seed = 0` would be asking the config to declare that it is the
    config.
    """
    return any(
        isinstance(base, ast.Name) and base.id in {"StrictModel", "BaseModel"}
        for base in node.bases
    ) or any(
        isinstance(base, ast.Name) and base.id.endswith("Config") for base in node.bases
    )


def numeric_sites(path: Path) -> list[Site]:
    """Numeric constants at module level, in class bodies, and as defaults.

    Class-body defaults are included because a dataclass field default IS a
    signature default — `@dataclass` generates `__init__` from it — and `Plan`
    carries four of them straight into the cost estimate the approval gate reads.
    Leaving them out would have shipped this test with a hole the same shape as
    the defect it exists to catch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sites: list[Site] = []

    def collect_assignments(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Assign) and is_numeric(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        sites.append(Site(path, node.lineno, target.id, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if is_numeric(node.value) and isinstance(node.target, ast.Name):
                    sites.append(Site(path, node.lineno, node.target.id, node.value))

    collect_assignments(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not is_config_schema(node):
            collect_assignments(node.body)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = args.args[len(args.args) - len(args.defaults):]
        pairs = list(zip(positional, args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for arg, default in pairs:
            if is_numeric(default):
                sites.append(Site(
                    path, default.lineno, f"{node.name}({arg.arg}=)", default,
                    param=arg.arg, owner_line=node.lineno,
                ))
    return sites


def comment_block_above(lines: list[str], line: int) -> list[re.Match]:
    """Markers in the contiguous comment block ending just above `line`."""
    found: list[re.Match] = []
    index = line - 2
    while index >= 0 and lines[index].strip().startswith("#"):
        if (match := MARKER.search(lines[index])) is not None:
            found.append(match)
        index -= 1
    return found


def justification_text(path: Path, site: Site) -> str:
    """Every comment line governing `site`, joined.

    A justification is normally several lines — the marker opens it and the
    reasoning follows. Reading only the marker's own line would force real
    arguments onto one line to satisfy a checker, which is how a check starts
    degrading the thing it checks.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = [lines[site.line - 1]]

    def block(end: int) -> None:
        index = end - 2
        while index >= 0 and lines[index].strip().startswith("#"):
            parts.append(lines[index])
            index -= 1

    block(site.line)
    if site.owner_line is not None:
        block(site.owner_line)
    return "\n".join(parts)


def markers_for(path: Path, site: Site) -> list[re.Match]:
    """Markers governing `site`.

    Three places, because a multi-line signature has two natural homes for one:
    trailing on the default's own line, in the comment block directly above it
    (inside the signature), or in the block above the enclosing `def`. The first
    version walked up from the default and stopped at the first ordinary
    argument line, so a marker above a long `def` never reached its own default —
    a checker whose annotations silently fail to attach is worse than none.

    One block may carry several markers when a `def` has several numeric
    defaults, which is why signature markers name their parameter.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    found: list[re.Match] = []

    if (trailing := MARKER.search(lines[site.line - 1])) is not None:
        found.append(trailing)
    found += comment_block_above(lines, site.line)
    if site.owner_line is not None:
        found += comment_block_above(lines, site.owner_line)

    if site.param is None:
        return [m for m in found if m.group("param") is None]
    return [m for m in found if m.group("param") == site.param]


def resolve_config_key(key: str):
    """Resolve a dotted key against the config a marker names.

    The root selects the file, so a marker says which YAML it mirrors as well as
    which field: `measurements.*`, `cost.*`, or `model.<name>.*`.
    """
    parts = key.split(".")
    root, rest = parts[0], parts[1:]
    if root == "measurements":
        current = load_measurements_config()
    elif root == "cost":
        current = load_cost_config()
    elif root == "model":
        current, rest = load_model_config(rest[0]), rest[1:]
    else:
        raise KeyError(f"unknown config root {root!r} in {key!r}")
    for part in rest:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            # Loud and specific: a marker naming a key that does not exist is a
            # mirror pointing at nothing, which is exactly as broken as a mirror
            # pointing at the wrong value.
            available = sorted(getattr(type(current), "model_fields", {}) or [])
            raise KeyError(
                f"{key}: no field {part!r} on {type(current).__name__}"
                + (f"; has {available}" if available else "")
            )
    return current


ALL_SITES = [site for path in source_files() for site in numeric_sites(path)]


def test_the_sweep_actually_finds_sites():
    """A scope bug that matched nothing would make every test below vacuous."""
    assert len(ALL_SITES) > 20


@pytest.mark.parametrize("site", ALL_SITES, ids=str)
def test_every_numeric_site_declares_why_it_is_not_in_yaml(site):
    assert markers_for(site.path, site), (
        f"{site}: numeric literal with no marker.\n"
        f"  If it is tunable it belongs in conf/*.yaml — that is the default answer "
        f"and it is almost always the right one.\n"
        f"  If it does not, say why on the line above or trailing:\n"
        f"    # config: measurements.probes.some_key      (and the values are compared)\n"
        f"    # constant: <a fact about the world or a spec>\n"
        f"    # unit: <a unit conversion>\n"
        f"    # derived: <what the live value is computed from>\n"
        f"    # definitional: <what it defines> ... tuning path: <how it would be tuned>\n"
        f"    # plumbing: <why it cannot change any reported number>\n"
        + (f"  Signature defaults name their parameter: # config({site.param}): ...\n"
           if site.param else "")
    )


@pytest.mark.parametrize(
    "site", [s for s in ALL_SITES if any(m.group(1) == "config" for m in markers_for(s.path, s))],
    ids=str,
)
def test_a_declared_mirror_matches_the_value_it_mirrors(site):
    """⚠️ THE ONE THAT WOULD HAVE CAUGHT `DEFAULT_LENGTH_BINS`.

    A mirror is a second copy of a value, and this is what stops the two
    drifting apart in silence. Asserting the marker exists would not have caught
    it; resolving the key and comparing does.
    """
    for marker in markers_for(site.path, site):
        if marker.group(1) != "config":
            continue
        # Several comma-separated keys mirror a tuple — `NGRAM_RANGE = (2, 5)`
        # is one literal over two YAML scalars, and both halves must be checked.
        spec = marker.group("detail").split()[0].strip("`,")
        keys = [k.strip("` ") for k in spec.split(",") if k.strip("` ")]
        literal = ast.literal_eval(site.value)
        if len(keys) > 1:
            configured = [resolve_config_key(k) for k in keys]
            literal = list(literal)
        else:
            configured = resolve_config_key(keys[0])
            if isinstance(configured, (list, tuple)):
                configured, literal = list(configured), list(literal)
        key = spec
        assert configured == pytest.approx(literal), (
            f"{site}: fail-safe default is {literal!r} but {key} is {configured!r}. "
            f"Two copies of one setting have drifted — fix the code copy, since the "
            f"YAML is what drives the run."
        )


@pytest.mark.parametrize(
    "site",
    [s for s in ALL_SITES if any(m.group(1) == "definitional" for m in markers_for(s.path, s))],
    ids=str,
)
def test_a_definitional_number_names_its_tuning_path(site):
    """`definitional` is the marker most able to launder a cut into code, so it
    carries the duty the tuning-path law imposes on every heuristic number."""
    assert TUNING_PATH.search(justification_text(site.path, site)), (
        f"{site}: marked definitional without naming a tuning path. "
        f"Add 'tuning path: <how it would be tuned>' — or, if there is none, it "
        f"is a cut and belongs in YAML."
    )


class TestTheCheckerItself:
    """A checker that silently passes is worse than no checker.

    Every assertion below is a way this file could look green while enforcing
    nothing, and two of them are mistakes that were actually made while building
    it: the marker walk stopped at the first ordinary argument line, so markers
    on long signatures never attached; and the tuning-path check read only the
    marker's own line, so a multi-line justification failed.
    """

    def _site(self, tmp_path: Path, source: str) -> Site:
        path = tmp_path / "sample.py"
        path.write_text(source, encoding="utf-8")
        sites = numeric_sites(path)
        assert len(sites) == 1, f"fixture should hold exactly one site, found {len(sites)}"
        return sites[0]

    def test_an_unmarked_constant_is_caught(self, tmp_path):
        site = self._site(tmp_path, "THRESHOLD = 0.7\n")
        assert not markers_for(site.path, site)

    def test_a_marker_on_a_long_signature_reaches_its_own_default(self, tmp_path):
        """⚠️ REGRESSION. The first walk-up stopped at `b: int,` and the marker
        never attached — every signature annotation was silently inert."""
        site = self._site(
            tmp_path,
            "# config(cut): measurements.probes.alpha\n"
            "def f(\n    a: str,\n    b: int,\n    c: list,\n    cut: float = 0.05,\n): ...\n",
        )
        assert [m.group(1) for m in markers_for(site.path, site)] == ["config"]

    def test_a_marker_for_another_parameter_does_not_cover_this_one(self, tmp_path):
        site = self._site(
            tmp_path, "# plumbing(other): unrelated\ndef f(cut: float = 0.05): ...\n"
        )
        assert not markers_for(site.path, site)

    def test_a_module_level_marker_does_not_cover_a_signature_default(self, tmp_path):
        """A bare marker and a parameter marker are not interchangeable —
        otherwise one annotation would launder every default under a `def`."""
        site = self._site(tmp_path, "# plumbing: bare\ndef f(cut: float = 0.05): ...\n")
        assert not markers_for(site.path, site)

    def test_a_dataclass_field_default_is_a_site(self, tmp_path):
        site = self._site(
            tmp_path, "@dataclass\nclass P:\n    # plumbing: counter\n    n: int = 0\n"
        )
        assert site.name == "n" and markers_for(site.path, site)

    def test_a_pydantic_config_field_is_not_a_site(self, tmp_path):
        """The config schema IS the config; asking it to declare that would be
        circular, and would train everyone to annotate reflexively."""
        path = tmp_path / "sample.py"
        path.write_text("class ProbeConfig(StrictModel):\n    seed: int = 0\n", encoding="utf-8")
        assert numeric_sites(path) == []

    def test_a_bool_is_not_a_numeric_site(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text("ENABLED = True\n", encoding="utf-8")
        assert numeric_sites(path) == []

    def test_a_multi_line_justification_satisfies_the_tuning_path_rule(self, tmp_path):
        """⚠️ REGRESSION. Reading only the marker's line would force real
        arguments onto one line to satisfy the checker."""
        site = self._site(
            tmp_path,
            "# definitional: what a content token is — long enough to drop\n"
            "# stopwords, short enough to keep 'bomb'.\n"
            "# Tuning path: the pilot's cached responses, swept offline.\n"
            "MIN_TOKEN = 4\n",
        )
        assert TUNING_PATH.search(justification_text(site.path, site))

    def test_a_definitional_marker_with_no_tuning_path_is_caught(self, tmp_path):
        site = self._site(tmp_path, "# definitional: just because\nMIN_TOKEN = 4\n")
        assert not TUNING_PATH.search(justification_text(site.path, site))

    def test_a_config_marker_naming_a_missing_key_fails_loudly(self):
        with pytest.raises((KeyError, AttributeError)):
            resolve_config_key("measurements.probes.no_such_knob")

    def test_a_config_marker_resolves_a_real_key(self):
        assert resolve_config_key("measurements.probes.alpha") == 0.05
