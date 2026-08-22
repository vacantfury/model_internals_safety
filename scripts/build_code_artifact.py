#!/usr/bin/env python3
"""Assemble the blind code archive for an upload set.

    uv run python scripts/build_code_artifact.py --out <path>.zip [--stage <dir>]

**Why this is a script and not a checklist.** The archive has to be free of the
maintainer's identity, and "free of" is a property nobody can eyeball: the
identifying strings hide in a lockfile's resolved URL, a cost comment naming an
institution, a README's sibling-project links. So the tree is staged, rewritten,
and then SWEPT — and the sweep is fail-closed. A hit refuses the zip. An archive
that was never written cannot be uploaded by mistake, which a warning printed
above a finished file very much can.

**The vendoring rule** (house rule): a dependency pinned to the maintainer's own
forge carries their handle in the dependency URL, so pinning it is a
deanonymiser no matter how the paper is written. Such a dependency is copied in
as source and re-pointed at the local path.

**What the sweep deliberately does NOT match.** Author surnames that appear as
CITATIONS are not identity leaks, and a pattern broad enough to catch every
surname would fire on the bibliography of every module docstring. The patterns
below name the maintainer's own handles, institution, and mailboxes.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "internals-safety-code"

#: Copied verbatim into the archive. Anything not named here does not ship —
#: an allow-list, because a deny-list silently ships each new directory.
INCLUDE = ("src", "scripts", "conf", "tests", "pyproject.toml")

#: Never ship: the research record (dated internal reasoning), the paper tree,
#: run outputs, the corpora (redistribution is not ours to grant), the
#: gitignored ops layer, and the agent instructions.
EXCLUDE_NAMES = {
    "__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache",
    "op_refs", "outputs", "data", "paper", "text_docs", "knowledge",
    "ops", "CLAUDE.md", "TODO.md", "NOW.md", ".claude", "run",
    # Packaging machinery, not research code. This builder additionally MUST
    # exclude itself: `FORBIDDEN` below is a list of the maintainer's
    # identifying strings, so an archive containing the builder contains the
    # very thing the builder exists to remove. The sweep catches this, which is
    # how the exclusion was found rather than remembered.
    "build_code_artifact.py", "manifest.py",
}

#: Fail-closed sweep, case-INSENSITIVE. See the module docstring for why
#: surnames are absent.
FORBIDDEN = (
    r"vacantfury",
    r"haoyu",
    r"northeastern",
    r"claude_thomas",
    r"infinitecry|solarfury",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:edu|com|org)",
)

#: Fail-closed sweep, case-SENSITIVE. The named cluster and the institution
#: abbreviation identify the site, but only in their prose spellings: the
#: lowercase `nurc` is a config IDENTIFIER threaded through the cluster
#: registry and every preset, and renaming it in a staged copy would edit
#: config values to buy an acronym — a change that can break the archive to
#: remove a token that expands to nothing once the prose is redacted.
FORBIDDEN_EXACT = (
    r"\bNURC\b",
    r"\bExplorer\b",
    r"\bNEU\b",
)

#: Applied longest-key-first to every shipped text file. Lives HERE rather than
#: in `conf/` because `conf/` ships: a redaction map stored inside the archive
#: would carry the very strings it exists to remove. The builder is already
#: excluded from the archive for exactly this reason.
REDACTIONS = {
    "NURC / Explorer": "The shared GPU cluster",
    "NEU Explorer's": "The shared cluster's",
    "Northeastern students and faculty": "its users",
    "Explorer's": "the shared cluster's",
    "NURC's": "the shared cluster's",
    "Explorer": "the shared cluster",
    "Northeastern": "the host institution",
    "NURC": "the shared cluster",
    # Last, and empty on purpose. The institution abbreviation also occurs as a
    # bare modifier that a comment WRAP split across two lines ("... GPU in NEU
    # / # Explorer's public pool"), which no multi-word key can match. Deleting
    # the modifier leaves the sentence correct once the noun it modified has
    # itself been redacted.
    "NEU": "",
}

ARTIFACT_README = """\
# Code artifact

Research code for the accompanying paper. Anonymised for review: the
maintainer's identity, forge account, and institution are absent by
construction, and the archive is refused by its own builder if any reappear.

## Layout

    src/internals_safety/   the library: encodings, probes, measurements, judges
    scripts/                entrypoints; each is runnable and self-describing
    conf/                   every tunable, in YAML (no magic numbers in code)
    tests/                  hermetic suite: no network, no keys, no weights
    vendor/llm_utils/       a dependency vendored as source (see below)

## Running

    uv sync
    uv run pytest

The suite is hermetic and needs no credentials, no downloads, and no GPU.
Scripts that call a hosted judge read their key from the environment; scripts
that load model weights need the weights present locally. Both are noted in
each script's own `--help`.

Two entrypoints cost nothing and explain the rest:

    uv run python scripts/build_status.py          # instrument roster + state
    uv run python scripts/cost_model.py --help     # what a run would cost

## Deliberate deviations from the development tree

- **`vendor/llm_utils/`** is a dependency normally installed from a source
  control URL. That URL names the maintainer, so the package is vendored as
  source here and `pyproject.toml` points at the local path instead.
- **`uv.lock` is omitted.** The lock pins the same URL by resolved commit;
  rewriting it to match the vendored path would produce a lock that no
  resolver had ever verified, which is worse than none. Dependency ranges in
  `pyproject.toml` are unmodified.
- **`vendor/llm_utils/LICENSE` has its copyright holder redacted.** The
  licence text is otherwise unmodified and its terms are unchanged; the holder
  is the same party as this paper's authors, and is restored in the public
  release.
- **Corpora and run outputs are not included.** The prompt sets are third-party
  releases and are cited in the paper; redistribution is theirs to grant, not
  ours.
"""


def _staged_copy(stage: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {n for n in names if n in EXCLUDE_NAMES}

    for name in INCLUDE:
        source = REPO_ROOT / name
        if not source.exists():
            sys.exit(f"missing from the tree, refusing to build: {name}")
        target = stage / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        else:
            shutil.copy2(source, target)


def _dist_info(site_packages: Path, distribution: str) -> Path:
    matches = sorted(site_packages.glob(f"{distribution}-*.dist-info"))
    if len(matches) != 1:
        sys.exit(f"expected exactly one {distribution} dist-info, found {len(matches)}")
    return matches[0]


def _vendored_pyproject(dist_info: Path, package: str) -> str:
    """Build the vendored project root's manifest FROM the installed metadata.

    Derived, never invented: the version, the Python floor, and the runtime
    requirements are read out of the real `METADATA`. A hand-written manifest
    here would drift from the package it claims to describe, and the drift
    would only surface in a reviewer's `uv sync`.
    """
    version = python_floor = None
    requires: list[str] = []
    for line in (dist_info / "METADATA").read_text().splitlines():
        if line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
        elif line.startswith("Requires-Python: "):
            python_floor = line.removeprefix("Requires-Python: ").strip()
        elif line.startswith("Requires-Dist: ") and "; extra ==" not in line:
            requires.append(line.removeprefix("Requires-Dist: ").strip())
        elif not line.strip():
            break
    if not version or not python_floor:
        sys.exit(f"could not read version / python floor from {dist_info.name}")
    rendered = "\n".join(f'    "{r}",' for r in requires)
    return (
        "# Generated from the installed distribution metadata by\n"
        "# scripts/build_code_artifact.py. This dependency is normally resolved\n"
        "# from source control; it is vendored here so the archive carries no\n"
        "# account name. Sources are unmodified.\n"
        "[project]\n"
        f'name = "{package}"\n'
        f'version = "{version}"\n'
        f'requires-python = "{python_floor}"\n'
        "dependencies = [\n"
        f"{rendered}\n"
        "]\n\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["{package}"]\n'
    )


def _redacted_license(text: str) -> str:
    """Keep the licence, redact the holder.

    A permissive licence REQUIRES its copyright notice to travel with the
    source, and the notice names the maintainer. Dropping the file would strip
    a term we are bound by; shipping it verbatim would defeat the anonymity the
    archive exists to have. So the notice stays and the holder is redacted, in
    the open, with the restoration stated in the archive README.

    Exactly one notice line is expected. A licence whose format changed under
    us fails here rather than shipping a name.
    """
    lines = text.splitlines(keepends=True)
    marked = [n for n, line in enumerate(lines) if line.strip().lower().startswith("copyright")]
    if len(marked) != 1:
        sys.exit(f"expected exactly one copyright line in the vendored licence, found {len(marked)}")
    year = re.search(r"\b(19|20)\d{2}\b", lines[marked[0]])
    stamp = year.group(0) if year else ""
    lines[marked[0]] = f"Copyright (c) {stamp} the authors (identity withheld for review)\n"
    return "".join(lines)


def _vendor_llm_utils(stage: Path) -> None:
    """Copy the maintainer-hosted dependency in as source, and re-point at it.

    The vendored tree is a PROJECT ROOT (`vendor/llm_utils/pyproject.toml` plus
    `vendor/llm_utils/llm_utils/`), not a bare package directory. A path
    dependency pointing at a bare package has no build metadata, so it fails at
    the reviewer's first `uv sync` — which is exactly where a packaging defect
    must never be discovered.
    """
    out = subprocess.run(
        [sys.executable, "-c", "import llm_utils, pathlib; print(pathlib.Path(llm_utils.__file__).parent)"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if out.returncode != 0:
        sys.exit("llm_utils is not importable; run inside the project environment")
    source = Path(out.stdout.strip())
    root = stage / "vendor" / "llm_utils"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, root / source.name,
        ignore=lambda _d, names: {n for n in names if n in EXCLUDE_NAMES},
    )

    dist_info = _dist_info(source.parent, source.name)
    (root / "pyproject.toml").write_text(_vendored_pyproject(dist_info, source.name))
    license_file = dist_info / "licenses" / "LICENSE"
    if not license_file.exists():
        sys.exit(f"no LICENSE alongside {dist_info.name}; refusing to vendor unlicensed source")
    (root / "LICENSE").write_text(_redacted_license(license_file.read_text()))

    pyproject = stage / "pyproject.toml"
    text = pyproject.read_text()
    replaced, count = re.subn(
        r'^llm-utils = \{ git = .*$',
        'llm-utils = { path = "vendor/llm_utils" }',
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        sys.exit(f"expected exactly one llm-utils source line to rewrite, rewrote {count}")
    pyproject.write_text(replaced)


def _assert_vendored_dep_builds(stage: Path) -> None:
    """The postcondition: the vendored path dependency actually installs.

    Checked by BUILDING it, because "the files are present" is not the claim
    the README makes to a reviewer. Cheap: one wheel, no network for the
    dependency graph.
    """
    root = stage / "vendor" / "llm_utils"
    out = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(stage / ".build-check"), str(root)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"vendored dependency does not build:\n{out.stderr[-2000:]}")
    shutil.rmtree(stage / ".build-check", ignore_errors=True)


def _redact_site(stage: Path) -> None:
    """Redact the site's identifying prose, in place, token by token.

    Deliberately a substitution and NOT a line deletion. Dropping the comment
    line that named the institution left the next line starting mid-sentence —
    still anonymous, but visibly redacted and no longer readable, in an
    artifact whose whole job is to be read. Replacing the token keeps every
    sentence intact and every number in place.
    """
    ordered = sorted(REDACTIONS.items(), key=lambda kv: -len(kv[0]))
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix not in {".yaml", ".yml", ".py", ".toml", ".md"}:
            continue
        try:
            text = original = path.read_text()
        except (UnicodeDecodeError, ValueError):
            continue
        for needle, replacement in ordered:
            text = text.replace(needle, replacement)
        if text != original:
            # An emptied token can leave trailing space. Comment lines are the
            # only place redaction lands, and trailing space there is inert.
            text = "".join(
                (line.rstrip() + "\n") if line.lstrip().startswith("#") else line
                for line in text.splitlines(keepends=True)
            )
            path.write_text(_recapitalise(text))


#: A redaction that lands at the start of a sentence arrives lowercase, because
#: the token it replaced was a proper noun. Terminal punctuation on the PREVIOUS
#: comment line is what separates a sentence start from a wrapped continuation —
#: a continuation must stay lowercase, so a blanket capitalise would be wrong
#: exactly as often as it was right.
_SENTENCE_END = (".", ":", "!", "?")


def _recapitalise(text: str) -> str:
    replacements = tuple(v for v in REDACTIONS.values() if v and v[0].islower())
    lines = text.splitlines(keepends=True)
    for n, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        body = stripped[1:].lstrip()
        if not body.startswith(replacements):
            continue
        previous = lines[n - 1].strip() if n else ""
        starts_sentence = (
            not previous.startswith("#")
            or previous.rstrip("#").strip() == ""
            or previous.endswith(_SENTENCE_END)
        )
        if starts_sentence:
            head = line[: len(line) - len(body.lstrip())]
            lines[n] = head + body[0].upper() + body[1:]
    return "".join(lines)


def _sweep(stage: Path) -> list[str]:
    hits: list[str] = []
    patterns = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN]
    patterns += [re.compile(p) for p in FORBIDDEN_EXACT]
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, ValueError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(stage)}:{number}: {line.strip()[:120]}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", type=Path, default=None)
    args = parser.parse_args()

    stage = args.stage or (REPO_ROOT / ".artifact-stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    _staged_copy(stage)
    _vendor_llm_utils(stage)
    _assert_vendored_dep_builds(stage)
    _redact_site(stage)
    (stage / "README.md").write_text(ARTIFACT_README)

    hits = _sweep(stage)
    if hits:
        print(f"REFUSED: {len(hits)} identifying string(s) survived staging", file=sys.stderr)
        for hit in hits[:40]:
            print(f"  {hit}", file=sys.stderr)
        sys.exit(1)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, Path(ARCHIVE_ROOT) / path.relative_to(stage))
                count += 1

    print(f"swept        {count} files, 0 identifying strings")
    print(f"wrote        {out.relative_to(REPO_ROOT)}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
