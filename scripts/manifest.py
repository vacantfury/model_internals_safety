#!/usr/bin/env python3
"""The submission manifest — machine identity for an upload set.

    uv run python scripts/manifest.py build --dir <manifest dir> \
        --source paper/as-5/<kit> --file <path>:<channel> ...
    uv run python scripts/manifest.py verify --dir <manifest dir> [--require-claims]
    uv run python scripts/manifest.py set-claims --dir <manifest dir> --clone pass ...

**Why identity is CODE and the narrative is prose.** A by-hand hash table rots
the day after it is written, and it rots silently: a sibling's manifest carried
one day's hashes through the next day's edits with nobody noticing, because
nothing re-read them. So the hashes live here, are only ever written by this
script, and `verify` is the staleness key the final check records and the submit
step refuses to proceed without.

**The source-tree hash is independent of git, deliberately.** `paper/` is
gitignored in this repo, so a git HEAD alone covers none of the paper source. We
hash the tree; HEAD is recorded alongside when present, as context rather than
as the identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "manifest.json"

#: Files that are BUILD OUTPUT or editor residue and are not part of a source
#: tree's identity. Hashing them would make the key change on a rebuild that
#: altered nothing an author wrote, which trains the reader to ignore a
#: mismatch — the failure mode this whole file exists to prevent.
SOURCE_SKIP_SUFFIXES = {
    ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out",
    ".pdf", ".synctex", ".gz", ".toc", ".DS_Store",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(source: Path) -> list[Path]:
    return sorted(
        p for p in source.rglob("*")
        if p.is_file()
        and p.suffix not in SOURCE_SKIP_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(source).parts)
    )


def _source_digest(source: Path) -> tuple[str, list[str]]:
    """One hash over the whole source tree, plus the file list it covered.

    The list is recorded because a digest alone cannot say what it covered, and
    "the hash matches" over a shrinking file set is the quiet way a check passes
    while missing the file that changed.
    """
    files = _source_files(source)
    rolling = hashlib.sha256()
    for path in files:
        rolling.update(str(path.relative_to(source)).encode())
        rolling.update(b"\x00")
        rolling.update(_sha256(path).encode())
        rolling.update(b"\x00")
    return rolling.hexdigest(), [str(p.relative_to(source)) for p in files]


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def build(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    uploads = []
    for spec in args.file:
        path_text, _, channel = spec.partition(":")
        if not channel:
            raise SystemExit(f"--file wants PATH:CHANNEL, got {spec!r}")
        path = Path(path_text)
        if not path.is_file():
            raise SystemExit(f"no such upload file: {path}")
        uploads.append({
            "path": str(path), "channel": channel,
            "sha256": _sha256(path), "bytes": path.stat().st_size,
        })

    facts = []
    for spec in args.fact:
        parts = spec.split("|")
        if len(parts) != 3:
            raise SystemExit(f'--fact wants "FACT|URL|YYYY-MM-DD", got {spec!r}')
        facts.append({"fact": parts[0], "url": parts[1], "checked": parts[2]})

    digest, covered = _source_digest(source)
    payload = {
        "source": {
            "dir": str(source), "sha256": digest,
            "file_count": len(covered), "files": covered,
            "git_head": _git_head(),
        },
        "uploads": uploads,
        "packaging_facts": facts,
        # Written by `set-claims`, from paper-check FINAL. `pending` at build so
        # `verify --require-claims` cannot pass on a package nobody checked.
        "claim_integrity": {"clone": "pending", "artifact": "pending", "host": None},
    }
    (directory / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {directory / MANIFEST}")
    print(f"  source tree   {digest[:16]} over {len(covered)} files")
    print(f"  uploads       {len(uploads)}")
    print(f"  facts         {len(facts)}")
    return 0


def set_claims(args: argparse.Namespace) -> int:
    path = Path(args.dir) / MANIFEST
    payload = json.loads(path.read_text())
    payload["claim_integrity"] = {
        "clone": args.clone, "artifact": args.artifact, "host": args.host,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"claim_integrity: clone={args.clone} artifact={args.artifact}")
    return 0


def verify(args: argparse.Namespace) -> int:
    path = Path(args.dir) / MANIFEST
    if not path.exists():
        print(f"FAIL: no manifest at {path}")
        return 1
    payload = json.loads(path.read_text())
    problems: list[str] = []

    source = Path(payload["source"]["dir"])
    digest, covered = _source_digest(source)
    # Every check prints its DENOMINATOR: a pass over nothing is not a pass.
    print(f"source tree   {len(covered)} files (manifest recorded "
          f"{payload['source']['file_count']})")
    if digest != payload["source"]["sha256"]:
        problems.append(
            f"source tree changed since packaging: {digest[:16]} != "
            f"{payload['source']['sha256'][:16]}"
        )

    print(f"uploads       {len(payload['uploads'])}")
    for entry in payload["uploads"]:
        upload = Path(entry["path"])
        if not upload.is_file():
            problems.append(f"upload missing: {upload}")
        elif _sha256(upload) != entry["sha256"]:
            problems.append(f"upload changed since packaging: {upload}")

    print(f"facts         {len(payload['packaging_facts'])}")
    if args.require_claims:
        claims = payload["claim_integrity"]
        print(f"claims        clone={claims['clone']} artifact={claims['artifact']}")
        if claims["clone"] != "pass":
            problems.append(f"claim_integrity.clone is {claims['clone']!r}, not 'pass'")
        if claims["artifact"] not in {"pass", "not-run"}:
            problems.append(f"claim_integrity.artifact is {claims['artifact']!r}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--dir", required=True)
    b.add_argument("--source", required=True, help="the paper SOURCE tree to hash")
    b.add_argument("--file", action="append", default=[], help="PATH:CHANNEL")
    b.add_argument("--fact", action="append", default=[], help='"FACT|URL|YYYY-MM-DD"')
    b.set_defaults(func=build)

    s = sub.add_parser("set-claims")
    s.add_argument("--dir", required=True)
    s.add_argument("--clone", required=True, choices=["pass", "fail"])
    s.add_argument("--artifact", required=True, choices=["pass", "fail", "not-run"])
    s.add_argument("--host", default=None)
    s.set_defaults(func=set_claims)

    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--require-claims", action="store_true")
    v.add_argument("--marks", action="store_true", help="accepted for interface parity")
    v.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
