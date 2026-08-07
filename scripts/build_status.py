"""Answer "is the instrument build done?" in one command.

    uv run python scripts/build_status.py

No model, no keys, no network — it reads the tree. Exit code 1 while anything is
outstanding, so it also works as a check rather than only as a report.

What it does NOT answer: whether a number may appear in a paper. That is a
per-run property and the contract already reports it (`reportable` per `Reading`,
`withheld` in every run record). An instrument can be finished and still produce
withheld readings; keeping the two apart is deliberate, because "the build is
done" must not come to mean "the numbers are good".
"""

from __future__ import annotations

import sys

from internals_safety.completion import build_status

SYMBOL = {
    "wired": "OK  ",
    "built-not-wired": "WIRE",
    "partial": "PART",
    "not-built": "TODO",
}


def main() -> int:
    status = build_status()

    print("INSTRUMENT ROSTER               (instrument_build_plan.md §3)")
    for entry in status.roster:
        detail = entry.item.incomplete or entry.item.note
        note = f"  — {detail}" if detail else ""
        print(f"  [{SYMBOL[entry.state]}] {entry.item.key:<4} {entry.item.what:<32}{note}")

    print("\nCONTROL BATTERY                 (instrument_build_plan.md §4)")
    for entry in status.controls:
        unwired = f"  — not reachable from any script" if entry.state == "built-not-wired" else ""
        print(f"  [{SYMBOL[entry.state]}] {entry.item.key:<16} {entry.item.what:<28}{unwired}")

    print(f"\nPLACEHOLDER KNOBS               ({len(status.placeholders)} untuned)")
    for name, locations in sorted(status.placeholders.items()):
        # Locations after the first are the fail-safe code mirror of a live YAML
        # value, listed rather than counted — one knob, marked twice.
        print(f"  {name:<26} {'  '.join(locations)}")

    if status.complete:
        print("\nBUILD COMPLETE.")
        return 0

    print(f"\nBUILD NOT COMPLETE — {len(status.outstanding)} item(s) outstanding:")
    for reason in status.outstanding:
        print(f"  - {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
