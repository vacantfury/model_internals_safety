#!/usr/bin/env python3
"""Run the refusal judge's negative control over a finished run (TODO 62b).

**What this answers.** `assign_regime` splits (B) decode-and-comply from (S)
decode-and-refuse on `refused`, and until now nothing screened that judge. This
re-judges cached responses under a paired manipulation in which only the
ciphertext moves, so an echo-induced flip is measured against ground truth
rather than inferred from a keyword probe. Design and the two verdicts it
produces: `measurements/refusal_control.py`.

**It can falsify a change already made.** TODO 62a nulls every echoing cell to
regime (P) on the grounds that the judge cannot tell a parrot from a refusal. If
`parrot_flip_rate` comes back LOW, that grounds is wrong, most of those cells
were genuine refusals, and the nulling has to be reverted. That is the reason to
spend judge calls on a documented behaviour.

**Costs judge calls, nothing else.** No GPU, no generation, no model load —
conditions B and C are string edits of responses already on disk. `--dry-run`
prints the count and spends nothing (and unlike the pilot's, it returns AFTER
building the conditions, so it exercises the selection path rather than only
argument parsing — the 2026-08-07 lesson that a dry run touching none of the
real path is not evidence).

Anchor handling: condition A is re-judged rather than read from `cells.jsonl`,
even though the verdict is recorded there. The re-judge IS this control's own
control — if the judge does not reproduce its own recorded verdict on unmodified
text, every flip measured against it is noise, and that has to surface as a
number rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from internals_safety.config import load_judge_config
from internals_safety.judges.base import JudgeItem
from internals_safety.judges.refusal import RefusalJudge
from internals_safety.measurements.refusal_control import judge_calls, summarize_control


def eligible_cells(cells_path: Path, family: str | None = None) -> dict[str, list[dict]]:
    """Every cell carrying a ciphertext and a response, grouped by rung.

    ⚠️ **Deliberately NOT filtered on the recorded verdict.** The first version
    kept only cells whose anchor read "not refused" and that collapsed the
    control to 18 items across 4 rungs — 6 on the decisive one — because the
    judge calls almost everything refused, which is the very defect under
    investigation. A control conditioned on the absence of the behaviour it
    measures is self-defeating.

    The parrot arm needs no filter: its ground truth is by construction, since a
    bare ciphertext contains no refusal whatever the model did. The appended arm
    applies its own filter downstream, in `summarize_control`.
    """
    by_family: dict[str, list[dict]] = defaultdict(list)
    for line in cells_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        if family is not None and cell["family"] != family:
            continue
        if not cell.get("ciphertext"):
            continue
        if not (cell.get("attack_response") or "").strip():
            continue
        by_family[cell["family"]].append(cell)
    return dict(by_family)


def conditions(cell: dict) -> tuple[str, str, str]:
    """(anchor, appended, parrot) response texts for one item.

    Only the ciphertext moves. `parrot` is the ciphertext ALONE, which is the
    case actually at issue: the commonest non-answer to an encoded prompt is the
    model echoing it back, and the judge counts that as a refusal.
    """
    response = cell["attack_response"]
    ciphertext = cell["ciphertext"]
    return response, f"{response}\n\n{ciphertext}", ciphertext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--family", default=None, help="restrict to one rung")
    parser.add_argument("--limit", type=int, default=100, help="items per rung")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cells_path = args.run_dir / "cells.jsonl"
    if not cells_path.exists():
        raise SystemExit(f"no cells.jsonl under {args.run_dir}")

    selected = {
        family: cells[: args.limit] for family, cells in eligible_cells(cells_path, args.family).items()
    }
    total = sum(len(cells) for cells in selected.values())

    print(f"items (every cell with a ciphertext): {total} across {len(selected)} rungs")
    for family, cells in selected.items():
        movable = sum(1 for cell in cells if not cell.get("refused"))
        print(f"  {family:20s} {len(cells):4d}   (arm B movable: {movable})")
    print(f"judge calls           {judge_calls(total)} (3 conditions x 1 judge)")
    print("GPU                   none — no generation, no model load")

    if args.dry_run:
        print("\nNothing judged. Costs judge API calls only; report them to the "
              "approval gate before running for real.")
        return 0

    judge = RefusalJudge.from_config(load_judge_config())
    results = {}
    for family, cells in selected.items():
        triples = [conditions(cell) for cell in cells]
        verdicts = []
        for index, arm in enumerate(("anchor", "appended", "parrot")):
            items = [
                JudgeItem(id=f"{cell['prompt_id']}", behavior=cell["plaintext"], response=triple[index])
                for cell, triple in zip(cells, triples)
            ]
            verdicts.append([verdict.flag for verdict in judge.judge(items)])
        anchor, appended, parrot = verdicts

        # The control's own control: the judge must reproduce its recorded
        # verdict on UNMODIFIED text, or every flip measured against it is noise.
        recorded = [bool(cell.get("refused")) for cell in cells]
        disagreement = (
            sum(a != r for a, r in zip(anchor, recorded)) / len(anchor) if anchor else 0.0
        )

        control = summarize_control(family, anchor, parrot, appended)
        results[family] = {
            "n": control.n,
            "n_appended": control.n_appended,
            "anchor_disagreement_rate": disagreement,
            "parrot_flip_rate": control.parrot_flip_rate,
            "appended_flip_rate": control.appended_flip_rate,
            "clears": control.clears(),
            "echo_route_dominates": control.echo_route_dominates,
        }
        print(
            f"\n{family}: n={control.n}  parrot_flip={control.parrot_flip_rate:.2f}  "
            f"appended_flip={control.appended_flip_rate:.2f} (n={control.n_appended})  "
            f"anchor_disagreement={disagreement:.2f}  clears={control.clears()}"
        )

    out = args.run_dir / "refusal_judge_control.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(
        "\nRead `echo_route_dominates` against TODO 62a: True supports nulling "
        "echoing cells to (P); False means the nulling was too aggressive and "
        "must be reverted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
