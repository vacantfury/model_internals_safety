#!/usr/bin/env python3
"""Turn a committed preset into a SLURM submission — dry-run by default.

    uv run python scripts/submit.py relicense_all          # print, submit nothing
    uv run python scripts/submit.py relicense_all --submit  # actually sbatch

**Dry-run by default is the approval gate expressed as a default.** The family
rule says report GPU count + type, money and wall-clock BEFORE launching, and
get an explicit go. A submitter that submits unless told not to inverts that:
the safe path becomes the one requiring an extra flag. The guardrail sibling's
`dispatch.py` made the same choice, and it is the one piece of their launcher
worth copying wholesale.

**No command line is built in bash.** `ops/run.sbatch` calls `--resolve` to be
handed the exact argv for its array index. That split exists because the thing
it replaced — `relicense.sbatch` — did array-index arithmetic over two bash
arrays, on the cluster, in a file that was never in git and never on the laptop.
Command construction is `PresetConfig.tasks`, which is tested.

Keyless and model-free: this script loads YAML and prints. It never imports
torch and never touches the network, so it runs on a login node in under a
second, and `--resolve` costs the job nothing.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from internals_safety.config import PresetConfig, list_presets, load_preset  # noqa: E402

# Where a cluster job writes. Activation caches are GB-scale and every sbatch in
# ops/ already exports this shape; kept as one string so the submitter and the
# launcher cannot disagree about it.
DEFAULT_OUTPUTS = "/scratch/$USER/internals_safety_outputs"

# How the preset name is written into the job's SLURM comment, and read back out.
COMMENT_PREFIX = "preset="


class SqueueUnavailable(RuntimeError):
    """The collision check could not run. Never silently treated as 'no collision'."""


def active_jobs_for(preset_name: str) -> list[tuple[str, str, str]]:
    """This account's queued or running jobs for one preset: (job id, state, how matched).

    **Two sessions share one SLURM account, and nothing stopped them colliding.**
    On 2026-08-07 one session was seconds from re-submitting a 30-task
    `relicense_all` array that another session already had running; it was caught
    by reading `squeue` by hand, which is not a mechanism. This is.

    Matching is on the COMMENT, not the job name, because the name is derived —
    `sbatch_flags` builds `is_<preset>` but a hand-typed `sbatch --job-name` can
    say anything, and matching a derived field means a job that lies about its
    name is invisible to the guard. The name is kept only as a FALLBACK for jobs
    submitted before the stamp existed, which is a real population right now.

    Returns empty when squeue cannot be reached; the caller decides what that
    means, because "I could not check" is not "there is nothing there".
    """
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-O", "JobID:24,Comment:64,Name:64,StateCompact:12"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise SqueueUnavailable("squeue could not be run")
    if result.returncode != 0:
        raise SqueueUnavailable(result.stderr.strip() or f"squeue exited {result.returncode}")

    stamped = f"{COMMENT_PREFIX}{preset_name}"
    derived = f"is_{preset_name}"
    found: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        job_id, comment, name, state = fields[0], fields[1], fields[2], fields[3]
        if comment == stamped:
            found.append((job_id, state, "comment"))
        elif comment in ("(null)", "") and name == derived:
            # Pre-stamp job. Reported as a weaker match so the human can see the
            # guard is inferring rather than reading an explicit declaration.
            found.append((job_id, state, "name (pre-stamp)"))
    return found


def sbatch_flags(preset: PresetConfig, name: str, n_tasks: int) -> list[str]:
    """The resource ask as CLI flags.

    CLI flags rather than `#SBATCH` directives in the file, because that is the
    direction that lets ONE generic launcher serve every preset — sbatch's own
    precedence rule is that command-line options override in-file directives.
    """
    resources = preset.resources
    flags = [
        f"--job-name=is_{name}",
        f"--partition={resources.partition}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output=logs/{name}_%A_%a.out",
        f"--error=logs/{name}_%A_%a.err",
    ]
    if resources.gres:
        flags.append(f"--gres={resources.gres}")
    # The preset name, stamped where a QUERY can see it. SLURM exposes the batch
    # script path but NOT its arguments — `squeue -o %o` and `scontrol show job`
    # both return `.../ops/run.sbatch` with the preset invisible — so without
    # this stamp the only thing left to match on is the derived job name, which
    # a hand-typed `--job-name` could contradict. See `active_jobs_for`.
    flags.append(f"--comment={COMMENT_PREFIX}{name}")
    if n_tasks > 1:
        # %10 throttles concurrency. The account's `normal` QOS allows 4
        # concurrent GPU jobs; CPU array tasks on `short` are not bound by that,
        # so the cap is politeness on a shared partition rather than a limit.
        flags.append(f"--array=0-{n_tasks - 1}%10")
    return flags


def describe(preset: PresetConfig, name: str, tasks: list[list[str]], show_all: bool = False) -> str:
    resources = preset.resources
    gpu = resources.gres or "none (CPU job)"
    lines = [
        f"PRESET  {name}",
        f"  {preset.description.strip()}",
        "",
        "GATES (what a result would change — the run-is-a-gate test):",
        f"  {preset.gates.strip()}",
        "",
        "RESOURCES",
        f"  partition   {resources.partition}",
        f"  gpu         {gpu}",
        f"  cpus/mem    {resources.cpus} / {resources.mem}",
        f"  wall-clock  {resources.time} per task (a CEILING, not an estimate)",
        f"  array tasks {len(tasks)}",
        "",
        f"COMMANDS ({len(tasks)})",
    ]
    # A 30-task array prints 30 near-identical lines and buries the gate text
    # above it. Truncation is display only — `--all` shows every one, and the
    # count is always stated, because a silently shortened list is how a run
    # gets approved for less than it does.
    shown = tasks if (show_all or len(tasks) <= 4) else tasks[:3]
    for index, argv in enumerate(shown):
        lines.append(f"  [{index}] ./run python {' '.join(shlex.quote(part) for part in argv)}")
    if len(shown) < len(tasks):
        lines.append(f"  ... {len(tasks) - len(shown)} more task(s); --all to list every one")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("preset", nargs="?", help=f"conf/experiment/<name>.yaml; available: {list_presets()}")
    parser.add_argument("--outputs-dir", default=None, help=f"default: {DEFAULT_OUTPUTS} expanded")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="actually call sbatch. Without this nothing is submitted — see the module docstring",
    )
    parser.add_argument(
        "--resolve",
        type=int,
        default=None,
        metavar="INDEX",
        help="print ONLY the argv for one array index, space-separated (used by ops/run.sbatch)",
    )
    parser.add_argument("--list", action="store_true", help="list presets and exit")
    parser.add_argument("--all", action="store_true", help="print every array task, not the first 3")
    parser.add_argument(
        "--force",
        action="store_true",
        help="submit even though this preset is already queued or running. Prints what it "
        "duplicates; never proceeds quietly",
    )
    args = parser.parse_args(argv)

    if args.list or not args.preset:
        for name in list_presets():
            print(f"  {name:24s} {load_preset(name).description.strip().splitlines()[0]}")
        return 0

    preset = load_preset(args.preset)
    outputs = args.outputs_dir or os.path.expandvars(DEFAULT_OUTPUTS)
    tasks = preset.tasks(outputs)

    if args.resolve is not None:
        if not 0 <= args.resolve < len(tasks):
            print(f"index {args.resolve} out of range for {len(tasks)} task(s)", file=sys.stderr)
            return 2
        print(" ".join(shlex.quote(part) for part in tasks[args.resolve]))
        return 0

    print(describe(preset, args.preset, tasks, show_all=args.all))

    command = ["sbatch", *sbatch_flags(preset, args.preset, len(tasks)), "ops/run.sbatch", args.preset]
    print("\nSBATCH\n  " + " ".join(shlex.quote(part) for part in command))

    if not args.submit:
        print(
            "\nNOTHING SUBMITTED. Cost this run with `--dry-run` on the entrypoint above,"
            "\nreport GPU/money/wall-clock, and re-run with --submit after the explicit go."
        )
        return 0

    # The collision check runs only on the submit path: it costs a squeue call,
    # and printing a preset is not an action that can collide with anything.
    try:
        active = active_jobs_for(args.preset)
    except SqueueUnavailable as error:
        # Loud, and NOT fatal. sbatch is the real backstop — on a machine with no
        # squeue there is no sbatch either, so refusing here would only block the
        # laptop while changing nothing about what reaches the cluster.
        print(f"\n⚠️  collision check did not run ({error}); submitting unchecked", file=sys.stderr)
        active = []

    if active:
        listing = "\n".join(f"    {job} {state:8s} (matched by {how})" for job, state, how in active)
        if not args.force:
            print(
                f"\nREFUSED — {args.preset} is already on the queue:\n{listing}\n"
                "\nAnother session may be running this. Check before duplicating; "
                "\n--force submits anyway.",
                file=sys.stderr,
            )
            return 3
        print(f"\n⚠️  --force: submitting a DUPLICATE of:\n{listing}")

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
