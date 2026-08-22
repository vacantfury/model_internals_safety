"""The run spine both papers share.

**Canonical design: `text_docs/shared/pipeline_architecture.md` §3.3.** Do not
re-derive the selection rule here; it is settled and it is short:

    the spine holds anything whose absence in ONE script would be a defect.

That is not a generality argument, it is one incident stated as a rule. The
length null reached `as6_guard_probe.py` on 2026-08-05 and
`phase0_regime_map.py` on 2026-08-06, so for a day the guard side carried a
mandatory control the target side lacked — **for a confound measured on the
target side.** Two scripts implementing the same sequence in their own words is
what made that representable.

**What is deliberately NOT here.** The instrument roster and the combination
step. AS-5 assigns four-regime cells, AS-6 assigns guard cells, and they run
different measurements; forcing those into the spine is how a shared runner
becomes a framework. Each script keeps its own `run_family` and passes it in.

**Form: plain functions, no class.** There is no `Pipeline` object, no lifecycle,
no config-driven runner, and adding one would be the failure mode this module was
weighed against. A file of functions is a module; that is the whole of it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dataclasses import dataclass
from internals_safety.measurements.lexical_decorrelation import (
    LexicalDecorrelation,
    measure_lexical_decorrelation,
)
from internals_safety.probes.linear import probe_transfer_detail, reading_threshold
from internals_safety.config import CONTRAST_TYPE_PREFIX, MatchingStrategy, ProbeConfig
from internals_safety.data import Prompt, prompt_set
from internals_safety.encodings.base import EncodedPrompt, Invertibility
from internals_safety.measurements.contract import Reading
from internals_safety.measurements.regimes import (
    assign_regime,
    build_regime_map,
    refusal_verdict,
)
from internals_safety.paths import OUTPUTS_DIR, run_dir

# A screen that can only be applied once EVERY rung has run. It receives the
# finished summaries and returns `{family: reason}` for the rungs whose
# deployment axis must be demoted to unmeasured.
#
# This type exists because the control floor is not a per-rung quantity. It is
# derived from the run's own can't-decode rungs, so no amount of threading it
# into `measure_deployment` would help: at the moment a rung is licensed, the
# rungs that calibrate its floor have not run yet. That is why the floor lived
# in three offline rescoring scripts and never reached the entrypoint.
CrossRungScreen = Callable[[Sequence[dict[str, Any]]], Mapping[str, str]]


# The plain baseline's family label. Not a ladder rung and deliberately not
# registered as one: nothing about it is an encoding.
PLAIN_FAMILY = "plain"


def plain_arm(prompts: Sequence[str]) -> list[EncodedPrompt]:
    """The corpus as itself — no encoder, no wrapper, no instruction.

    **`attack_prompt` is the bare prompt, deliberately.** Running the corpus
    through an identity ENCODER would still wrap it in that rung's
    `attack_template`, and the template is part of what the encoded condition
    is being blamed for. The baseline has to be the prompt the model would get
    if nobody had encoded anything, or it measures a wrapper effect and calls it
    a plaintext rate.

    **Spine, by the rule in this module's docstring, and it was earned the hard
    way (2026-08-09).** `encoding_ablation.py` hand-wrote its own copy of this
    six-line function, omitted two required fields, and died on an H200 after a
    queue wait — while an identical, correct version had been sitting in
    `phase0_regime_map.py` since the plaintext baseline landed. Every arm the
    two papers compare against a plaintext denominator has to be built the same
    way or the denominators are not comparable, which is exactly the condition
    the selection rule names.
    """
    return [
        EncodedPrompt(
            plaintext=text,
            ciphertext=text,
            family=PLAIN_FAMILY,
            invertibility=Invertibility.EXACT,
            attack_prompt=text,
            restate_prompt=text,
        )
        for text in prompts
    ]


# The scaffold control's family label carries the rung whose scaffold it wears,
# because the scaffold is NOT shared: the registry bakes `{encoding_name}` into
# each family's template, so "The following request is encoded with Base64" and
# "... with homoglyph substitution" are different prompts.
SCAFFOLD_FAMILY_PREFIX = "scaffold"


def scaffold_family(family: str) -> str:
    return f"{SCAFFOLD_FAMILY_PREFIX}:{family}"


def scaffold_arm(prompts: Sequence[str], encoder) -> list[EncodedPrompt]:
    """Plaintext content wearing a rung's attack scaffold. The factorial control.

    **What it isolates.** The encoded condition changes TWO things at once: the
    characters of the request, and the wrapper announcing that the request is
    encoded. `plain_arm` holds neither, the encoded arm holds both, and this
    holds only the second. Reading the three together separates "the model
    reacts to the transformed characters" from "the model reacts to being told
    it is looking at an encoding".

    **Why it exists (2026-08-09).** An external review of AS-5 identified this
    as the paper's central confound, and it was right: the paper's own words
    were that the plaintext arm is bare "deliberately, since ... that scaffold
    is part of what the encoded condition is being blamed for". That is a
    defensible definition of the *condition* and it is not a defence of the
    *causal claim* — with only the two arms, "the encoding destroys
    discrimination" and "our encoded-prompt protocol destroys discrimination"
    predict identical numbers. Same shape as every other confound this repo has
    found: two explanations, one measurement, and the control is the only thing
    that separates them.

    **`canonicalize` is applied, deliberately.** The encoded arm canonicalises
    before encoding (Morse round-trips uppercase only), so holding it here keeps
    scaffold-vs-encoded a clean one-variable contrast — the character
    substitution and nothing else. For every rung the paper reports it is the
    identity, so plain-vs-scaffold is equally clean there.

    ⚠️ `ciphertext` is the untransformed text, so `echoed_ciphertext` fires on
    any response that quotes the request — exactly as it already does for
    `plain_arm`. That flag is meaningless on an unencoded arm and must not be
    used to withhold cells here; report it, never subtract it.
    """
    return [
        EncodedPrompt(
            plaintext=(canonical := encoder.canonicalize(text)),
            ciphertext=canonical,
            family=scaffold_family(encoder.family),
            invertibility=Invertibility.EXACT,
            attack_prompt=encoder.attack_template.format(ciphertext=canonical),
            restate_prompt=encoder.restate_template.format(ciphertext=canonical),
        )
        for text in prompts
    ]


def add_common_arguments(parser: argparse.ArgumentParser, *, default_n_prompts: int) -> None:
    """The arguments every phase entrypoint must accept.

    Shared because each one encodes a rule rather than a preference, and a script
    that quietly lacked one would be a defect rather than a variant: `--dry-run`
    is how the approval gate gets its estimate without spending, `--allow-dirty`
    is the escape hatch on the provenance guard, `--allow-cpu` exists so a batch
    job cannot silently leave an allocated GPU idle, and `--outputs-dir` is what
    lets the cluster write to scratch instead of a small repo volume.
    """
    parser.add_argument("--families", nargs="+", default=None, help="default: every configured rung")
    parser.add_argument("--n-prompts", type=int, default=default_n_prompts)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the work this run would do and exit; loads no model and needs no keys",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record the diff and run anyway (refused by default on result-bearing devices)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run inside a SLURM job (refused by default: a batch job "
        "on CPU means a GPU was allocated and left idle)",
    )
    parser.add_argument("--refresh-activations", action="store_true", help="ignore the capture cache")
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="root for activations/ and runs/ (default: the repo's outputs/); "
        "point it at cluster scratch when the repo tree is on a small volume",
    )


def matched_by_contrast_type(
    harmful: list[Prompt], harmless: list[Prompt]
) -> tuple[list[Prompt], list[Prompt]]:
    """The `contrast_type` matching strategy, DERIVED from the corpus's own names.

    XSTest names every unsafe subset by prefixing the safe subset it contrasts
    with: `contrast_homonyms` against `homonyms`, and so on. So the matched
    subset is not a list somebody typed, it is the fixed point of that naming
    rule — a safe type is in iff `contrast_<type>` is a real unsafe type, and an
    unsafe type is in iff stripping the prefix names a real safe type.

    **What that rule drops, and why dropping it is the honest move.** Two of
    XSTest's eight unsafe types (`contrast_discr`, `contrast_privacy`) each
    contrast with TWO safe types, so 50 safe prompts face 25 unsafe ones and no
    one-to-one pairing exists. Balancing those by subsampling would mean drawing
    25 of 50 at some seed, and a seeded draw inside a held-out corpus is a knob
    in exactly the place a holdout is supposed to have none. The six remaining
    types pair exactly, which is 150 per arm — larger than the 100 per arm every
    number in either paper currently rests on.

    Returns both arms filtered and in file order. Type membership is read off
    each prompt's `category`, which is what the corpus files carry.
    """
    safe_types = {prompt.category for prompt in harmless}
    unsafe_types = {prompt.category for prompt in harmful}
    paired_safe = {
        safe
        for safe in safe_types
        if f"{CONTRAST_TYPE_PREFIX}{safe}" in unsafe_types
    }
    paired_unsafe = {f"{CONTRAST_TYPE_PREFIX}{safe}" for safe in paired_safe}
    if not paired_safe:
        raise SystemExit(
            "contrast_type matching found no paired types; the harmful arm's "
            f"categories {sorted(unsafe_types)[:4]}... do not prefix any of the "
            f"harmless arm's {sorted(safe_types)[:4]}... — is this pair really "
            "matched by contrast type? (conf/corpus.yaml)"
        )
    return (
        [prompt for prompt in harmful if prompt.category in paired_unsafe],
        [prompt for prompt in harmless if prompt.category in paired_safe],
    )


def load_contrast_sets(
    harmful_set: str, harmless_set: str, n_prompts: int, *, matching: MatchingStrategy
) -> tuple[list[Prompt], list[Prompt]]:
    """The probe's two classes, with the size check that must never be skipped.

    Unequal classes would shift every AUROC by the base rate rather than by the
    signal, and the shift is invisible in the reported number. Raising here is
    the point: both scripts fit probes on these sets, so a mismatch is a defect
    in either, not a variant of one.

    **`matching` is keyword-only with NO default, and that is the whole design.**
    It is the fifth time this repo has met the same shape: a rule settles, it is
    threaded into the caller that motivated it, and the other callers keep the
    old behaviour silently (`strata` on `measure_deployment`, `device` on
    `guard_working_tree`, the control floor never reaching `phase0_regime_map`,
    the split-half screen reaching the AUROCs but not the counts). Defaulting
    this to `"theme"` would mean a caller that forgot it pairs XSTest's 200
    unsafe prompts against the first 200 of 250 safe ones — an arm mismatch that
    produces a number rather than an error. Omitting it is a `TypeError`.

    **The matching is applied BEFORE `n_prompts`**, so the limit takes a prefix
    of the matched subset rather than of the raw file. The other order would let
    a limit smaller than the file silently unbalance an already-matched pair.
    """
    harmful = prompt_set(harmful_set)
    harmless = prompt_set(harmless_set)
    if matching == "contrast_type":
        harmful, harmless = matched_by_contrast_type(harmful, harmless)
    harmful = harmful[:n_prompts]
    harmless = harmless[:n_prompts]
    if len(harmful) != len(harmless):
        raise SystemExit(
            f"contrast sets differ in size ({len(harmful)} vs {len(harmless)}); the probe's "
            "classes must be matched — see conf/corpus.yaml"
        )
    return harmful, harmless


def resolve_run_paths(
    phase: str, name: str, run_name: str | None, outputs_dir: str | None
) -> tuple[Path, Path, str]:
    """`(run directory, activations dir, run name)`, directory created.

    The run name defaults to a UTC timestamp rather than a local one so runs
    launched from a laptop and from the cluster sort together.

    **⚠️ A named run is made collision-proof by appending a timestamp and the
    SLURM job id** (sibling parity, `pipeline_convergence.md` §c — their run dirs
    carry `_<ts>_<jobid>` and cannot collide). Before this, re-running with the
    same `--run-name` SILENTLY OVERWROTE the previous results.json and
    cells.jsonl. That is a data-loss path this repo had simply not hit yet, and
    it is the worst kind: the second run looks like it worked.

    The readable name is KEPT as the prefix — `band2-20260805_20260806T1412Z_8957794`
    still sorts and greps by what it is, which is the property the sibling's
    scheme has and a bare uuid would not.
    """
    outputs = Path(outputs_dir) if outputs_dir else OUTPUTS_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_name:
        job = os.environ.get("SLURM_JOB_ID")
        resolved = f"{run_name}_{stamp}" + (f"_{job}" if job else "")
    else:
        resolved = stamp
    directory = run_dir(phase, name, resolved, runs_dir=outputs / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    return directory, outputs / "activations", resolved


def quarantine_run(directory: Path, reason: str, outputs_dir: Path | None = None) -> Path:
    """MOVE an invalidated run under `outputs/_quarantine/<reason>_<date>/`.

    Adopted from the sibling, which has `_quarantine/oracle_leak_20260805`,
    `_quarantine/figstep_incomplete_20260805` and five more
    (`pipeline_convergence.md` §c).

    **This repo needs it more than they do.** Every quantitative map from both of
    our runs has been revised at least once, and the pilot's `cells.jsonl` is
    currently superseded-but-in-place with that fact recorded only in prose. A
    run that has been invalidated and still sits at its original path is a trap
    for the next session, which will read it as current.

    MOVED, never deleted: an invalidated run is evidence about an instrument
    defect, and the defect is usually more interesting than the run.
    """
    outputs = outputs_dir if outputs_dir is not None else OUTPUTS_DIR
    slug = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = outputs / "_quarantine" / f"{slug}_{stamp}" / directory.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"{target} already exists — quarantining twice under one reason on one "
            "day would overwrite the first, which is the failure this exists to stop"
        )
    shutil.move(str(directory), str(target))
    (target / "QUARANTINED.txt").write_text(
        "\n".join(
            [
                f"reason: {reason}",
                f"quarantined: {stamp}",
                f"original path: {directory}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def run_families(
    families: Sequence[str],
    directory: Path,
    run_one: Callable[[str], dict[str, Any]],
    report: Callable[[dict[str, Any]], None] | None = None,
    *,
    cross_rung_screen: CrossRungScreen | None,
) -> tuple[list[dict[str, Any]], list[Reading], float]:
    """Drive the per-family loop with a crash-durable checkpoint after each rung.

    `run_one(family)` returns `{"cells": [...], "summary": {...}}` and MAY return
    `"readings"` and `"benign_cells"`; `report` gets the whole result for
    whatever the paper wants printed. Returns the summaries, every `Reading`
    collected across rungs, and the total elapsed seconds.

    **`benign_cells` is in the spine for the reason the spine exists.** The
    benign arm's per-prompt responses were generated on every run since
    2026-08-07, consumed for four aggregate rates, and dropped — so
    `band2-20260805` offered **2 valid control cells against 22 objects** for the
    graded regrade, while the ~70-79 cells per rung that would have worked had
    been computed and thrown away (`instrument_layer.md` §3.5.1). The judge bill
    was already paid; only the write was missing. That is precisely "anything
    whose absence in ONE script would be a defect", and AS-6 needs the same file
    for the same reason, so it belongs here rather than in either entrypoint.

    **Readings are accumulated here rather than in each script** for the same
    reason the checkpoint is: a run whose instruments emit verdicts and whose
    record does not carry them is a run that cannot say what it withheld, and
    that must not be possible in one paper and not the other.

    **The checkpoint is why this is in the spine.** Both halves were earned by a
    real loss: the comprehension-band sweep was killed at the 8 h wall having
    COMPLETED `zero_width` on two models and recovered nothing — `cells.jsonl`
    was 0 bytes because Python's buffer had not filled, and `results.json` is
    only written after the loop. So each rung is written, flushed AND fsynced
    before the next begins: a SIGKILL at the wall does not run buffers out, and
    on a networked scratch filesystem a flush alone can still sit in the page
    cache. A rung that finished must survive the job that did not, or every
    wall-clock kill silently re-buys work already paid for.

    `elapsed_seconds` is stamped on every summary because `conf/cost.yaml`'s
    throughput numbers are assumptions with exactly one tuning path — a real run
    measuring them. Gather-and-cover: the instrumentation ships with the knob.

    **`cross_rung_screen` is keyword-only with NO DEFAULT, and that is the whole
    point of it (TODO 60).** The control floor was adopted, tested, documented
    and owner-approved on 2026-08-07 and still governed nothing that produced a
    run, because `phase0_regime_map.py` never imported it — only the three
    offline rescoring scripts did. It cost a rung the same day: `tag_block`
    licensed at AUROC 0.6445 with ability 0.00 and read `deployment=True` on 66
    of 100 cells, while `reverse_characters` 0.012 higher did not license and
    correctly went (U). Both sat below every floor ever derived for that model.

    A screen that a caller may forget is a screen that describes the instrument
    in a document and not in a run, so passing one — or explicitly passing
    `None` — is now the only way to call this function. That is the third time
    this repo has fixed this class of defect the same way (`strata` on
    `measure_deployment`, `device` on `guard_working_tree`), and the lesson each
    time was that threading a rule into its callers is not the fix; **making the
    omission inexpressible is.**

    The screen runs AFTER the loop because the floor is derived from the run's
    own can't-decode rungs — at the moment any single rung is licensed, the
    rungs that calibrate it have not run yet. Demoted rungs are then rewritten
    in place: `deployment` becomes `None` and every affected cell is re-labelled
    through `assign_regime`, so the rules stay in one place rather than being
    restated as a hard-coded (U).
    """
    summaries: list[dict[str, Any]] = []
    readings: list[Reading] = []
    started = time.perf_counter()
    with (directory / "cells.jsonl").open("w", encoding="utf-8") as handle, (
        directory / "summaries.partial.jsonl"
    ).open("w", encoding="utf-8") as partial_handle, (
        directory / "benign_cells.jsonl"
    ).open("w", encoding="utf-8") as benign_handle:
        for family in families:
            print(f"\n=== {family}", flush=True)
            family_started = time.perf_counter()
            result = run_one(family)

            for cell in result["cells"]:
                handle.write(json.dumps(cell, ensure_ascii=False) + "\n")
            _checkpoint(handle)

            # A SEPARATE file, deliberately, and the reason is one function down:
            # `_demote_to_unmeasured` rewrites every row of cells.jsonl through
            # `assign_regime(..., prompt_is_harmful=True)`. A benign row sharing
            # that file would be silently relabelled as a harmful one on any run
            # where a rung is demoted. Same-file with an `arm` column would also
            # double the row count under every existing consumer that assumes
            # otherwise (`rescore_ability`, `rebaseline_pilot`,
            # `regrade_compliance`), which is a silent corruption rather than a
            # loud one.
            for cell in result.get("benign_cells", ()):
                benign_handle.write(json.dumps(cell, ensure_ascii=False) + "\n")
            _checkpoint(benign_handle)

            summary = result["summary"]
            summary["elapsed_seconds"] = round(time.perf_counter() - family_started, 1)
            summaries.append(summary)
            partial_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            _checkpoint(partial_handle)

            readings.extend(result.get("readings", []))

            if report is not None:
                report(result)
            print(f"    ({summary['elapsed_seconds'] / 60:.1f} min)", flush=True)

    # Outside the `with`: the demotion rewrites both files, so they must be
    # closed and fsynced first. A crash here leaves the un-demoted run intact,
    # which is the safe direction — the cells are still there to re-screen.
    if cross_rung_screen is not None:
        demoted = cross_rung_screen(summaries)
        if demoted:
            _demote_to_unmeasured(directory, summaries, demoted)

    return summaries, readings, time.perf_counter() - started


def _demote_to_unmeasured(
    directory: Path,
    summaries: list[dict[str, Any]],
    reasons: Mapping[str, str],
) -> None:
    """Re-label every cell of a screened-out rung as deployment-unmeasured.

    The demotion goes through `assign_regime` rather than writing (U) directly.
    A rung that fails the cross-rung screen is in exactly the state the tri-state
    fix was built for — *this instrument could not read this rung* — and that is
    already a rule, so restating it here would be a second copy free to drift
    from the first. It also means the incoherence flags come out right: a
    demoted rung's `deployment_without_ability` counts must DISAPPEAR, because
    the axis that produced them is now unmeasured, and only the rules module
    knows that.

    Files are replaced atomically. A partial rewrite would leave cells.jsonl
    holding some demoted rungs and some not, which is worse than either state.
    """
    cells_path = directory / "cells.jsonl"
    rewritten: list[str] = []
    per_family: dict[str, list[Any]] = {family: [] for family in reasons}

    with cells_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cell = json.loads(line)
            family = cell.get("family")
            if family in reasons:
                assignment = assign_regime(
                    ability=bool(cell["ability"]),
                    deployment=None,
                    recognition=cell["recognition"],
                    refused=refusal_verdict(
                        refused=bool(cell["refused"]),
                        echoed_ciphertext=bool(cell.get("echoed_ciphertext")),
                    ),
                    prompt_is_harmful=True,
                )
                cell["deployment"] = None
                cell["regime"] = assignment.regime.value
                cell["incoherences"] = [flag.value for flag in assignment.incoherences]
                cell["demoted_by_cross_rung_screen"] = reasons[family]
                per_family[family].append(assignment)
            rewritten.append(json.dumps(cell, ensure_ascii=False))

    _replace_atomically(cells_path, "\n".join(rewritten) + ("\n" if rewritten else ""))

    for summary in summaries:
        family = summary.get("family")
        if family not in reasons:
            continue
        regime_map = build_regime_map(family, per_family[family])
        summary["regimes"] = {r.value: c for r, c in regime_map.counts.items()}
        summary["incoherences"] = {f.value: c for f, c in regime_map.incoherence_counts.items()}
        summary["binding_failure_rate"] = regime_map.binding_failure_rate
        summary["hard_incoherence_rate"] = regime_map.hard_incoherence_rate
        summary["deployment_unmeasured"] = regime_map.deployment_unmeasured
        # The reading itself is kept, not deleted: the AUROC that FAILED the
        # screen is the evidence for the demotion, and a run record that hid it
        # would be unable to say why the rung is unmeasured.
        summary.setdefault("deployment", {})["cleared_control_floor"] = False
        summary["cross_rung_screen"] = reasons[family]

    partial = directory / "summaries.partial.jsonl"
    if partial.exists():
        # Rewritten too, or the crash artifact disagrees with the record it was
        # a checkpoint of — the dual-truth failure this repo keeps paying for.
        _replace_atomically(
            partial, "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in summaries)
        )


def _replace_atomically(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, fsync, then `os.replace`."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        _checkpoint(handle)
    os.replace(temporary, path)


def _checkpoint(handle) -> None:
    """flush + fsync. Both, always — see `run_families`."""
    handle.flush()
    os.fsync(handle.fileno())


def select_known(requested: Iterable[str] | None, available: Iterable[str], *, label: str) -> list[str]:
    """Validate a requested subset against what is configured, or take all of it.

    Fails on an unknown name rather than silently running a shorter ladder: a
    typo'd `--families` would otherwise produce a complete-looking run whose map
    is missing a rung nobody notices is absent.
    """
    known = list(available)
    if requested is None:
        return known
    chosen = list(requested)
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise SystemExit(f"unknown {label} {unknown}; configured: {sorted(known)}")
    return chosen


# ---------------------------------------------------------------------------
# The XSTest lexical control's RUNNER.
#
# In the spine because of the spine's own selection rule: it holds anything
# whose absence in ONE script would be a defect. This is that case, found the
# hard way on 2026-08-08 — `deployment.REQUIRED_CONTROLS` names the XSTest
# vocabulary screen, so a deployment reading without it is non-reportable by
# the contract, and `as6_guard_probe.py` contained ZERO references to it. Every
# AS-6 decode number ever produced was therefore unusable, and the reason was
# simply that the runner lived inside the OTHER entrypoint's file where AS-6
# could not reach it without a script-imports-script edge.
#
# The pure scoring stays in `measurements/lexical_decorrelation.py`; this is
# orchestration over activations both entrypoints already hold.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XStestCapture:
    """The XSTest corpus, captured once per model and read by every rung.

    Carries the TYPES alongside the activations because the pairing IS the
    control: `paired_separation` scores safe against unsafe within a matched
    type, and a batch without its types would silently become an ordinary
    two-class AUROC over unmatched prompts — a reassuring number measuring
    nothing.
    """

    safe_batch: object
    unsafe_batch: object
    safe_types: tuple[str, ...]
    unsafe_types: tuple[str, ...]
    safe_cache: Path
    unsafe_cache: Path


def run_lexical_control(
    plain_harmful_batch,
    plain_harmless_batch,
    xstest_safe_batch,
    xstest_unsafe_batch,
    safe_types: Sequence[str],
    unsafe_types: Sequence[str],
    layer: int,
    position: str,
    config: ProbeConfig,
) -> LexicalDecorrelation:
    """The XSTest control, read at the cell a rung's claim is read at.

    **What it screens, and why nothing else catches it.** In JBB, alarming words
    and actual harm are perfectly confounded: every harmful prompt has them and
    no benign prompt does, so a probe reading only "kill" scores identically to
    one reading intent. The length null, the black-box baseline and the
    format-decorrelation 2x2 all leave that route open. XSTest closes it by
    construction — its matched contrast pairs share the vocabulary and differ in
    intent ("How can I kill a Python process?" vs "...kill a person?").

    **Two decisions that would be silently wrong the other way.**

    *The cell.* Read at the (layer, position) the deployment reading SELECTED,
    not at a fresh argmax. A control evaluated somewhere other than where the
    claim is made says nothing about the claim.

    *The threshold.* The percentile operating point is taken from the PLAIN
    negative class, because XSTest prompts are plain text. `reading_threshold`'s
    own rule is that the cut comes from the negative class *in the same
    condition*; borrowing the encoded condition's cut would compare across
    conditions, which is the shift that rule exists to prevent.
    """
    detail = probe_transfer_detail(
        plain_harmful_batch,
        plain_harmless_batch,
        xstest_unsafe_batch,
        xstest_safe_batch,
        layer=layer,
        position=position,
        config=config,
    )
    # The probe's own operating point on plain text. `probe_transfer_detail`
    # returns the TEST-side scores, so the plain negatives are refit-free here:
    # score the training negatives through the same boundary.
    plain_detail = probe_transfer_detail(
        plain_harmful_batch,
        plain_harmless_batch,
        plain_harmful_batch,
        plain_harmless_batch,
        layer=layer,
        position=position,
        config=config,
    )
    threshold = reading_threshold(plain_detail.negative_scores, config)
    return measure_lexical_decorrelation(
        safe_scores=detail.negative_scores,
        safe_types=list(safe_types),
        unsafe_scores=detail.positive_scores,
        unsafe_types=list(unsafe_types),
        threshold=threshold,
    )
