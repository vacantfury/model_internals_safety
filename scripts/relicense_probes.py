"""Re-license the phase-0 probes from CACHED activations — no GPU, no judge calls.

Why this exists (2026-08-05). The pilot's `results.json` records probe licensing
computed under the retired rule: a fixed `auroc >= 0.70` cut applied per cell,
with `deployed = any(cell)` over the whole (layer x position) grid. Both halves
were wrong in opposite directions — the number was a guess, and `any()` over
hundreds of cells is an uncorrected multiple comparison. Recognition moved to a
permutation test in `4d3e78d`; deployment followed on 2026-08-05.

That matters more for deployment than for recognition, because deployment is the
axis every regime label is decided on: an under-powered licensing rule there does
not weaken a reported number, it turns whole rungs into (U). Under the old cut,
13 of 15 rungs on Llama-3.1-8B licensed nowhere — and `hex` missed by 0.009
(0.691 against 0.70) on a rung whose ability is 84/100.

This script re-asks licensing against the cached activation tensors, so it needs
no model weights, no generation and no judge: it loads `.pt` batches, refits the
probes, draws the permutation nulls, and prints old-versus-new licensing per
family. CPU only.

Usage (from a machine with the activation cache mounted):

    uv run python scripts/relicense_probes.py \
        --activations /path/to/activations/<model> \
        --results outputs/runs/phase0/<model>/<run>/results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from internals_safety.config import load_measurements_config
from internals_safety.measurements.deployment import measure_deployment
from internals_safety.measurements.length_null import length_strata
from internals_safety.measurements.recognition import measure_recognition
from internals_safety.models.capture import ActivationBatch


@dataclass(frozen=True)
class Resolved:
    """A cache file plus HOW it was found. The route is recorded in the output
    because a licensing decision resting on a guessed cache and one resting on
    the run record's own statement are not the same claim."""

    path: Path
    route: str  # "record" | "unique"


def resolve_batch(activations: Path, prefix: str, recorded: str | None) -> Resolved | None:
    """Find a cached batch, preferring what the run record NAMES.

    **The record is authoritative and the glob is the fallback — that ordering is
    the fix for TODO item 21** (2026-08-07). Cached files are `<condition>-<digest>.pt`
    and the digest hashes the capture request, so it is not reproducible here;
    the original implementation could therefore only prefix-match, and it refused
    outright when more than one file matched. That refusal was correct and it
    blocked all 15 Qwen rungs: the head run was killed at the 8h wall and the
    tail run re-captured, leaving two `plain-harmful`, two `plain-harmless` and
    two `base64` captures with nothing to tell them apart.

    But `results.json` records `activations_path` — the run naming its own
    caches, which is the authoritative link the glob was standing in for.
    Measured 2026-08-07: on Qwen exactly THREE conditions are duplicated, not the
    whole set, so preferring the record recovers 14 of 15 rungs and leaves only
    `base64` genuinely undecidable.

    Falls back to a glob ONLY when it is unambiguous, and still refuses when it
    is not — an unrecorded, duplicated capture is exactly the case where guessing
    would put a licensing decision on an unknown tensor.
    """
    if recorded:
        named = Path(recorded)
        if named.exists():
            return Resolved(named, "record")
        # The record stores absolute paths from the machine that wrote it; a
        # cache reachable under a different mount is still the same file, and
        # the digest in the NAME is what identifies it.
        relocated = activations / named.name
        if relocated.exists():
            return Resolved(relocated, "record")

    matches = sorted(activations.glob(f"{prefix}-*.pt"))
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous cache for {prefix!r}: {[m.name for m in matches]}, and the run "
            "record does not name one. Refusing to guess which capture a licensing "
            "decision is based on — re-capture this condition, or point --results at "
            "the record that wrote it."
        )
    return Resolved(matches[0], "unique") if matches else None


def recorded_paths(results_path: Path) -> dict:
    """`activations_path` from the run record, or `{}` on an older record.

    Empty is a legitimate answer, not an error: the Qwen HEAD run wrote no
    `results.json` at all — the 8h wall killed it first — so nine of its rungs
    have caches on disk that no record names. Those still resolve, by unique
    glob, and their route says so.
    """
    record = json.loads(results_path.read_text())
    paths = record.get("activations_path")
    return paths if isinstance(paths, dict) else {}


def old_licensing(results_path: Path) -> dict[str, dict]:
    record = json.loads(results_path.read_text())
    return {e["family"]: e for e in record.get("metrics", {}).get("families", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args()

    config = load_measurements_config().probes
    previous = old_licensing(args.results)
    recorded = recorded_paths(args.results)
    per_family_recorded = recorded.get("per_family", {})

    plain_harmful_ref = resolve_batch(args.activations, "plain-harmful", recorded.get("plain_harmful"))
    plain_harmless_ref = resolve_batch(args.activations, "plain-harmless", recorded.get("plain_harmless"))
    if not plain_harmful_ref or not plain_harmless_ref:
        raise SystemExit(f"missing plain-condition caches under {args.activations}")
    plain_harmful = ActivationBatch.load(plain_harmful_ref.path)
    plain_harmless = ActivationBatch.load(plain_harmless_ref.path)
    print(
        f"plain contrast set: {plain_harmful_ref.path.name} / {plain_harmless_ref.path.name} "
        f"(via {plain_harmful_ref.route})\n"
    )

    families = args.families or sorted(
        p.name.split("-")[2]
        for p in args.activations.glob("encoded-harmful-*.pt")
    )

    print(f"alpha={config.alpha}  n_permutations={config.n_permutations}  "
          f"effect-size bar (reported, not licensing) auroc>={config.auroc_threshold}\n")
    header = (
        f"{'rung':<20} {'dep OLD':>8} {'dep NEW':>8} {'dep p':>8} {'maxAUROC':>9}   "
        f"{'rec OLD':>8} {'rec NEW':>8} {'rec p':>8}   change"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for family in families:
        family_record = per_family_recorded.get(family, {})
        harmful_ref = resolve_batch(
            args.activations, f"encoded-harmful-{family}", family_record.get("encoded_harmful")
        )
        harmless_ref = resolve_batch(
            args.activations, f"encoded-harmless-{family}", family_record.get("encoded_harmless")
        )
        if not harmful_ref or not harmless_ref:
            print(f"{family:<20} (no cached activations — skipped)")
            continue
        encoded_harmful = ActivationBatch.load(harmful_ref.path)
        encoded_harmless = ActivationBatch.load(harmless_ref.path)

        # The length-MATCHED null, rebuilt from the ciphertexts the cache itself
        # carries. `user_messages` is what the model was actually sent, so the
        # strata match the tensors being re-licensed rather than a corpus name.
        #
        # ⚠️ REFUSE rather than fall back. An older cache with no `user_messages`
        # cannot support the matched null, and running the unmatched one instead
        # would silently reproduce the defect this fix exists to close — job
        # 8995184 licensed 14/15 rungs under exactly that superseded test.
        if not encoded_harmful.user_messages or not encoded_harmless.user_messages:
            raise SystemExit(
                f"{family}: cached batch carries no `user_messages`, so the length-matched "
                "permutation null cannot be rebuilt. Re-capture this condition — do NOT "
                "re-license it under the unmatched null."
            )
        strata = length_strata(
            encoded_harmful.user_messages,
            encoded_harmless.user_messages,
            config.length_strata_bins,
        )
        curve = measure_deployment(
            family,
            plain_harmful,
            plain_harmless,
            encoded_harmful,
            encoded_harmless,
            config,
            strata=strata,
        )
        recognition = measure_recognition(encoded_harmful, encoded_harmless, config)

        # `None` = this family is absent from the run record, so its OLD
        # licensing is UNKNOWN. Reporting it as False would invent a
        # comparison — and the Qwen head run has no record at all, because the
        # job was killed at the 8h wall before it could write one.
        prior = previous.get(family)
        dep_old = bool(prior["deployment"]["licensed"]) if prior else None
        rec_old = bool(prior["recognition"]["licensed"]) if prior else None
        dep_new, rec_new = curve.deployed, recognition.recognized

        change = []
        if dep_old is None:
            change.append("old licensing UNKNOWN (family absent from run record)")
        elif dep_new != dep_old:
            change.append(f"deployment {'GAINED' if dep_new else 'LOST'}")
        if rec_old is not None and rec_new != rec_old:
            change.append(f"recognition {'GAINED' if rec_new else 'LOST'}")

        fmt = lambda v: "?" if v is None else str(v)
        print(
            f"{family:<20} {fmt(dep_old):>8} {str(dep_new):>8} {curve.p_value:>8.4f} "
            f"{curve.observed_max_transfer_auroc:>9.4f}   "
            f"{fmt(rec_old):>8} {str(rec_new):>8} {recognition.p_value:>8.4f}   "
            + ", ".join(change)
        )
        rows.append(
            {
                "family": family,
                "deployment_licensed_old": dep_old,
                "deployment_licensed_new": dep_new,
                "deployment_p_value": curve.p_value,
                "deployment_max_transfer_auroc": curve.observed_max_transfer_auroc,
                "deployment_meets_effect_size_bar": curve.meets_effect_size_bar,
                "recognition_licensed_old": rec_old,
                "recognition_licensed_new": rec_new,
                "recognition_p_value": recognition.p_value,
                # Provenance of the tensors this row rests on. "record" means the
                # run named its own cache; "unique" means only one file matched
                # and nothing contradicted it. A reader can tell the two apart.
                "encoded_harmful_cache": harmful_ref.path.name,
                "encoded_harmless_cache": harmless_ref.path.name,
                "cache_route": harmful_ref.route,
            }
        )

    known = [r for r in rows if r["deployment_licensed_old"] is not None]
    gained = [r["family"] for r in known if r["deployment_licensed_new"] and not r["deployment_licensed_old"]]
    lost = [r["family"] for r in known if r["deployment_licensed_old"] and not r["deployment_licensed_new"]]
    unknown = [r["family"] for r in rows if r["deployment_licensed_old"] is None]
    print(
        f"\ndeployment licensing NOW: {sum(r['deployment_licensed_new'] for r in rows)}/{len(rows)} rungs"
    )
    if known:
        print(f"  comparable rungs (in the run record): was {sum(r['deployment_licensed_old'] for r in known)}/{len(known)}")
    if unknown:
        print(f"  old licensing UNKNOWN for {len(unknown)}: {', '.join(unknown)}")
    if gained:
        print(f"  GAINED: {', '.join(gained)}")
    if lost:
        print(f"  LOST:   {', '.join(lost)}")

    by_route: dict[str, int] = {}
    for row in rows:
        by_route[row["cache_route"]] = by_route.get(row["cache_route"], 0) + 1
    if by_route:
        print("  cache resolution: " + ", ".join(f"{n} via {route}" for route, n in sorted(by_route.items())))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
