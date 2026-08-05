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
from pathlib import Path

from internals_safety.config import load_measurements_config
from internals_safety.measurements.deployment import measure_deployment
from internals_safety.measurements.recognition import measure_recognition
from internals_safety.models.capture import ActivationBatch


def find_batch(activations: Path, prefix: str) -> Path | None:
    """Cached files are `<condition>-<digest>.pt`; the digest is not reproducible
    here (it hashes the capture request), so match on the condition prefix."""
    matches = sorted(activations.glob(f"{prefix}-*.pt"))
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous cache for {prefix!r}: {[m.name for m in matches]}. "
            "Refusing to guess which capture a licensing decision is based on."
        )
    return matches[0] if matches else None


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

    plain_harmful_path = find_batch(args.activations, "plain-harmful")
    plain_harmless_path = find_batch(args.activations, "plain-harmless")
    if not plain_harmful_path or not plain_harmless_path:
        raise SystemExit(f"missing plain-condition caches under {args.activations}")
    plain_harmful = ActivationBatch.load(plain_harmful_path)
    plain_harmless = ActivationBatch.load(plain_harmless_path)

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
        encoded_harmful_path = find_batch(args.activations, f"encoded-harmful-{family}")
        encoded_harmless_path = find_batch(args.activations, f"encoded-harmless-{family}")
        if not encoded_harmful_path or not encoded_harmless_path:
            print(f"{family:<20} (no cached activations — skipped)")
            continue
        encoded_harmful = ActivationBatch.load(encoded_harmful_path)
        encoded_harmless = ActivationBatch.load(encoded_harmless_path)

        curve = measure_deployment(
            family, plain_harmful, plain_harmless, encoded_harmful, encoded_harmless, config
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

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
