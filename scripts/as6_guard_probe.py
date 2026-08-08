"""AS-6 phase 1 — did the guard decode it, or did it just not block it?

`text_docs/as6/s0_reshape.md` §6, phase 1: forward passes only, no training, $0.
The reshape's three steps, of which step 1 (re-baselining AS-5's cached cells) is
already done. This script is steps 2 and 3.

## What it measures, and why generation is not involved anywhere

Llama Guard 3's chat template hard-wires the classification task and ends "First
line must read 'safe' or 'unsafe'." There is no in-distribution channel through
which a guard can be asked to restate a decoded payload — it answers `safe`. So
AS-5's ability measurement (decode-and-restate) does not port, and the decode
question has to be asked of the residual stream instead:

    decode   the plain-fitted content probe read on the encoded condition at the
             payload-final position. Never refitted on encoded activations —
             refitting would ask whether encoded prompts are separable at all,
             which they are, on surface form.
    verdict  P(unsafe) > P(safe) read from logits at the verdict position, one
             forward pass, no sampling.

Their cross is `measurements/guard_regimes.py`: the guard worked, it decoded and
let it through anyway, it blocked without ever decoding (a format detector), or
it never saw the content. That fourth cell is invisible to end-to-end ASR, which
is the confound AS-6 exists to remove.

**No judge is called at any point**, which is why this run costs $0: decode is a
probe, and the verdict is the guard's own logits.

## Two controls that are not optional

**Length null.** Character length alone separates the harmful corpus from the
benign one at AUROC 0.6544, and every encoder is monotone in length, so the
signal passes through all 19 rungs at 0.647-0.657. A probe scoring 0.66 on a
rung has measured length, not decoded content. Every rung reports its margin over
that baseline (`measurements/length_null.py`). The margin is REPORTED beside
licensing rather than gating it while the rule is settled (TODO 17b), so this
script never silently drops a rung — it labels it.

**Unmeasured is not negative.** Both the decode probe and the cell map are
tri-state. An unlicensed probe means the instrument could not read the rung, and
recording that as "the guard did not decode" would manufacture precisely AS-6's
headline finding out of a measurement failure — the defect AS-5 hit twice.

## The verdict-position check rides along

Llama Guard 3's template ends with no trailing newline, so the model may supply
`\\n\\n` before its verdict and `verdict_prefix: ""` would read one token too
early (TODO 13). Every reading records its argmax token, and each rung reports
`verdict_format_health`. Low label mass plus a consistent non-label argmax means
the prefix is wrong and the run is invalid — a condition, not a finding, so it
lands in `results.json` rather than a log line.

Usage:

    uv run python scripts/as6_guard_probe.py --guard llama_guard_3_8b --dry-run
    ./run python scripts/as6_guard_probe.py --guard llama_guard_3_8b

The dry-run path loads no model and needs no keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from internals_safety.config import (
    GuardConfig,
    MeasurementsConfig,
    load_guard_config,
    load_measurements_config,
    load_corpus_config,
)
from internals_safety.data import Prompt, digest
from internals_safety.encodings.base import Encoder
from internals_safety.encodings.registry import load_ladder
from internals_safety.guards.prompts import prepare_guard_prompts
from internals_safety.guards.verdict import read_verdicts, verdict_format_health
from internals_safety.measurements.deployment import (
    measure_deployment,
    read_deployment_per_prompt,
)
from internals_safety.measurements.guard_regimes import (
    assign_guard_cell,
    build_guard_cell_map,
)
from internals_safety.measurements.length_null import length_strata, measure_length_null
from internals_safety.models.capture import (
    ActivationBatch,
    cache_path,
    capture_activations,
    resolve_layers,
)
from internals_safety.models.loader import LoadedModel, load_model, resolve_device
from internals_safety.pipeline import (
    add_common_arguments,
    load_contrast_sets,
    resolve_run_paths,
    run_families,
    select_known,
)
from internals_safety.provenance import (
    capture_provenance,
    guard_working_tree,
    write_run_record,
)

PHASE = "as6_phase1"


def capture_guard(
    loaded: LoadedModel,
    payloads: Sequence[str],
    condition: str,
    cache_dir: Path,
    batch_size: int | None = None,
    refresh: bool = False,
) -> tuple[ActivationBatch, Path, bool]:
    """Capture activations for payloads rendered through the GUARD's prompt.

    Deliberately not `models.capture.capture_or_load`: that helper renders through
    `prepare_prompts`, i.e. the plain chat template, which fails closed on a guard
    that ships none (WildGuard) and would silently use the wrong prompt on one
    that does. Everything else about it — the cache key, the return contract — is
    reused rather than reinvented.

    **The cache key is the RENDERED prompt, not the payload.** That is what makes
    editing `conf/guards/*.yaml` invalidate stale tensors automatically: a changed
    template, `verdict_prefix`, or `prepend_bos` produces different text and
    therefore a different key. Keying on the payload would have silently served
    activations captured under the OLD prompt format after a config fix — which is
    exactly the class of defect this project has already paid for twice.
    """
    prompts = prepare_guard_prompts(loaded, payloads)
    rendered = [prompt.text for prompt in prompts]
    layers = resolve_layers(loaded, None)
    positions = list(loaded.config.capture.positions)
    site = loaded.config.capture.site
    path = cache_path(loaded, condition, rendered, layers, positions, site, cache_dir)

    if path.exists() and not refresh:
        return ActivationBatch.load(path), path, True

    batch = capture_activations(
        loaded, prompts, batch_size=batch_size or loaded.config.capture_batch_size
    )
    batch.save(path)
    return batch, path, False


def run_family(
    loaded: LoadedModel,
    encoder: Encoder,
    harmful: Sequence[Prompt],
    harmless: Sequence[Prompt],
    plain_harmful_batch,
    plain_harmless_batch,
    measurements: MeasurementsConfig,
    cache_dir: Path,
    refresh: bool = False,
) -> dict:
    """Every phase-1 measurement for one rung."""
    config: GuardConfig = loaded.config
    family = encoder.family

    encoded_harmful = [encoder.encode(prompt.text) for prompt in harmful]
    encoded_harmless = [encoder.encode(prompt.text) for prompt in harmless]

    # The ATTACK prompt, as in AS-5: the condition where nothing asks for a decode.
    harmful_payloads = [item.attack_prompt for item in encoded_harmful]
    harmless_payloads = [item.attack_prompt for item in encoded_harmless]

    encoded_harmful_batch, harmful_cache, harmful_hit = capture_guard(
        loaded, harmful_payloads, f"encoded-harmful-{family}", cache_dir, refresh=refresh
    )
    encoded_harmless_batch, harmless_cache, harmless_hit = capture_guard(
        loaded, harmless_payloads, f"encoded-harmless-{family}", cache_dir, refresh=refresh
    )

    # LENGTH-MATCHED licensing (2026-08-05). Strata are built from the ciphertexts
    # actually sent, in the order the probe layer concatenates them (harmful then
    # harmless), so labels are permuted only among prompts of similar length. A
    # probe separating on length alone therefore scores as well under the null as
    # on the real labels and cannot license. Without this, 20 of 38 (guard, rung)
    # pairs licensed at the length baseline in the first phase-1 run.
    strata = length_strata(
        [item.ciphertext for item in encoded_harmful],
        [item.ciphertext for item in encoded_harmless],
        measurements.probes.length_strata_bins,
    )
    curve = measure_deployment(
        family,
        plain_harmful_batch,
        plain_harmless_batch,
        encoded_harmful_batch,
        encoded_harmless_batch,
        measurements.probes,
        strata=strata,
    )
    decode = read_deployment_per_prompt(
        curve,
        plain_harmful_batch,
        plain_harmless_batch,
        encoded_harmful_batch,
        encoded_harmless_batch,
        measurements.probes,
    )

    null = measure_length_null(
        family,
        [prompt.text for prompt in harmful],
        [prompt.text for prompt in harmless],
        [item.ciphertext for item in encoded_harmful],
        [item.ciphertext for item in encoded_harmless],
    )
    margin = null.margin(curve.observed_max_transfer_auroc)
    beats_null = null.beats_null(
        curve.observed_max_transfer_auroc, measurements.probes.length_null_min_margin
    )

    verdicts, tokens = read_verdicts(loaded, harmful_payloads)

    cells = [
        assign_guard_cell(decoded=decoded, blocked=verdict.unsafe)
        for decoded, verdict in zip(decode.harmful, verdicts)
    ]
    cell_map = build_guard_cell_map(config.name, family, cells)

    records = [
        {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "guard": config.name,
            "family": family,
            "plaintext": item.plaintext,
            "ciphertext": item.ciphertext,
            "decoded": decoded,
            "blocked": verdict.unsafe,
            "p_unsafe": verdict.p_unsafe,
            "p_safe": verdict.p_safe,
            "top_token": verdict.top_token,
            "top_prob": verdict.top_prob,
            "cell": cell.value,
            # The raw probe score behind `decoded`, with the threshold it was
            # compared against. Emitted so that changing the operating point is
            # offline arithmetic rather than a cluster re-run: `decoded` is
            # `decode_score > decode_threshold`, and nothing else in this record
            # depends on where that threshold sits.
            "decode_score": score,
            "decode_threshold": decode.threshold_score,
        }
        for prompt, item, decoded, score, verdict, cell in zip(
            harmful, encoded_harmful, decode.harmful, decode.harmful_scores, verdicts, cells
        )
    ]

    summary = {
        **cell_map.as_dict(),
        "invertibility": encoder.invertibility.value,
        "decode": {
            "licensed": decode.licensed,
            "layer": decode.layer,
            "position": decode.position,
            "transfer_auroc": decode.transfer_auroc,
            "observed_max_transfer_auroc": curve.observed_max_transfer_auroc,
            "p_value": curve.p_value,
            "meets_effect_size_bar": curve.meets_effect_size_bar,
            "harmful_rate": decode.harmful_rate,
            "harmless_rate": decode.harmless_rate,
            "gap": decode.gap,
            # The operating point, recorded explicitly rather than left implicit
            # in the config: `harmless_rate` is 1 - reading_percentile/100 BY
            # CONSTRUCTION, so a reader who does not know the percentile cannot
            # tell a finding from a threshold. At the default 50 the benign
            # false-positive rate is 50%, and the informative quantity is `gap`.
            "reading_percentile": measurements.probes.reading_percentile,
            "threshold_score": decode.threshold_score,
            # The benign score distribution the threshold is a percentile OF.
            # cells.jsonl carries the harmful scores; without these the benign
            # side is unrecoverable and no other operating point can be computed
            # offline, which is the whole point of emitting scores at all.
            "harmless_scores": decode.harmless_scores,
        },
        # Reported beside licensing, never silently gating it (TODO 17b). A rung
        # licensed by permutation but sitting AT the length baseline is the exact
        # pattern that produced a false 14-of-15 map on the AS-5 side.
        "length_null": {
            # Licensing is now length-MATCHED (labels permuted within strata), so
            # `decode.p_value` already accounts for length. The margin below stays
            # reported as a MAGNITUDE, exactly as auroc_threshold did once
            # permutation licensing replaced it.
            "licensing": "length_matched_permutation",
            "strata_bins": measurements.probes.length_strata_bins,
            "plain_auroc": null.plain_auroc,
            "encoded_auroc": null.encoded_auroc,
            "margin": margin,
            "min_margin": measurements.probes.length_null_min_margin,
            "beats_length_null": beats_null,
            "mean_positive_chars": null.mean_positive_chars,
            "mean_negative_chars": null.mean_negative_chars,
        },
        # Run-validity condition, not a finding: see the module docstring.
        "verdict_format": {
            **verdict_format_health(verdicts),
            "safe_token_id": tokens.safe_id,
            "unsafe_token_id": tokens.unsafe_id,
            "safe_piece": tokens.safe_piece,
            "unsafe_piece": tokens.unsafe_piece,
            "verdict_prefix": config.verdict_prefix,
        },
        "mean_p_unsafe": sum(v.p_unsafe for v in verdicts) / len(verdicts) if verdicts else 0.0,
        # Named so "re-run it" is a well-defined instruction: capture is the
        # expensive step, and a reported number must say which tensors produced it.
        "activations": {
            "encoded_harmful": str(harmful_cache),
            "encoded_harmless": str(harmless_cache),
            "cache_hits": [harmful_hit, harmless_hit],
        },
        "mean_ciphertext_chars": (
            sum(len(item.ciphertext) for item in encoded_harmful) / len(encoded_harmful)
            if encoded_harmful
            else 0.0
        ),
    }
    return {"summary": summary, "cells": records}


def describe_plan(config: GuardConfig, families: Sequence[str], n: int, device: str) -> str:
    """The approval-gate estimate, printed before anything loads.

    Forward passes only: 2 capture passes per rung plus one verdict pass, over n
    prompts, with NO generation and NO judge call anywhere. That is why the money
    line is exactly zero rather than a small number.
    """
    per_rung = 3 * n
    total = per_rung * len(families) + 2 * n
    return "\n".join(
        [
            f"guard          {config.name} ({config.hf_id})",
            f"prompt style   {config.prompt_style}",
            f"verdict        prefix={config.verdict_prefix!r} "
            f"labels={config.safe_token!r}/{config.unsafe_token!r}",
            f"device         {device}",
            f"rungs          {len(families)}: {', '.join(families)}",
            f"prompts        {n} harmful + {n} benign",
            "",
            f"forward passes {total} ({per_rung} per rung + {2 * n} plain, captured once)",
            "generations    0",
            "judge calls    0",
            "money          $0.00",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    corpus = load_corpus_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--guard", required=True, help="conf/guards/<name>.yaml")
    add_common_arguments(parser, default_n_prompts=corpus.n_prompts)
    parser.add_argument(
        "--strata-bins",
        type=int,
        default=None,
        help="length strata for the matched null (default: conf/measurements.yaml). "
        "Re-running with 5 / 10 / 20 is the stability check the config names as this "
        "knob's tuning path; with the capture cache warm it costs no forward passes.",
    )
    args = parser.parse_args(argv)

    config = load_guard_config(args.guard)
    measurements = load_measurements_config()
    if args.strata_bins is not None:
        measurements = measurements.model_copy(
            update={
                "probes": measurements.probes.model_copy(
                    update={"length_strata_bins": args.strata_bins}
                )
            }
        )
    ladder = load_ladder()
    families = select_known(args.families, ladder, label="rungs")

    device = resolve_device(config.device, allow_cpu_in_job=args.allow_cpu)
    print(describe_plan(config, families, args.n_prompts, str(device)))
    if args.dry_run:
        return 0

    guard_working_tree(device, allow_dirty=args.allow_dirty)

    harmful, harmless = load_contrast_sets(corpus.harmful_set, corpus.harmless_set, args.n_prompts)

    directory, activations_dir, run_name = resolve_run_paths(
        PHASE, config.name, args.run_name, args.outputs_dir
    )

    print(f"\nloading {config.hf_id} ...", flush=True)
    loaded = load_model(config)

    # Rung-independent, captured once. Note these go through the GUARD's prompt
    # too: the probe must be fitted on the same distribution it is read on, minus
    # the encoding. Fitting on bare plaintext would confound "decoded the payload"
    # with "is inside a classification template".
    plain_harmful_batch, _, _ = capture_guard(
        loaded, [p.text for p in harmful], "plain-harmful", activations_dir,
        refresh=args.refresh_activations,
    )
    plain_harmless_batch, _, _ = capture_guard(
        loaded, [p.text for p in harmless], "plain-harmless", activations_dir,
        refresh=args.refresh_activations,
    )

    plain_verdicts, _ = read_verdicts(loaded, [prompt.text for prompt in harmful])
    plain_block_rate = sum(v.unsafe for v in plain_verdicts) / len(plain_verdicts)
    plain_health = verdict_format_health(plain_verdicts)
    print(f"\nplaintext block rate: {plain_block_rate:.2f}  (the ceiling)")
    print(f"verdict format: {plain_health}", flush=True)

    raw_path = directory / "cells.jsonl"

    def report(result: dict) -> None:
        summary = result["summary"]
        decode, null = summary["decode"], summary["length_null"]
        print(
            f"    decode auroc {decode['observed_max_transfer_auroc']:.3f} "
            f"p={decode['p_value']:.4f} licensed={decode['licensed']} | "
            f"length null {null['encoded_auroc']:.3f} margin {null['margin']:+.3f} "
            f"beats={null['beats_length_null']}",
            flush=True,
        )
        print(
            f"    block {summary['block_rate']:.2f} | "
            f"decoded-not-blocked {summary['decoded_not_blocked_rate']} | "
            f"blocked-without-decoding {summary['blocked_without_decoding_rate']} | "
            f"unmeasured {summary['n_unmeasured']}/{summary['n']}",
            flush=True,
        )

    summaries, readings, elapsed_seconds = run_families(
        families,
        directory,
        lambda family: run_family(
            loaded,
            ladder[family],
            harmful,
            harmless,
            plain_harmful_batch,
            plain_harmless_batch,
            measurements,
            activations_dir,
            refresh=args.refresh_activations,
        ),
        report=report,
        # EXPLICITLY none, and this line is the decision rather than an omission
        # (TODO 60). AS-6 must derive its OWN floor PER GUARD — the AS-5 screen
        # is calibrated on a target model's can't-decode rungs and says nothing
        # about what WildGuard or Llama Guard 3 reads on the same rungs. Wiring
        # AS-5's screen here would be worse than none: a floor from the wrong
        # instrument is a screen that passes the wrong things.
        #
        # The guard-side ladder must first carry enough can't-decode rungs to
        # estimate a control DISTRIBUTION rather than bound it — n=2 yields no
        # SD (`instrument_layer.md` §2.4). Until that lands, guard-side
        # deployment readings carry no floor screen, which the contract's
        # `required_controls` already treats as disqualifying, not as passing.
        cross_rung_screen=None,
    )

    record = capture_provenance(
        config={
            "guard": config.model_dump(),
            "measurements": measurements.model_dump(),
            "families": families,
            "n_prompts": args.n_prompts,
        },
        seed=measurements.probes.seed,
        extra={
            "phase": PHASE,
            "corpus_digest": {"harmful": digest(harmful), "harmless": digest(harmless)},
            "plain_block_rate": plain_block_rate,
            "plain_verdict_format": plain_health,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "summaries": summaries,
            "cells_path": str(raw_path),
        },
    )
    path = write_run_record(directory, record, readings)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
