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
import dataclasses
from pathlib import Path
from typing import Sequence

from internals_safety.config import (
    GuardConfig,
    MeasurementsConfig,
    load_guard_config,
    load_measurements_config,
    load_corpus_config,
)
from internals_safety.data import Prompt, digest, prompt_set as load_prompt_set
from internals_safety.encodings.base import Encoder
from internals_safety.encodings.registry import load_ladder
from internals_safety.guards.prompts import prepare_guard_prompts
from internals_safety.guards.verdict import read_verdicts, verdict_format_health
from internals_safety.measurements.causal import (
    forward_passes as causal_forward_passes,
    guard_verdict_probe,
    run_causal_gate,
)
from internals_safety.measurements.deployment import (
    measure_deployment,
    read_deployment_per_prompt,
)
from internals_safety.measurements.guard_benign_control import (
    summarize_control as summarize_guard_benign_control,
)
from internals_safety.measurements.guard_scaffold_control import (
    summarize_control as summarize_guard_scaffold_control,
    verdict_passes as scaffold_verdict_passes,
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
    XStestCapture,
    add_common_arguments,
    run_lexical_control,
    load_contrast_sets,
    resolve_run_paths,
    run_families,
    scaffold_arm,
    select_known,
)
from internals_safety.provenance import (
    capture_provenance,
    guard_working_tree,
    write_run_record,
)

PHASE = "as6_phase1"

# The optional roster. `lexical` is not decorative: `deployment.REQUIRED_CONTROLS`
# names the XSTest vocabulary screen, so a decode reading produced without it is
# WITHHELD by the contract. This script had no way to run it until 2026-08-08,
# which means every AS-6 decode number before that date was non-reportable — not
# wrong, unusable, and nothing said so because the entrypoint could not express
# the control at all.
OPTIONAL_INSTRUMENTS = ("lexical", "causal_license")


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
    # Keyword-only with NO default, so omitting the plaintext arm is a
    # `TypeError` rather than a silent zero. Fifth application of the rule this
    # repo has paid for four times in a week (`strata`, `device`, `inherited`,
    # the control floor): a rule threaded into its callers is a rule that will
    # be dropped by the next caller; a rule whose omission does not compile is
    # not. A default of 0.0 here would report the guard as discriminating
    # nothing in plaintext and make every wrapper term wrong by that amount.
    *,
    plain_harmful_block_rate: float,
    plain_benign_block_rate: float,
    refresh: bool = False,
    xstest: XStestCapture | None = None,
) -> dict:
    """Every phase-1 measurement for one rung.

    `xstest` carries the model-level capture the lexical control reads, captured
    once in `main` and passed down. `None` means the control was not requested,
    and then the decode reading is non-reportable by the contract — which is
    what every AS-6 run before 2026-08-08 was, silently.

    The two plain block rates are model-level and identical for every rung; they
    are passed down rather than re-measured so the scaffold decomposition can be
    read off one rung's summary without joining it back to the run record.
    """
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

    # THE BENIGN ARM (2026-08-08). One prefill-only verdict pass over payloads
    # already built and already captured for the probe fit — no generation, no
    # judge, no money. Until this existed the entrypoint read verdicts on
    # `harmful_payloads` ALONE, so every `blocked` number AS-6 had was a rate
    # with no benign denominator, and "the guard recognised harm through the
    # encoding" was indistinguishable from "the guard flags anything that looks
    # encoded". Rationale: `measurements/guard_benign_control.py`.
    benign_verdicts, _ = read_verdicts(loaded, harmless_payloads)
    harmful_block_rate = sum(v.unsafe for v in verdicts) / len(verdicts) if verdicts else 0.0
    benign_control = summarize_guard_benign_control(
        family=family,
        guard=config.name,
        benign_blocked=[v.unsafe for v in benign_verdicts],
        harmful_block_rate=harmful_block_rate,
    )

    # THE SCAFFOLD ARM (2026-08-09, TODO 65). Plaintext content wearing THIS
    # rung's attack wrapper — the factorial cell that separates "the guard
    # blocked the encoded content" from "the guard flags anything asking about
    # an encoding". Two prefill-only verdict passes, no capture, no judge, $0.
    #
    # ⚠️ NOT behind `--instruments`, deliberately, and that is the whole lesson
    # of TODO 61: measurement #4's benign arm sat behind a flag for a month and
    # therefore never ran, so every ASR the repo held was unscreened. A control
    # that can be omitted is a control that will be. The plain baseline was made
    # mandatory for the same reason a day earlier.
    #
    # The wrapper is per-rung, not shared: `scaffold_family` bakes the rung's
    # own `attack_template` in, so "encoded with Base64" and "encoded with
    # homoglyph substitution" are different prompts and one scaffold arm cannot
    # stand in for another's.
    scaffold_items = scaffold_arm([prompt.text for prompt in harmful], encoder)
    scaffold_benign_items = scaffold_arm([prompt.text for prompt in harmless], encoder)
    scaffold_harmful_verdicts, _ = read_verdicts(
        loaded, [item.attack_prompt for item in scaffold_items]
    )
    scaffold_benign_verdicts, _ = read_verdicts(
        loaded, [item.attack_prompt for item in scaffold_benign_items]
    )
    scaffold_control = summarize_guard_scaffold_control(
        family=family,
        guard=config.name,
        scaffold_harmful_blocked=[v.unsafe for v in scaffold_harmful_verdicts],
        scaffold_benign_blocked=[v.unsafe for v in scaffold_benign_verdicts],
        encoded_harmful_block_rate=harmful_block_rate,
        encoded_benign_block_rate=benign_control.benign_block_rate,
        plain_harmful_block_rate=plain_harmful_block_rate,
        plain_benign_block_rate=plain_benign_block_rate,
    )

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

    # The XSTest lexical control, read at the cell this rung's decode claim is
    # read at. Costs no forward pass here — the corpus was captured once at model
    # level and this is a logistic fit over activations already in hand.
    #
    # In JBB, alarming words and actual harm are perfectly confounded, so a guard
    # probe reading only "bomb" scores identically to one reading intent. The
    # length null and the ability-0 floor both leave that route open; XSTest
    # closes it by construction with matched pairs that share the vocabulary and
    # differ in intent.
    lexical_detail: dict = {}
    if xstest is not None:
        best = curve.best()
        lexical = run_lexical_control(
            plain_harmful_batch,
            plain_harmless_batch,
            xstest.safe_batch,
            xstest.unsafe_batch,
            xstest.safe_types,
            xstest.unsafe_types,
            layer=best.layer,
            position=best.position,
            config=measurements.probes,
        )
        lexical_detail = {
            "lexical_pooled_auroc": lexical.pooled_auroc,
            "lexical_false_positive_rate": lexical.lexical_false_positive_rate,
            "lexical_reads_vocabulary": lexical.reads_vocabulary,
            "lexical_clears": lexical.clears(
                measurements.controls.lexical_min_margin,
                measurements.controls.vocabulary_reader_floor,
            ),
            "lexical_floor": measurements.controls.vocabulary_reader_floor,
            "lexical_pairs": {pair.pair: pair.auroc for pair in lexical.pairs},
            # Zero scorable pairs reads exactly like a failure otherwise, and
            # `reads_vocabulary` fails CLOSED to True — the tri-state defect
            # this repo has now found in three instruments.
            "lexical_n_pairs": len(lexical.pairs),
            "lexical_cell": {"layer": best.layer, "position": best.position},
        }

    summary = {
        **cell_map.as_dict(),
        "invertibility": encoder.invertibility.value,
        "lexical": lexical_detail,
        # THE CONTROL ON THE BLOCK AXIS. Reported beside the cell counts and
        # never silently gating them — the same posture the length null takes
        # below, and for the same reason: a reader must be able to see that the
        # block rate was screened and by how much, not merely that a number
        # survived. `clears` is TRI-STATE (None = the benign arm was empty),
        # because an absent control is not a failed one.
        "benign_arm": {
            "n": benign_control.n,
            "benign_block_rate": benign_control.benign_block_rate,
            "harmful_block_rate": benign_control.harmful_block_rate,
            "margin": benign_control.margin,
            "bar": benign_control.bar,
            "clears": benign_control.clears(),
            # The verdict that decides what AS-6's central cell MEANS: a guard
            # blocking benign and harmful content at the same rate through this
            # encoding is a format detector, and `blocked_on_content` is then a
            # misnomer rather than a finding.
            "is_format_detector": benign_control.is_format_detector,
            "screen": dataclasses.asdict(benign_control.screen()),
        },
        # THE CONTROL ON WHAT THE BLOCK IS ABOUT. The benign arm above asks
        # whether the guard flags anything WEARING the encoding; this asks
        # whether it flags anything ASKING ABOUT one. A guard can clear the first
        # and fail this — discriminating harm inside the encoded condition says
        # nothing about whether the LEVEL of blocking was set by the wrapper.
        #
        # The decomposition is reported and never screened: `clears` reads the
        # single margin above the wrapper floor, while `wrapper_term` and
        # `character_term` describe how the loss splits. One number, one
        # question — the boundary `refusal.py` drew for the same attribution.
        "scaffold_arm": {
            "n": scaffold_control.n,
            "scaffold_harmful_block_rate": scaffold_control.scaffold_harmful_block_rate,
            "scaffold_benign_block_rate": scaffold_control.scaffold_benign_block_rate,
            "plain_gap": scaffold_control.plain_gap,
            "scaffold_gap": scaffold_control.scaffold_gap,
            "encoded_gap": scaffold_control.encoded_gap,
            "wrapper_term": scaffold_control.wrapper_term,
            "character_term": scaffold_control.character_term,
            "total_loss": scaffold_control.total_loss,
            "margin": scaffold_control.margin,
            "bar": scaffold_control.bar,
            "clears": scaffold_control.clears(),
            "is_wrapper_responder": scaffold_control.is_wrapper_responder,
            "screen": dataclasses.asdict(scaffold_control.screen()),
        },
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


def causal_passes(n_layers: int, positions: int, measurements, enabled: bool) -> tuple[int, int]:
    """(candidate directions, forward passes) the causal gate adds. 0 when off.

    Priced from `causal.forward_passes` and `causal.causal_candidate_cells`
    rather than restated, so the number the approval gate is shown comes from
    the same code that will run. The negative control is INCLUDED: the
    random-direction null is a second sweep, it is not optional in the code, and
    a cost estimate that omitted it under-reported the run — a mistake this repo
    made once already on the AS-5 side.
    """
    if not enabled:
        return 0, 0
    config = measurements.causal_license
    eligible = int(n_layers * (1.0 - config.prune_layer_percentage))
    if eligible <= 0:
        return 0, 0
    stride = max(1, -(-eligible // config.max_sweep_layers))
    candidates = len(range(0, eligible, stride)) * positions
    passes = causal_forward_passes(candidates)
    if config.n_random_directions:
        passes += causal_forward_passes(config.n_random_directions)
    return candidates, passes


def describe_plan(
    config: GuardConfig,
    families: Sequence[str],
    n: int,
    device: str,
    *,
    # plumbing(causal): (candidates, pass-units) from `causal_passes`, which
    # derives both from conf/measurements.yaml. (0, 0) means the gate is off —
    # a disabled-stage sentinel, not a tunable.
    causal: tuple[int, int] = (0, 0),
) -> str:
    """The approval-gate estimate, printed before anything loads.

    Forward passes only: per rung, 2 capture passes plus FOUR verdict passes —
    harmful and benign on the encoded arm, harmful and benign on the scaffold
    arm — over n prompts, with NO generation and NO judge call anywhere. That is
    why the money line is exactly zero rather than a small number.

    ⚠️ **Every arm is counted here, and that is not a detail.** When the benign
    arm landed (2026-08-08) this function still read `3 * n`, so the estimate
    under-reported the run by a quarter and the approval gate would have been
    shown a number for a run that no longer existed. `behavior_control` made the
    identical correction about itself on 2026-08-07 — *a control the cost
    estimate cannot see is a control nobody approved*.

    ⚠️ **And the model-level line was wrong in the same way until 2026-08-09.**
    It read `2 * n`, counting the two plain CAPTURE passes and silently omitting
    the plaintext verdict pass `main` has always run to print the block-rate
    ceiling. Adding the plain benign verdict for the scaffold decomposition
    turned one missing pass into two, which is what made it visible — a census
    that is only ever checked when it changes is not a census. It is `4 * n` now
    and derived below rather than restated.
    """
    # Derived from the controls themselves so the estimate cannot drift from
    # what runs: two encoded verdicts + the scaffold arm's own count.
    per_rung = 2 * n + 2 * n + scaffold_verdict_passes(n, n)
    # 2 captures + 2 verdicts, both harmful and benign, all four rung-independent.
    model_level = 2 * n + 2 * n
    n_candidates, causal_pass_units = causal
    # Each causal "pass" in `causal.forward_passes` is one sweep over a prompt
    # SET, so it costs n prompts. Converting here rather than in the counter
    # keeps that function comparable across entrypoints.
    causal_total = causal_pass_units * n
    total = per_rung * len(families) + model_level + causal_total
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
            f"forward passes {total} ({per_rung} per rung, incl. the scaffold arm"
            f" + {model_level} plain, captured and judged once"
            + (f" + {causal_total} causal)" if causal_total else ")"),
        ]
        + (
            [
                f"causal gate    {n_candidates} candidate directions x 3 passes + 2 baselines,"
                f" plus the random-direction null",
            ]
            if causal_total
            else []
        )
        + [
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
        "--instruments",
        nargs="+",
        default=None,
        help=f"optional instruments to run: {', '.join(OPTIONAL_INSTRUMENTS)}. Default "
        "none. `lexical` is the XSTest vocabulary screen, which "
        "`deployment.REQUIRED_CONTROLS` names as MANDATORY for a reportable decode "
        "reading — without it every decode number this script produces is withheld "
        "by the contract, which is what happened to every AS-6 run before "
        "2026-08-08.",
    )
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
    instruments = (
        select_known(args.instruments, OPTIONAL_INSTRUMENTS, label="instruments")
        if args.instruments
        else []
    )

    device = resolve_device(config.device, allow_cpu_in_job=args.allow_cpu)
    print(
        describe_plan(
            config,
            families,
            args.n_prompts,
            str(device),
            causal=causal_passes(
                config.n_layers or 0,
                len(config.capture.positions),
                measurements,
                enabled="causal_license" in instruments,
            ),
        )
    )
    if args.dry_run:
        return 0

    tree = guard_working_tree(device, allow_dirty=args.allow_dirty)

    _, pair = corpus.pair(None)
    harmful, harmless = load_contrast_sets(
        pair.harmful_set, pair.harmless_set, args.n_prompts, matching=pair.matching
    )

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

    # ---- the XSTest lexical control, MODEL-level ----------------------------
    # Captured once and read by every rung: the decode probe is plain-fitted, so
    # the control is the same corpus for every rung and capturing per family
    # would pay 11x for one answer.
    #
    # ⚠️ Through `capture_guard`, NOT the plain chat template. The probe is
    # fitted on plaintext rendered through the GUARD's classification prompt, so
    # the control has to sit in that same distribution — reading it on bare
    # plaintext would compare across conditions and answer a question nobody
    # asked. Same reason the plain contrast sets above go through it.
    xstest: XStestCapture | None = None
    if "lexical" in instruments:
        safe_prompts = load_prompt_set("xstest_safe_prompts.jsonl")
        unsafe_prompts = load_prompt_set("xstest_unsafe_prompts.jsonl")
        safe_batch, safe_cache, _ = capture_guard(
            loaded, [p.text for p in safe_prompts], "xstest-safe", activations_dir,
            refresh=args.refresh_activations,
        )
        unsafe_batch, unsafe_cache, _ = capture_guard(
            loaded, [p.text for p in unsafe_prompts], "xstest-unsafe", activations_dir,
            refresh=args.refresh_activations,
        )
        xstest = XStestCapture(
            safe_batch=safe_batch,
            unsafe_batch=unsafe_batch,
            safe_types=tuple(p.category for p in safe_prompts),
            unsafe_types=tuple(p.category for p in unsafe_prompts),
            safe_cache=safe_cache,
            unsafe_cache=unsafe_cache,
        )

    # ---- the causal gate, MODEL-level ---------------------------------------
    # Runs ONCE, between the plain captures and the family loop, for the reason
    # `causal.run_causal_gate` states: the direction is fit on plain harmful vs
    # plain harmless, is identical for every rung, and gates which direction the
    # downstream reads may use at all. Behind a flag rather than mandatory
    # because it is the only stage here that costs real GPU time.
    #
    # The probe is the GUARD one — guard rendering plus verdict prefix, P(unsafe)
    # renormalised over the two labels. Passing the generating-model probe would
    # read a refusal opening at a position the guard never answers at.
    causal_readings: list = []
    if "causal_license" in instruments:
        print("\nrunning the causal gate (ablate + add + random-direction null) ...", flush=True)
        causal_readings.append(
            run_causal_gate(
                loaded,
                plain_harmful_batch,
                plain_harmless_batch,
                [prompt.text for prompt in harmful],
                [prompt.text for prompt in harmless],
                probe=guard_verdict_probe(loaded),
                measurements=measurements,
                batch_size=config.capture_batch_size,
            )
        )
        gate = causal_readings[0]
        print(f"causal gate: licensed={gate.licensed} {gate.detail}", flush=True)

    plain_verdicts, _ = read_verdicts(loaded, [prompt.text for prompt in harmful])
    plain_block_rate = sum(v.unsafe for v in plain_verdicts) / len(plain_verdicts)
    plain_health = verdict_format_health(plain_verdicts)

    # THE PLAIN BENIGN ARM. Rung-independent, one prefill-only pass, and until
    # 2026-08-09 it did not exist — so the guard's plaintext ceiling was a rate
    # with no denominator, exactly what AS-5 §4d found on the target side and
    # made mandatory there. Without it the scaffold decomposition has no
    # baseline to measure the wrapper term against, and "the guard discriminates
    # harm in plaintext" was an assumption rather than a measurement.
    plain_benign_verdicts, _ = read_verdicts(loaded, [prompt.text for prompt in harmless])
    plain_benign_block_rate = sum(v.unsafe for v in plain_benign_verdicts) / len(
        plain_benign_verdicts
    )
    print(f"\nplaintext block rate: {plain_block_rate:.2f}  (the ceiling)")
    print(
        f"plaintext benign:     {plain_benign_block_rate:.2f}  "
        f"(gap {plain_block_rate - plain_benign_block_rate:+.2f})"
    )
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
        scaffold = summary["scaffold_arm"]
        print(
            f"    scaffold benign {scaffold['scaffold_benign_block_rate']:.2f} "
            f"clears={scaffold['clears']} wrapper-responder="
            f"{scaffold['is_wrapper_responder']} | gap plain {scaffold['plain_gap']:+.2f} "
            f"-> scaffold {scaffold['scaffold_gap']:+.2f} -> encoded "
            f"{scaffold['encoded_gap']:+.2f} (wrapper {scaffold['wrapper_term']:+.2f}, "
            f"chars {scaffold['character_term']:+.2f})",
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
            plain_harmful_block_rate=plain_block_rate,
            plain_benign_block_rate=plain_benign_block_rate,
            refresh=args.refresh_activations,
            xstest=xstest,
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
            "plain_benign_block_rate": plain_benign_block_rate,
            "plain_verdict_format": plain_health,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "summaries": summaries,
            "cells_path": str(raw_path),
        },
        tree=tree,
    )
    path = write_run_record(directory, record, causal_readings + readings)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
