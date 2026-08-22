#!/usr/bin/env python3
"""I7 — ablate the encoding direction and re-measure harm discrimination.

**The question.** AS-5's behavioural result cannot separate two hypotheses that
predict the same refusal rates and carry opposite safety implications: the
encoding either DESTROYS the model's harm recognition, or merely SUPPRESSES its
expression behind a refusal triggered by surface form. Llama-3.1-8B refuses
benign and harmful `homoglyph` prompts at an identical 0.99 either way. Design
and the reading: `measurements/encoding_direction.py`.

**Four conditions, none optional** — encoded baseline, encoded with the direction
ablated, encoded with a matched-norm RANDOM direction ablated, and plaintext.
The control because "ablation worked" and "perturbing anything by this much
worked" are otherwise one observation; the plaintext arm because a restored gap
of +0.20 is most of the way back on a model whose plaintext gap is +0.28 and a
quarter of the way on one at +0.82.

⚠️ **THE DIRECTION IS FITTED ON THE POOLED ARMS, and that is load-bearing.**
The contrast is {encoded harmful + encoded benign} vs {plain harmful + plain
benign} — harm balanced on both sides, so its component cancels and what remains
is the response to surface form. Fitting on the harmful arm alone would give a
direction confounded with harm, and ablating THAT would destroy discrimination
by construction: a guaranteed "recognition destroyed" verdict that is an artefact
of the fit, not a property of the model.

**Selection is by the outcome and the control is what makes that sound.** Arditi
et al. select the same way; the protection is that `select_cell` runs over the
random directions too, over the same candidate cells, by the same criterion. A
best-of-N real direction against one fixed random direction would report the
maximum of a sample as an effect.

Costs GPU time and judge calls. `--dry-run` prints the work and spends nothing;
it returns AFTER building the conditions, so it exercises the selection path
rather than only argument parsing (the 2026-08-07 lesson that a dry run touching
none of the real path is not evidence).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from internals_safety.provenance import capture_provenance, guard_working_tree, write_run_record
from internals_safety.config import (
    load_corpus_config,
    load_judge_config,
    load_measurements_config,
    load_model_config,
)
from internals_safety.encodings.base import EncodedPrompt
from internals_safety.encodings.registry import get_encoder
from internals_safety.judges.refusal import RefusalJudge
from internals_safety.measurements.ability import measure_ability
from internals_safety.measurements.behavior import measure_behavior
from internals_safety.measurements.causal import resolve_refusal_tokens
from internals_safety.measurements.causal_license import (
    kl_divergence,
    matched_norm_random_direction,
    refusal_probability,
)
from internals_safety.measurements.encoding_direction import (
    AblationReading,
    ArmRates,
    CellEvidence,
    select_cell,
)
from internals_safety.models.capture import capture_activations
from internals_safety.models.generate import generate
from internals_safety.models.interventions import ablate_direction
from internals_safety.models.loader import load_model, prepare_prompts
from internals_safety.pipeline import load_contrast_sets, plain_arm, resolve_run_paths
from internals_safety.probes.directions import Direction, difference_in_means

POSITION = "instruction_final"


def sweep_layers(n_layers: int, budget: int) -> list[int]:
    """Evenly spaced layers under a CAP, never a stride.

    A stride of 4 visits 7 layers of a 32-layer model and exactly one of a
    3-layer model, whose `resid_pre` is the raw embedding — measured 2026-08-06,
    the whole sweep came back degenerate.
    """
    usable = max(1, n_layers - 1)
    if budget >= usable:
        return list(range(usable))
    step = usable / budget
    return sorted({int(i * step) for i in range(budget)})


def separation_auroc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """AUROC of the projection scores, computed by rank rather than by sklearn.

    Keeps this module free of a fit: the direction is already fitted, and all we
    ask is how well its projections order the two classes.
    """
    from sklearn.metrics import roc_auc_score

    scores = torch.cat([positive, negative]).float().numpy()
    labels = [1] * len(positive) + [0] * len(negative)
    return float(roc_auc_score(labels, scores))


def cell_evidence(
    loaded,
    direction: Direction,
    encoded_harmful_prompts: list[str],
    encoded_benign_prompts: list[str],
    plain_benign_prompts: list[str],
    baseline_plain_logits: torch.Tensor,
    refusal_ids: list[int],
    encoded_pooled: torch.Tensor,
    plain_pooled: torch.Tensor,
    batch_size: int,
) -> CellEvidence:
    """Forward-only evidence for one candidate direction.

    Cheap by construction — `refusal_probability` reads first-token mass rather
    than judging a completion, which is what makes a sweep affordable at all. The
    judge remains the arbiter for anything reported.
    """
    with ablate_direction(loaded, direction.vector):
        harmful_logits = final_logits(loaded, encoded_harmful_prompts, batch_size)
        benign_logits = final_logits(loaded, encoded_benign_prompts, batch_size)
        plain_logits = final_logits(loaded, plain_benign_prompts, batch_size)

    return CellEvidence(
        layer=direction.layer,
        position=direction.position,
        proxy_harmful_refusal=refusal_probability(harmful_logits, refusal_ids),
        proxy_benign_refusal=refusal_probability(benign_logits, refusal_ids),
        kl=kl_divergence(baseline_plain_logits, plain_logits),
        separation_auroc=separation_auroc(
            direction.project(encoded_pooled), direction.project(plain_pooled)
        ),
        raw_norm=direction.raw_norm,
    )


def final_logits(loaded, prompts: list[str], batch_size: int) -> torch.Tensor:
    """`[batch, seq, vocab]` at the last position, batched, no generation."""
    chunks = []
    for start in range(0, len(prompts), batch_size):
        prepared = prepare_prompts(loaded, prompts[start : start + batch_size], positions=[POSITION])
        encoded = loaded.tokenizer(
            [item.text for item in prepared],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(loaded.model.device)
        with torch.no_grad():
            chunks.append(loaded.model(**encoded).logits[:, -1:, :].float().cpu())
    return torch.cat(chunks, dim=0)


def arm_rates(
    loaded,
    harmful: list[EncodedPrompt],
    benign: list[EncodedPrompt],
    judge: RefusalJudge,
    behavior_config,
    ability_config=None,
    direction: torch.Tensor | None = None,
    recovered=None,
) -> ArmRates:
    """Refusal on both arms under an optional ablation, plus comprehension.

    Generation happens INSIDE the intervention context and judging outside it —
    the judge is a separate model and must never see an ablated forward pass.
    """
    from contextlib import nullcontext

    context = nullcontext() if direction is None else ablate_direction(loaded, direction)
    with context:
        harmful_responses = generate(
            loaded,
            [item.attack_prompt for item in harmful],
            max_new_tokens=behavior_config.max_new_tokens,
            batch_size=behavior_config.batch_size,
        )
        benign_responses = generate(
            loaded,
            [item.attack_prompt for item in benign],
            max_new_tokens=behavior_config.max_new_tokens,
            batch_size=behavior_config.batch_size,
        )
        ability = None
        if ability_config is not None:
            records = measure_ability(loaded, harmful, config=ability_config)
            # The cuts come from ONE place (`recovered`), so this arm and the
            # regime map can never disagree about what ability meant.
            ability = sum(recovered(r.score) for r in records) / len(records)

    harmful_records = measure_behavior(
        loaded, harmful, judge, None, config=behavior_config, responses=harmful_responses
    )
    benign_records = measure_behavior(
        loaded, benign, judge, None, config=behavior_config, responses=benign_responses
    )
    return ArmRates(
        harmful_refusal=sum(r.refused for r in harmful_records) / len(harmful_records),
        benign_refusal=sum(r.refused for r in benign_records) / len(benign_records),
        n_harmful=len(harmful_records),
        n_benign=len(benign_records),
        ability=ability,
        # TODO 95. This script computes its rates straight off the records and
        # never builds a `Reading`, so `Reading.mechanism_errors` cannot
        # withhold anything here — `ArmRates.harm_gap` reads `None` instead.
        mechanism_errors=sum(
            r.judge_mechanism_error for r in (*harmful_records, *benign_records)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--families", nargs="+", required=True,
        help="encoding rungs; each gets its own direction fit and four conditions",
    )
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    measurements = load_measurements_config()
    config = measurements.encoding_direction
    corpus = load_corpus_config()
    model_config = load_model_config(args.model)
    harmful, harmless = load_contrast_sets(
        corpus.harmful_set, corpus.harmless_set, args.n_prompts
    )
    encoders = {family: get_encoder(family) for family in args.families}

    if not model_config.n_layers:
        # Fail closed rather than guess. A wrong depth silently changes which
        # layers the sweep visits AND where the late-layer prune falls, and both
        # failures return plausible numbers.
        raise SystemExit(
            f"{args.model} declares no n_layers; the sweep and the late-layer prune "
            "both key off it, so guessing a depth would move the answer silently"
        )
    n_layers = model_config.n_layers
    cells = sweep_layers(n_layers, config.max_sweep_layers)

    # ⚠️ ONE random direction per cell, not `n_random_directions` per cell.
    # Matched selection means the control searches the SAME space the real arm
    # searches: best-of-8 against best-of-8. Drawing 20 per cell would let the
    # control search a space 20x larger and then report the maximum of a bigger
    # sample as the null — which biases AGAINST finding an effect, but is just as
    # wrong as the reverse and costs 20x the forward passes.
    # (`n_random_directions` stays what it is in `causal_license`: replicate
    # count for a p-value null, a different instrument's question.)
    n_forward_sweeps = 2 * len(cells)
    # Per family: 3 encoded conditions x 2 arms. The PLAINTEXT arm is
    # model-level and computed once — it is the same two numbers whatever
    # rung is being ablated, and re-measuring it per family would spend
    # generations to re-derive a constant.
    n_families = len(args.families)
    n_behavioural = n_families * 3 * 2 * args.n_prompts + 2 * args.n_prompts
    # Ability runs on the baseline and ablated arms only, harmful side, and is
    # scored by STRING SIMILARITY — `RecoveryScore` is config-free by design, so
    # these generations cost GPU time and NO judge money. Counting them as judge
    # calls (as the first version did) overstates the money line by 25%.
    n_ability = n_families * 2 * args.n_prompts
    n_generations = n_behavioural + n_ability
    n_judge_calls = n_behavioural
    print(f"model                 {args.model}   families {args.families}   n {args.n_prompts}")
    print(f"candidate cells       {cells}")
    print(f"forward-only sweeps   {n_forward_sweeps * n_families}  (per family: {len(cells)} "
          f"real + {len(cells)} matched random, same search space)")
    print(f"generations           {n_generations}  ({n_behavioural} behavioural + "
          f"{n_ability} ability; ability is string-scored, no judge)")
    print(f"judge calls           {n_judge_calls}")
    print("GPU                   one; no training, forward passes and generation only")

    if args.dry_run:
        print("\nNothing run. Report GPU/$/wall-clock to the approval gate before launching.")
        return 0

    tree = guard_working_tree(device=model_config.device)
    directory, activations_dir, run_name = resolve_run_paths(
        "phase0", args.model, args.run_name, args.outputs_dir
    )
    loaded = load_model(model_config)
    behavior_config = measurements.behavior
    judge = RefusalJudge(load_judge_config())

    plain_harmful, plain_benign = plain_arm([p.text for p in harmful]), plain_arm(
        [p.text for p in harmless]
    )
    plain_batch = capture_activations(
        loaded,
        prepare_prompts(
            loaded, [i.attack_prompt for i in plain_harmful + plain_benign], positions=[POSITION]
        ),
        positions=[POSITION],
    )
    refusal_ids = resolve_refusal_tokens(loaded, model_config.refusal_openings)
    baseline_plain_logits = final_logits(
        loaded, [i.attack_prompt for i in plain_benign], behavior_config.batch_size
    )
    ability_config = measurements.ability
    generator = torch.Generator().manual_seed(measurements.probes.seed)

    # MODEL-LEVEL and computed ONCE. The plaintext arm is the same two numbers
    # whatever rung is being ablated, so re-measuring it per family would spend
    # generations re-deriving a constant — and worse, would let two families
    # disagree about a quantity that cannot differ.
    plaintext_rates = arm_rates(loaded, plain_harmful, plain_benign, judge, behavior_config)
    # TODO 95: `harm_gap` is None when a judge call failed on either arm, so it
    # cannot go straight into a format spec. Loud rather than crashing — the run
    # should finish and record the failure, not die three hours in.
    plain_gap = plaintext_rates.harm_gap
    print(f"plaintext  harmful {plaintext_rates.harmful_refusal:.2f}  "
          f"benign {plaintext_rates.benign_refusal:.2f}  "
          f"gap {'UNMEASURED' if plain_gap is None else f'{plain_gap:+.2f}'}",
          flush=True)
    if plaintext_rates.mechanism_errors:
        print(
            f"⚠️  JUDGE CALLS FAILED on the plaintext arm: "
            f"{plaintext_rates.mechanism_errors}. Every rate in this run is computed "
            "over verdicts fabricated from error text; the gaps read UNMEASURED.",
            flush=True,
        )

    readings: list[AblationReading] = []
    record: dict = {"plaintext": asdict(plaintext_rates), "families": {}}

    def checkpoint() -> None:
        """Persist after EVERY family, measured or not.

        Originally this ran only on the success branch, so a run whose first
        rung came back UNMEASURED and whose second hit the wall left no
        checkpoint at all — losing the negative result, which is a finding.
        The plaintext arm is in here too and it costs generations, so a kill
        before the first successful rung would have thrown that away as well.
        """
        (directory / "encoding_ablation.json").write_text(json.dumps(record, indent=2))

    checkpoint()  # the plaintext arm is already paid for; land it before the loop
    for family in args.families:
        encoded_harmful = [encoders[family].encode(p.text) for p in harmful]
        encoded_benign = [encoders[family].encode(p.text) for p in harmless]

        # ⚠️ POOLED. Harm is balanced on both sides of this contrast, so its
        # component cancels and the direction is about SURFACE FORM. Fitting on
        # the harmful arm alone would guarantee the "destroyed" verdict by
        # construction — an artefact of the fit, not a property of the model.
        encoded_batch = capture_activations(
            loaded,
            prepare_prompts(
                loaded,
                [i.attack_prompt for i in encoded_harmful + encoded_benign],
                positions=[POSITION],
            ),
            positions=[POSITION],
        )

        def evidence_for(direction: Direction) -> CellEvidence:
            return cell_evidence(
                loaded,
                direction,
                [i.attack_prompt for i in encoded_harmful],
                [i.attack_prompt for i in encoded_benign],
                [i.attack_prompt for i in plain_benign],
                baseline_plain_logits,
                refusal_ids,
                encoded_batch.select(direction.layer, POSITION),
                plain_batch.select(direction.layer, POSITION),
                behavior_config.batch_size,
            )

        real: list[tuple[CellEvidence, Direction]] = []
        control: list[tuple[CellEvidence, Direction]] = []
        degenerate: list[int] = []
        for layer in cells:
            direction = difference_in_means(encoded_batch, plain_batch, layer, POSITION)
            # ⚠️ A DEGENERATE CELL IS COVERAGE, NOT AN EXCEPTION — the same rule
            # the causal-licensing wiring landed on 2026-08-06, re-derived here
            # the hard way. `difference_in_means` returns a ZERO vector when the
            # two conditions do not separate at that layer and says in its own
            # comment that the caller sees `raw_norm` and discards the cell;
            # this caller did not, so `project_out` raised four frames down and
            # took the whole run with it. "The encoded and plain conditions were
            # indistinguishable at layer L" is a measurement about the model,
            # and it must not be able to lose the other fifteen cells.
            if direction.raw_norm <= 0.0:
                degenerate.append(layer)
                continue
            real.append((evidence_for(direction), direction))
            # ONE matched-norm random per cell, so the control searches the same
            # space the real arm searches: best-of-N against best-of-N.
            random = Direction(
                vector=matched_norm_random_direction(direction.vector, generator),
                layer=layer,
                position=POSITION,
                n_positive=direction.n_positive,
                n_negative=direction.n_negative,
                raw_norm=direction.raw_norm,
            )
            control.append((evidence_for(random), random))

        if degenerate:
            print(f"{family}: {len(degenerate)}/{len(cells)} cells degenerate "
                  f"(zero separation) at layers {degenerate}", flush=True)

        # Selection can legitimately find NOTHING, and that is a finding rather
        # than an error: no valid direction exists to intervene on. Recorded and
        # skipped, so one dead rung does not lose the rest of the run. `real`
        # being EMPTY — every cell degenerate — lands here too, which is why
        # the message distinguishes the two: "nothing separated" and "something
        # separated but nothing passed the screen" are different facts about
        # the model and a reader must not have to guess which one happened.
        try:
            if not real:
                raise ValueError(
                    f"all {len(cells)} candidate cells degenerate: the encoded and "
                    "plain conditions did not separate at any swept layer"
                )
            best_real = select_cell([e for e, _ in real], n_layers, config)
            best_control = select_cell([e for e, _ in control], n_layers, config)
        except ValueError as error:
            print(f"{family}: UNMEASURED — {error}", flush=True)
            record["families"][family] = {
                "unmeasured": str(error),
                "degenerate_layers": degenerate,
                "candidates": [asdict(e) for e, _ in real],
                # ⚠️ The CONTROL arm's candidates too, and this branch omitted
                # them until 2026-08-09. Job 9033243 came back UNMEASURED and
                # the record could not say WHICH arm's filter had emptied —
                # the real arm's eligibility had to be recomputed offline from
                # the recorded numbers to find out. The success branch records
                # both arms because a reader needs the search space; the
                # failure branch needs it MORE, because the search space is the
                # entire result.
                "control_candidates": [asdict(e) for e, _ in control],
            }
            checkpoint()
            continue

        real_direction = next(d for e, d in real if e is best_real)
        control_direction = next(d for e, d in control if e is best_control)

        reading = AblationReading(
            family=family,
            model=args.model,
            layer=best_real.layer,
            position=best_real.position,
            separation_auroc=best_real.separation_auroc,
            kl=best_real.kl,
            baseline=arm_rates(
                loaded, encoded_harmful, encoded_benign, judge, behavior_config,
                ability_config, recovered=recovered,
            ),
            ablated=arm_rates(
                loaded, encoded_harmful, encoded_benign, judge, behavior_config,
                ability_config, direction=real_direction.vector, recovered=recovered,
            ),
            control=arm_rates(
                loaded, encoded_harmful, encoded_benign, judge, behavior_config,
                direction=control_direction.vector,
            ),
            plaintext=plaintext_rates,
        )
        readings.append(reading)
        record["families"][family] = {
            **asdict(reading),
            "gap_destroyed": reading.gap_destroyed,
            "gap_restored": reading.gap_restored,
            "control_gap_restored": reading.control_gap_restored,
            "margin": reading.margin,
            "restored_fraction": reading.restored_fraction,
            "ability_shift": reading.ability_shift,
            "resolution": reading.resolution,
            "verdict": reading.verdict(config),
            # Recorded even when empty: the reader needs the SEARCH SPACE the
            # winner won against, and "no cell was discarded" is part of it.
            "degenerate_layers": degenerate,
            "candidates": [asdict(e) for e, _ in real],
            "control_candidates": [asdict(e) for e, _ in control],
        }
        # Checkpoint per family — a wall kill must not lose a finished rung.
        checkpoint()
        margin = reading.margin
        print(f"{family}: L{reading.layer}  "
              f"margin {'UNMEASURED' if margin is None else f'{margin:+.3f}'}  "
              f"restored_fraction {reading.restored_fraction}  "
              f"ability_shift {reading.ability_shift}  verdict {reading.verdict(config)}",
              flush=True)

    write_run_record(
        directory,
        capture_provenance(
            config={
                "corpus": corpus.model_dump(),
                "model": model_config.model_dump(),
                "measurements": measurements.model_dump(),
                "judges": load_judge_config().model_dump(),
            },
            seed=measurements.probes.seed,
            extra={
                "phase": "phase0",
                "run_name": run_name,
                "instrument": "encoding_direction",
                "families": args.families,
                "encoding_ablation": record,
            },
            tree=tree,
        ),
    )
    print(f"\nwrote {directory}  ({len(readings)}/{len(args.families)} rungs measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
