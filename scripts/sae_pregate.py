"""I4's pre-gate — may a Llama Scope dictionary be read on OUR model at all?

**This can refuse the whole instrument, which is why it is its own entrypoint.**
Llama Scope is trained on Llama-3.1-8B-**Base**; AS-5's target is **Instruct**
and AS-6's is Llama Guard 3 8B (a safety fine-tune of the same base). A
base-trained dictionary applied to a fine-tune fails *silently* — it returns a
normal-looking reconstruction and normal-looking feature activations — so no
feature claim may be made before this passes.

## Run the BASE arm first, and read nothing until it passes

The pre-gate has two arms and they answer different questions:

    --model llama_3_1_8b_base        does OUR LOADER work?
    --model llama_3_1_8b_instruct    does the dictionary TRANSFER?

A dictionary cannot fail to transfer to the model it was fitted on, so a poor
Base result is a bug in `models/sae_loader.py` — most likely the dataset-wise
normalisation convention, which was derived from the checkpoint rather than read
from upstream's code (`other_repos` has no clone of it; repo clones are the
owner's call). Reading the Instruct arm before the Base arm passes would attribute
our own bug to the science.

## Cost

Two forward passes per prompt per arm (clean, and the SAE substitution) plus one
ablation pass, over a few hundred prompts — minutes on one GPU. The dictionary
download is ~540 MB per layer. No judge calls, no spend.

    ./run python scripts/sae_pregate.py --model llama_3_1_8b_base --sae-layer 15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

import torch

from internals_safety.config import load_measurements_config, load_model_config
from internals_safety.data import prompt_set
from internals_safety.measurements.sae_reconstruction import (
    RandomDictionary,
    measure_reconstruction,
    reading,
    unmeasured_reading,
)
from internals_safety.models.loader import load_model, resolve_device
from internals_safety.models.sae_loader import download_llama_scope_sae
from internals_safety.paths import PROJECT_ROOT
from internals_safety.provenance import capture_provenance, guard_working_tree, write_run_record

PHASE = "sae_pregate"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=15,
        help="LLAMA SCOPE's layer index (the block whose resid_post it reads), NOT ours",
    )
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--run-name", default="pregate")
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    model_config = load_model_config(args.model)
    measurements = load_measurements_config()
    prompts = [p.text for p in prompt_set("jbb_prompts.jsonl", limit=args.n_prompts)]

    if args.dry_run:
        print(f"model            {model_config.name}")
        print(f"SAE              Llama Scope layer {args.sae_layer} (~540 MB download)")
        print(f"prompts          {len(prompts)}")
        # Three passes: clean, SAE-substituted, ablated. Plus the same again for
        # the matched random-dictionary control — priced, because a control the
        # estimate cannot see is a cost nobody approved.
        #
        # DOUBLED since 2026-08-07: the run measures both selection rules
        # (jumprelu and topk) on the same weights, which is a second full
        # `measure_reconstruction`. Same rule as the control — work the estimate
        # cannot see is work nobody approved, and this is the larger half.
        print(f"forward passes   {12 * len(prompts)} "
              f"(clean/SAE/ablated x real + control, x2 selection rules)")
        print("selection rules  jumprelu (derived) AND topk (nominal_top_k) — "
              "the comparison IS the gate")
        print("judge calls      0 — no spend")
        print("\n⚠️  Run --model llama_3_1_8b_base FIRST. A poor Base result is a")
        print("    loader bug, not a transfer finding, and reading the Instruct arm")
        print("    before Base passes attributes our bug to the science.")
        return 0

    # Device FIRST: the guard needs it to decide whether this run is
    # result-bearing, which is the whole basis on which it raises rather than
    # warns. Called unconditionally with `allow_dirty` passed through, matching
    # `phase0_regime_map.py` and `as6_guard_probe.py` — the `if not
    # args.allow_dirty` wrapper that stood here was wrong twice over: it dropped
    # the required `device` argument (a TypeError that killed all three tasks of
    # job 8995805 in 20 seconds), and it made `--allow-dirty` skip the guard
    # ENTIRELY rather than downgrade it to a warning, so no diff would have been
    # recorded alongside the results.
    device = resolve_device(model_config.device, allow_cpu_in_job=args.allow_cpu)
    guard_working_tree(str(device), allow_dirty=args.allow_dirty)
    print(f"loading {model_config.hf_id} on {device} ...", flush=True)
    loaded = load_model(model_config)

    print(f"loading Llama Scope L{args.sae_layer} ...", flush=True)
    sae = download_llama_scope_sae(layer=args.sae_layer, device=str(device))
    layer = sae.our_layer
    if layer >= loaded.n_layers:
        print(
            f"SAE maps to our layer {layer} but the model has {loaded.n_layers}", file=sys.stderr
        )
        return 1
    print(f"  {sae.hook_point} -> our resid_pre layer {layer}", flush=True)

    # The control's sparsity is matched to what the dictionary ACTUALLY does on
    # this data — never to `nominal_top_k`, which is a training-schedule leftover
    # in a jumprelu checkpoint. A denser control reconstructs better for reasons
    # unrelated to having learned anything, making it trivially easy to beat.
    from internals_safety.models.capture import capture_activations
    from internals_safety.models.loader import prepare_prompts

    probe_batch = capture_activations(
        loaded, prepare_prompts(loaded, prompts[:16]), layers=[layer], positions=["last"]
    )
    observed_k = round(sae.observed_l0(probe_batch.select(layer, "last").float()))
    print(f"  observed L0 {observed_k} (nominal top_k {sae.nominal_top_k}, unused)", flush=True)

    control = RandomDictionary(
        d_model=sae.d_model,
        n_features=sae.d_sae,
        k=max(1, observed_k),
        generator=torch.Generator().manual_seed(measurements.probes.seed),
    )

    # BOTH selection rules, on the same weights and the same activations.
    #
    # The loader DERIVED `jumprelu` from the artifact and never verified it
    # against upstream's forward pass; `top_k: 50` sits in the same hyperparams
    # file, dismissed in a comment as a training-schedule leftover. Job 9009119
    # made that dismissal the leading suspect: the normalisation is off by only
    # 1.4-1.7x (corpus offset) while the round trip emits a vector ~90x too
    # large, and jumprelu selects 549 features where nominal is 50.
    #
    # Run as ONE job rather than two, because the question is a comparison and a
    # within-run comparison holds the weights, the prompts, the activations and
    # the control fixed. Two jobs would differ by whatever else drifted.
    qualities = {
        variant.selection: measure_reconstruction(
            loaded, variant, prompts, layer=layer, config=measurements.sae,
            control=control, batch_size=model_config.capture_batch_size,
        )
        for variant in (sae, replace(sae, selection="topk"))
    }
    verdicts = [
        reading(quality, measurements.sae)
        if quality.n_prompts
        else unmeasured_reading(f"no prompts scored ({name})")
        for name, quality in qualities.items()
    ]

    for name, quality in qualities.items():
        print(f"\n--- selection: {name}")
        print(f"variance explained  {quality.variance_explained:.4f}")
        print(f"control (random)    {quality.control_variance_explained}")
        print(f"KL recovered        {quality.kl_recovered}")
        print(f"L0                  {quality.l0:.1f}  (nominal {sae.nominal_top_k})")
        print(f"activation norm     {quality.observed_activation_norm:.3f} observed "
              f"vs {quality.declared_input_norm} declared (ratio {quality.norm_ratio:.3f})")

    for verdict, (name, _) in zip(verdicts, qualities.items()):
        print(f"\nLICENSED ({name}): {verdict.licensed}")
        for why in verdict.why_not_reportable():
            print(f"  - {why}")
    if not any(verdict.licensed is True for verdict in verdicts):
        print("\n⚠️  Feature-level claims are NOT licensed on this model under EITHER rule.")

    quality = qualities[sae.selection]

    directory = (
        (PROJECT_ROOT / args.outputs_dir if args.outputs_dir else PROJECT_ROOT / "outputs")
        / "runs" / PHASE / model_config.name / args.run_name
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_run_record(
        directory,
        {
            "phase": PHASE,
            "model": model_config.name,
            "sae": {
                "trained_on": sae.trained_on,
                "hook_point": sae.hook_point,
                "our_layer": layer,
                "observed_l0": observed_k,
                "nominal_top_k": sae.nominal_top_k,
            },
            "n_prompts": quality.n_prompts,
            "provenance": capture_provenance(str(device)),
        },
        readings=verdicts,
    )
    print(f"\nwrote {directory / 'results.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
