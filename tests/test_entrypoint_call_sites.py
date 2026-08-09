"""Every entrypoint's calls into the library must BIND against the live signature.

**Why this exists (2026-08-07).** `scripts/sae_pregate.py` called
`guard_working_tree()` with no arguments against a signature requiring `device`.
It killed all three tasks of job `8995805` twenty seconds in, after a queue wait,
on a run that had passed the approval gate.

Nothing caught it, and the reason is the thing worth fixing:

1. **969 tests were green.** No test invokes this script's `main()`.
2. **`--dry-run` passed and proved nothing.** The dry-run branch `return 0`s
   BEFORE the guard, so it exercises argument parsing and a cost printout and
   not one line of the real path. The approval gate is built on `--dry-run`,
   so a preset can be approved on evidence that touches none of the code.
3. **`build_status.py` reported the instrument "built, tested and reachable".**
   Reachability is an import-graph property; it cannot see a call that is wrong.

A real smoke test needs weights and a GPU, which is exactly the cost this is
avoiding. So this checks statically what the failure actually was — an argument
list that cannot bind — across every script, for the shared library functions an
entrypoint must get right. It is cheap, it needs no model, and it fails at
`pytest` time rather than after a queue wait.

Deliberately NOT a whole-program type check: the aim is the narrow, recurring
failure of a shared function whose signature moved out from under one caller.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from internals_safety import pipeline, provenance
from internals_safety.measurements import (
    causal,
    control_floor,
    deployment,
    length_null,
    sae_reconstruction,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# The shared functions whose call sites an entrypoint must get right, mapped to
# the live object. Grow this when a signature change breaks a caller — that is
# the trigger, and this session hit it THREE times in one day (`strata` on
# `measure_deployment`, `device` on `guard_working_tree`, and `cross_rung_screen`
# on `run_families` — the last one being the control floor that never reached
# the entrypoint at all, TODO 60).
WATCHED = {
    "guard_working_tree": provenance.guard_working_tree,
    "write_run_record": provenance.write_run_record,
    "capture_provenance": provenance.capture_provenance,
    "measure_deployment": deployment.measure_deployment,
    "length_strata": length_null.length_strata,
    "run_families": pipeline.run_families,
    # Added 2026-08-08 by the same trigger: `drop_bos: bool` became
    # `bos_token_id` keyword-only with no default, so a stale caller is now a
    # TypeError — and this is what makes it a `pytest` TypeError rather than a
    # cluster one. `scored_positions` itself is deliberately NOT here: no script
    # calls it, and the watchlist's own vacuity guard rejects a name that matches
    # nothing. It is reached through these two.
    "measure_reconstruction": sae_reconstruction.measure_reconstruction,
    "observed_sparsity": sae_reconstruction.observed_sparsity,
    # Added 2026-08-08 by the trigger, for the fifth time: `ability_rate:
    # Mapping` became `ability: AbilitySource` keyword-only with no default, so
    # every floor now has to say what model its controls were selected on and
    # what model it screens. Four scripts call it; a stale positional call is a
    # TypeError here rather than a floor quoted without its inheritance.
    # Keyed by the LOCAL alias — all four scripts import it as this name.
    "derive_control_floor": control_floor.derive,
    "sigma_bounds": control_floor.sigma_bounds,
    # Added 2026-08-09, sixth application of the trigger. `refusal_token_ids`
    # positional became `probe: BehaviourProbe` keyword-only with no default,
    # because the causal test now serves two model KINDS that render and score
    # differently. A stale positional call would pass a token-id list where a
    # probe belongs; more dangerously, a DEFAULT would have let the guard
    # entrypoint silently score a generating model's refusal opening at a
    # position a guard never answers at. `run_causal_gate` moved out of
    # phase0_regime_map.py into the library in the same change, so BOTH
    # entrypoints call it and both are checked here.
    # `measure_causal_evidence` is deliberately NOT here, and the watchlist's own
    # vacuity guard is what said so: after the move, no script calls it directly
    # — both entrypoints reach it through `run_causal_gate`. Same reasoning as
    # `scored_positions` above. Watching it would have passed vacuously.
    "run_causal_gate": causal.run_causal_gate,
    "guard_verdict_probe": causal.guard_verdict_probe,
    "refusal_probe": causal.refusal_probe,
}


def call_sites(path: Path):
    """Every call to a WATCHED name in `path`, as (name, node)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in WATCHED:
            yield name, node


def script_paths() -> list[Path]:
    # rglob, not glob: `completion.py` was fixed the same way after a subdir of
    # scripts went unseen. A checker that misses files reports false health.
    return sorted(p for p in SCRIPTS.rglob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", script_paths(), ids=lambda p: p.name)
def test_every_watched_call_binds_to_the_live_signature(path: Path):
    for name, node in call_sites(path):
        signature = inspect.signature(WATCHED[name])
        positional = [object()] * len(node.args)
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # *args unpacking: arity is not statically known
        keywords = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
        if any(kw.arg is None for kw in node.keywords):
            continue  # **kwargs unpacking: likewise
        try:
            signature.bind(*positional, **keywords)
        except TypeError as error:
            raise AssertionError(
                f"{path.name}:{node.lineno} calls {name}() in a way that cannot bind "
                f"to {name}{signature}: {error}. This is the class of defect that "
                f"costs a queue cycle — job 8995805 died on exactly it."
            ) from error


def test_the_watchlist_actually_matches_something():
    """A watchlist that matches nothing passes vacuously and looks like health.

    The same failure mode as the coverage sweep measuring absence against a
    narrow index: a green check over an empty set is not evidence.
    """
    found = {name for path in script_paths() for name, _ in call_sites(path)}
    missing = sorted(set(WATCHED) - found)
    assert not missing, f"watched but never called in scripts/: {missing}"


def test_no_script_constructs_an_EncodedPrompt_by_hand():
    """Scripts build prompts through an encoder or `pipeline.plain_arm`. Never raw.

    **Paid for by job 9032777 (2026-08-09), which died on an H200 in 76 seconds
    after a queue wait.** `encoding_ablation.py` hand-wrote its own six-line
    `plain_arm`, omitting `invertibility` and `restate_prompt`, while a correct
    copy had been sitting in `phase0_regime_map.py` since the plaintext baseline
    landed the day before. Two scripts, one function, one of them wrong.

    Note what could NOT have caught it. `EncodedPrompt` is a frozen dataclass,
    so `WATCHED` above would have bound its signature and rejected the call —
    but only if someone had thought to watch it, and the watchlist's trigger is
    *a signature change breaking a caller*, which never happened here. The
    signature was right from the start; the caller was wrong from the start.
    So the guard is the stronger one: the raw constructor is not a thing a
    script may reach for, which makes the whole class unrepresentable rather
    than making one instance of it detectable.

    `plain_arm` now lives in the spine by the selection rule in `pipeline`'s
    docstring — absence in ONE script would be a defect — and this test is what
    stops the next script from re-growing it.
    """
    offenders = []
    for path in script_paths():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "EncodedPrompt":
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"{offenders} construct EncodedPrompt directly. Use an encoder from the "
        "registry, or `pipeline.plain_arm` for the plaintext arm — a second "
        "hand-rolled copy is what killed job 9032777."
    )


def second_device() -> str | None:
    """A real device that is not CPU, or None.

    ⚠️ `meta` does NOT work here and the first version of this test used it.
    `cpu @ meta` silently SUCCEEDS — meta propagates through matmul instead of
    rejecting the mix — so the test passed on the pre-fix code and guarded
    nothing. Verified before trusting it, which is the rule this file's own
    `test_the_watchlist_actually_matches_something` states: a green check over
    an empty set is not evidence.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(second_device() is None, reason="needs a non-CPU device")
class TestDictionarySeamsCoerceDevice:
    """A CPU batch against a non-CPU dictionary must work, not raise.

    Job `9006556` loaded an 8B model and a 540 MB dictionary and then died on
    `mat2 is on cuda:0, different from other tensors on cpu` — because
    `measure_reconstruction` hands the round trip `h.float().cpu()` while
    `sae_pregate.py` loads the dictionary with `device=cuda`. Two call sites,
    each locally reasonable, contradicting each other. Verified to REJECT the
    pre-fix code on this device pair, not merely to pass on the fixed one.
    """

    def test_the_trained_dictionary_moves_the_batch_to_its_own_device(self):
        import torch

        from internals_safety.models.sae_loader import LlamaScopeSAE

        device, d_model, d_sae = second_device(), 8, 16
        sae = LlamaScopeSAE(
            encoder_weight=torch.zeros(d_sae, d_model, device=device),
            encoder_bias=torch.zeros(d_sae, device=device),
            decoder_weight=torch.zeros(d_model, d_sae, device=device),
            decoder_bias=torch.zeros(d_model, device=device),
            jump_relu_threshold=0.0,
            input_norm=1.0,
            output_norm=1.0,
            d_model=d_model,
            d_sae=d_sae,
            hook_point="blocks.17.hook_resid_post",
            trained_on="test",
            nominal_top_k=4,
        )
        features = sae.encode(torch.zeros(3, d_model))   # CPU in, GPU dictionary
        assert features.device.type == device
        assert sae.decode(features).device.type == device

    def test_the_random_control_coerces_the_same_way(self):
        """Symmetry matters: a control that breaks where the real dictionary
        works is not a matched control."""
        import torch

        from internals_safety.measurements.sae_reconstruction import RandomDictionary

        device = second_device()
        control = RandomDictionary(
            d_model=8, n_features=16, k=4, generator=torch.Generator().manual_seed(0)
        )
        control.weights = control.weights.to(device)
        assert control.encode(torch.zeros(3, 8)).device.type == device


@pytest.mark.skipif(second_device() is None, reason="needs a non-CPU device")
def test_measure_reconstruction_runs_with_an_OFF_DEVICE_dictionary(tiny_model):
    """The end-to-end path the cluster actually runs: CPU pipeline, GPU dictionary.

    THE test that was missing. Three separate device defects reached the queue
    in `sae_pregate` (jobs 8995805, 9006556, 9006846) and every unit test passed
    through all of them, because they all place the dictionary on CPU — where
    the two halves of the invariant are indistinguishable:

        encode/decode  -> move activations TO the dictionary (weights stay put)
        round_trip     -> bring the result BACK to the caller

    Fixing only the first half turned an error inside `encode` into an error one
    line later at `hidden - reconstruction`. Only an off-device dictionary can
    tell the difference, so this test uses one.
    """
    import torch

    from internals_safety.config import SAEConfig
    from internals_safety.measurements.sae_reconstruction import (
        RandomDictionary,
        measure_reconstruction,
    )

    device = second_device()
    dictionary = RandomDictionary(
        d_model=tiny_model.model.config.hidden_size,
        n_features=32,
        k=4,
        generator=torch.Generator().manual_seed(0),
    )
    dictionary.weights = dictionary.weights.to(device)   # the cluster's shape

    quality = measure_reconstruction(
        tiny_model,
        dictionary,
        ["the cat sat", "a dog ran"],
        layer=1,
        config=SAEConfig(trained_on="t", min_kl_recovered=0.8, min_transfer_ratio=0.8),
        batch_size=2,
    )
    assert quality.n_prompts == 2
    assert quality.variance_explained == quality.variance_explained   # not NaN
