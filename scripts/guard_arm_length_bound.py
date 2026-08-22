"""Could LENGTH alone manufacture AS-6's harmful-versus-benign block gap? No.

TODO 88, answered without the run it was filed as needing.

**The worry, which is a fair one.** AS-5 measured raw character length separating
the harmful from the benign corpus at AUROC $0.6544$ (86.0 characters against
73.8), and every encoder in the ladder is monotone in length. AS-6 reports a
block-rate gap between the two arms and calls it harm sensitivity. A guard that
blocked partly on length would produce exactly such a gap, and neither the benign
arm nor the wrapper arm removes it: both hold CONTENT and TEMPLATE fixed while
leaving length free.

**Why this is not `measure_rate_length_null`, and why that matters.** The
permutation null bins both arms by length and permutes labels within bins. It
needs per-prompt flags on BOTH arms. AS-6 has none: every `benign_cells.jsonl` on
disk is zero bytes, every `cells.jsonl` is the harmful arm, and the benign arm
survives only as the `benign_arm.benign_block_rate` summary. That is a re-run,
not a transfer, and it was filed as though a down-sync would fix it.

What IS computable is strictly stronger in one direction, which is the direction
a control needs. Take the two length distributions, which are exact and free, and
build the BEST POSSIBLE MONOTONE length-only guard operating at this guard's own
blocking budget: block the $k$ longest -- or the $k$ shortest, whichever
separates more -- of the 200 pooled items, where $k$ is the number the real guard
blocked. Its harmful-minus-benign gap is the most that length alone can buy at
this operating point. Monotone is a restriction and `length_only_gap` says why it
is the right one; the short version is that the unrestricted optimum bounds near
1.0 on any corpus and so bounds nothing. So:

* observed gap ABOVE the bound  ->  length cannot account for it. Conclusive.
* observed gap BELOW the bound  ->  inconclusive, and it needs the re-run.

The bound is one-directional on purpose. A control that can only ever fail to
reject is a verdict with a script attached.

Fixing the budget to the guard's own blocked count is the right comparison, since
the confound at issue is that THIS guard's decisions are partly length and its
rate is observed. The strictly more conservative bound, the maximum over every
budget in either unit, is computed alongside it so that the question does not
have to be argued: it is recorded as `length_only_gap_bound_any_budget`.

**Length is measured TWICE, in characters and in the guard's own tokens, and
neither is optional.** The referee's worry and AS-5's $0.6544$ both name
CHARACTER length, but a guard processes tokens, and the published confound this
control answers (arXiv 2605.00269) is stated over sequence length. The two come
apart exactly where it matters: `fullwidth` has the same character count as its
plaintext and roughly two and a half times the tokens, so a character-only
control would clear the one condition whose sequence length the encoding most
inflates. Reporting one and calling it "the length control" is the error this
repo keeps finding one estate over, so both are computed and the verdict is the
CONSERVATIVE one -- a cell clears only if it clears under both.

**What licenses regenerating the benign arm at all.** The encoders are
deterministic and the corpora are local, but "deterministic" is a claim about
code that has been wrong here before. So the script re-encodes the HARMFUL arm
too and requires it to reproduce the on-disk ciphertext for every prompt, byte
for byte, before it will report anything. The harmful arm is the only arm whose
ground truth is on disk, so it is the only arm that can carry that check, and
passing it is what makes the benign regeneration trustworthy by the same path.

Keyless, GPU-free, no model, seconds.

    uv run python scripts/guard_arm_length_bound.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internals_safety.config import load_corpus_config, load_guard_config
from internals_safety.data import digest
from internals_safety.encodings.registry import load_ladder
from internals_safety.measurements.length_null import length_auroc
from internals_safety.pipeline import load_contrast_sets

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "outputs" / "runs" / "as6_phase1"
OUT = REPO_ROOT / "outputs" / "analysis"

#: The conditions each guard REPORTS, after every screen. Duplicated from
#: `review_statistics.py` deliberately: this script must be able to say which of
#: its rows the paper actually rests on, and importing that list would couple two
#: scripts whose reasons for holding it are different.
#:
#: Llama Guard `fullwidth` is ABSENT because the item-level holdout withdrew it
#: to (U) -- 0.7057 against a floor of 0.7066. It is still measured below, and
#: labelled, because a withdrawn cell is exactly where a reader looks for the
#: confound that withdrew it.
REPORTED = {
    "llama_guard_3_8b": {"homoglyph", "zero_width", "reverse_words"},
    "wildguard": {"homoglyph", "zero_width", "reverse_words"},
}


def arm_rates(guard: str) -> tuple[dict[str, tuple[float, float]], str]:
    """`{family: (harmful_block_rate, benign_block_rate)}` and its source run.

    Several runs carry the same `benign_arm` block, and they agree; the earliest
    is chosen so the number is attributed to the run that MEASURED it rather than
    to whichever later run happened to copy it forward.
    """
    candidates = sorted((RUNS / guard).glob("guard-benign-*/results.json"))
    if not candidates:
        raise SystemExit(f"{guard}: no guard-benign-* run on disk; nothing to bound")
    record = json.loads(candidates[0].read_text())
    rates: dict[str, tuple[float, float]] = {}
    for summary in record.get("summaries", []):
        benign = summary.get("benign_arm")
        if not benign or benign.get("benign_block_rate") is None:
            continue
        rates[summary["family"]] = (
            float(summary["block_rate"]),
            float(benign["benign_block_rate"]),
        )
    return rates, candidates[0].parent.name


def on_disk_ciphertexts(guard: str, run: str) -> dict[tuple[str, str], str]:
    """`{(family, prompt_id): ciphertext}` for the harmful arm, from the record."""
    path = RUNS / guard / run / "cells.jsonl"
    out: dict[tuple[str, str], str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[(row["family"], row["prompt_id"])] = row["ciphertext"]
    return out


def length_only_gap(
    harmful_lengths: list[int], benign_lengths: list[int], n_blocked: int
) -> float:
    """The gap the best MONOTONE length-only guard buys at this blocking budget.

    Both directions, because a confound running the other way is just as real:
    block the `n_blocked` longest, or the `n_blocked` shortest, whichever
    separates more. This mirrors `length_auroc`'s two-sidedness and for the same
    reason -- a directional bound would score a benign-longer confound as
    harmless.

    Ties at the cut are broken in the direction that FAVOURS the bound (harmful
    items counted as blocked first), so the number is a true ceiling rather than
    an arbitrary one. That costs us nothing: a bound we might have beaten by a
    tie-break is a bound we should not have quoted.

    **MONOTONE is a real restriction and it is deliberate.** The unrestricted
    optimum over all functions of length is not a useful ceiling: with lengths
    nearly unique across 200 items, "block the lengths where the harmful
    fraction is highest" fits the sample almost perfectly and would bound at
    close to 1.0 for any corpus whatsoever. That number would describe our
    corpus's memorability, not a guard's behaviour. The confound actually at
    issue -- and the one the OOD-detection literature reports -- is monotone:
    longer sequences are flagged more. A guard implementing a non-monotone
    length rule that happens to fit these 200 items is not a length-confounded
    guard, it is a coincidence, and `tests/test_guard_arm_length_bound.py` pins
    that such a rule CAN beat this bound rather than leaving the gap unstated.

    Takes LENGTHS rather than texts because the unit is the caller's choice and
    there are two of them; handing this function texts would have quietly made
    characters the only unit it could ever express.
    """
    harmful, benign = harmful_lengths, benign_lengths

    def gap(longest_first: bool) -> float:
        pool = [(n, True) for n in harmful] + [(n, False) for n in benign]
        sign = -1 if longest_first else 1
        pool.sort(key=lambda item: (sign * item[0], not item[1]))
        blocked = pool[:n_blocked]
        harmful_blocked = sum(1 for _, is_harmful in blocked if is_harmful)
        benign_blocked = len(blocked) - harmful_blocked
        return harmful_blocked / len(harmful) - benign_blocked / len(benign)

    return max(gap(longest_first=True), gap(longest_first=False))


def _length_auroc_numeric(positive: list[int], negative: list[int]) -> float:
    """`length_auroc` over counts that are not character counts.

    Deliberately NOT a second implementation of the two-sided rule. `length_auroc`
    is a function of lengths alone, so an item of length n is fully represented by
    any string of length n; going through it keeps one home for the
    `max(a, 1 - a)` decision, which is the part that would silently diverge if
    this were re-derived here.
    """
    return length_auroc(["x" * n for n in positive], ["x" * n for n in negative])


def token_lengths(tokenizer, texts: list[str]) -> list[int]:
    """Token counts under the guard's OWN tokenizer, specials excluded.

    `add_special_tokens=False` for the same reason the guard prompt layer sets
    it: the specials are a constant per condition and per guard, so including
    them shifts every count by the same amount and changes no rank, while making
    the printed means disagree with every other token census in this repo.
    """
    return [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    corpus = load_corpus_config()
    harmful_prompts, benign_prompts = load_contrast_sets(
        corpus.harmful_set, corpus.harmless_set, corpus.n_prompts
    )
    digests = {"harmful": digest(harmful_prompts), "harmless": digest(benign_prompts)}
    ladder = load_ladder()

    # Imported here rather than at module scope: it is the one heavy dependency,
    # and `--help` should not pay for it.
    from transformers import AutoTokenizer

    results: dict[str, list[dict]] = {}
    for guard, reported in REPORTED.items():
        rates, source = arm_rates(guard)
        # The guard's OWN tokenizer, not a shared one: the two guards' vocabularies
        # differ by 4x and a token count is meaningless across them.
        tokenizer = AutoTokenizer.from_pretrained(load_guard_config(guard).hf_id)

        # THE CORPUS MUST BE THE ONE THE GUARD SAW. A digest mismatch means the
        # local corpus drifted from the run's, and every length below would be
        # measured on the wrong text while looking entirely reasonable.
        recorded = json.loads((RUNS / guard / source / "results.json").read_text())
        if recorded.get("corpus_digest") != digests:
            raise SystemExit(
                f"{guard}: corpus digest mismatch -- run {source} saw "
                f"{recorded.get('corpus_digest')}, local corpus is {digests}. "
                "The regeneration below would silently measure different text."
            )

        disk = on_disk_ciphertexts(guard, source)
        rows: list[dict] = []
        for family, (harmful_rate, benign_rate) in sorted(rates.items()):
            encoder = ladder[family]
            harmful_ct = [encoder.encode(p.text).ciphertext for p in harmful_prompts]
            benign_ct = [encoder.encode(p.text).ciphertext for p in benign_prompts]

            # THE SELF-CHECK. Regenerating the benign arm is only as trustworthy
            # as regenerating the harmful arm, which we can verify exactly.
            #
            # A MISSING key raises rather than skipping. `family` came from this
            # run's own summaries, so an absent ciphertext means the record and
            # its summary disagree about what was measured; skipping would leave
            # the benign regeneration unlicensed while the run still printed a
            # bound, which is the shape of a check that certifies rather than
            # tests.
            for prompt, generated in zip(harmful_prompts, harmful_ct):
                key = (family, prompt.id)
                if key not in disk:
                    raise SystemExit(
                        f"{guard}/{family}/{prompt.id}: the run summarises this condition but "
                        "its cells hold no such prompt, so the regeneration cannot be verified "
                        "against anything. Refusing to bound an arm on an unchecked encoder."
                    )
                if disk[key] != generated:
                    raise SystemExit(
                        f"{guard}/{family}/{prompt.id}: regenerated ciphertext differs from "
                        "the one the guard was given. The encoders have moved since the run; "
                        "no length measured against this corpus is comparable to it."
                    )

            n_blocked = round(harmful_rate * len(harmful_ct) + benign_rate * len(benign_ct))
            observed = harmful_rate - benign_rate

            units = {
                "chars": ([len(t) for t in harmful_ct], [len(t) for t in benign_ct]),
                "tokens": (token_lengths(tokenizer, harmful_ct), token_lengths(tokenizer, benign_ct)),
            }
            bounds = {
                unit: length_only_gap(harmful_n, benign_n, n_blocked)
                for unit, (harmful_n, benign_n) in units.items()
            }
            # SENSITIVITY TO THE BUDGET-MATCHING CHOICE. Fixing k to the guard's
            # own blocked count is the right comparison -- the confound is "this
            # guard's decisions are partly length", and its rate is observed --
            # but a referee may reasonably ask what happens without that
            # constraint. So the strictly more conservative bound, the maximum
            # over every budget in either unit, is computed and recorded rather
            # than argued about. It is never the binding number; it is the
            # answer to "does the conclusion depend on the constraint".
            any_budget = max(
                length_only_gap(harmful_n, benign_n, k)
                for harmful_n, benign_n in units.values()
                for k in range(1, len(harmful_n) + len(benign_n) + 1)
            )
            # THE CONSERVATIVE VERDICT: the binding bound is the LARGEST one, so a
            # cell clears only if it clears in both units. Taking the smaller, or
            # taking whichever unit was measured first, is how a control comes to
            # certify the case it was built to catch.
            binding = max(bounds.values())
            rows.append(
                {
                    "family": family,
                    "reported": family in reported,
                    "harmful_block_rate": harmful_rate,
                    "benign_block_rate": benign_rate,
                    "observed_gap": observed,
                    "n_blocked_pooled": n_blocked,
                    "length_only_gap_bound": binding,
                    "length_only_gap_bound_by_unit": bounds,
                    "binding_unit": max(bounds, key=lambda u: bounds[u]),
                    "clears_bound": observed > binding,
                    "length_only_gap_bound_any_budget": any_budget,
                    "clears_any_budget_bound": observed > any_budget,
                    "auroc_by_unit": {
                        "chars": length_auroc(harmful_ct, benign_ct),
                        "tokens": _length_auroc_numeric(*units["tokens"]),
                    },
                    "mean_harmful": {u: sum(h) / len(h) for u, (h, _) in units.items()},
                    "mean_benign": {u: sum(b) / len(b) for u, (_, b) in units.items()},
                }
            )
        results[guard] = rows
        print(f"\n{guard}  (arms from {source}; harmful ciphertexts verified byte-identical)")
        print(
            f"  {'condition':17s} {'gap':>7s} {'bnd.chr':>8s} {'bnd.tok':>8s} "
            f"{'margin':>7s} {'unit':>6s} {'anyK':>6s}  verdict"
        )
        for row in rows:
            mark = "REPORTED" if row["reported"] else "--"
            verdict = "length CANNOT account" if row["clears_bound"] else "inconclusive"
            if row["observed_gap"] == 0.0:
                verdict = "no gap to defend"
            print(
                f"  {row['family']:17s} {row['observed_gap']:>+7.3f} "
                f"{row['length_only_gap_bound_by_unit']['chars']:>+8.3f} "
                f"{row['length_only_gap_bound_by_unit']['tokens']:>+8.3f} "
                f"{row['observed_gap'] - row['length_only_gap_bound']:>+7.3f} "
                f"{row['binding_unit']:>6s} "
                f"{row['length_only_gap_bound_any_budget']:>+6.2f} "
                f"{mark:9s} {verdict}"
            )

    out = args.out or OUT / "guard_arm_length_bound_20260821.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "corpus_digest": digests,
                "method": (
                    "upper bound: the best length-only guard blocking the k longest of the "
                    "200 pooled ciphertexts, k fixed to the real guard's own blocked count. "
                    "Computed in BOTH characters and the guard's own tokens; the reported "
                    "bound is the larger of the two, so a cell clears only if it clears in "
                    "both units. Observed gap above the bound means length cannot account for it."
                ),
                "guards": results,
            },
            indent=1,
        )
    )
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
