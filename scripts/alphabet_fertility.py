"""Order the naturalness band by MEASURED tokenizer fertility, not by intuition.

## What this answers

The band in `encodings/deterministic/alphabets.py` varies one thing — which
complete A-Z/a-z alphabet a rung substitutes — in order to un-confound the two
explanations for `fullwidth`'s 31-point harm-sensitivity gap
(`instrument_layer.md` §3.6). But "corpus naturalness" is not observable, and a
band ordered by the author's guess about which script looks more familiar is a
heuristic with no tuning path.

**Fertility is observable, keylessly and in seconds:** tokens per character
under the target model's own tokenizer. A script a tokenizer saw often during
training earns efficient merges; a rare one falls back to UTF-8 bytes, costing
~3-4 tokens for a single codepoint. That number, per rung per model, is the
band's ordering variable.

## What it deliberately does NOT claim

Fertility and corpus exposure are **not separable by this design**. Fertility is
a proxy for exposure and may equally be the causal variable itself: a rung
costing four tokens per character might fail to transmit harm because the
content is smeared across a long token sequence, with nothing to do with how
familiar the script looks. Both readings are live and the write-up names them
rather than assuming one.

It is also NOT the length null. `measurements/length_null.py` controls for
*character* length separating harmful from benign content (the AUROC-0.654
confound). This measures *token* cost of a fixed string under a fixed
tokenizer, which is a property of the rung, not of the corpus split. CLAUDE.md
already records that these are different controls; running one is not running
the other.

## Why the ratio, not the raw number

Raw tokens-per-character is not comparable across tokenizers — Llama and Qwen
have different vocabularies and different baseline efficiency on plain English.
Every rung is therefore reported as a ratio to the SAME model's plaintext
fertility, so 1.0 means "as cheap as ordinary English for this tokenizer" and
4.0 means "four times the tokens for the same characters".

Keyless except for the gated Llama repos, which need the launcher:

    ./run python scripts/alphabet_fertility.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from internals_safety.config import load_model_config
from internals_safety.data import prompt_set
from internals_safety.encodings.registry import load_ladder
from internals_safety.paths import OUTPUTS_DIR

# The band, plus the rungs it has to be commensurable with: the three sound
# rungs from the 2026-08-05 comprehension band, and tag_block as the measured
# floor (ability 0.00 on both models — a script neither can read at all).
DEFAULT_FAMILIES = [
    "fullwidth_letters", "math_bold", "math_sans", "math_monospace",
    "circled", "math_fraktur",
    "fullwidth", "homoglyph", "zero_width", "tag_block",
]

DEFAULT_MODELS = ["llama3_1_8b_instruct", "qwen2_5_7b_instruct"]


def fertility(tokenizer, texts: list[str]) -> float:
    """Mean tokens per character over a corpus.

    Per-string means are averaged rather than pooling all tokens over all
    characters, so one long prompt cannot dominate the statistic.
    """
    ratios = []
    for text in texts:
        if not text:
            continue
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        ratios.append(n_tokens / len(text))
    return statistics.fmean(ratios)


def measure(model_key: str, families: list[str], n_prompts: int) -> dict:
    config = load_model_config(model_key)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=config.trust_remote_code)
    ladder = load_ladder()
    prompts = [p.text for p in prompt_set("jbb_prompts.jsonl", limit=n_prompts)]

    # The denominator. Every rung is scored against ITS OWN canonicalised
    # plaintext, not the raw corpus, because a rung that strips characters would
    # otherwise be credited with the cheapness of a shorter string.
    rows = {}
    plain = fertility(tokenizer, prompts)
    for family in families:
        encoder = ladder[family]
        encoded = [encoder.encode(text) for text in prompts]
        cipher_f = fertility(tokenizer, [e.ciphertext for e in encoded])
        canon_f = fertility(tokenizer, [e.plaintext for e in encoded])
        rows[family] = {
            "fertility": cipher_f,
            "canonical_plaintext_fertility": canon_f,
            "ratio_to_own_plaintext": cipher_f / canon_f,
            "ratio_to_corpus_plaintext": cipher_f / plain,
        }
    return {"model": model_key, "hf_id": config.hf_id, "plaintext_fertility": plain, "rungs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = [measure(m, args.families, args.n_prompts) for m in args.models]

    width = max(len(f) for f in args.families) + 2
    for result in results:
        print(f"\n{result['model']}  (plain English = {result['plaintext_fertility']:.3f} tok/char)")
        print(f"{'rung':<{width}} {'tok/char':>9} {'xplain':>8}")
        ordered = sorted(result["rungs"].items(), key=lambda kv: kv[1]["ratio_to_own_plaintext"])
        for family, row in ordered:
            print(f"{family:<{width}} {row['fertility']:>9.3f} {row['ratio_to_own_plaintext']:>7.2f}x")

    out = args.output or OUTPUTS_DIR / "analysis" / "alphabet_fertility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_prompts": args.n_prompts, "models": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
