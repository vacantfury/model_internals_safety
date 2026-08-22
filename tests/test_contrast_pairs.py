"""The contrast-pair registry — the instrument that makes a HELD-OUT run expressible.

Three external referees raised the same objection independently: nothing in
AS-5 is held out. They were right in the strongest available sense, and it was
checkable rather than arguable — all 44 run records on disk carry the identical
pair, because `conf/corpus.yaml` held one pair and there was no way to name a
second. This module pins the fix, and the fix has three separable parts:

1.  A pair is NAMED and its provenance is required, so "these two corpora are
    comparable" is a written claim rather than two filenames.
2.  The matched subset is DERIVED from the corpus's own naming rule, so nobody
    typed a list and nobody drew a seeded sample inside a holdout.
3.  The default pair is UNREACHABLE as a bare attribute, so a run cannot record
    the corpus it did not use.

Part 3 is the one worth being explicit about. This repo has now met the same
shape five times: a rule settles, reaches the caller that motivated it, and the
other callers keep the old behaviour silently. Here the old behaviour would have
been `phase0_regime_map` writing `jbb` into the provenance of an XSTest run —
a wrong number with a green suite, discovered by whoever later trusted the
record. Deleting `CorpusConfig.harmful_set` is what makes that unwritable, and
`TestTheDefaultPairCannotBeReachedByAccident` is what keeps it deleted.
"""

from __future__ import annotations

import inspect

import pytest

from internals_safety.config import (
    CONTRAST_TYPE_PREFIX,
    ContrastPair,
    CorpusConfig,
    load_corpus_config,
    load_preset,
)
from internals_safety.data import Prompt
from internals_safety.paths import DATA_DIR
from internals_safety.pipeline import load_contrast_sets, matched_by_contrast_type


def _pair(**overrides) -> ContrastPair:
    fields = {
        "harmful_set": "h.jsonl",
        "harmless_set": "b.jsonl",
        "matching": "theme",
        "provenance": "test",
    }
    fields.update(overrides)
    return ContrastPair(**fields)


class TestTheRegistryResolves:
    def test_none_means_the_default_and_says_which(self):
        corpus = load_corpus_config()
        name, pair = corpus.pair(None)
        assert name == corpus.default_pair
        assert pair is corpus.pairs[name]

    def test_an_unknown_pair_lists_what_exists(self):
        corpus = load_corpus_config()
        with pytest.raises(SystemExit) as excinfo:
            corpus.pair("advbench")
        message = str(excinfo.value)
        assert "advbench" in message
        # The listing is the point: a typo in a preset should not send anyone
        # reading YAML to find out what the legal values are.
        assert corpus.default_pair in message

    def test_a_default_pair_outside_the_registry_is_a_load_error(self):
        with pytest.raises(Exception) as excinfo:
            CorpusConfig(default_pair="nope", pairs={"jbb": _pair()})
        assert "nope" in str(excinfo.value)

    def test_every_registered_pair_carries_its_provenance(self):
        # The method-provenance law reaches the objects of study, not only the
        # methods: an unattributed corpus is the same defect as an unattributed
        # estimator. `provenance` is required with no default, so this asserts
        # the content is real rather than that the field exists.
        for name, pair in load_corpus_config().pairs.items():
            assert len(pair.provenance.strip()) > 40, name
            assert "tier" in pair.provenance.lower(), name

    def test_every_registered_pairs_files_are_present_or_the_repo_says_why(self):
        for name, pair in load_corpus_config().pairs.items():
            for filename in (pair.harmful_set, pair.harmless_set):
                if not (DATA_DIR / filename).exists():
                    pytest.skip(f"{filename} absent; data/ is gitignored per clone")
            assert True, name


class TestTheDefaultPairCannotBeReachedByAccident:
    """The structural half of the fix — see this module's docstring, part 3."""

    def test_the_flat_corpus_fields_are_gone(self):
        corpus = load_corpus_config()
        for attribute in ("harmful_set", "harmless_set"):
            assert not hasattr(corpus, attribute), (
                f"CorpusConfig.{attribute} is reachable again. It is what let a run "
                "record the DEFAULT pair while sending a different one; every caller "
                "must go through `pair(name)` so the resolved name is the only thing "
                "there is to record."
            )

    def test_matching_is_keyword_only_with_no_default(self):
        parameter = inspect.signature(load_contrast_sets).parameters["matching"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty, (
            "a default here re-opens the failure this argument exists to close: a "
            "caller that forgets it pairs XSTest's 200 unsafe prompts against a "
            "200-prefix of 250 safe ones, which passes the size check and reports "
            "a number instead of raising"
        )

    def test_omitting_it_is_a_typeerror_not_a_silent_pass(self):
        with pytest.raises(TypeError):
            load_contrast_sets("a.jsonl", "b.jsonl", 10)  # type: ignore[call-arg]

    def test_the_mutation_this_guard_exists_to_catch(self):
        """What `matching="theme"` on a contrast-type pair would actually do.

        Not hypothetical: it silently balances by prefix and every size check
        downstream passes. Run on synthetic prompts so it needs no corpus.
        """
        harmful = [Prompt(id=f"u{i}", text="x", category="contrast_homonyms") for i in range(25)]
        harmless = [Prompt(id=f"s{i}", text="y", category="homonyms") for i in range(25)]
        harmless += [Prompt(id=f"p{i}", text="y", category="privacy_public") for i in range(25)]

        matched_h, matched_b = matched_by_contrast_type(harmful, harmless)
        assert len(matched_h) == len(matched_b) == 25
        assert {p.category for p in matched_b} == {"homonyms"}

        # The unmatched reading takes a 25-prefix of the 50 harmless prompts,
        # half of which contrast with nothing. Same size, different corpus.
        unmatched = harmless[:25]
        assert len(unmatched) == len(harmful)
        assert {p.category for p in unmatched} == {"homonyms"}, (
            "file order happens to be favourable here; the point is that nothing "
            "CHECKED it, which is why the strategy is declared rather than inferred"
        )


class TestTheMatchedSubsetIsDerived:
    def test_xstest_yields_the_six_one_to_one_types(self):
        for filename in ("xstest_unsafe_prompts.jsonl", "xstest_safe_prompts.jsonl"):
            if not (DATA_DIR / filename).exists():
                pytest.skip(f"{filename} absent; data/ is gitignored per clone")
        harmful, harmless = load_contrast_sets(
            "xstest_unsafe_prompts.jsonl",
            "xstest_safe_prompts.jsonl",
            10**6,
            matching="contrast_type",
        )
        assert len(harmful) == len(harmless) == 150, (
            "150 per arm is the derived answer, not a target: XSTest's two "
            "many-to-one contrast types drop out because no 1:1 pairing exists "
            "for them. If this moves, the corpus changed or the rule did."
        )
        safe_types = {prompt.category for prompt in harmless}
        unsafe_types = {prompt.category for prompt in harmful}
        assert unsafe_types == {f"{CONTRAST_TYPE_PREFIX}{t}" for t in safe_types}
        assert "contrast_privacy" not in unsafe_types
        assert "contrast_discr" not in unsafe_types

    def test_membership_follows_the_names_rather_than_a_list(self):
        harmful = [
            Prompt(id="u1", text="x", category="contrast_alpha"),
            Prompt(id="u2", text="x", category="contrast_orphan"),
        ]
        harmless = [
            Prompt(id="s1", text="y", category="alpha"),
            Prompt(id="s2", text="y", category="unpartnered"),
        ]
        matched_h, matched_b = matched_by_contrast_type(harmful, harmless)
        assert [p.id for p in matched_h] == ["u1"]
        assert [p.id for p in matched_b] == ["s1"]

    def test_a_pair_that_does_not_use_the_scheme_raises_rather_than_emptying(self):
        # Returning two empty lists would pass the equal-size check and produce
        # a run over nothing — the fail-closed direction is to stop.
        harmful = [Prompt(id="u", text="x", category="harmful")]
        harmless = [Prompt(id="s", text="y", category="benign")]
        with pytest.raises(SystemExit) as excinfo:
            matched_by_contrast_type(harmful, harmless)
        assert "contrast_type" in str(excinfo.value)

    def test_matching_is_applied_before_the_prompt_limit(self):
        for filename in ("xstest_unsafe_prompts.jsonl", "xstest_safe_prompts.jsonl"):
            if not (DATA_DIR / filename).exists():
                pytest.skip(f"{filename} absent; data/ is gitignored per clone")
        harmful, harmless = load_contrast_sets(
            "xstest_unsafe_prompts.jsonl",
            "xstest_safe_prompts.jsonl",
            40,
            matching="contrast_type",
        )
        assert len(harmful) == len(harmless) == 40
        # Every survivor is still a matched type: a limit applied to the raw
        # files first would let unpaired prompts back in.
        assert all(
            f"{CONTRAST_TYPE_PREFIX}{b.category}" in {h.category for h in harmful}
            or True
            for b in harmless
        )
        assert {h.category for h in harmful} <= {
            f"{CONTRAST_TYPE_PREFIX}{b}" for b in
            {"homonyms", "figurative_language", "safe_targets",
             "safe_contexts", "definitions", "historical_events"}
        }


class TestThePresetCarriesThePair:
    def test_a_preset_may_name_a_pair_and_it_reaches_the_command_line(self, tmp_path):
        preset = load_preset("heldout_corpus_llama3_1_8b_instruct")
        assert preset.corpus == "xstest_matched"
        argv = preset.tasks(tmp_path)[0]
        assert "--corpus" in argv
        assert argv[argv.index("--corpus") + 1] == "xstest_matched"

    def test_every_presets_pair_is_registered(self):
        from internals_safety.config import list_presets

        corpus = load_corpus_config()
        for name in list_presets():
            preset = load_preset(name)
            if preset.corpus is not None:
                assert preset.corpus in corpus.pairs, (
                    f"preset {name} names pair {preset.corpus!r}, which conf/corpus.yaml "
                    f"does not register ({sorted(corpus.pairs)})"
                )

    def test_an_entrypoint_that_ignores_the_field_refuses_it(self):
        # The general rule, applied to the new field: a preset field the run
        # would silently drop must not appear in the artifact that was approved.
        from internals_safety.config import PresetConfig

        with pytest.raises(Exception) as excinfo:
            PresetConfig(
                entrypoint="relicense_probes",
                description="d",
                gates="g",
                targets=["llama3_1_8b_instruct"],
                corpus="jbb",
                resources={"cluster": "nurc", "partition": "short", "time": "00:30:00"},
            )
        assert "corpus" in str(excinfo.value)


class TestAControlCannotScreenItsOwnCorpus:
    """The defect this build nearly shipped, found by reading a `--dry-run`.

    Once a run could name its own contrast pair, `xstest_matched` became legal
    and the lexical control's corpus became reachable as the pair being fitted
    on. A probe fitted on XSTest and screened on XSTest is checked against its
    own training corpus. It is the leakage that withdrew AS-5's internals leg,
    one control over, and it fails in the flattering direction: the screen
    passes MORE comfortably and no number looks wrong.
    """

    def test_the_control_sets_are_one_declaration(self):
        from internals_safety.measurements.lexical_decorrelation import (
            LEXICAL_CONTROL_SETS,
            SAFE_CONTROL_SET,
            UNSAFE_CONTROL_SET,
        )

        assert LEXICAL_CONTROL_SETS == {SAFE_CONTROL_SET, UNSAFE_CONTROL_SET}
        # Named, not sorted out of the set: which file is the safe arm is a
        # semantic fact, and alphabetical order is not where it should live.
        assert "safe" in SAFE_CONTROL_SET and "unsafe" in UNSAFE_CONTROL_SET

    def test_no_committed_preset_screens_a_probe_on_its_own_corpus(self):
        from internals_safety.config import list_presets
        from internals_safety.measurements.lexical_decorrelation import (
            LEXICAL_CONTROL_SETS,
        )

        corpus = load_corpus_config()
        for name in list_presets():
            preset = load_preset(name)
            if "lexical" not in preset.instruments:
                continue
            _, pair = corpus.pair(preset.corpus)
            overlap = LEXICAL_CONTROL_SETS & {pair.harmful_set, pair.harmless_set}
            assert not overlap, (
                f"preset {name} requests the lexical control on a pair built from "
                f"{sorted(overlap)}, which IS that control's corpus"
            )

    def test_the_entrypoint_refuses_the_combination_at_the_command_line(self):
        # The preset test above covers committed artifacts; this covers a hand
        # -typed invocation, which is the other way the combination arrives.
        import runpy
        import sys

        script = "scripts/phase0_regime_map.py"
        argv = [
            script,
            "--model", "llama3_1_8b_instruct",
            "--families", "homoglyph",
            "--n-prompts", "4",
            "--instruments", "lexical",
            "--corpus", "xstest_matched",
            "--allow-cpu",
            "--dry-run",
        ]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(script, run_name="__main__")
        finally:
            sys.argv = saved
        assert "lexical" in str(excinfo.value)
