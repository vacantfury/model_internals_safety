"""The venue bib is build output, and this pins the two ways it could lie.

A generated bib fails in exactly two directions: it can drop a cited key (which
renders as a bare ``?`` in the PDF) or it can truncate an entry (which renders as
a malformed reference). Both compile. So the tests below are mutation tests on
the two mechanisms that prevent them -- fail-loud on unresolved keys, and brace
matching rather than a line-anchored terminator.

The fixtures are built to the STRICTEST real shapes in the corpus, per this
repo's fixture rule: a hyphenated ACL key, a ``note`` field containing braces, a
``\\citep`` carrying optional arguments, and TWO direction bibs rather than one.
A fixture more permissive than the real thing certifies rather than tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_venue_bib as bvb  # noqa: E402

# --- the corpus shapes the extractor must survive ---------------------------

HYPHENATED_ACL_ENTRY = """@inproceedings{tasawong-etal-2025-shortcut,
    title = "Shortcut Learning in Safety",
    author = "Tasawong, Panuthep",
    year = "2025"
}"""

# The note field's braces are the reason parse_entries matches braces instead of
# looking for a line-anchored `}` -- a `(?m)^\\}` terminator truncates here.
ENTRY_WITH_BRACES_IN_NOTE = """@misc{feng2026refusalcueshortcut,
  title        = {When Refusal Looks Safe},
  author       = {Feng, Yu},
  note         = {Response-side shortcut in {WildGuardMix}; see {Sec. 4}.
}
still inside the note, and this line starts with a brace-free word}
}"""

PLAIN_ENTRY = """@misc{jiao2026siren,
  title        = {LLM Safety From Within},
  author       = {Jiao, Difan}
}"""


def _corpus(tmp_path: Path) -> Path:
    """A corpus with TWO directions, because the real one has two."""
    corpus = tmp_path / "literature"
    security = corpus / "llm-security"
    internals = corpus / "model-internals"
    security.mkdir(parents=True)
    internals.mkdir(parents=True)
    (security / "references.bib").write_text(HYPHENATED_ACL_ENTRY, encoding="utf-8")
    (internals / "references.bib").write_text(
        ENTRY_WITH_BRACES_IN_NOTE + "\n\n" + PLAIN_ENTRY, encoding="utf-8"
    )
    return corpus


class TestCitedKeys:
    def test_it_reads_every_natbib_cite_form(self) -> None:
        tex = r"""
        \citep{alpha} and \citet{beta} and \citep{gamma,delta}
        and \citep[see][p.~4]{epsilon} and \citealp*{zeta}
        """
        assert bvb.cited_keys(tex) == {
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
        }

    def test_a_commented_out_citation_is_not_cited(self) -> None:
        tex = "\n".join(
            [
                r"% \citep{retracted_claim}",
                r"   % \citep{also_retracted}",
                r"\citep{live}",
            ]
        )
        assert bvb.cited_keys(tex) == {"live"}

    def test_a_percent_line_continuation_inside_a_key_list_survives(self) -> None:
        # The real paper wraps a three-key \citep with a trailing `%`. That is
        # not a whole-line comment, so the strip above must not eat the keys.
        tex = "\\citep{first,second,%\nthird}"
        assert bvb.cited_keys(tex) == {"first", "second", "third"}

    def test_hyphenated_acl_keys_survive(self) -> None:
        assert bvb.cited_keys(r"\citep{tasawong-etal-2025-shortcut}") == {
            "tasawong-etal-2025-shortcut"
        }

    def test_a_TRAILING_comment_does_not_leak_a_citation(self) -> None:
        # The whole-line-only strip this replaced returned {live, parked}: a
        # key we deliberately stopped citing came back, and had it been dropped
        # from the corpus at the same time the build would have aborted on a
        # key nothing cites.
        tex = r"\citep{live} % superseded, was \citep{parked}"
        assert bvb.cited_keys(tex) == {"live"}

    def test_an_ESCAPED_percent_is_not_a_comment(self) -> None:
        # The reason the strip cannot be a bare `%.*$`: prose says "40\% of".
        tex = r"Refusal fell by 40\% here \citep{beta}"
        assert bvb.cited_keys(tex) == {"beta"}

    def test_the_continuation_still_survives_the_stricter_strip(self) -> None:
        # Stripping to end of line must LEAVE the newline, or the brace group
        # of a wrapped \citep never closes and every key in it is lost.
        assert bvb.cited_keys("\\citep{first,second,%\n  third}") == {
            "first",
            "second",
            "third",
        }


class TestParseEntries:
    def test_a_note_field_with_braces_is_not_truncated(self) -> None:
        entries = bvb.parse_entries(ENTRY_WITH_BRACES_IN_NOTE)
        entry = entries["feng2026refusalcueshortcut"]
        assert entry.endswith("}")
        # The tell of truncation: the closing brace arrives before the note does.
        assert "still inside the note" in entry

    def test_a_line_anchored_terminator_would_have_truncated_it(self) -> None:
        # Mutation: this is the implementation the brace matcher replaced.
        import re

        naive = re.search(
            r"(?ms)^@\w+\{feng2026refusalcueshortcut,.*?^\}",
            ENTRY_WITH_BRACES_IN_NOTE,
        )
        assert naive is not None
        assert "still inside the note" not in naive.group(0)

    def test_hyphenated_keys_parse(self) -> None:
        assert "tasawong-etal-2025-shortcut" in bvb.parse_entries(HYPHENATED_ACL_ENTRY)


class TestBuild:
    def _paper(self, tmp_path: Path, body: str) -> Path:
        kit = tmp_path / "paper" / "as-x" / "some_kit"
        kit.mkdir(parents=True)
        (kit / "paper.tex").write_text(body, encoding="utf-8")
        return kit

    def test_it_reads_every_direction_not_just_the_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # §5.2's defect: keys were checked against ONE direction bib and four
        # entries were nearly duplicated. One key here lives in each direction.
        kit = self._paper(
            tmp_path,
            r"\citep{tasawong-etal-2025-shortcut} \citet{jiao2026siren}",
        )
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        count, written = bvb.build("as-x", corpus=_corpus(tmp_path))
        assert count == 2
        generated = (kit / "paper.bib").read_text(encoding="utf-8")
        assert "tasawong-etal-2025-shortcut" in generated
        assert "jiao2026siren" in generated
        assert written == [kit / "paper.bib"]

    def test_an_unresolved_key_is_an_ERROR_not_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dropped key renders as `?` and compiles, which is why this is fatal.
        self._paper(tmp_path, r"\citep{jiao2026siren,never_added_to_the_corpus}")
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        with pytest.raises(SystemExit) as caught:
            bvb.build("as-x", corpus=_corpus(tmp_path))
        assert "never_added_to_the_corpus" in str(caught.value)

    def test_an_empty_corpus_is_an_ERROR(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._paper(tmp_path, r"\citep{jiao2026siren}")
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit) as caught:
            bvb.build("as-x", corpus=empty)
        assert "no direction bibs" in str(caught.value)

    def test_every_kit_of_a_paper_gets_the_same_bib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The kits are dual-truth by construction (test_paper_source_parity);
        # their bibs must not become a third copy that drifts.
        root = tmp_path / "paper" / "as-x"
        for name in ("arxiv_latex", "venue_latex"):
            kit = root / name
            kit.mkdir(parents=True)
            (kit / "paper.tex").write_text(r"\citep{jiao2026siren}", encoding="utf-8")
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        _, written = bvb.build("as-x", corpus=_corpus(tmp_path))
        assert len(written) == 2
        assert len({path.read_text(encoding="utf-8") for path in written}) == 1


def _generated_kits() -> list[Path]:
    """Kits whose `paper.bib` claims to be generated, and are therefore in scope.

    Scoping by the marker rather than by a hard-coded paper list is what keeps
    this honest: a kit still on a hand-written bib is visibly out of coverage,
    and joins it the moment it is migrated -- no test edit, no venue token, no
    silent gap. AS-5's kits are out today (TODO 74).
    """
    if not bvb.PAPER_ROOT.is_dir():
        return []
    return [
        path.parent
        for path in sorted(bvb.PAPER_ROOT.rglob("paper.bib"))
        if path.read_text(encoding="utf-8").startswith(bvb.GENERATED_MARKER)
    ]


class TestAgainstTheRealCorpus:
    """Skipped on a fresh clone: `paper/` is gitignored and the corpus is external."""

    def test_every_key_a_generated_kit_cites_resolves(self) -> None:
        corpus = bvb.corpus_dir()
        kits = _generated_kits()
        if not kits or not bvb.master_bibs(corpus):
            pytest.skip("no generated kit, or the shared corpus is not present here")
        # resolve_corpus, not a fresh merge: a test that merges differently
        # from the generator can disagree with it on a duplicated key.
        available, _ = bvb.resolve_corpus(bvb.master_bibs(corpus))
        unresolved: list[str] = []
        for kit in kits:
            for key in bvb.cited_keys((kit / "paper.tex").read_text(encoding="utf-8")):
                if key not in available:
                    unresolved.append(f"{kit.name}: {key}")
        assert not unresolved, "cited keys missing from the shared corpus: " + ", ".join(
            unresolved
        )

    def test_a_generated_bib_is_not_stale(self) -> None:
        """Regenerating must be a no-op -- else the PDF and the corpus disagree."""
        corpus = bvb.corpus_dir()
        kits = _generated_kits()
        if not kits or not bvb.master_bibs(corpus):
            pytest.skip("no generated kit, or the shared corpus is not present here")
        sources = bvb.master_bibs(corpus)
        available, _ = bvb.resolve_corpus(sources)
        for kit in kits:
            keys = bvb.cited_keys((kit / "paper.tex").read_text(encoding="utf-8"))
            on_disk = (kit / "paper.bib").read_text(encoding="utf-8")
            # Compare against what the RENDERER produces, not against the raw
            # master entries. The earlier form asserted the verbatim master
            # entry appeared on disk, which held only while the generator was a
            # pure copy; it went red the first time the generator legitimately
            # transformed an entry (curation-field stripping, 2026-08-21). A
            # staleness check must follow the generator, never re-derive it.
            assert on_disk == bvb.render(available, keys, sources), (
                f"{kit.name}: paper.bib differs from what the corpus renders -- "
                "re-run scripts/build_venue_bib.py <paper-id>"
            )


class TestDuplicateKeys:
    """One key, one entry -- in either scope.

    Across directions, a second copy silently stops being the citation. WITHIN
    one direction it is worse: ``parse_entries`` returns a dict, so the last
    copy wins and nothing is reported at all. Both are the same violation and
    the check has to see both, which is why ``entry_keys`` keeps the repeats.
    """

    def test_a_key_in_two_directions_is_an_ERROR(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = _corpus(tmp_path)
        # This is the near-miss that motivated the check: a second entry for one
        # paper, minted in a different direction under a different-looking key.
        (corpus / "llm-security" / "references.bib").write_text(
            HYPHENATED_ACL_ENTRY + "\n\n" + PLAIN_ENTRY, encoding="utf-8"
        )
        kit = tmp_path / "paper" / "as-x" / "kit"
        kit.mkdir(parents=True)
        (kit / "paper.tex").write_text(r"\citep{jiao2026siren}", encoding="utf-8")
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        with pytest.raises(SystemExit) as caught:
            bvb.build("as-x", corpus=corpus)
        message = str(caught.value)
        assert "jiao2026siren" in message
        assert "llm-security" in message and "model-internals" in message

    def test_a_key_defined_TWICE_IN_ONE_direction_is_an_ERROR(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = _corpus(tmp_path)
        (corpus / "llm-security" / "references.bib").write_text(
            HYPHENATED_ACL_ENTRY + "\n\n" + HYPHENATED_ACL_ENTRY, encoding="utf-8"
        )
        kit = tmp_path / "paper" / "as-x" / "kit"
        kit.mkdir(parents=True)
        (kit / "paper.tex").write_text(r"\citep{jiao2026siren}", encoding="utf-8")
        monkeypatch.setattr(bvb, "PAPER_ROOT", tmp_path / "paper")
        with pytest.raises(SystemExit) as caught:
            bvb.build("as-x", corpus=corpus)
        message = str(caught.value)
        assert "tasawong-etal-2025-shortcut" in message
        # The two scopes must be told apart in the message, or the operator
        # goes looking in the wrong file for a duplicate that is in front of
        # them. This one is within a single direction.
        assert "within" in message and "across" not in message

    def test_a_dict_merge_would_have_hidden_the_within_file_case(self) -> None:
        # Mutation: this is what resolve_corpus did before entry_keys existed.
        doubled = HYPHENATED_ACL_ENTRY + "\n\n" + HYPHENATED_ACL_ENTRY
        assert len(bvb.parse_entries(doubled)) == 1
        assert len(bvb.entry_keys(doubled)) == 2

    def test_the_real_corpus_has_no_duplicate_keys(self) -> None:
        corpus = bvb.corpus_dir()
        bibs = bvb.master_bibs(corpus)
        if not bibs:
            pytest.skip("the shared corpus is not present in this checkout")
        _, duplicated = bvb.resolve_corpus(bibs)
        assert not duplicated, f"keys defined in two direction bibs: {duplicated}"


class TestCurationFieldsNeverReachTheVenueBib:
    """The masters are a curation record; a venue bib is a citation artifact.

    natbib RENDERS ``note``, so entries carrying ``CANDIDATE -- verify+download``
    typeset that sentence into the bibliography of a built PDF. An automated desk
    check found it in AS-6 before the suite did, which is why these tests exist at
    the GENERATOR rather than as a lint over the output: a stripped field cannot
    be forgotten, an output lint can be.
    """

    def test_a_braced_note_is_stripped(self) -> None:
        entry = '@misc{k,\n  author = {A},\n  note = {CANDIDATE -- verify},\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "note" not in out and "CANDIDATE" not in out
        assert "author" in out and "year" in out

    def test_a_QUOTED_note_is_stripped(self) -> None:
        entry = '@misc{k,\n  author = {A},\n  note = "taken verbatim from the anthology",\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "note" not in out and "verbatim" not in out

    def test_a_note_with_NESTED_BRACES_is_stripped_whole(self) -> None:
        entry = '@misc{k,\n  note = {see {Foo} and {Bar}},\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "Foo" not in out and "Bar" not in out and "year" in out

    def test_a_MULTILINE_note_is_stripped_whole(self) -> None:
        entry = '@misc{k,\n  note = {THE paper: two failure\n modes, and it was ABSENT\n until 2026-08-07},\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "ABSENT" not in out and "failure" not in out and "year" in out

    def test_a_note_as_the_LAST_field_still_parses(self) -> None:
        entry = '@misc{k,\n  year = {2026},\n  note = {trailing}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "trailing" not in out
        assert list(bvb.parse_entries(out)) == ["k"]

    def test_a_note_as_the_FIRST_field_still_parses(self) -> None:
        entry = '@misc{k,\n  note = {leading},\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "leading" not in out
        assert list(bvb.parse_entries(out)) == ["k"]

    def test_abstract_is_stripped_too(self) -> None:
        entry = '@misc{k,\n  abstract = {We show that...},\n  year = {2026}\n}'
        assert "We show" not in bvb.strip_curation_fields(entry)

    def test_a_TITLE_containing_the_word_note_is_NOT_touched(self) -> None:
        """The strip is top-level-only. A value that merely looks like a field
        is a value, and a stripper that reads inside one silently truncates a
        title -- worse than the leak it fixes."""
        entry = '@misc{k,\n  title = {On {A}, note = the second movement},\n  year = {2026}\n}'
        out = bvb.strip_curation_fields(entry)
        assert "second movement" in out and "title" in out

    def test_a_field_named_ANNOTE_is_not_mistaken_for_note(self) -> None:
        entry = '@misc{k,\n  annote = {keep me},\n  year = {2026}\n}'
        assert "keep me" in bvb.strip_curation_fields(entry)

    def test_the_REAL_corpus_generates_a_bib_with_no_curation_fields(self) -> None:
        """The end-to-end property: whatever the masters carry, the kit does not."""
        bibs = bvb.master_bibs(bvb.corpus_dir())
        if not bibs:
            pytest.skip("the shared corpus is not present in this checkout")
        available, _ = bvb.resolve_corpus(bibs)
        assert available, "corpus resolved empty"
        leaked = [
            key for key, entry in available.items()
            if any(f"{name}" in bvb.strip_curation_fields(entry).lower()
                   for name in ("candidate — verify", "candidate --- verify", "candidate -- verify"))
        ]
        assert not leaked, f"curation text survived the strip: {leaked}"
