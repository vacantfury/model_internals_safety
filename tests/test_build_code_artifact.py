"""The blind archive's guard rails, pinned by mutation.

An anonymised artifact fails silently in one direction only: it ships. Nobody
re-reads a 300-file zip, so every property below is enforced by the builder at
build time and pinned here by breaking the mechanism and watching the build
refuse -- not by asserting that today's archive happens to be clean.

Fixtures follow this repo's rule and model the STRICTEST real shapes: the
licence fixture carries a real MIT header, the redaction fixtures carry a
comment WRAPPED mid-phrase (the shape that defeated a multi-word key), and the
citation fixture carries a surname that must survive (an author cited in the
module docstrings, not the maintainer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_code_artifact as bca  # noqa: E402

MIT_LICENCE = """MIT License

Copyright (c) 2026 Ada Lovelace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""


class TestTheSweepIsFailClosed:
    """The sweep must FIRE, not warn. A build that prints a warning above a
    finished zip has shipped the thing it warned about."""

    @pytest.mark.parametrize(
        "planted",
        [
            "# contact: vacantfury on the forge",
            "# written by Haoyu",
            "# run on the Northeastern cluster",
            "# mail me at someone@example.edu",
            "# submitted from NEU",
            "# the NURC scheduler",
            "# Explorer's gpu partition",
        ],
    )
    def test_a_planted_identifying_string_is_caught(self, tmp_path, planted):
        (tmp_path / "mod.py").write_text(f"{planted}\nx = 1\n")
        assert bca._sweep(tmp_path), f"sweep missed {planted!r}"

    def test_a_clean_tree_sweeps_empty(self, tmp_path):
        (tmp_path / "mod.py").write_text("# nothing identifying here\nx = 1\n")
        assert bca._sweep(tmp_path) == []

    def test_a_cited_surname_is_NOT_treated_as_identity(self, tmp_path):
        """The deliberate non-match. A pattern broad enough to catch every
        surname fires on the bibliography of every module docstring, and a
        sweep that always fires is a refusal with a script attached."""
        (tmp_path / "mod.py").write_text(
            '"""Zhang & Nanda, ICLR 2023, recommend logit difference."""\n'
        )
        assert bca._sweep(tmp_path) == []

    def test_the_lowercase_config_identifier_survives(self, tmp_path):
        """`nurc` is a registry key threaded through the cluster config and
        every preset. It is deliberately NOT swept: the prose that expands the
        acronym is redacted, which leaves an opaque token."""
        (tmp_path / "preset.yaml").write_text("resources:\n  cluster: nurc\n")
        assert bca._sweep(tmp_path) == []


class TestTheBuilderExcludesItself:
    """`FORBIDDEN` is a list of the maintainer's identifying strings, so an
    archive containing the builder contains what the builder removes."""

    def test_the_builder_and_the_manifest_tool_do_not_ship(self):
        assert "build_code_artifact.py" in bca.EXCLUDE_NAMES
        assert "manifest.py" in bca.EXCLUDE_NAMES

    def test_shipping_the_builder_would_trip_its_own_sweep(self, tmp_path):
        """The mutation: drop the self-exclusion and the sweep must object."""
        shipped = tmp_path / "build_code_artifact.py"
        shipped.write_text(Path(bca.__file__).read_text())
        assert bca._sweep(tmp_path), "the builder's own pattern list swept clean"


class TestTheResearchRecordNeverShips:
    @pytest.mark.parametrize(
        "name", ["text_docs", "paper", "outputs", "data", "knowledge", "ops", "run", "CLAUDE.md", "TODO.md", ".git"]
    )
    def test_it_is_excluded(self, name):
        assert name in bca.EXCLUDE_NAMES

    def test_the_include_list_is_an_allow_list(self):
        """A deny-list ships each new top-level directory by default."""
        assert set(bca.INCLUDE) == {"src", "scripts", "conf", "tests", "pyproject.toml"}


class TestTheVendoredLicence:
    def test_the_holder_is_redacted_and_the_terms_are_not(self):
        out = bca._redacted_license(MIT_LICENCE)
        assert "Ada Lovelace" not in out
        assert "identity withheld for review" in out
        assert "Permission is hereby granted" in out
        assert out.startswith("MIT License")

    def test_the_year_is_preserved(self):
        assert "2026" in bca._redacted_license(MIT_LICENCE)

    @pytest.mark.parametrize("count", [0, 2])
    def test_an_unexpected_notice_count_refuses_rather_than_ships(self, count):
        """A licence whose format changed under us must fail here, not ship a
        name it silently failed to find."""
        body = "MIT License\n\n" + "Copyright (c) 2026 Ada Lovelace\n" * count + "\nPermission...\n"
        with pytest.raises(SystemExit):
            bca._redacted_license(body)


class TestRedactionSurvivesWrapping:
    def test_a_phrase_split_across_two_comment_lines_is_still_redacted(self, tmp_path):
        """The shape that defeated the first map: no multi-word key can match
        a phrase a comment wrap has split."""
        target = tmp_path / "cost.yaml"
        target.write_text(
            "  # This matters because V100 is the highest-availability GPU in NEU\n"
            "  # Explorer's public pool, which makes it the obvious fallback.\n"
        )
        bca._redact_site(tmp_path)
        assert bca._sweep(tmp_path) == []
        assert "public pool" in target.read_text()

    def test_redaction_substitutes_rather_than_deletes_the_line(self, tmp_path):
        """Deleting the line left the NEXT line starting mid-sentence: still
        anonymous, no longer readable, in an artifact whose job is to be read."""
        target = tmp_path / "cost.yaml"
        target.write_text(
            "  # Explorer is free of charge to Northeastern students and faculty, so the\n"
            "  # money line of the approval gate is judge API spend only.\n"
        )
        before = len(target.read_text().splitlines())
        bca._redact_site(tmp_path)
        after = target.read_text()
        assert len(after.splitlines()) == before
        assert "money line of the approval gate" in after


class TestRecapitalisation:
    def test_a_sentence_start_is_capitalised(self):
        out = bca._recapitalise("gres_hardware:\n  # the shared cluster's `gpu` partition holds cards.\n")
        assert "# The shared cluster's" in out

    def test_a_wrapped_continuation_stays_lowercase(self):
        """The previous line ends mid-sentence, so capitalising would break it."""
        out = bca._recapitalise(
            "    # This matters because V100 is the highest-availability GPU in\n"
            "    # the shared cluster's public pool, which makes it the fallback.\n"
        )
        assert "# the shared cluster's public pool" in out

    def test_a_line_after_a_full_stop_is_capitalised(self):
        out = bca._recapitalise(
            "# Multi-GPU work schedules here.\n# the shared cluster's QOS never does.\n"
        )
        assert "# The shared cluster's QOS" in out


class TestTheVendoredMetadataIsDerived:
    def test_it_is_read_from_the_installed_distribution(self, tmp_path):
        """Derived, never invented: a hand-written manifest drifts from the
        package it describes, and the drift surfaces in a reviewer's sync."""
        dist = tmp_path / "widget-1.2.3.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.4\n"
            "Name: widget\n"
            "Version: 1.2.3\n"
            "Requires-Python: >=3.11\n"
            "Requires-Dist: anthropic>=0.116\n"
            "Requires-Dist: boto3>=1.34; extra == 'bedrock'\n"
            "\n"
            "# readme body\n"
        )
        out = bca._vendored_pyproject(dist, "widget")
        assert 'version = "1.2.3"' in out
        assert 'requires-python = ">=3.11"' in out
        assert '"anthropic>=0.116",' in out
        assert "boto3" not in out, "an extra was pulled into the runtime deps"

    def test_an_ambiguous_dist_info_refuses(self, tmp_path):
        for name in ("widget-1.2.3.dist-info", "widget-1.2.4.dist-info"):
            (tmp_path / name).mkdir()
        with pytest.raises(SystemExit):
            bca._dist_info(tmp_path, "widget")
