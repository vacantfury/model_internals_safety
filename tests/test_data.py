"""Prompt-set loading tests.

Two of these guard reproducibility rather than correctness. `limit` must be a
prefix, not a sample, or two runs at different sizes stop being comparable; and
the digest must cover prompt *text*, or an edited corpus file silently reuses a
stale activation cache — the one artifact expensive enough that nobody would
notice.
"""

from __future__ import annotations

import json

import pytest

from internals_safety.data import digest, load_prompts, prompt_set


def write_set(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    return write_set(
        tmp_path / "small.jsonl",
        [
            {"id": f"p{index}", "category": "c", "source": "s", "prompt": f"prompt {index}"}
            for index in range(5)
        ],
    )


class TestLoading:
    def test_reads_every_field(self, corpus):
        prompts = load_prompts(corpus)
        assert len(prompts) == 5
        assert prompts[0].id == "p0"
        assert prompts[0].text == "prompt 0"
        assert prompts[0].category == "c"
        assert prompts[0].source == "s"

    def test_limit_takes_a_prefix_not_a_sample(self, corpus):
        assert [prompt.id for prompt in load_prompts(corpus, limit=3)] == ["p0", "p1", "p2"]

    def test_limit_beyond_the_file_is_an_error(self, corpus):
        with pytest.raises(ValueError, match="fewer than the requested"):
            load_prompts(corpus, limit=99)

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "gappy.jsonl"
        path.write_text('{"id":"a","prompt":"x"}\n\n{"id":"b","prompt":"y"}\n', encoding="utf-8")
        assert len(load_prompts(path)) == 2

    def test_a_record_without_a_prompt_field_names_its_line(self, tmp_path):
        path = write_set(tmp_path / "bad.jsonl", [{"id": "a", "prompt": "x"}, {"id": "b"}])
        with pytest.raises(ValueError, match=":2 has no 'prompt' field"):
            load_prompts(path)

    def test_a_missing_file_says_where_the_corpus_comes_from(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="guardrail sibling"):
            load_prompts(tmp_path / "absent.jsonl")

    def test_prompt_set_resolves_against_a_data_dir(self, corpus, tmp_path):
        assert len(prompt_set("small.jsonl", data_dir=tmp_path)) == 5


class TestDigest:
    def test_is_stable_for_the_same_content(self, corpus):
        assert digest(load_prompts(corpus)) == digest(load_prompts(corpus))

    def test_changes_when_prompt_text_changes(self, tmp_path):
        first = load_prompts(write_set(tmp_path / "a.jsonl", [{"id": "x", "prompt": "one"}]))
        second = load_prompts(write_set(tmp_path / "b.jsonl", [{"id": "x", "prompt": "two"}]))
        # Same id, different text — an id-only digest would collide here and the
        # activation cache would serve tensors for the wrong prompts.
        assert digest(first) != digest(second)

    def test_changes_with_the_prefix_length(self, corpus):
        assert digest(load_prompts(corpus, limit=3)) != digest(load_prompts(corpus, limit=4))
