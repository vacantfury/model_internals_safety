"""Cache resolution for re-licensing: the record is authoritative, the glob is a
fallback, and an unresolvable duplicate is still refused.

**The defect this fixes (TODO item 21).** `relicense_probes.py` could only
prefix-match `<condition>-<digest>.pt`, because the digest hashes the capture
request and is not reproducible offline. When the Qwen phase-0 head run was
killed at the 8 h wall and the tail run re-captured, three conditions ended up
with two files each — and the script refused, correctly, rather than guess which
tensor a licensing decision rested on. That refusal blocked all 15 Qwen rungs for
two days.

The record was naming its own caches the whole time. `results.json` carries
`activations_path`, which is the authoritative link the glob was standing in for.

**Measured before fixing, not assumed:** on Qwen exactly THREE conditions are
duplicated (`plain-harmful`, `plain-harmless`, `base64`) — not the whole set, as
item 21 read. So preferring the record recovers 14 of 15 rungs, and `base64`
stays genuinely undecidable and is still refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from relicense_probes import resolve_batch  # noqa: E402


def touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


class TestTheRecordWins:
    def test_a_recorded_path_is_used_even_when_the_glob_is_ambiguous(self, tmp_path):
        """The whole point: two candidates, and the record decides."""
        touch(tmp_path, "plain-harmful-970dfebff4fa9c29.pt")
        chosen = touch(tmp_path, "plain-harmful-ca74b3d107539b67.pt")
        resolved = resolve_batch(tmp_path, "plain-harmful", str(chosen))
        assert resolved.path == chosen
        assert resolved.route == "record"

    def test_a_recorded_path_from_another_mount_resolves_by_its_digest_name(self, tmp_path):
        """Records store absolute paths from the machine that wrote them.

        A cache reached under a different mount is the same file — the digest in
        the NAME is what identifies it — so a relocated outputs tree must not
        send the resolver back to guessing.
        """
        touch(tmp_path, "plain-harmful-970dfebff4fa9c29.pt")
        real = touch(tmp_path, "plain-harmful-ca74b3d107539b67.pt")
        stale = "/scratch/someone-else/outputs/activations/m/plain-harmful-ca74b3d107539b67.pt"
        resolved = resolve_batch(tmp_path, "plain-harmful", stale)
        assert resolved.path == real
        assert resolved.route == "record"

    def test_a_recorded_path_naming_a_file_that_does_not_exist_falls_through(self, tmp_path):
        """Not an error: the run may have been re-captured under a new digest."""
        only = touch(tmp_path, "encoded-harmful-hex-aaaaaaaaaaaaaaaa.pt")
        resolved = resolve_batch(tmp_path, "encoded-harmful-hex", "/gone/encoded-harmful-hex-bbbb.pt")
        assert resolved.path == only
        assert resolved.route == "unique"


class TestTheGlobFallback:
    def test_a_unique_match_resolves_and_says_so(self, tmp_path):
        """The nine Qwen head-run rungs: caches on disk, no record naming them,
        exactly one candidate each. They must run, and their route must be
        distinguishable from a recorded one."""
        only = touch(tmp_path, "encoded-harmful-hex-1234567890abcdef.pt")
        resolved = resolve_batch(tmp_path, "encoded-harmful-hex", None)
        assert resolved.path == only
        assert resolved.route == "unique"

    def test_no_match_is_None_not_an_error(self, tmp_path):
        assert resolve_batch(tmp_path, "encoded-harmful-nothing", None) is None

    def test_a_prefix_does_not_match_a_LONGER_family_name(self, tmp_path):
        """`reverse_words` must not pick up `reverse_words_extra`'s cache.

        Guarded because the glob is `<prefix>-*` and family names share stems;
        a resolver that silently crossed rungs would put one rung's licensing on
        another rung's tensors.
        """
        touch(tmp_path, "encoded-harmful-reverse_characters-aaaaaaaaaaaaaaaa.pt")
        mine = touch(tmp_path, "encoded-harmful-reverse_words-bbbbbbbbbbbbbbbb.pt")
        resolved = resolve_batch(tmp_path, "encoded-harmful-reverse_words", None)
        assert resolved.path == mine


class TestItStillRefusesWhenItGenuinelyCannotTell:
    def test_ambiguous_and_unrecorded_raises(self, tmp_path):
        """`base64` on Qwen — two captures, no record. Still refused.

        This is the half of item 21's original behaviour that was RIGHT and must
        survive the fix: the failure mode being prevented is a licensing decision
        resting on an unknown tensor, and having a fallback does not make that
        acceptable.
        """
        touch(tmp_path, "encoded-harmful-base64-1111111111111111.pt")
        touch(tmp_path, "encoded-harmful-base64-2222222222222222.pt")
        with pytest.raises(SystemExit) as caught:
            resolve_batch(tmp_path, "encoded-harmful-base64", None)
        assert "ambiguous" in str(caught.value)

    def test_the_refusal_names_both_candidates_and_the_way_out(self, tmp_path):
        """An error a reader cannot act on costs another session to diagnose."""
        touch(tmp_path, "plain-harmful-1111111111111111.pt")
        touch(tmp_path, "plain-harmful-2222222222222222.pt")
        with pytest.raises(SystemExit) as caught:
            resolve_batch(tmp_path, "plain-harmful", None)
        message = str(caught.value)
        assert "1111111111111111" in message and "2222222222222222" in message
        assert "--results" in message
