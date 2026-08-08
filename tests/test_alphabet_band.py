"""The naturalness band is only an experiment while its ONE variable stays one.

`tests/test_encodings.py` already round-trips every configured rung and checks
canonicalize idempotence ladder-wide, so these six inherit that. What it cannot
check is the thing this band exists for: that the rungs differ in the *alphabet*
and in nothing else.

That is not a tidiness concern. `fullwidth`'s 31-point harm-sensitivity gap
(`instrument_layer.md` §3.6) is confounded between completeness of substitution
and corpus naturalness, and this band breaks the confound by holding
completeness fixed. A rung that quietly also maps digits, or drops a letter, or
emits a different script than it claims, re-introduces the confound while every
round-trip test stays green — and the resulting run would look like a clean
answer to a question it stopped asking.
"""

from __future__ import annotations

import unicodedata

import pytest

from internals_safety.encodings.deterministic.alphabets import (
    ASCII_LOWER,
    ASCII_UPPER,
    LETTER_ALPHABETS,
    AlphabetMapEncoder,
    _build_alphabet,
)
from internals_safety.encodings.registry import load_ladder

BAND = [
    "fullwidth_letters", "math_bold", "math_sans",
    "math_monospace", "circled", "math_fraktur",
]

# Every non-letter class the band must leave alone, plus a letter to prove the
# encoder is doing anything at all.
MIXED = "Attack 123 the-zebra, at 07:45! (x) [y] {z} 50% #tag \"quoted\" 'q' a_b"


def skeleton(plain: str, cipher: str) -> str:
    """The ciphertext at the positions the PLAINTEXT had no ASCII letter.

    Filtering the ciphertext by `str.isalpha()` instead looks equivalent and is
    not: `circled` glyphs are Unicode category So (Symbol), not Ll/Lu, so Ⓐ is
    NOT alpha and the whole rung would report as untouched punctuation. Reading
    positions off the plaintext is script-independent, and it is the mapping's
    own guarantee — every rung in this band is 1:1 and length-preserving, which
    `test_the_maps_are_length_preserving` pins so this zip stays aligned.
    """
    return "".join(c for p, c in zip(plain, cipher, strict=True) if not p.isalpha())


@pytest.fixture(scope="module")
def ladder():
    return load_ladder()


class TestTheAlphabetTablesAreVerifiedNotTyped:
    """The NFKC check is the only thing separating a right table from a
    self-consistently wrong one: encode and decode share an inverse, so a
    mistyped offset round-trips perfectly while showing the model another
    script."""

    def test_a_wrong_offset_RAISES_rather_than_encoding_to_the_wrong_glyph(self):
        with pytest.raises(ValueError, match="alphabet is wrong at"):
            _build_alphabet(0x1D400 + 1, 0x1D41A)  # off by one: A -> bold B

    def test_the_error_names_the_letter_and_what_it_actually_folded_to(self):
        with pytest.raises(ValueError, match=r"'A'.*folds to 'B'"):
            _build_alphabet(0x1D400 + 1, 0x1D41A)

    @pytest.mark.parametrize("name", BAND)
    def test_every_alphabet_is_injective(self, name):
        """Asserted as a PROPERTY, not as a guard firing. A collision would make
        the inverse a relation rather than a function, but `_build_alphabet`
        carries no injectivity check because the NFKC fold already proves it:
        two letters sharing a glyph would need that glyph to fold to both, and
        the fold loop rejects that first. The property is what matters, so the
        property is what is tested."""
        table = LETTER_ALPHABETS[name]
        assert len(set(table.values())) == len(table)

    def test_a_collision_is_UNREACHABLE_because_the_fold_catches_it_first(self):
        """Pins the reasoning above rather than trusting it — an override that
        collides must fail on the fold, naming the letter."""
        with pytest.raises(ValueError, match=r"alphabet is wrong at 'a'"):
            _build_alphabet(0x1D400, 0x1D41A, overrides={"a": "𝐀"})

    @pytest.mark.parametrize("name", BAND)
    def test_every_letter_is_present_in_both_cases(self, name):
        table = LETTER_ALPHABETS[name]
        assert set(table) == set(ASCII_UPPER + ASCII_LOWER)

    @pytest.mark.parametrize("name", BAND)
    def test_every_glyph_folds_back_to_its_own_letter(self, name):
        for letter, glyph in LETTER_ALPHABETS[name].items():
            assert unicodedata.normalize("NFKC", glyph) == letter

    def test_fraktur_uses_letterlike_symbols_for_its_five_reserved_capitals(self):
        """U+1D506/50B/50C/515/51D are RESERVED. A contiguous range emits them
        and they are not characters — this is the case that justifies
        generating the tables instead of typing them."""
        table = LETTER_ALPHABETS["math_fraktur"]
        assert [table[c] for c in "CHIRZ"] == ["ℭ", "ℌ", "ℑ", "ℜ", "ℨ"]
        for letter in "CHIRZ":
            assert not (0x1D504 <= ord(table[letter]) <= 0x1D51D)


class TestTheBandsONEVariable:
    """Completeness is held fixed so that naturalness is the only thing moving.
    Each test here is a way that could stop being true without any round-trip
    test noticing."""

    @pytest.mark.parametrize("name", BAND)
    def test_the_maps_are_length_preserving(self, ladder, name):
        """Unlike `zero_width` and `combining_marks`, which change length. Length
        is a content-correlated property this repo has already been burned by
        (the AUROC-0.654 length confound), so a band meant to isolate the
        alphabet must not move it."""
        assert len(ladder[name].encode(MIXED).ciphertext) == len(MIXED)

    @pytest.mark.parametrize("name", BAND)
    def test_digits_punctuation_and_spacing_pass_through_untouched(self, ladder, name):
        cipher = ladder[name].encode(MIXED).ciphertext
        assert skeleton(MIXED, cipher) == skeleton(MIXED, MIXED)

    @pytest.mark.parametrize("name", BAND)
    def test_every_ascii_letter_is_actually_replaced(self, ladder, name):
        cipher = ladder[name].encode(MIXED).ciphertext
        assert not any(c in ASCII_UPPER + ASCII_LOWER for c in cipher)

    def test_all_six_rungs_agree_EXACTLY_on_their_non_letter_output(self, ladder):
        """The controlled-variable check stated directly. If one rung ever maps
        a space or a digit, this band stops isolating the alphabet and silently
        becomes the same two-axis comparison `fullwidth` already was."""
        skeletons = {
            name: skeleton(MIXED, ladder[name].encode(MIXED).ciphertext) for name in BAND
        }
        assert len(set(skeletons.values())) == 1

    def test_the_rungs_produce_SIX_different_ciphertexts(self, ladder):
        """The mirror of the test above: identical skeletons must not mean
        identical rungs."""
        assert len({ladder[n].encode(MIXED).ciphertext for n in BAND}) == len(BAND)


class TestTheBridgeToFullwidth:
    def test_fullwidth_letters_differs_from_fullwidth_ONLY_in_non_letters(self, ladder):
        """The pair isolates the punctuation-and-space half of `fullwidth`'s
        substitution, which is the other candidate explanation for its gap."""
        full = ladder["fullwidth"].encode(MIXED).ciphertext
        letters_only = ladder["fullwidth_letters"].encode(MIXED).ciphertext
        assert [c for c in full if c.isalpha()] == [c for c in letters_only if c.isalpha()]
        assert full != letters_only

    def test_fullwidth_maps_the_space_and_fullwidth_letters_does_not(self, ladder):
        assert "　" in ladder["fullwidth"].encode("a b").ciphertext
        assert " " in ladder["fullwidth_letters"].encode("a b").ciphertext


class TestRecoveryPreconditions:
    @pytest.mark.parametrize("name", BAND)
    def test_canonicalize_strips_the_rungs_OWN_codomain(self, ladder, name):
        """A plaintext already containing this alphabet could not be told from
        encoded content, so it is not recoverable and must leave the reference."""
        glyph = LETTER_ALPHABETS[name]["a"]
        assert glyph not in ladder[name].encode(f"attack {glyph} now").plaintext

    @pytest.mark.parametrize("name", BAND)
    def test_a_rung_does_not_strip_a_DIFFERENT_bands_glyphs(self, ladder, name):
        """Over-broad canonicalisation would shrink the corpus for every rung at
        once and make the bands incomparable on different references."""
        other = next(n for n in BAND if n != name)
        foreign = LETTER_ALPHABETS[other]["q"]
        assert foreign in ladder[name].encode(f"attack {foreign} now").plaintext


class TestTheEncoderRejectsWhatItCannotBuild:
    def test_an_unknown_alphabet_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown alphabet"):
            AlphabetMapEncoder(
                family="f", attack_template="{ciphertext}",
                restate_template="{ciphertext}", alphabet="klingon",
            )

    def test_the_configured_band_is_all_reachable_from_the_ladder(self, ladder):
        missing = [name for name in BAND if name not in ladder]
        assert not missing, f"configured but unreachable: {missing}"
