"""Chunk boundaries -- sentence first, and never inside a tag.

A chunk boundary is a place where the speech stops and starts again, so it has
to be somewhere a speaker would also stop: after a sentence, after a line, at
a space. And it has to be somewhere the *tags* are not, because half a
``[chuckle]`` is not a tag at all -- it is punctuation the model reads out
loud, in two pieces, neither of which means anything.

The rule under test is therefore twofold: prefer the best boundary available,
and move a boundary that landed inside a tag out of it. The one documented
exception is a tag run that is longer than the chunk limit by itself: it stays
whole and the chunk grows past the ceiling, because splitting it would be
worse than exceeding it.
"""

from __future__ import annotations

import pytest

from ComfyUI_Chatterbox.modules.text_processing import chunk_text


def _assert_no_split_tag(chunks: list[str]) -> None:
    """No chunk may contain an opening bracket without its closing one.

    That is the whole property: a tag that is split across two chunks shows up
    here as a chunk that opens a bracket and never closes it.
    """
    for chunk in chunks:
        opened = chunk.count("[") - chunk.count("]")
        assert opened <= 0, f"a square tag was split: {chunk!r}"
        opened = chunk.count("<") - chunk.count(">")
        assert opened <= 0, f"an angle tag was split: {chunk!r}"


class TestBoundaries:
    def test_a_sentence_end_is_preferred_over_a_space(self) -> None:
        """The cut lands after the sentence, not at the last word before it."""
        text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa."
        chunks = chunk_text(text, 45)
        assert chunks[0] == "Alpha beta gamma delta.", (
            "the cut took the last space instead of the sentence end"
        )

    def test_an_exclamation_mark_ends_a_sentence_too(self) -> None:
        chunks = chunk_text("Alpha beta gamma delta! Epsilon zeta eta theta!", 45)
        assert chunks[0] == "Alpha beta gamma delta!"

    def test_a_question_mark_ends_a_sentence_too(self) -> None:
        chunks = chunk_text("Alpha beta gamma delta? Epsilon zeta eta theta?", 45)
        assert chunks[0] == "Alpha beta gamma delta?"

    def test_a_space_is_used_when_no_sentence_end_exists(self) -> None:
        chunks = chunk_text("alpha beta gamma delta epsilon zeta", 20)
        assert chunks[0] == "alpha beta gamma"

    def test_a_line_break_is_used_when_no_sentence_end_exists(self) -> None:
        chunks = chunk_text("alpha beta gamma\ndelta epsilon zeta", 20)
        assert chunks[0] == "alpha beta gamma"

    def test_a_hard_cut_only_happens_for_an_unbreakable_run(self) -> None:
        """A URL has no spaces to cut at, so the limit is the only boundary."""
        url = "https://example.com/" + "a" * 480
        chunks = chunk_text(url, 100)
        assert len(chunks) == 5
        assert all(len(chunk) == 100 for chunk in chunks)

    def test_every_chunk_is_at_most_the_limit(self) -> None:
        text = "A sentence here. " * 100
        chunks = chunk_text(text, 200)
        assert all(len(chunk) <= 200 for chunk in chunks)


class TestTagSafeBoundaries:
    def test_a_chunk_boundary_never_splits_a_square_tag(self) -> None:
        """The tag sits where a naive cut would land inside it."""
        text = "x" * 398 + "[chuckle]" + "y" * 500
        chunks = chunk_text(text, 400)
        _assert_no_split_tag(chunks)
        assert "[chuckle]" in "".join(chunks)

    def test_a_chunk_boundary_never_splits_an_angle_tag(self) -> None:
        text = "x" * 398 + "<giggle>" + "y" * 500
        chunks = chunk_text(text, 400)
        _assert_no_split_tag(chunks)
        assert "<giggle>" in "".join(chunks)

    def test_a_tag_with_a_space_in_it_is_never_split(self) -> None:
        """``[clear throat]`` is the one Turbo tag with an internal space,
        which is exactly where a space-based cut would land."""
        text = "[clear throat] and the rest of the script goes on here"
        chunks = chunk_text(text, 12)
        _assert_no_split_tag(chunks)
        assert "[clear throat]" in chunks[0]

    def test_a_tag_longer_than_the_limit_is_not_split(self) -> None:
        """The one documented exception: the chunk grows past the ceiling."""
        tag = "[" + "a" * 498 + "]"
        chunks = chunk_text(tag, 100)
        assert chunks == [tag]
        assert len(chunks[0]) > 100, "the exception is supposed to exceed the limit"

    def test_a_tag_at_the_very_start_of_a_script_is_not_split(self) -> None:
        text = "[chuckle]" + "y" * 500
        chunks = chunk_text(text, 400)
        _assert_no_split_tag(chunks)
        assert chunks[0].startswith("[chuckle]")

    def test_a_tag_is_never_split_in_a_real_script(self) -> None:
        """The whole thing end to end, with tags scattered through it."""
        script = (
            "First sentence here. [laugh] Second sentence here. "
            "<giggle> Third sentence here. [clear throat] Fourth sentence. "
        ) * 12
        chunks = chunk_text(script, 120)
        _assert_no_split_tag(chunks)
        assert "[laugh]" in "".join(chunks)
        assert "[clear throat]" in "".join(chunks)


class TestNothingIsLost:
    def test_the_concatenation_of_the_chunks_loses_no_words(self) -> None:
        text = "A sentence here. " * 100
        chunks = chunk_text(text, 200)
        assert text.split() == " ".join(chunks).split()

    def test_no_words_are_lost_when_tags_are_in_the_script(self) -> None:
        script = ("One two three. [laugh] Four five six. <giggle> Seven. ") * 20
        chunks = chunk_text(script, 100)
        assert script.split() == " ".join(chunks).split()

    def test_chunking_can_still_be_disabled(self) -> None:
        text = "A sentence here. " * 100
        assert chunk_text(text, 200, enabled=False) == [text]

    @pytest.mark.parametrize("limit", [1, 7, 40, 400])
    def test_the_chunker_always_makes_progress(self, limit: int) -> None:
        """A limit of one is the pathological case: it must still terminate."""
        chunks = chunk_text("a bb ccc [d] eeeee", limit)
        assert "".join(chunks).replace(" ", "") == "abbccc[d]eeeee"


class TestThePlanCutsTheScript:
    """The engine's own view of a long script.

    The model-level half of this -- a stub that says more when the script is
    longer -- lives in ``tests/test_generation.py``, because the module that
    drives the model imports ComfyUI and this file deliberately does not.
    """

    def test_a_long_script_is_cut_into_several_chunks(self) -> None:
        from ComfyUI_Chatterbox.modules.text_processing import plan_text

        plan = plan_text(
            "A sentence here. " * 100,
            crash_protection_template="",
            max_chars_per_chunk=200,
        )
        assert len(plan.chunks) > 1

    def test_the_cut_loses_no_words(self) -> None:
        from ComfyUI_Chatterbox.modules.text_processing import plan_text

        script = "A sentence here. " * 100
        plan = plan_text(
            script,
            crash_protection_template="",
            max_chars_per_chunk=200,
        )
        assert " ".join(plan.chunks).split() == script.split()

    def test_the_cut_can_be_switched_off(self) -> None:
        from ComfyUI_Chatterbox.modules.text_processing import plan_text

        plan = plan_text(
            "A sentence here. " * 100,
            crash_protection_template="",
            enable_chunking=False,
            max_chars_per_chunk=200,
        )
        assert len(plan.chunks) == 1
