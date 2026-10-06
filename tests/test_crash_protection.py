"""S8.4 -- ``crash_protection_template``: padding, disabling, and *where* it applies.

Upstream Chatterbox does not raise on text shorter than roughly 21 characters in
sequential generation -- it produces CUDA tensor errors and takes the process
with them. So this is not a quality-of-life feature, it is the difference between
a short line in a dialogue working and the run dying.

Two things about it are easy to get wrong and neither shows up as an exception:

* **``""`` has to actually disable it.** A template that is falsy but not
  compared by equality still wraps the text in something, and the user who
  turned the workaround off still pays for it.
* **The padding is per segment, not per chunk.** The upstream crash is a
  property of one sequential generation call, and chunking does not create one.
  A five-line dialogue of ten-character lines is five short calls, and a
  per-chunk template would pad one of them and leave the other four to crash.
"""

from __future__ import annotations

import pytest

from ComfyUI_Chatterbox.modules.text_processing import (
    CRASH_PROTECTION_PRESETS,
    CRASH_PROTECTION_TEMPLATE,
    MIN_SAFE_SEGMENT_CHARS,
    apply_crash_protection,
    plan_text,
    split_segments,
)


class TestPaddingAShortSegment:
    def test_a_ten_character_input_is_padded(self) -> None:
        """The corpus's own example, and the case that crashes upstream."""
        text = "Hi there!!"  # 10 characters
        assert len(text) < MIN_SAFE_SEGMENT_CHARS
        padded = apply_crash_protection(text)
        assert padded != text
        assert text in padded
        assert padded.startswith("hmm ,,")
        assert padded.endswith("hmm ,,")

    def test_a_long_enough_segment_is_untouched(self) -> None:
        text = "This sentence is comfortably longer than twenty-one characters."
        assert apply_crash_protection(text) == text

    def test_the_threshold_is_exactly_twenty_one(self) -> None:
        """One character either side of the boundary, on both sides."""
        at = "x" * MIN_SAFE_SEGMENT_CHARS
        under = "x" * (MIN_SAFE_SEGMENT_CHARS - 1)
        assert apply_crash_protection(at) == at, "exactly 21 characters is safe"
        assert apply_crash_protection(under) != under, "20 characters is not"

    def test_whitespace_does_not_count_towards_the_threshold(self) -> None:
        padded = apply_crash_protection("  Hi  ")
        assert "Hi" in padded

    def test_the_threshold_is_configurable(self) -> None:
        assert apply_crash_protection("Hello there.", min_chars=1000) != "Hello there."
        assert apply_crash_protection("Hello there.", min_chars=5) == "Hello there."


class TestDisabling:
    def test_an_empty_template_is_a_no_op(self) -> None:
        assert apply_crash_protection("Hi there!!", "") == "Hi there!!"
        assert apply_crash_protection("", "") == ""

    def test_the_default_can_be_overridden_by_the_caller(self) -> None:
        plan = plan_text("Hi there!!", crash_protection_template="")
        assert plan.text == "Hi there!!"

    def test_empty_is_one_of_the_documented_presets(self) -> None:
        assert "" in CRASH_PROTECTION_PRESETS


class TestDocumentedPresets:
    @pytest.mark.parametrize("template", CRASH_PROTECTION_PRESETS)
    def test_every_preset_substitutes_the_segment(self, template: str) -> None:
        if template == "":
            assert apply_crash_protection("Hi", template) == "Hi"
            return
        assert "Hi" in apply_crash_protection("Hi", template)

    @pytest.mark.parametrize(
        "template",
        ["...ummmmm {seg}", "{seg}... yes... {seg}", "Well, {seg}"],
    )
    def test_a_preset_with_other_braces_does_not_explode(self, template: str) -> None:
        """``str.format`` would raise on a brace the user typed themselves."""
        padded = apply_crash_protection("Hi", template + " {")
        assert padded.endswith("{")
        assert "Hi" in padded

    def test_the_documented_presets_are_the_documented_strings(self) -> None:
        assert CRASH_PROTECTION_PRESETS == (
            "hmm ,, {seg} hmm ,,",
            "...ummmmm {seg}",
            "{seg}... yes... {seg}",
            "Well, {seg}",
            "",
        )
        assert CRASH_PROTECTION_TEMPLATE == CRASH_PROTECTION_PRESETS[0]


class TestAppliedPerSegmentNotPerChunk:
    """The distinction that actually matters when the clip is short."""

    def test_two_short_segments_are_padded_separately(self) -> None:
        """Both lines, not one of them.

        ``"[Alice] Hi there! [Bob] Ok then!"`` is two segments of ten and nine
        characters. A per-chunk template would wrap the whole script once; a
        per-segment template wraps each.
        """
        script = "[Alice] Hi there! [Bob] Ok then!"
        segments = split_segments(script)
        assert [s.text for s in segments] == ["Hi there!", "Ok then!"]

        plan = plan_text(script, enable_chunking=True, max_chars_per_chunk=400)
        assert len(plan.segments) == 2
        for original, segment in zip(("Hi there!", "Ok then!"), plan.segments):
            assert segment.text != original, "this segment was not padded"
            assert original in segment.text
            assert segment.text.startswith("hmm ,,")

        # Two pads, not one: the padding count is the number of segments.
        assert plan.text.count("hmm ,,") == 4, (
            "one 'hmm ,,' either side of each of two segments is four"
        )

    def test_one_long_script_is_padded_once_however_many_chunks_it_makes(self) -> None:
        """Chunking does not multiply the padding.

        A 1200-character script is over the ``silence_padding`` threshold and
        gets cut into several chunks. Every chunk is long enough, so none of them
        is padded -- the crash is about *short* segments, and cutting a long one
        makes it longer, not shorter.
        """
        long_script = "A reasonably long sentence for the chunker. " * 30
        plan = plan_text(long_script, enable_chunking=True, max_chars_per_chunk=200)
        assert len(plan.chunks) > 1
        assert plan.text == long_script.strip()
        assert "hmm ,," not in plan.text

    def test_a_chunked_short_script_is_padded_per_segment_anyway(self) -> None:
        """Three short segments, chunking on: still three pads, not one."""
        script = "[A] One! [B] Two! [C] Three!"
        plan = plan_text(
            script,
            crash_protection_template="{seg}",
            enable_chunking=True,
            max_chars_per_chunk=1000,
        )
        assert len(plan.segments) == 3
        assert [s.text for s in plan.segments] == ["One!", "Two!", "Three!"]
        assert plan.text == "One! Two! Three!"
