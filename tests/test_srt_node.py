"""S8.5 -- ``ChatterboxSRT``: four timing modes, four outputs, one promise.

The interesting property of this node is not that it makes audio; it is that the
subtitle file it returns **describes the audio next to it**. Every test here is
about that pair staying honest, which is why they are built on synthetic clips
of a known length rather than on a model:

* each mode is pinned to the timings it must produce, so a change in the layout
  algorithm has to be a deliberate one;
* ``stretch_to_fit`` is checked against ``timing_tolerance`` -- the bound the
  widget advertises, asserted rather than assumed;
* the overlap fallback is checked for the thing it exists to prevent: a file
  where cue 2 starts before cue 1 ends;
* ``use_native_duration_targeting`` is asserted **absent**. The reference
  documentation scopes that widget to engines with a native duration parameter,
  and Chatterbox has none, so the socket would be a control with no effect.

Nothing here imports the vendored model, and no test touches the network.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest
import torch

from ComfyUI_Chatterbox.modules.audio_utils import S3GEN_SAMPLE_RATE
from ComfyUI_Chatterbox.modules.subtitles import (
    DEFAULT_FADE_FOR_STRETCH_TO_FIT,
    DEFAULT_MAX_STRETCH_RATIO,
    DEFAULT_MIN_STRETCH_RATIO,
    DEFAULT_TIMING_TOLERANCE,
    TIMING_MODE_DEFAULT,
    TIMING_MODES,
    SrtCue,
    SpokenCue,
    apply_timing_mode,
    format_timestamp,
    has_overlap,
    overlapping_pairs,
    parse_srt,
    render_srt,
    stretch_to_duration,
)
from ComfyUI_Chatterbox.nodes import srt_node as srt_module
from ComfyUI_Chatterbox.nodes.srt_node import (
    SRT_TIMING_MODE_DEFAULT,
    SRT_TIMING_MODES,
    ChatterboxSRTNode,
)

from conftest import ROOT_DIR

SRT_NODE_ID = "ChatterboxSRT"
FROZEN_NODE_IDS = ("ChatterboxTTS", "ChatterboxVC")


def _required() -> dict:
    return ChatterboxSRTNode.GET_NODE_INFO_V1()["input"]["required"]


def _optional() -> dict:
    return ChatterboxSRTNode.GET_NODE_INFO_V1()["input"].get("optional", {})


def _tone(seconds: float, value: float = 0.5) -> torch.Tensor:
    """A ``[1, T]`` constant clip whose value identifies it in a stem."""
    return torch.full((1, int(seconds * S3GEN_SAMPLE_RATE)), value, dtype=torch.float32)


def _clip(text: str, start: float, end: float, seconds: float, value: float = 0.5) -> SpokenCue:
    return SpokenCue(
        text=text,
        waveform=_tone(seconds, value),
        slot_start=start,
        slot_end=end,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_node_id_is_new_and_additive(self) -> None:
        schema = ChatterboxSRTNode.GET_SCHEMA()
        assert schema.node_id == SRT_NODE_ID
        assert SRT_NODE_ID not in FROZEN_NODE_IDS
        assert schema.is_output_node is False

    def test_the_frozen_ids_are_untouched(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

        assert ChatterboxTTSNode.GET_SCHEMA().node_id == "ChatterboxTTS"
        assert ChatterboxVCNode.GET_SCHEMA().node_id == "ChatterboxVC"

    def test_the_frozen_snapshot_still_holds_only_the_original_two(self) -> None:
        import json

        with open(
            os.path.join(ROOT_DIR, "tests", "frozen_widget_snapshot.json"),
            encoding="utf-8",
        ) as handle:
            snapshot = json.load(handle)
        assert set(snapshot["nodes"]) == set(FROZEN_NODE_IDS)
        assert SRT_NODE_ID not in snapshot["nodes"]

    def test_there_are_exactly_four_outputs_in_the_documented_order(self) -> None:
        schema = ChatterboxSRTNode.FINALIZE_SCHEMA()
        assert [output.display_name for output in schema.outputs] == [
            "audio",
            "generation_info",
            "timing_report",
            "Adjusted_SRT",
        ]
        assert all(output.id for output in schema.outputs)

    def test_the_schema_validates(self) -> None:
        assert ChatterboxSRTNode.validate_inputs() is True

    def test_an_unknown_timing_mode_is_refused_with_a_message(self) -> None:
        result = ChatterboxSRTNode.validate_inputs(timing_mode="stretch_hard")
        assert isinstance(result, str)
        assert "stretch_hard" in result
        assert "smart_natural" in result

    def test_every_advertised_timing_mode_is_accepted(self) -> None:
        for mode in _required()["timing_mode"][1]["options"]:
            assert ChatterboxSRTNode.validate_inputs(timing_mode=mode) is True


class TestTimingModeEnum:
    def test_the_four_modes_and_the_default(self) -> None:
        assert TIMING_MODES == (
            "stretch_to_fit",
            "pad_with_silence",
            "smart_natural",
            "concatenate",
        )
        assert TIMING_MODE_DEFAULT == "smart_natural"
        assert SRT_TIMING_MODES == TIMING_MODES
        assert SRT_TIMING_MODE_DEFAULT == "smart_natural"

    def test_the_combo_offers_exactly_those_in_that_order(self) -> None:
        config = _required()["timing_mode"]
        assert config[0] == "COMBO"
        assert tuple(config[1]["options"]) == TIMING_MODES
        assert config[1]["default"] == "smart_natural"

    def test_smart_natural_is_also_the_execute_default(self) -> None:
        parameters = inspect.signature(ChatterboxSRTNode.execute).parameters
        assert parameters["timing_mode"].default == "smart_natural"


class TestSupportingSocketDefaults:
    """The four documented defaults, asserted rather than eyeballed."""

    @pytest.mark.parametrize(
        ("name", "expected", "low", "high"),
        [
            ("timing_tolerance", 2.0, 0.5, 10.0),
            ("max_stretch_ratio", 1.0, 0.5, 5.0),
            ("min_stretch_ratio", 0.5, 0.1, 2.0),
            ("fade_for_StretchToFit", 0.01, 0.0, 0.1),
        ],
    )
    def test_the_default_and_the_range(self, name, expected, low, high) -> None:
        config = _required()[name]
        assert config[0] == "FLOAT"
        assert config[1]["default"] == expected
        assert (config[1]["min"], config[1]["max"]) == (low, high)

    def test_the_module_constants_agree_with_the_widgets(self) -> None:
        assert DEFAULT_TIMING_TOLERANCE == 2.0
        assert DEFAULT_MAX_STRETCH_RATIO == 1.0
        assert DEFAULT_MIN_STRETCH_RATIO == 0.5
        assert DEFAULT_FADE_FOR_STRETCH_TO_FIT == 0.01


class TestNativeDurationTargetingIsNotPorted:
    """The reference corpus scopes this widget to other engines. It is not ours."""

    def test_the_socket_is_absent_from_the_schema(self) -> None:
        names = set(_required()) | set(_optional())
        assert "use_native_duration_targeting" not in names

    def test_it_is_absent_from_execute(self) -> None:
        assert "use_native_duration_targeting" not in inspect.signature(
            ChatterboxSRTNode.execute
        ).parameters

    def test_the_widget_name_is_not_a_string_anywhere_in_the_node(self) -> None:
        """Not in the schema, not in ``execute``, and not as a stray literal.

        The docstring has to *name* the widget to explain why it is missing, so
        the scan is AST-wise and skips module, class and function docstrings.
        """
        path = os.path.join(ROOT_DIR, "nodes", "srt_node.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and id(node) not in docstrings:
                assert node.value != "use_native_duration_targeting"


class TestSrtParsing:
    def test_a_real_srt_round_trips(self) -> None:
        content = (
            "1\n00:00:00,000 --> 00:00:02,500\nFirst line.\n\n"
            "2\n00:00:02,500 --> 00:00:05,000\nSecond line.\n\n"
            "3\n00:00:05,000 --> 00:00:07,125\nThird line.\n"
        )
        cues = parse_srt(content)
        assert [cue.index for cue in cues] == [1, 2, 3]
        assert cues[0].start == 0.0
        assert cues[0].end == 2.5
        assert cues[2].end == pytest.approx(7.125)
        assert cues[1].duration == pytest.approx(2.5)

    def test_timestamps_use_a_comma(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"
        assert format_timestamp(1.5) == "00:00:01,500"
        assert format_timestamp(3661.007) == "01:01:01,007"

    def test_a_rendered_file_parses_back_to_the_same_cues(self) -> None:
        cues = [
            SrtCue(1, 0.0, 1.25, "One."),
            SrtCue(2, 1.25, 3.75, "Two."),
        ]
        assert parse_srt(render_srt(cues)) == cues

    def test_a_multi_line_cue_is_one_cue(self) -> None:
        cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nline one\nline two\n")
        assert len(cues) == 1
        assert cues[0].text == "line one line two"

    def test_an_unreadable_block_is_dropped_not_guessed(self) -> None:
        cues = parse_srt(
            "1\nnot a timestamp at all --> neither is this\nNope.\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nKept.\n"
        )
        assert [cue.text for cue in cues] == ["Kept."]

    def test_empty_input_yields_nothing(self) -> None:
        assert parse_srt("") == []
        assert parse_srt("   \n  \n") == []


# ---------------------------------------------------------------------------
# The four modes
# ---------------------------------------------------------------------------


class TestStretchToFit:
    """The SRT is the truth; the audio is time-stretched onto it.

    "Onto it" is bounded by ``min_stretch_ratio`` / ``max_stretch_ratio``, and
    the fixtures below are deliberately all *compression* cases so the exact
    fit is achievable with the default ``max_stretch_ratio`` of 1.0 -- which
    refuses to make a cue longer, because slowing speech down is the one
    adjustment a listener always notices.
    """

    def test_each_cue_lands_exactly_on_its_slot(self) -> None:
        clips = [
            _clip("One.", 0.0, 0.5, 1.0),
            _clip("Two.", 0.5, 1.5, 2.0),
        ]
        result = apply_timing_mode(clips, timing_mode="stretch_to_fit")
        assert [(c.start, c.end) for c in result.cues] == [(0.0, 0.5), (0.5, 1.5)]

    def test_the_audio_is_as_long_as_the_file_says(self) -> None:
        clips = [_clip("One.", 0.0, 0.5, 1.0), _clip("Two.", 0.5, 2.5, 4.0)]
        result = apply_timing_mode(clips, timing_mode="stretch_to_fit")
        assert result.total_seconds == pytest.approx(2.5, abs=0.001)

    @pytest.mark.parametrize("natural", [1.0, 1.5, 2.0, 4.0])
    def test_durations_match_within_the_advertised_tolerance(
        self, natural: float
    ) -> None:
        """The bound the widget advertises, checked against the module's own default."""
        clips = [_clip("One.", 0.0, 1.0, natural)]
        result = apply_timing_mode(clips, timing_mode="stretch_to_fit")
        realised = result.cues[0].end - result.cues[0].start
        assert abs(realised - 1.0) / 1.0 <= DEFAULT_TIMING_TOLERANCE

    def test_the_stretch_is_bounded_by_the_ratio_widgets(self) -> None:
        """A cue four times too long is compressed, but never past the floor."""
        clips = [_clip("Long.", 0.0, 0.25, 2.0)]
        result = apply_timing_mode(
            clips, timing_mode="stretch_to_fit", min_stretch_ratio=0.5
        )
        realised = result.cues[0].end - result.cues[0].start
        # 0.25 s slot / 2.0 s natural = 0.125, clamped up to the 0.5 floor.
        assert result.stretches == (0.5,)
        assert realised == pytest.approx(1.0)  # 2.0 s x 0.5

    def test_max_stretch_ratio_of_one_never_lengthens(self) -> None:
        clips = [_clip("Short.", 0.0, 4.0, 0.4)]
        result = apply_timing_mode(
            clips, timing_mode="stretch_to_fit", max_stretch_ratio=1.0
        )
        assert result.stretches == (1.0,)
        assert result.cues[0].end - result.cues[0].start == pytest.approx(0.4)

    def test_raising_max_stretch_ratio_enables_the_exact_fit(self) -> None:
        """The bounds are the whole mechanism, so moving one has to do something."""
        clips = [_clip("Short.", 0.0, 4.0, 0.4)]
        result = apply_timing_mode(
            clips, timing_mode="stretch_to_fit", max_stretch_ratio=5.0
        )
        assert result.stretches == (5.0,)
        assert result.cues[0].end == pytest.approx(2.0)


class TestPadWithSilence:
    """Natural audio at the written starts; the gaps are real silence."""

    def test_the_audio_keeps_its_natural_length(self) -> None:
        clips = [_clip("One.", 0.0, 0.5, 0.5), _clip("Two.", 2.0, 3.0, 1.0)]
        result = apply_timing_mode(clips, timing_mode="pad_with_silence")
        assert result.stretches == (1.0, 1.0)
        assert result.cues[0].end - result.cues[0].start == pytest.approx(0.5)
        assert result.cues[1].end - result.cues[1].start == pytest.approx(1.0)

    def test_the_track_is_long_enough_to_hold_the_last_written_slot(self) -> None:
        clips = [_clip("One.", 5.0, 6.0, 0.5)]
        result = apply_timing_mode(clips, timing_mode="pad_with_silence")
        assert result.total_seconds == pytest.approx(5.5, abs=0.001)
        assert result.cues[0].start == 5.0

    def test_the_gap_is_silence_not_audio(self) -> None:
        clips = [_clip("One.", 0.0, 0.5, 0.5, value=0.5), _clip("Two.", 2.0, 3.0, 1.0, 0.25)]
        result = apply_timing_mode(clips, timing_mode="pad_with_silence")
        between = result.waveform[:, int(0.5 * S3GEN_SAMPLE_RATE) : 2 * S3GEN_SAMPLE_RATE]
        assert torch.count_nonzero(between) == 0
        assert torch.allclose(result.waveform[:, 0], torch.full((1,), 0.5))
        assert torch.allclose(result.waveform[:, -1], torch.full((1,), 0.25))


class TestSmartNatural:
    """Natural audio, written timings honoured, no overlaps ever emitted."""

    def test_a_clean_script_is_left_on_its_written_timings(self) -> None:
        clips = [_clip("One.", 0.0, 1.5, 1.0), _clip("Two.", 1.5, 3.0, 1.0)]
        result = apply_timing_mode(clips, timing_mode="smart_natural")
        assert result.fallback_used is False
        assert [(c.start, c.end) for c in result.cues] == [(0.0, 1.0), (1.5, 2.5)]
        assert result.stretches == (1.0, 1.0)

    def test_a_line_that_ran_long_is_not_stretched_within_tolerance(self) -> None:
        """Tolerance exists precisely so a modest overrun is left alone."""
        clips = [_clip("One.", 0.0, 1.0, 1.4), _clip("Two.", 1.5, 2.5, 1.0)]
        result = apply_timing_mode(clips, timing_mode="smart_natural", timing_tolerance=2.0)
        assert result.stretches == (1.0, 1.0)
        assert result.cues[0].end == pytest.approx(1.4)

    def test_a_runaway_line_is_compressed_within_the_ratio_bounds(self) -> None:
        clips = [_clip("One.", 0.0, 0.5, 4.0)]
        result = apply_timing_mode(
            clips,
            timing_mode="smart_natural",
            timing_tolerance=0.5,
            min_stretch_ratio=0.5,
            max_stretch_ratio=1.0,
        )
        assert result.stretches == (0.5,)
        assert result.cues[0].end == pytest.approx(2.0)


class TestOverlapFallback:
    """Honouring the written timings can overlap. It must never ship that."""

    @staticmethod
    def _overlapping_clips() -> list[SpokenCue]:
        # Cue 2 is written to start at 0.5 s, while cue 1's audio runs 1.2 s.
        return [_clip("One.", 0.0, 1.0, 1.2, 0.5), _clip("Two.", 0.5, 1.5, 1.2, 0.25)]

    def test_the_written_timings_would_overlap(self) -> None:
        """Guard the fixture: without an overlap there is no fallback to test."""
        clips = self._overlapping_clips()
        honoured = [
            SrtCue(position + 1, clip.slot_start, clip.slot_start + clip.natural_seconds, clip.text)
            for position, clip in enumerate(clips)
        ]
        assert has_overlap(honoured)
        assert overlapping_pairs(honoured) == [(0, 1)]

    def test_smart_natural_falls_back_rather_than_overlapping(self) -> None:
        result = apply_timing_mode(
            self._overlapping_clips(), timing_mode="smart_natural"
        )
        assert result.fallback_used is True

    def test_no_cue_in_the_fallback_output_overlaps(self) -> None:
        result = apply_timing_mode(
            self._overlapping_clips(), timing_mode="smart_natural"
        )
        assert overlapping_pairs(result.cues) == []
        for previous, following in zip(result.cues, result.cues[1:]):
            assert following.start >= previous.end - 1e-6

    def test_the_fallback_is_recorded_in_the_timing_report(self) -> None:
        from ComfyUI_Chatterbox.modules.subtitles import build_timing_report

        clips = self._overlapping_clips()
        result = apply_timing_mode(clips, timing_mode="smart_natural")
        report = build_timing_report(clips, result, timing_mode="smart_natural")
        assert "FALLBACK" in report
        assert "No cue overlaps" in report

    def test_the_fallback_shifts_only_the_later_cue(self) -> None:
        """Cue 1 keeps its written start; cue 2 moves to clear it."""
        result = apply_timing_mode(
            self._overlapping_clips(), timing_mode="smart_natural"
        )
        assert result.cues[0].start == 0.0
        assert result.cues[1].start == pytest.approx(result.cues[0].end)

    def test_the_rendered_srt_has_no_overlap_either(self) -> None:
        """The file is the deliverable; the fallback has to reach it."""
        result = apply_timing_mode(
            self._overlapping_clips(), timing_mode="smart_natural"
        )
        reparsed = parse_srt(render_srt(result.cues))
        assert has_overlap(reparsed) is False
        assert len(reparsed) == 2

    def test_a_script_with_no_overlap_never_triggers_the_fallback(self) -> None:
        clips = [_clip("One.", 0.0, 1.5, 1.2), _clip("Two.", 2.0, 3.0, 1.0)]
        result = apply_timing_mode(clips, timing_mode="smart_natural")
        assert result.fallback_used is False


class TestConcatenate:
    """The original timings are discarded and a new file is written."""

    def test_the_file_starts_at_zero_and_is_contiguous(self) -> None:
        clips = [
            _clip("One.", 10.0, 11.0, 0.5),
            _clip("Two.", 90.0, 95.0, 1.25),
            _clip("Three.", 100.0, 101.0, 0.75),
        ]
        result = apply_timing_mode(clips, timing_mode="concatenate")
        assert result.cues[0].start == 0.0
        assert result.cues[0].end == pytest.approx(0.5)
        assert result.cues[1].start == pytest.approx(0.5)
        assert result.cues[1].end == pytest.approx(1.75)
        assert result.cues[2].start == pytest.approx(1.75)

    def test_the_timings_come_from_the_actual_audio(self) -> None:
        clips = [_clip("One.", 0.0, 9.0, 0.4), _clip("Two.", 20.0, 21.0, 0.6)]
        result = apply_timing_mode(clips, timing_mode="concatenate")
        for cue, clip in zip(result.cues, clips):
            assert cue.end - cue.start == pytest.approx(clip.natural_seconds)
        assert result.total_seconds == pytest.approx(1.0, abs=0.001)

    def test_the_audio_is_back_to_back_with_no_gap(self) -> None:
        clips = [
            _clip("One.", 0.0, 5.0, 0.5, 0.5),
            _clip("Two.", 9.0, 10.0, 0.5, 0.25),
        ]
        result = apply_timing_mode(clips, timing_mode="concatenate")
        assert result.waveform.shape[-1] == pytest.approx(1.0 * S3GEN_SAMPLE_RATE, abs=2)
        assert torch.allclose(result.waveform[:, : S3GEN_SAMPLE_RATE // 2], torch.full((1, S3GEN_SAMPLE_RATE // 2), 0.5))
        assert torch.allclose(result.waveform[:, S3GEN_SAMPLE_RATE // 2 :], torch.full((1, S3GEN_SAMPLE_RATE // 2), 0.25))

    def test_no_stretch_is_ever_applied(self) -> None:
        clips = [_clip("One.", 0.0, 0.1, 3.0)]
        result = apply_timing_mode(clips, timing_mode="concatenate")
        assert result.stretches == (1.0,)
        assert result.cues[0].end == pytest.approx(3.0)


class TestAllFourModesAgreeOnContent:
    """Whatever the mode, every cue's text survives into the output file."""

    @pytest.mark.parametrize("mode", list(TIMING_MODES))
    def test_the_output_srt_carries_every_cue(self, mode: str) -> None:
        clips = [
            _clip("First line.", 0.0, 1.0, 0.9),
            _clip("Second line.", 1.5, 2.5, 1.0),
            _clip("Third line.", 3.0, 4.5, 1.4),
        ]
        result = apply_timing_mode(clips, timing_mode=mode)
        parsed = parse_srt(render_srt(result.cues))
        assert [cue.text for cue in parsed] == ["First line.", "Second line.", "Third line."]
        assert [cue.index for cue in parsed] == [1, 2, 3]

    @pytest.mark.parametrize("mode", list(TIMING_MODES))
    def test_the_track_is_as_long_as_the_last_cue_ends(self, mode: str) -> None:
        clips = [_clip("One.", 0.0, 1.0, 0.5), _clip("Two.", 1.5, 2.5, 1.0)]
        result = apply_timing_mode(clips, timing_mode=mode)
        last_end = max(cue.end for cue in result.cues)
        assert result.total_seconds >= last_end - 1e-3

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown timing_mode"):
            apply_timing_mode([_clip("One.", 0.0, 1.0, 1.0)], timing_mode="nope")


class TestStretchHelper:
    def test_the_output_length_is_exactly_what_was_asked_for(self) -> None:
        assert stretch_to_duration(_tone(1.0), 4800).shape[-1] == 4800

    def test_a_fade_is_applied_at_the_head_and_the_tail(self) -> None:
        clipped = stretch_to_duration(_tone(1.0, 1.0), 12000, fade_fraction=0.01)
        assert clipped.shape[-1] == 12000
        assert clipped[0, 0] == pytest.approx(0.0)
        assert clipped[0, -1] == pytest.approx(0.0)
        assert clipped[0, 6000] == pytest.approx(1.0)

    def test_a_zero_fade_leaves_the_step(self) -> None:
        clipped = stretch_to_duration(_tone(1.0, 1.0), 12000, fade_fraction=0.0)
        assert clipped[0, 0] == pytest.approx(1.0)

    def test_a_one_sample_target_is_refused_rather_than_raised(self) -> None:
        assert stretch_to_duration(_tone(1.0), 1).shape[-1] == S3GEN_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_generation(monkeypatch):
    """Replace the loader and the synthesis with fixed-length clips.

    One clip of ``0.4 s`` per cue regardless of its text, so every timing
    assertion below is arithmetic rather than a property of a sampling run.
    """
    calls: list[dict] = []

    def _load(model_name, **kwargs):
        calls.append({"model": model_name, **kwargs})
        return type("P", (), {"cache_key": "srt-test-key"})(), object(), None

    def _generate(model, text, **kwargs):
        calls.append({"text": text, **kwargs})
        return _tone(0.4).clone(), S3GEN_SAMPLE_RATE

    monkeypatch.setattr(srt_module, "load_chatterbox_model", _load)
    monkeypatch.setattr(srt_module, "generate_audio", _generate)
    return calls


_DEMO = (
    "1\n00:00:00,000 --> 00:00:01,000\nOne.\n\n"
    "2\n00:00:02,000 --> 00:00:03,000\nTwo.\n\n"
    "3\n00:00:04,000 --> 00:00:05,000\nThree.\n"
)


def _execute(srt_content: str, timing_mode: str = "smart_natural", **kwargs):
    return ChatterboxSRTNode.execute(
        model_pack_name="pack",
        srt_content=srt_content,
        timing_mode=timing_mode,
        max_new_tokens=1000,
        flow_cfg_scale=0.7,
        exaggeration=0.5,
        temperature=0.8,
        seed=1,
        use_watermark=False,
        timing_tolerance=kwargs.pop("timing_tolerance", DEFAULT_TIMING_TOLERANCE),
        max_stretch_ratio=kwargs.pop("max_stretch_ratio", DEFAULT_MAX_STRETCH_RATIO),
        min_stretch_ratio=kwargs.pop("min_stretch_ratio", DEFAULT_MIN_STRETCH_RATIO),
        fade_for_StretchToFit=kwargs.pop(
            "fade_for_StretchToFit", DEFAULT_FADE_FOR_STRETCH_TO_FIT
        ),
        **kwargs,
    )


class TestExecute:
    def test_four_outputs_come_back(self, stub_generation) -> None:
        result = _execute(_DEMO)
        assert len(result.args) == 4
        audio, generation_info, timing_report, adjusted_srt = result.args
        assert audio["sample_rate"] == S3GEN_SAMPLE_RATE
        assert "pack: pack" in generation_info
        assert "timing_mode: smart_natural" in timing_report
        assert adjusted_srt.startswith("1\n")

    @pytest.mark.parametrize("mode", list(TIMING_MODES))
    def test_every_mode_produces_a_parseable_adjusted_srt(
        self, stub_generation, mode: str
    ) -> None:
        result = _execute(_DEMO, timing_mode=mode)
        cues = parse_srt(result.args[3])
        assert [cue.text for cue in cues] == ["One.", "Two.", "Three."]

    def test_each_cue_is_synthesised_separately(self, stub_generation) -> None:
        _execute(_DEMO)
        spoken = [call["text"] for call in stub_generation if "text" in call]
        assert len(spoken) == 3
        assert all("hmm" in text for text in spoken), (
            "crash protection is applied per cue: a subtitle cue is short by "
            "construction, which is the input upstream segfaults on"
        )

    def test_crash_protection_can_be_disabled(self, stub_generation) -> None:
        _execute(_DEMO, crash_protection_template="")
        spoken = [call["text"] for call in stub_generation if "text" in call]
        assert spoken == ["One.", "Two.", "Three."]

    def test_an_empty_box_returns_silence_without_loading_a_model(
        self, stub_generation
    ) -> None:
        result = _execute("")
        assert result.args[3] == ""
        assert [call for call in stub_generation if "model" in call] == []

    def test_srt_with_no_text_cue_is_refused(self, stub_generation) -> None:
        with pytest.raises(ValueError, match="none of them has any text"):
            _execute("1\n00:00:00,000 --> 00:00:01,000\n\n")

    def test_a_cancelled_run_returns_silence_not_a_failure(self, monkeypatch) -> None:
        """The interrupt subclasses ``BaseException``, so the clause must name it."""
        from ComfyUI_Chatterbox.modules.audio_utils import interrupt_exception_class

        interrupt_error = interrupt_exception_class()
        assert issubclass(interrupt_error, BaseException)
        assert not issubclass(interrupt_error, Exception)

        monkeypatch.setattr(
            srt_module,
            "load_chatterbox_model",
            lambda name, **kwargs: (type("P", (), {"cache_key": "k"})(), object(), None),
        )

        def _cancel(model, text, **kwargs):
            raise interrupt_error("cancelled")

        monkeypatch.setattr(srt_module, "generate_audio", _cancel)
        result = _execute(_DEMO)
        assert result.args[3] == ""
        assert torch.count_nonzero(result.args[0]["waveform"]) == 0


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


class TestModuleHygiene:
    def test_subtitles_does_not_import_the_node_api(self) -> None:
        """``modules/`` stays free of the ComfyUI node surface.

        ``modules/subtitles.py`` holds the SRT layout algorithm, which is pure
        arithmetic over waveforms. A node-API import there would make the
        algorithm importable only inside a running ComfyUI, and the tests above
        -- which is every test of that algorithm -- would stop running at all.
        """
        path = os.path.join(ROOT_DIR, "modules", "subtitles.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("comfy_api" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "comfy_api" not in (node.module or "")

    def test_the_node_talks_to_the_node_api_and_the_module_does_not(self) -> None:
        path = os.path.join(ROOT_DIR, "nodes", "srt_node.py")
        with open(path, encoding="utf-8") as handle:
            assert "from comfy_api.latest import" in handle.read()