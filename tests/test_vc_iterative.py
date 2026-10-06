"""S8.6 -- iterative and chunked voice conversion on ``ChatterboxVC``.

Three new widgets, each with a failure mode that does not raise:

* ``refinement_passes`` feeds each iteration's output back in. If the passes are
  not deterministic, the cache is a correctness bug rather than a speed-up --
  so determinism is asserted by running the same job twice and comparing
  samples, not by counting calls.
* The cache is what makes the widget usable at all: six passes after five are
  already computed must cost one conversion, and three passes again must cost
  nothing. Both are asserted by counting model calls, which is the only way to
  tell a cache hit from a fast stub.
* ``max_chunk_duration=0`` disables chunking. A default of 30 s silently applied
  to a 4-second clip would be indistinguishable from working, so the assertion
  is that a clip under the ceiling comes back as **one** piece.

``smart`` cutting lands in silence, ``fixed`` does not, and that difference is
the reason both modes exist -- so it is asserted on the audio rather than on the
cut positions.
"""

from __future__ import annotations

import pytest
import torch

from ComfyUI_Chatterbox.modules.audio_utils import S3GEN_SAMPLE_RATE
from ComfyUI_Chatterbox.modules.generation import (
    REFINEMENT_CACHE_DEPTH,
    RefinementCache,
    combine_audio_chunks,
    iterative_convert_voice,
    split_audio_for_conversion,
)
from ComfyUI_Chatterbox.nodes.vc_node import (
    CHUNK_METHOD_SMART,
    ChatterboxVCNode,
)


class _CountingVC:
    """A conversion stand-in that records what it was handed.

    Deterministic *and* order-dependent: each call returns a distinct constant,
    so the final result tells you how many passes ran, and the waveform read back
    out of the temporary file tells you what the previous pass produced. That is
    what makes "each pass is fed the previous output" an assertion rather than a
    claim.

    ``convert_voice_audio`` hands the vendored models a *file path*, so the input
    is read back from the ``.wav`` the helper wrote rather than inspected
    in-process.
    """

    def __init__(self, base: float = 0.5) -> None:
        self.base = base
        self.calls: list[dict] = []

    def generate(self, source, target_voice_path=None, **kwargs):
        import numpy as np
        import soundfile as sf

        samples, sample_rate = sf.read(source, dtype="float32")
        self.calls.append(
            {
                "path": source,
                "input_mean": float(np.mean(samples)) if samples.size else 0.0,
                **kwargs,
            }
        )
        value = self.base * len(self.calls)
        return torch.full((1, 240), value, dtype=torch.float32)


@pytest.fixture
def source_audio() -> dict:
    return {
        "waveform": torch.ones((1, 1, S3GEN_SAMPLE_RATE), dtype=torch.float32) * 0.25,
        "sample_rate": S3GEN_SAMPLE_RATE,
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_three_widgets_are_appended_after_the_frozen_block(self) -> None:
        from .conftest_helpers import load_frozen_snapshot, schema_signatures

        snapshot = load_frozen_snapshot()
        frozen = snapshot["nodes"]["ChatterboxVC"]["required"]
        actual = schema_signatures(ChatterboxVCNode, "required")

        assert tuple(entry[0] for entry in actual[: len(frozen)]) == tuple(
            entry["name"] for entry in frozen
        ), "the frozen block moved"
        assert tuple(entry[0] for entry in actual[len(frozen) :]) == (
            "device",
            "dtype",
            "attention_mode",
            "refinement_passes",
            "max_chunk_duration",
            "chunk_method",
        )

    def test_the_documented_ranges_and_defaults(self) -> None:
        required = ChatterboxVCNode.GET_NODE_INFO_V1()["input"]["required"]

        refinement = required["refinement_passes"]
        assert refinement[0] == "INT"
        assert (refinement[1]["min"], refinement[1]["max"], refinement[1]["default"]) == (
            1,
            30,
            1,
        )
        assert "5" in refinement[1]["tooltip"], "the tooltip states the recommended max"

        chunk = required["max_chunk_duration"]
        assert chunk[0] == "FLOAT"
        assert (chunk[1]["min"], chunk[1]["max"], chunk[1]["default"]) == (0.0, 300.0, 30.0)

        method = required["chunk_method"]
        assert method[0] == "COMBO"
        assert method[1]["options"] == ["smart", "fixed"]
        assert method[1]["default"] == CHUNK_METHOD_SMART == "smart"

    def test_the_widget_defaults_match_the_execute_signature(self) -> None:
        import inspect

        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(
                ChatterboxVCNode.execute
            ).parameters.items()
        }
        assert defaults["refinement_passes"] == 1
        assert defaults["max_chunk_duration"] == 0.0
        assert defaults["chunk_method"] == "smart"

    def test_the_cache_is_capped_at_five_iterations(self) -> None:
        assert REFINEMENT_CACHE_DEPTH == 5
        cache = RefinementCache()
        assert cache.depth == 5
        cache.put("job", [{"audio": i} for i in range(9)])
        assert len(cache.get("job")) == 5
        assert [entry["audio"] for entry in cache.get("job")] == [4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


class TestRefinementPasses:
    def test_one_pass_is_one_conversion(self, source_audio) -> None:
        model = _CountingVC()
        iterative_convert_voice(model, source_audio)
        assert len(model.calls) == 1

    @pytest.mark.parametrize("passes", [1, 2, 3, 5])
    def test_n_passes_are_n_conversions(self, source_audio, passes: int) -> None:
        model = _CountingVC()
        iterative_convert_voice(model, source_audio, refinement_passes=passes)
        assert len(model.calls) == passes

    def test_the_pass_number_is_clamped_up_to_one(self, source_audio) -> None:
        model = _CountingVC()
        iterative_convert_voice(model, source_audio, refinement_passes=0)
        assert len(model.calls) == 1

    def test_each_pass_is_fed_the_previous_output(self, source_audio) -> None:
        """That is what refinement *is*: a fold, not a loop over the source."""
        model = _CountingVC()
        iterative_convert_voice(model, source_audio, refinement_passes=3)

        # Pass 1 sees the source clip (mean 0.25); passes 2 and 3 see the
        # previous pass's output (0.5 then 1.0), not the source again.
        means = [call["input_mean"] for call in model.calls]
        assert means[0] == pytest.approx(0.25, abs=1e-4), means
        assert means[1] == pytest.approx(0.5, abs=1e-4), means
        assert means[2] == pytest.approx(1.0, abs=1e-4), means
        assert means[1:] != [means[0]] * 2, "the source was re-converted every pass"

    def test_n_passes_are_deterministic(self, source_audio) -> None:
        """The property the cache rests on.

        Run twice, from scratch, and compare samples. A cache that returned a
        different iteration than recomputing *would have been* a defect, and
        counting calls would not detect it.
        """
        first_model = _CountingVC()
        first, first_sr = iterative_convert_voice(
            first_model, source_audio, refinement_passes=4
        )
        second_model = _CountingVC()
        second, second_sr = iterative_convert_voice(
            second_model, source_audio, refinement_passes=4
        )
        assert torch.equal(first, second)
        assert first_sr == second_sr == S3GEN_SAMPLE_RATE
        assert len(first_model.calls) == len(second_model.calls) == 4

    def test_the_result_is_the_nth_pass(self, source_audio) -> None:
        """And the number of passes is visible in the result."""
        model = _CountingVC()
        waveform, _ = iterative_convert_voice(model, source_audio, refinement_passes=3)
        assert float(waveform[0, 0]) == pytest.approx(1.5), (
            "three passes of base 0.5 must end at 1.5"
        )

    def test_a_different_number_of_passes_gives_a_different_result(self, source_audio) -> None:
        """Otherwise "n passes" would be a knob with no effect."""
        one, _ = iterative_convert_voice(_CountingVC(), source_audio)
        three, _ = iterative_convert_voice(_CountingVC(), source_audio, refinement_passes=3)
        assert not torch.equal(one, three)


class TestRefinementCache:
    def test_a_repeat_request_returns_the_cached_iteration(self, source_audio) -> None:
        cache = RefinementCache()
        model = _CountingVC()

        iterative_convert_voice(
            model,
            source_audio,
            refinement_passes=3,
            cache=cache,
            cache_key="job",
        )
        assert len(model.calls) == 3

        first, _ = iterative_convert_voice(
            model, source_audio, refinement_passes=3, cache=cache, cache_key="job"
        )
        assert len(model.calls) == 3, "a cached pass still converted"

        second, _ = iterative_convert_voice(
            model, source_audio, refinement_passes=3, cache=cache, cache_key="job"
        )
        assert len(model.calls) == 3
        assert torch.equal(first, second)

    def test_asking_for_more_costs_only_the_difference(self, source_audio) -> None:
        cache = RefinementCache()
        model = _CountingVC()
        iterative_convert_voice(
            model, source_audio, refinement_passes=3, cache=cache, cache_key="job"
        )
        iterative_convert_voice(
            model, source_audio, refinement_passes=5, cache=cache, cache_key="job"
        )
        assert len(model.calls) == 5, "two more passes should cost two conversions"

    def test_different_keys_do_not_share_iterations(self, source_audio) -> None:
        cache = RefinementCache()
        model = _CountingVC()
        iterative_convert_voice(
            model, source_audio, refinement_passes=2, cache=cache, cache_key="a"
        )
        iterative_convert_voice(
            model, source_audio, refinement_passes=2, cache=cache, cache_key="b"
        )
        assert len(model.calls) == 4

    def test_different_settings_do_not_share_iterations(self, source_audio) -> None:
        cache = RefinementCache()
        model = _CountingVC()
        iterative_convert_voice(
            model, source_audio, n_timesteps=10, cache=cache, cache_key="job"
        )
        iterative_convert_voice(
            model, source_audio, n_timesteps=20, cache=cache, cache_key="job"
        )
        assert len(model.calls) == 2

    def test_the_cache_is_capped(self, source_audio) -> None:
        """Nine passes retain five iterations -- the documented cap, not memory."""
        cache = RefinementCache()
        model = _CountingVC()
        iterative_convert_voice(
            model, source_audio, refinement_passes=9, cache=cache, cache_key="job"
        )
        assert len(model.calls) == 9
        assert len(cache) == 1
        assert list(cache.iteration_counts().values()) == [5]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _speech_like(seconds: float, period: float = 0.02) -> torch.Tensor:
    """A clip of tone bursts separated by digital silence.

    Silence is what ``smart`` is looking for, so the fixture has to actually
    contain some -- and the silence has to sit *near* the ideal cut rather than
    exactly on it, or the test would pass on any implementation.
    """
    sample_rate = S3GEN_SAMPLE_RATE
    total = int(seconds * sample_rate)
    samples = torch.zeros(total, dtype=torch.float32)
    burst = int(period * sample_rate)
    for start in range(0, total, int(0.1 * sample_rate)):
        end = min(start + burst, total)
        samples[start:end] = 0.5
    return samples


class TestChunking:
    def test_zero_disables_chunking(self) -> None:
        clip = _speech_like(60.0)
        chunks = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 0)
        assert len(chunks) == 1
        assert chunks[0].numel() == clip.numel()

    def test_a_clip_under_the_ceiling_is_not_split(self) -> None:
        clip = _speech_like(4.0)
        chunks = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30)
        assert len(chunks) == 1

    def test_a_long_clip_is_split_into_bounded_pieces(self) -> None:
        clip = _speech_like(120.0)
        limit = int(30.0 * S3GEN_SAMPLE_RATE)
        for method in ("smart", "fixed"):
            chunks = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, method)
            # ``fixed`` lands exactly four 30-second chunks. ``smart`` may cut
            # *earlier* than the ideal boundary to land in a pause, which leaves a
            # longer remainder, so its count is bounded rather than exact.
            assert len(chunks) >= 4, method
            assert sum(c.numel() for c in chunks) == clip.numel(), (
                "chunking must not drop or duplicate samples"
            )

        assert len(split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "fixed")) == 4
        assert all(
            chunk.numel() <= limit * 1.3
            for chunk in split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "smart")
        ), "a smart cut looks 25% either side of the ideal boundary, not further"

    def test_smart_cuts_land_in_silence_and_fixed_does_not(self) -> None:
        """The reason both modes exist."""
        clip = _speech_like(60.0)
        window = int(0.01 * S3GEN_SAMPLE_RATE)

        smart = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "smart")
        fixed = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "fixed")

        def peak_at(chunks, edge: int) -> float:
            position = 0
            for index, chunk in enumerate(chunks):
                if index == edge:
                    return float(chunk[:window].abs().max())
                position += chunk.numel()
            return 1.0

        smart_edges = [peak_at(smart, i) for i in range(1, len(smart))]
        fixed_edges = [peak_at(fixed, i) for i in range(1, len(fixed))]

        assert max(smart_edges) < 0.1, (
            f"a smart cut landed on signal at amplitude {max(smart_edges)}"
        )
        assert max(fixed_edges) > 0.1, (
            "fixed cuts were expected to land on signal; the fixture no longer "
            "has enough speech for the comparison to mean anything"
        )

    def test_fixed_cuts_are_at_exact_intervals(self) -> None:
        clip = _speech_like(120.0)
        chunks = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "fixed")
        exact = int(30.0 * S3GEN_SAMPLE_RATE)
        assert len(chunks) == 4
        assert all(chunk.numel() == exact for chunk in chunks)

    def test_a_clip_with_no_silence_still_chunks(self) -> None:
        """``smart`` must not fail on continuous speech."""
        clip = torch.full((S3GEN_SAMPLE_RATE * 60,), 0.5, dtype=torch.float32)
        chunks = split_audio_for_conversion(clip, S3GEN_SAMPLE_RATE, 30, "smart")
        assert len(chunks) >= 2
        assert sum(c.numel() for c in chunks) == clip.numel()

    def test_an_empty_clip_is_refused_by_the_conversion_not_the_splitter(self, source_audio) -> None:
        model = _CountingVC()
        with pytest.raises(ValueError):
            iterative_convert_voice(model, {"waveform": torch.zeros(0), "sample_rate": 24000})


class TestChunkJoining:
    def _pieces(self):
        return [
            torch.full((1, 100), 0.5),
            torch.full((1, 100), -0.5),
            torch.full((1, 100), 0.25),
        ]

    def test_concatenate_is_a_hard_join(self) -> None:
        joined = combine_audio_chunks(self._pieces(), S3GEN_SAMPLE_RATE, "concatenate")
        assert joined.shape[-1] == 300
        assert float(joined[0, 100]) == -0.5

    def test_silence_padding_inserts_the_configured_gap(self) -> None:
        joined = combine_audio_chunks(
            self._pieces(), S3GEN_SAMPLE_RATE, "silence_padding", 100
        )
        gap = int(S3GEN_SAMPLE_RATE * 100 / 1000)
        assert joined.shape[-1] == 300 + 2 * gap
        assert float(joined[0, 150]) == 0.0

    def test_zero_padding_is_legal(self) -> None:
        joined = combine_audio_chunks(
            self._pieces(), S3GEN_SAMPLE_RATE, "silence_padding", 0
        )
        assert joined.shape[-1] == 300

    def test_crossfade_overlaps_and_stays_finite(self) -> None:
        joined = combine_audio_chunks(self._pieces(), S3GEN_SAMPLE_RATE, "crossfade", 10)
        assert joined.shape[-1] < 300, "a crossfade is shorter than a concatenation"
        assert torch.isfinite(joined).all()

    def test_one_piece_comes_back_untouched(self) -> None:
        only = torch.ones((1, 100))
        assert combine_audio_chunks([only], S3GEN_SAMPLE_RATE, "crossfade") is only

    def test_no_pieces_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            combine_audio_chunks([], S3GEN_SAMPLE_RATE, "concatenate")
