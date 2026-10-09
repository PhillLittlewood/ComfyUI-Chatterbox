"""S5.8 -- ``modules/audio_utils.py``.

The conversion helpers are the seam between ComfyUI's ``AUDIO`` socket and
Chatterbox's tensor-based entry points, and every mistake on that seam is silent:
a channel that should have been mixed down sounds phasey, a waveform in the
wrong rank throws inside the vendor, and a resample to the wrong sample rate
stretches the speech without any error at all.

The regression this module exists for is :func:`preprocess_comfy_audio`. V1
wrote every voice prompt to a temporary ``.wav`` and handed the vendor a path:
the clip was quantised to 16-bit PCM on the way out, and a run that died
between the write and the cleanup leaked a file carrying the user's voice into
the temp directory -- twice, in V1's own history. The vendor's loader now takes
an in-memory waveform (``load_audio_at``, ``src/chatterbox/models/dsp.py``), so
the prompt crosses this seam as a mono ``float32`` array and there is no longer
a file to write, to remove, or to leak. :class:`TestModuleDiscipline` pins that
the file-writing machinery is gone for good.

Nothing here touches the network or loads a model.
"""

from __future__ import annotations

import importlib
import logging
import math
import random

import numpy as np
import pytest
import torch

from modules import audio_utils
from modules.audio_utils import (
    S3GEN_SAMPLE_RATE,
    check_for_interrupt,
    comfy_audio_to_numpy,
    downmix_to_mono,
    extract_waveform,
    interrupt_exception_class,
    numpy_to_comfy_audio,
    preprocess_comfy_audio,
    resample_waveform,
    set_seed,
    to_comfy_audio,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def make_audio(channels: int = 1, samples: int = 480, sample_rate: int = 24000):
    """A ``[1, C, T]`` AUDIO dict with a deterministic non-square-wave signal."""
    generator = torch.Generator().manual_seed(1234)
    waveform = torch.rand((1, channels, samples), generator=generator)
    return {"waveform": waveform, "sample_rate": sample_rate}


class FakeModelManagement:
    """Stands in for ``comfy.model_management`` with a scripted interrupt."""

    #: Mirrors core's: a ``BaseException``, deliberately not an ``Exception``.
    class InterruptProcessingException(BaseException):
        pass

    def __init__(self, interrupt: bool = False) -> None:
        self._interrupt = interrupt
        self.checks = 0

    def throw_exception_if_processing_interrupted(self) -> None:
        self.checks += 1
        if self._interrupt:
            raise self.InterruptProcessingException("cancelled")


@pytest.fixture
def fake_mm(monkeypatch):
    """Install a scripted ``comfy.model_management`` into the module."""

    def install(interrupt: bool) -> FakeModelManagement:
        instance = FakeModelManagement(interrupt=interrupt)
        monkeypatch.setattr(audio_utils, "_get_model_management", lambda: instance)
        return instance

    install(False)
    return install


# ---------------------------------------------------------------------------
# Rank and channel handling
# ---------------------------------------------------------------------------


class TestWaveformPlumbing:
    def test_extract_strips_the_batch_dimension(self) -> None:
        waveform = extract_waveform(make_audio(channels=2))
        assert waveform.shape == (2, 480)

    def test_extract_promotes_a_bare_1d_waveform(self) -> None:
        waveform = extract_waveform({"waveform": torch.zeros(10), "sample_rate": 16000})
        assert waveform.shape == (1, 10)

    def test_extract_accepts_a_bare_tensor(self) -> None:
        assert extract_waveform(torch.zeros(1, 2, 8)).shape == (2, 8)

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"waveform": torch.zeros(10)},
            {"sample_rate": 24000},
            {"waveform": None, "sample_rate": 24000},
            {"waveform": torch.zeros(0, 1, 10), "sample_rate": 24000},
            "not audio at all",
        ],
    )
    def test_extract_rejects_malformed_input(self, bad) -> None:
        with pytest.raises(ValueError):
            extract_waveform(bad, name="voice prompt")

    def test_extract_rejects_an_unusable_rank(self) -> None:
        with pytest.raises(ValueError, match=r"expected \[B, C, T\]"):
            extract_waveform({"waveform": torch.zeros(2, 3, 4, 5), "sample_rate": 1})

    def test_downmix_means_the_channels(self) -> None:
        waveform = torch.stack(
            [torch.ones(4), torch.zeros(4), -torch.ones(4)], dim=0
        )  # [3, 4]
        mixed = downmix_to_mono(waveform)
        torch.testing.assert_close(mixed, torch.zeros(4))

    def test_downmix_on_a_batched_waveform(self) -> None:
        waveform = torch.tensor([[[1.0, 3.0], [3.0, 1.0]]])  # [1, 2, 2]
        mixed = downmix_to_mono(waveform)
        assert mixed.shape == (1, 2)
        torch.testing.assert_close(mixed, torch.tensor([[2.0, 2.0]]))

    def test_downmix_on_a_single_channel_is_a_squeeze(self) -> None:
        waveform = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        torch.testing.assert_close(downmix_to_mono(waveform), waveform[0])

    def test_downmix_passes_1d_through(self) -> None:
        waveform = torch.arange(3, dtype=torch.float32)
        torch.testing.assert_close(downmix_to_mono(waveform), waveform)

    def test_downmix_rejects_a_bad_rank(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            downmix_to_mono(torch.zeros(2, 2, 2, 2))


# ---------------------------------------------------------------------------
# AUDIO round trips
# ---------------------------------------------------------------------------


class TestAudioRoundTrip:
    def test_to_comfy_audio_normalises_every_rank(self) -> None:
        for shape in ((480,), (2, 480), (1, 2, 480)):
            result = to_comfy_audio(torch.zeros(shape), 24000)
            assert result["waveform"].shape == (1, 1, 480) if len(shape) == 1 else True
            assert result["waveform"].dim() == 3
            assert result["sample_rate"] == 24000
            assert result["waveform"].device.type == "cpu"
            assert result["waveform"].dtype is torch.float32

    def test_to_comfy_audio_rejects_a_bad_rank(self) -> None:
        with pytest.raises(ValueError, match="AUDIO dict"):
            to_comfy_audio(torch.zeros(2, 2, 2, 2), 24000)

    def test_tensor_round_trip_is_bit_exact(self) -> None:
        original = torch.rand((1, 2, 256), generator=torch.Generator().manual_seed(7))
        audio = to_comfy_audio(original, 24000)
        restored = extract_waveform(audio)
        torch.testing.assert_close(restored, original[0])

    def test_mono_numpy_round_trip(self) -> None:
        audio = make_audio(channels=1, samples=256)
        mono = comfy_audio_to_numpy(audio)

        assert mono.ndim == 1
        assert mono.shape == (256,)
        assert mono.dtype == np.float32

        rebuilt = numpy_to_comfy_audio(mono, audio["sample_rate"])
        assert rebuilt["waveform"].shape == (1, 1, 256)
        np.testing.assert_allclose(
            comfy_audio_to_numpy(rebuilt), mono, rtol=0, atol=0
        )

    def test_multichannel_numpy_round_trip_downmixes(self) -> None:
        audio = make_audio(channels=2, samples=64)
        mono = comfy_audio_to_numpy(audio)

        expected = audio["waveform"][0].mean(dim=0).numpy()
        np.testing.assert_allclose(mono, expected, rtol=1e-6, atol=1e-6)

        rebuilt = numpy_to_comfy_audio(mono, 24000)
        np.testing.assert_allclose(
            comfy_audio_to_numpy(rebuilt), mono, rtol=0, atol=0
        )

    def test_a_four_dimensional_tensor_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            comfy_audio_to_numpy(torch.zeros(1, 1, 1, 5))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_none_and_empty_are_none(self) -> None:
        assert preprocess_comfy_audio(None) is None
        assert preprocess_comfy_audio({"waveform": None, "sample_rate": 1}) is None
        assert preprocess_comfy_audio(
            {"waveform": torch.zeros(0), "sample_rate": 24000}
        ) is None

    def test_same_rate_passes_through(self) -> None:
        audio = make_audio(samples=128, sample_rate=S3GEN_SAMPLE_RATE)
        result = preprocess_comfy_audio(audio)
        assert result.shape == (128,)
        np.testing.assert_allclose(result, audio["waveform"][0, 0].numpy(), atol=1e-7)

    def test_resampling_changes_the_length_proportionally(self) -> None:
        audio = make_audio(samples=16000, sample_rate=16000)
        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        # 1 second at 16 kHz becomes 1 second at 24 kHz (within one sample).
        assert abs(result.shape[0] - 24000) <= 1
        assert result.dtype == np.float32

    def test_resampling_a_real_signal_preserves_its_amplitude(self) -> None:
        rate = 16000
        time = torch.arange(rate, dtype=torch.float32) / rate
        tone = (0.5 * torch.sin(2 * math.pi * 440.0 * time)).unsqueeze(0).unsqueeze(0)
        audio = {"waveform": tone, "sample_rate": rate}

        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert float(np.abs(result).max()) == pytest.approx(0.5, abs=0.05)

    def test_non_finite_samples_are_scrubbed_before_resampling(self) -> None:
        waveform = torch.rand((1, 1, 1600), generator=torch.Generator().manual_seed(3))
        waveform[0, 0, ::7] = float("nan")
        waveform[0, 0, ::11] = float("inf")
        audio = {"waveform": waveform, "sample_rate": 16000}

        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert np.all(np.isfinite(result))
        assert float(np.abs(result).max()) > 0.0, (
            "scrubbing replaced the whole signal; only the bad samples should "
            "have been zeroed"
        )

    def test_huge_peaks_are_scaled_down(self) -> None:
        waveform = (torch.rand((1, 1, 64)) + 1.0) * 50.0
        audio = {"waveform": waveform, "sample_rate": 24000}
        result = preprocess_comfy_audio(audio)
        assert result is not None
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-5)

    def test_a_missing_sample_rate_is_an_error_not_a_guess(self) -> None:
        audio = {"waveform": torch.zeros((1, 1, 64)), "sample_rate": 0}
        with pytest.raises(ValueError, match="sample_rate"):
            preprocess_comfy_audio(audio, target_sr=24000)

    def test_resample_rejects_a_bad_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            resample_waveform(torch.zeros(4), 0, 24000)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_the_same_seed_reproduces_the_same_draws(self) -> None:
        set_seed(1234)
        first = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())

        set_seed(1234)
        second = (torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random())

        assert first == second

    def test_a_different_seed_does_not(self) -> None:
        set_seed(1)
        first = torch.rand(4).tolist()
        set_seed(2)
        assert first != torch.rand(4).tolist()

    def test_seed_zero_picks_a_real_random_seed(self) -> None:
        set_seed(0)
        first = torch.rand(4).tolist()
        set_seed(0)
        assert first != torch.rand(4).tolist()

    def test_a_seed_above_numpy_s_range_is_clamped_not_rejected(self) -> None:
        # 0xffffffffffffffff is the node widget's maximum; numpy's legacy seeder
        # only accepts 32 bits, and this must not raise.
        set_seed(0xFFFFFFFFFFFFFFFF)
        assert np.random.rand() >= 0.0


    def test_the_re_export_shim_was_removed(self) -> None:
        """This module holds the one and only ``set_seed``.

        ``modules/utils.py`` used to re-export it under a shorter name for
        the V1 nodes. That shim is gone, so anything importing it has to
        come here -- which is what stops the two spellings drifting.
        """
        with pytest.raises(ImportError):
            importlib.import_module("modules.utils")


# ---------------------------------------------------------------------------
# Interruption
# ---------------------------------------------------------------------------


class TestCheckForInterrupt:
    def test_false_when_not_interrupted(self, fake_mm) -> None:
        instance = fake_mm(interrupt=False)
        assert check_for_interrupt() is False
        assert instance.checks == 1

    def test_true_when_interrupted(self, fake_mm) -> None:
        fake_mm(interrupt=True)
        assert check_for_interrupt() is True

    def test_it_returns_the_core_class_when_available(self) -> None:
        import comfy.model_management as model_management

        expected = model_management.InterruptProcessingException
        assert interrupt_exception_class() is expected

    def test_a_keyboard_interrupt_is_not_reported_as_a_cancel(self, fake_mm) -> None:
        """Only the interrupt class is swallowed.

        ``except BaseException`` here would report a Ctrl-C in the console as
        "the user pressed stop" and let the run continue.
        """
        instance = fake_mm(interrupt=False)

        def raise_keyboard():
            instance.checks += 1
            raise KeyboardInterrupt()

        instance.throw_exception_if_processing_interrupted = raise_keyboard

        with pytest.raises(KeyboardInterrupt):
            check_for_interrupt()


# ---------------------------------------------------------------------------
# Module discipline
# ---------------------------------------------------------------------------


class TestModuleDiscipline:
    @staticmethod
    def _source(module) -> str:
        with open(module.__file__, "r", encoding="utf-8") as handle:
            return handle.read()

    def test_it_imports_no_comfy_api(self) -> None:
        source = self._source(audio_utils)
        assert "comfy_api" not in source
        assert "import folder_paths" not in source

    def test_comfy_is_reached_through_the_shared_lazy_helper(self) -> None:
        from modules import device_utils

        assert audio_utils._get_model_management is device_utils._get_model_management
        assert "import comfy.model_management" not in self._source(audio_utils)

    def test_no_logger_objects_or_handlers(self) -> None:
        source = self._source(audio_utils)
        for forbidden in (
            "getLogger(",
            "logging.Formatter(",
            "LoggerAdapter(",
            ".addHandler(",
            ".setLevel(",
        ):
            assert forbidden not in source, f"audio_utils.py must not use {forbidden}"

    def test_no_temp_file_machinery_remains(self) -> None:
        """The in-memory handoff, pinned so it cannot quietly come back.

        The deleted :class:`TestTempWavPath` proved the temp file was *removed*
        -- on the normal path and on the error path, because V1 got both wrong.
        The stronger guarantee is that there is no file at all: the vendor's
        loader takes a waveform, so nothing here may import or reference the
        machinery that used to write the clip out as a 16-bit PCM ``.wav``.
        """
        source = self._source(audio_utils)
        for forbidden in (
            "temp_wav_path",
            "tempfile",
            "soundfile",
            "sf.write",
            "os.remove",
        ):
            assert forbidden not in source, (
                f"audio_utils.py references {forbidden}; the voice prompt must "
                "cross to the vendor as an in-memory array, not a file"
            )

    def test_scrubbing_is_logged_through_the_root_logger(self, caplog) -> None:
        waveform = torch.zeros((1, 1, 64))
        waveform[0, 0, 0] = float("nan")
        with caplog.at_level(logging.ERROR, logger=""):
            preprocess_comfy_audio(
                {"waveform": waveform, "sample_rate": 24000}, target_sr=24000
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any("NaN" in message for message in messages)