"""Unit tests for the vendored ``models/dsp.py`` DSP replicas.

``conftest.py`` registers a ``MagicMock`` for every dotted name under
``src/chatterbox`` -- ``chatterbox.models.dsp`` included -- so the module under
test is loaded straight from disk under a private name, the same mechanism the
one-off librosa-vs-replica probe uses. No test in the shipped path imports
librosa or scipy: bit-exactness against librosa 1.0 was proven once by that
probe, and what these tests pin is what must not silently drift afterwards --
shapes, dtypes, and the padding/trim branch structure -- using only numpy,
torch and soundfile.

The parameters are the vendored call sites' own: the three baked mel
filterbanks (s3tokenizer, s3gen mel, voice encoder), the voice encoder's STFT
(n_fft=400, hop=160, win=400, center=True, reflect) and trim (top_db=20), and
the periodic Hann behind the scipy ``get_window`` mirror. The signals come
from the probe's seed and draw order, so the inputs pinned here are the inputs
that were proven bit-exact there.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest
import torch

from conftest import ROOT_DIR

DSP_PATH = os.path.join(ROOT_DIR, "src", "chatterbox", "models", "dsp.py")

#: The voice encoder's STFT geometry (``VoiceEncConfig`` in
#: ``models/voice_encoder/config.py``); shared by every stft test below.
N_FFT = 400
HOP_LENGTH = 160

#: The one-off probe's seed. Drawing in its order keeps every signal these
#: tests use byte-identical to the signal the librosa comparison used.
_PROBE_SEED = 20261009


def _load_dsp_module():
    """Execute ``models/dsp.py`` under a private top-level name.

    The conftest mock covers ``chatterbox.models.dsp``, so a package import
    would hand back a MagicMock; loading the file directly is the only way to
    exercise the real thing without waking the model stack.
    """
    spec = importlib.util.spec_from_file_location("dsp_under_test", DSP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dsp = _load_dsp_module()

#: The three filterbanks the pretrained featurization bakes in, as
#: ``(label, sr, n_fft, n_mels, fmin, fmax)`` -- one per vendored call site.
FILTERBANKS = (
    ("s3tokenizer", 16000, 400, 128, 0.0, None),
    ("s3gen", 24000, 1920, 80, 0.0, 8000),
    ("voice_encoder", 16000, 400, 40, 0.0, 8000),
)

#: Input lengths that hit every centred-padding branch: shorter than
#: ``n_fft`` (warns), shorter than ``n_fft + 2*hop`` (head/tail overlap ->
#: copy-pad), and longer (pre/post frame split).
STFT_LENGTHS = (300, 400, 450, 521, 1000, 1600, 2423, 24000, 24001, 100000)

_probe_rng = np.random.default_rng(_PROBE_SEED)
_STFT_INPUTS = {
    length: _probe_rng.standard_normal(length).astype(np.float32)
    for length in STFT_LENGTHS
}
_NOISE = _probe_rng.standard_normal(24000).astype(np.float32)


def _speech_like(length: int, sr: int) -> np.ndarray:
    """Three bursts of a 220 Hz tone at 0.3 amplitude, silence between.

    The probe's speech generator: RMS-rich bursts separated by long silences,
    which is what puts the trim edges to work.
    """
    t = np.arange(length) / sr
    y = np.zeros(length, dtype=np.float64)
    for lo, hi in [(0.05, 0.3), (0.4, 0.7), (0.8, 0.95)]:
        mask = (t >= lo) & (t < hi)
        y[mask] = 0.3 * np.sin(2 * np.pi * 220 * t[mask])
    return y.astype(np.float32)


def _naive_centered_stft(y: np.ndarray) -> np.ndarray:
    """librosa's centred STFT as plain pad-frame-window-rfft.

    ``dsp.stft`` splits long signals into pre/post frame groups and copy-pads
    short ones; both are optimizations that must reproduce this direct
    definition. The window comes from the module under test because it is
    pinned separately (:class:`TestHannWindow`) -- what this reference checks
    is the padding and frame layout, bit for bit.
    """
    pad = N_FFT // 2
    padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    frames = np.stack(
        [padded[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] for i in range(n_frames)],
        axis=-1,
    )
    window = dsp.hann_window(N_FFT)[:, np.newaxis]
    return np.fft.rfft(window * frames, axis=-2).astype(np.complex64)


class TestMelFilterbank:
    """The Slaney triangles the three featurizers bake in at import time."""

    @pytest.mark.parametrize(
        ("label", "sr", "n_fft", "n_mels", "fmin", "fmax"),
        FILTERBANKS,
        ids=[case[0] for case in FILTERBANKS],
    )
    def test_shape_dtype_and_triangle_structure(
        self,
        label: str,
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float,
        fmax: float | None,
    ) -> None:
        weights = dsp.mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
        # librosa's default output: float32, one row per mel band.
        assert weights.shape == (n_mels, 1 + n_fft // 2)
        assert weights.dtype == np.float32
        assert (weights >= 0).all(), f"{label}: negative weight"
        assert (weights > 0).any(axis=1).all(), f"{label}: an all-zero mel band"
        for row in weights:
            support = np.flatnonzero(row)
            # A Slaney triangle is one contiguous run of bins, never scatter.
            assert support.size > 0, f"{label}: an all-zero mel band"
            assert support[-1] - support[0] + 1 == support.size, (
                f"{label}: non-contiguous band support"
            )

    def test_repeat_call_is_identical(self) -> None:
        first = dsp.mel_filterbank(16000, 400, 128)
        second = dsp.mel_filterbank(16000, 400, 128)
        assert np.array_equal(first, second)


class TestHannWindow:
    """The periodic Hann: ``scipy.signal.get_window('hann', M, fftbins=True)``."""

    def test_matches_the_periodic_formula(self) -> None:
        window = dsp.hann_window(400)
        reference = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
        assert window.shape == (400,)
        assert window.dtype == np.float64
        assert window[0] == 0.0  # the dropped M+1th sample would be 0 too
        assert np.allclose(window, reference, atol=1e-12)

    def test_matches_torchs_periodic_hann(self) -> None:
        # ``torch.hann_window`` is periodic by default -- the very window the
        # s3tokenizer and s3gen paths build themselves -- so this is the
        # in-stack cross-check that needs no scipy.
        reference = torch.hann_window(400, periodic=True, dtype=torch.float64)
        assert np.allclose(dsp.hann_window(400), reference.numpy(), atol=1e-12)


class TestStft:
    """The voice encoder's STFT geometry across every padding branch."""

    @pytest.mark.parametrize("length", STFT_LENGTHS)
    def test_shape_dtype_and_warning_branch(self, length: int) -> None:
        y = _STFT_INPUTS[length]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spectrum = dsp.stft(
                y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT
            )
        # librosa's centred frame count, for every padding branch.
        assert spectrum.shape == (1 + N_FFT // 2, 1 + length // HOP_LENGTH)
        assert spectrum.dtype == np.complex64  # float32 in -> complex64 out
        # n_fft longer than the signal is the only warning librosa raises here.
        assert bool(caught) == (length < N_FFT)

    @pytest.mark.parametrize("length", STFT_LENGTHS)
    def test_matches_the_direct_definition(self, length: int) -> None:
        y = _STFT_INPUTS[length]
        assert np.array_equal(
            dsp.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT),
            _naive_centered_stft(y),
        )

    def test_uncentered_shape(self) -> None:
        y = _STFT_INPUTS[24000]
        spectrum = dsp.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False)
        assert spectrum.shape == (
            1 + N_FFT // 2,
            1 + (24000 - N_FFT) // HOP_LENGTH,
        )
        assert spectrum.dtype == np.complex64

    def test_uncentered_rejects_signal_shorter_than_n_fft(self) -> None:
        with pytest.raises(ValueError):
            dsp.stft(
                _STFT_INPUTS[300], n_fft=N_FFT, hop_length=HOP_LENGTH, center=False
            )

    def test_float64_input_promotes_to_complex128(self) -> None:
        y = _STFT_INPUTS[1000].astype(np.float64)
        spectrum = dsp.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        assert spectrum.dtype == np.complex128


class TestTrimSilence:
    """Silence trimming at the voice encoder's ``top_db=20``."""

    TOP_DB = 20.0

    def test_noise_survives_unchanged(self) -> None:
        trimmed = dsp.trim_silence(_NOISE, top_db=self.TOP_DB)
        assert np.array_equal(trimmed, _NOISE)

    def test_short_clip_survives_unchanged(self) -> None:
        # L=2423: the first and last frames already reach the tone, so no
        # frame qualifies as silent and the clip passes through whole.
        y = _speech_like(2423, 16000)
        assert np.array_equal(dsp.trim_silence(y, top_db=self.TOP_DB), y)

    def test_edge_silence_shrinks_the_clip(self) -> None:
        y = _speech_like(24000, 16000)
        trimmed = dsp.trim_silence(y, top_db=self.TOP_DB)
        assert trimmed.size < y.size
        assert not np.array_equal(trimmed, y[y.size - trimmed.size :])

    def test_leading_silence_is_trimmed_at_the_head(self) -> None:
        y = np.concatenate(
            [np.zeros(5000, np.float32), _speech_like(19000, 16000)]
        )
        trimmed = dsp.trim_silence(y, top_db=self.TOP_DB)
        assert trimmed.size < y.size
        assert not np.array_equal(trimmed, y[: trimmed.size])

    def test_trailing_silence_is_trimmed_at_the_tail(self) -> None:
        y = np.concatenate(
            [_speech_like(19000, 16000), np.zeros(5000, np.float32)]
        )
        trimmed = dsp.trim_silence(y, top_db=self.TOP_DB)
        assert trimmed.size < y.size
        assert np.array_equal(trimmed, y[: trimmed.size])
        assert not np.array_equal(trimmed, y[y.size - trimmed.size :])

    def test_all_silent_clip_is_returned_untrimmed(self) -> None:
        # librosa.effects.trim behaves the same way: with no non-silent frame
        # there is no interval to return, so the input passes through.
        y = np.zeros(24000, np.float32)
        trimmed = dsp.trim_silence(y, top_db=self.TOP_DB)
        assert np.array_equal(trimmed, y)

    def test_dtype_is_preserved(self) -> None:
        y = _speech_like(24000, 16000)
        assert dsp.trim_silence(y, top_db=self.TOP_DB).dtype == y.dtype


class TestResample:
    """The same-rate no-op; the sinc path is measured by the probe, not here."""

    def test_same_rate_is_a_no_op(self) -> None:
        y = _NOISE[:1024]
        assert np.array_equal(dsp.resample(y, 16000.0, 16000.0), y)


class TestLoadAudioAt:
    """Reference-clip loading: path, in-memory waveform, cuda tensor."""

    def test_path_reads_float32_mono(self, tmp_path) -> None:
        import soundfile as sf

        y = _NOISE[:1600]
        path = os.path.join(str(tmp_path), "ref.wav")
        sf.write(path, y, 16000, subtype="FLOAT")
        loaded = dsp.load_audio_at(path, 16000)
        assert loaded.dtype == np.float32
        assert loaded.shape == (1600,)
        assert np.array_equal(loaded, y)

    def test_path_downmixes_stereo_by_channel_mean(self, tmp_path) -> None:
        import soundfile as sf

        rng = np.random.default_rng(11)
        stereo = rng.standard_normal((800, 2)).astype(np.float32)
        path = os.path.join(str(tmp_path), "stereo.wav")
        sf.write(path, stereo, 16000, subtype="FLOAT")
        loaded = dsp.load_audio_at(path, 16000)
        assert loaded.shape == (800,)
        assert np.array_equal(loaded, stereo.mean(axis=1))

    def test_ndarray_becomes_float32_and_keeps_its_values(self) -> None:
        y = _NOISE[:500].astype(np.float64)
        loaded = dsp.load_audio_at(y, 24000)
        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == np.float32
        assert np.array_equal(loaded, y.astype(np.float32))

    def test_target_sr_is_ignored_for_in_memory_audio(self) -> None:
        # The node hands over audio already resampled to target_sr; only the
        # on-disk path resamples.
        y = _NOISE[:500]
        assert np.array_equal(dsp.load_audio_at(y, 24000), y)

    def test_cpu_tensor_becomes_float32_ndarray(self) -> None:
        tensor = torch.from_numpy(_NOISE[:100])
        loaded = dsp.load_audio_at(tensor, 16000)
        assert isinstance(loaded, np.ndarray)
        assert not torch.is_tensor(loaded)
        assert loaded.dtype == np.float32
        assert np.array_equal(loaded, tensor.numpy())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_cuda_tensor_is_moved_to_cpu(self) -> None:
        tensor = torch.from_numpy(_NOISE[:100]).cuda()
        loaded = dsp.load_audio_at(tensor, 16000)
        assert isinstance(loaded, np.ndarray)
        assert not torch.is_tensor(loaded)
        assert loaded.dtype == np.float32
        assert np.array_equal(loaded, tensor.cpu().numpy())
