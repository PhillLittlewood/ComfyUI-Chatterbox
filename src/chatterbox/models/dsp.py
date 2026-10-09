# SPDX-License-Identifier: ISC
"""Pure-numpy/torch DSP for the vendored Chatterbox models.

Adapted from librosa (https://librosa.org), ISC license:

    Permission to use, copy, modify, and/or distribute this software for
    any purpose with or without fee is hereby granted, provided that the
    above copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
    WARRANTIES WITH REGARD TO THIS SOFTWARE.

librosa is heavy (numba/llvmlite at import time) and only a few of its
functions are actually used: the Slaney mel filterbank, the STFT behind the
voice encoder, and silence trimming. Each mirror below reproduces librosa
1.0's exact operation order: the mel filterbanks and featurization are
training-baked, so the replicas must be bit-for-bit identical to librosa's
output on the parameters this package uses (verified against librosa
1.0.0). Loading a reference clip from disk resamples with torchaudio's sinc
kernel instead of soxr -- the one intentional numerical difference, and it
only touches conditioning clips supplied at call time.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "mel_filterbank",
    "stft",
    "trim_silence",
    "preemphasis",
    "hann_window",
    "resample",
    "load_audio_at",
]


# ---------------------------------------------------------------------------
# Slaney Hz <-> mel conversion (librosa.core.convert mirrors)
# ---------------------------------------------------------------------------


def _hz_to_mel(frequencies) -> np.ndarray:
    """Slaney Hz->mel, mirroring librosa's ``hz_to_mel`` (htk=False)."""
    f_min = 0.0
    f_sp = 200.0 / 3

    freqs = np.asarray(frequencies, dtype=np.float64)
    mels = (freqs - f_min) / f_sp

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    # Same values as librosa's masked assignment; np.maximum keeps the
    # untaken branch's log() argument in domain so no warning is raised.
    log_t = freqs >= min_log_hz
    return np.where(
        log_t,
        min_log_mel + np.log(np.maximum(freqs, min_log_hz) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Slaney mel->Hz, mirroring librosa's ``mel_to_hz`` (htk=False)."""
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Center freqs of mel bands, uniformly spaced in mel space."""
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    return _mel_to_hz(np.linspace(min_mel, max_mel, n_mels))


def mel_filterbank(
    sr: float,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax=None,
) -> np.ndarray:
    """Slaney mel filterbank, mirroring ``librosa.filters.mel`` with
    ``norm="slaney"`` and the default float32 output.

    The op order matters: librosa builds the triangles in float64, assigns
    them into a float32 array, then multiplies the float64 enorm in-place
    (cast to float32 per element). Any other order shifts features by ~1e-7.
    """
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        # lower and upper slopes for all bins, intersected and zeroed
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # Slaney-style: approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


# ---------------------------------------------------------------------------
# STFT (librosa.core.spectrum mirror, as used by the voice encoder)
# ---------------------------------------------------------------------------


def hann_window(win_length: int) -> np.ndarray:
    """Periodic Hann, mirroring ``scipy.signal.get_window("hann", M,
    fftbins=True)`` bit-for-bit: the symmetric formula on M+1 linspace
    points with the last sample dropped."""
    fac = np.linspace(-np.pi, np.pi, win_length + 1)
    w = 0.5 + 0.5 * np.cos(fac)
    return w[:-1].copy()


def _frame(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Overlapping frames of a 1-D signal as ``(frame_length, n_frames)``.

    Mirrors ``librosa.util.frame``: the tail past the last complete frame is
    dropped, and an input shorter than one frame raises.
    """
    if len(y) < frame_length:
        raise ValueError(
            f"input signal of length={len(y)} is too short to frame with "
            f"frame_length={frame_length}"
        )
    n_frames = 1 + (len(y) - frame_length) // hop_length
    return np.lib.stride_tricks.as_strided(
        y,
        shape=(frame_length, n_frames),
        strides=(y.strides[0], hop_length * y.strides[0]),
    )


def stft(
    y: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """librosa's ``stft`` for a 1-D signal, reproducing its exact output.

    The float64 Hann window promotes the frame product to float64, the FFT
    runs in float64, and the result is stored into a complex64 buffer when
    the input is float32 (complex128 for float64) -- exactly librosa's
    dtype chain. Center padding uses librosa's head/tail split (not a plain
    both-sides pad), so the frame count and content match for every input
    length. The FFT columns are independent, so the whole spectrum is
    transformed in one call where librosa batches it.
    """
    if win_length is None:
        win_length = n_fft
    fft_window = hann_window(win_length)
    if win_length != n_fft:
        lpad = (n_fft - win_length) // 2
        fft_window = np.pad(fft_window, (lpad, n_fft - win_length - lpad))
    fft_window = fft_window[:, np.newaxis]

    y_frames_pre = None
    y_frames_post = None
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of "
                f"length={y.shape[-1]}",
                stacklevel=2,
            )
        # How many frames depend on left padding / extra right padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1
        if tail_k <= start_k:
            # head and tail overlap: copy-pad the whole signal and carry on
            start = 0
            extra = 0
            y = np.pad(y, (n_fft // 2, n_fft // 2), mode=pad_mode)
        else:
            start = start_k * hop_length - n_fft // 2
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                (n_fft // 2, 0),
                mode=pad_mode,
            )
            y_frames_pre = _frame(y_pre, n_fft, hop_length)[..., :start_k]
            extra = y_frames_pre.shape[-1]
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                y_post = np.pad(
                    y[..., tail_k * hop_length - n_fft // 2 :],
                    (0, n_fft // 2),
                    mode=pad_mode,
                )
                y_frames_post = _frame(y_post, n_fft, hop_length)
                extra += y_frames_post.shape[-1]
            else:
                y_frames_post = np.empty((n_fft, 0), dtype=y.dtype)
    else:
        if n_fft > y.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of "
                f"input signal of length={y.shape[-1]}"
            )
        start = 0
        extra = 0

    dtype = np.complex64 if y.dtype == np.float32 else np.complex128
    y_frames = _frame(y[..., start:], n_fft, hop_length)
    out = np.zeros((1 + n_fft // 2, y_frames.shape[-1] + extra), dtype=dtype)

    off_start = 0
    if extra > 0:
        # warm-up: head frames at the front, tail frames at the end
        off_start = y_frames_pre.shape[-1]
        out[..., :off_start] = np.fft.rfft(fft_window * y_frames_pre, axis=-2)
        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            out[..., -off_end:] = np.fft.rfft(fft_window * y_frames_post, axis=-2)
    out[..., off_start : off_start + y_frames.shape[-1]] = np.fft.rfft(
        fft_window * y_frames, axis=-2
    )
    return out


# ---------------------------------------------------------------------------
# Silence trimming (librosa.feature.rms + librosa.effects.trim mirrors)
# ---------------------------------------------------------------------------


def _frames_nonsilent(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
    top_db: float,
) -> np.ndarray:
    """Frame-wise non-silent indicator, mirroring librosa's
    ``feature.rms`` (center=True, pad_mode="constant", float32) composed
    with ``amplitude_to_db(ref=np.max, top_db=None)``."""
    padded = np.pad(y, frame_length // 2, mode="constant")
    frames = _frame(padded, frame_length, hop_length)
    power = np.mean(
        np.square(frames.astype(np.float32, copy=False)), axis=-2, keepdims=True
    )
    rms = np.sqrt(power)[..., 0, :]
    ref_value = np.max(rms)
    db = 10.0 * np.log10(np.maximum(1e-10, np.square(rms)))
    db -= 10.0 * np.log10(np.maximum(1e-10, ref_value**2))
    return db > -top_db


def trim_silence(
    y: np.ndarray,
    *,
    top_db: float = 60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Drop leading and trailing silence, mirroring librosa's
    ``effects.trim`` with its default ``ref=np.max``. An all-silent clip is
    returned untrimmed, matching librosa's own behavior.
    """
    y = np.asarray(y)
    non_silent = _frames_nonsilent(y, frame_length, hop_length, top_db)
    if not non_silent.any():
        return y
    edges = np.flatnonzero(np.diff(non_silent.astype(int)))
    edges = [edges + 1]
    if non_silent[0]:
        edges.insert(0, np.array([0]))
    if non_silent[-1]:
        edges.append(np.array([len(non_silent)]))
    edges = np.concatenate(edges) * hop_length
    edges = np.minimum(edges, y.shape[-1])
    interval = edges.reshape(-1, 2)[0]
    return y[interval[0] : interval[1]]


def preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """librosa's ``effects.preemphasis`` for the FIR case (a=[1]):
    ``y[n] - coef * y[n-1]``."""
    y = np.asarray(y)
    out = np.array(y, copy=True)
    out[1:] = y[1:] - coef * y[:-1]
    return out


# ---------------------------------------------------------------------------
# Loading and resampling
# ---------------------------------------------------------------------------


def resample(wav: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    """Resample a float32 waveform with torchaudio's sinc kernel.

    Replaces librosa's soxr resampler on the reference-clip path; the
    kernel difference is imperceptible and only touches conditioning clips
    supplied at call time.
    """
    if orig_sr == target_sr:
        return wav
    import torchaudio

    wav_t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    out = torchaudio.functional.resample(
        wav_t, orig_freq=int(orig_sr), new_freq=int(target_sr)
    )
    return out.numpy()


def load_audio_at(audio, target_sr: float) -> np.ndarray:
    """Reference clip as float32 mono numpy at ``target_sr``.

    A path (str or ``Path``) is read with soundfile -- mirroring
    librosa.load's float32 read and channel-mean downmix -- and resampled
    with :func:`resample` when its rate differs. An in-memory waveform
    (torch tensor or ndarray) is taken as-is: the ComfyUI node passes audio
    already downmixed and resampled to ``target_sr``.
    """
    if isinstance(audio, (str, Path)):
        import soundfile

        wav, sr = soundfile.read(audio, dtype="float32", always_2d=True)
        wav = wav.T
        wav = wav.mean(axis=0) if wav.shape[0] > 1 else wav[0]
        return resample(wav, float(sr), float(target_sr))
    if isinstance(audio, torch.Tensor) and audio.is_cuda:
        audio = audio.cpu()
    return np.asarray(audio, dtype=np.float32)
