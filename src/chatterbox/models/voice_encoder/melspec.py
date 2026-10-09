from functools import lru_cache

import numpy as np

from .. import dsp


@lru_cache()
def mel_basis(hp):
    assert hp.fmax <= hp.sample_rate // 2
    return dsp.mel_filterbank(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


def preemphasis(wav, hp):
    assert hp.preemphasis != 0
    wav = dsp.preemphasis(wav, coef=hp.preemphasis)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
    return mel   # (M, T)


def _stft(y, hp, pad=True):
    # librosa.stft mirror in models/dsp.py; reflect kept for historical
    # consistency with the pretrained featurization
    return dsp.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def _amp_to_db(x, hp):
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
