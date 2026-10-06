"""Audio conversion helpers for the Chatterbox nodes.

Everything that turns a ComfyUI ``AUDIO`` dict into something Chatterbox can
consume -- and back again -- lives here, in one place:

* :func:`extract_waveform` / :func:`to_comfy_audio` -- ``AUDIO`` <-> tensor.
* :func:`downmix_to_mono` -- the ``[B, C, T]`` mean-downmix Chatterbox's
  tokenizer needs.
* :func:`comfy_audio_to_numpy` / :func:`numpy_to_comfy_audio` -- ``AUDIO`` <->
  numpy.
* :func:`preprocess_comfy_audio` -- the whole chain, resampled and scrubbed,
  in one call.
* :func:`temp_wav_path` -- a context manager handing Chatterbox a real ``.wav``
  file path and **guaranteeing** the file is deleted, including when the body
  raises. The V1 nodes wrote that by hand in two ``try``/``finally`` blocks
  (``nodes.py:181-186`` and ``:246-251``), one of which was on the error path
  of a function that returned ``None`` without ever removing anything.
* :func:`set_seed` -- the seeding routine that replaces the deleted
  ``set_chatterbox_seed``.
* :func:`check_for_interrupt` -- a cooperative cancellation poll.

``comfy.model_management`` is reached through
:func:`modules.device_utils._get_model_management`, never imported at module
scope, so this module is importable without ComfyUI booted. ``torchaudio`` and
``soundfile`` are imported inside the functions that need them: neither is a
node-load-time cost, and both are heavier than the rest of the module.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import tempfile
from typing import Iterator, Optional, Union

import numpy as np
import torch

from .device_utils import _get_model_management

__all__ = [
    "S3GEN_SAMPLE_RATE",
    "set_seed",
    "check_for_interrupt",
    "interrupt_exception_class",
    "resample_waveform",
    "downmix_to_mono",
    "extract_waveform",
    "to_comfy_audio",
    "comfy_audio_to_numpy",
    "numpy_to_comfy_audio",
    "preprocess_comfy_audio",
    "temp_wav_path",
]


# Every Chatterbox model synthesises through S3Gen at 24 kHz
# (``src/chatterbox/models/s3gen/const.py::S3GEN_SR``). Reference audio is
# resampled to it so the tokenizer's token-rate and mel-rate invariants hold.
S3GEN_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Seeding and cancellation
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed torch, numpy and ``random`` from one widget value.

    Replaces the deleted ``set_chatterbox_seed``. ``seed == 0`` means "random"
    (the node widget's convention), which is resolved here to a real seed so
    the run that actually happened can be logged.

    Args:
        seed: Non-negative integer. ``0`` picks a random seed.
    """
    if seed == 0:
        seed = random.randint(1, 0xFFFFFFFFFFFFFFFF)

    # numpy's legacy seeder only accepts a 32-bit value, and torch's accepts
    # the full range. Clamping numpy rather than the seed keeps the two RNGs
    # correlated for every value a 64-bit widget can produce.
    max_numpy_seed = 2**32 - 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % max_numpy_seed)
    random.seed(seed)


def interrupt_exception_class() -> type:
    """The exception class ComfyUI raises when a run is cancelled.

    ``comfy.model_management.InterruptProcessingException`` subclasses
    ``BaseException``, **not** ``Exception``. Anything written as
    ``except Exception`` therefore cannot catch it, and anything written as
    ``except BaseException`` catches it by accident. Generation code that wants
    one wants exactly this class.
    """
    candidate = getattr(_get_model_management(), "InterruptProcessingException", None)
    return candidate if isinstance(candidate, type) else BaseException


def check_for_interrupt() -> bool:
    """Poll for a user cancellation without ever raising.

    Only the interrupt class is caught: a ``KeyboardInterrupt`` arriving
    mid-generation must keep propagating rather than being reported as "the
    user pressed stop".

    Returns:
        ``True`` when the run was cancelled, ``False`` otherwise.
    """
    try:
        _get_model_management().throw_exception_if_processing_interrupted()
    except interrupt_exception_class():
        return True
    return False


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_waveform(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample ``waveform`` from ``orig_sr`` to ``target_sr``.

    Uses ``torchaudio.functional.resample`` -- the ComfyUI-core idiom -- rather
    than reaching for librosa, which is a much heavier import for a sinc
    resampler. Resampling in tensor space also avoids a numpy round-trip.

    Args:
        waveform: A float tensor of any shape; the last axis is time.
        orig_sr: Source sample rate.
        target_sr: Destination sample rate.

    Returns:
        The resampled tensor. The input object itself when the rates match.

    Raises:
        ValueError: Either sample rate is not positive.
        RuntimeError: ``torchaudio`` is not importable.
    """
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"Sample rates must be positive (got orig={orig_sr}, target={target_sr})"
        )
    if orig_sr == target_sr:
        return waveform

    try:
        import torchaudio.functional as audio_functional
    except Exception as exc:  # pragma: no cover - torchaudio ships with torch
        raise RuntimeError(
            "Resampling reference audio requires torchaudio, which could not be "
            f"imported: {exc}"
        ) from exc

    return audio_functional.resample(waveform, orig_sr, target_sr)


# ---------------------------------------------------------------------------
# Waveform plumbing
# ---------------------------------------------------------------------------


def extract_waveform(audio: Union[dict, torch.Tensor], name: str = "audio") -> torch.Tensor:
    """Pull the waveform tensor out of a ComfyUI ``AUDIO`` dict.

    The batch axis is stripped: ComfyUI hands over ``[B, C, T]`` and every
    Chatterbox entry point wants ``[C, T]``. Multi-channel audio is preserved
    here on purpose -- the downmix is a separate, explicit step
    (:func:`downmix_to_mono`), because whether to mix is the caller's
    decision, not a side effect of unpacking.

    Args:
        audio: A ComfyUI ``AUDIO`` dict, or a bare waveform tensor.
        name: Field name used in error messages.

    Returns:
        A ``[C, T]`` tensor, or ``[T]`` when the input was already
        single-channel and unbatched.

    Raises:
        ValueError: The dict is malformed or the waveform is empty.
    """
    if isinstance(audio, torch.Tensor):
        waveform = audio
    else:
        if not isinstance(audio, dict):
            raise ValueError(
                f"{name}: expected an AUDIO dict, got {type(audio).__name__}"
            )
        if "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError(
                f"{name}: an AUDIO dict needs both 'waveform' and 'sample_rate'"
            )
        waveform = audio["waveform"]
        if waveform is None:
            raise ValueError(f"{name}: 'waveform' is None")

    if waveform is None:
        raise ValueError(f"{name}: 'waveform' is None")
    if waveform.numel() == 0:
        raise ValueError(f"{name}: the waveform is empty")
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(
            f"{name}: expected [B, C, T] / [C, T] / [T], got {tuple(waveform.shape)}"
        )
    return waveform


def downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Mean-mix a ``[B, C, T]`` / ``[C, T]`` waveform down to one channel.

    Chatterbox's voice encoder and tokenizer are mono-only. Averaging (rather
    than taking channel 0) is what ComfyUI itself does and what keeps an
    anti-phase stereo pair from cancelling to silence.

    Args:
        waveform: ``[B, C, T]``, ``[C, T]`` or ``[T]``.

    Returns:
        ``[B, T]`` for a batched input, ``[T]`` otherwise.
    """
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() == 2:
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0)
        return waveform[0]
    if waveform.dim() == 3:
        return waveform.mean(dim=1)
    raise ValueError(
        f"Expected a waveform of rank 1-3, got shape {tuple(waveform.shape)}"
    )


def to_comfy_audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    """Build a ComfyUI ``AUDIO`` dict from a waveform tensor.

    Normalises ``[T]`` / ``[C, T]`` / ``[B, C, T]`` to ComfyUI's ``[B, C, T]``,
    so a node can hand back whatever its model returned.

    Args:
        waveform: The audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        ``{"waveform": FloatTensor[B, C, T], "sample_rate": int}`` on CPU.
    """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 3:
        raise ValueError(
            f"Cannot build an AUDIO dict from shape {tuple(waveform.shape)}"
        )
    return {
        "waveform": waveform.detach().to(device="cpu", dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


def comfy_audio_to_numpy(
    audio: Union[dict, torch.Tensor], name: str = "audio"
) -> np.ndarray:
    """Mono float32 numpy view of a ComfyUI ``AUDIO`` dict.

    Args:
        audio: A ComfyUI ``AUDIO`` dict, or a bare waveform tensor.
        name: Field name used in error messages.

    Returns:
        A 1-D ``float32`` array.
    """
    mono = downmix_to_mono(extract_waveform(audio, name=name))
    return mono.detach().to(device="cpu", dtype=torch.float32).numpy()


def numpy_to_comfy_audio(waveform: np.ndarray, sample_rate: int) -> dict:
    """Build a ComfyUI ``AUDIO`` dict from a numpy waveform.

    Args:
        waveform: ``[T]`` or ``[C, T]`` samples.
        sample_rate: Sample rate in Hz.

    Returns:
        ``{"waveform": FloatTensor[B, C, T], "sample_rate": int}``.
    """
    if not isinstance(waveform, np.ndarray):
        waveform = np.asarray(waveform)
    return to_comfy_audio(torch.from_numpy(np.ascontiguousarray(waveform)), sample_rate)


def preprocess_comfy_audio(
    audio: Optional[dict],
    target_sr: int = S3GEN_SAMPLE_RATE,
    name: str = "audio",
) -> Optional[np.ndarray]:
    """Turn a ComfyUI ``AUDIO`` dict into a mono, resampled float32 array.

    The full chain, in the order that has to happen: extract, downmix, scrub
    non-finite samples **before** resampling (one NaN would otherwise poison
    the whole sinc kernel), resample, scrub again.

    Args:
        audio: A ComfyUI ``AUDIO`` dict, or ``None``.
        target_sr: Sample rate the caller needs.
        name: Field name used in error messages.

    Returns:
        A mono ``float32`` numpy array, or ``None`` when ``audio`` is ``None``
        or carries no samples.

    Raises:
        RuntimeError: Resampling was required but no backend was available.
    """
    if audio is None:
        return None
    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        if waveform is None or waveform.numel() == 0:
            return None
        original_sr = int(audio.get("sample_rate") or 0)
    else:
        waveform = audio
        original_sr = 0
        if waveform is None or waveform.numel() == 0:
            return None

    if original_sr <= 0:
        raise ValueError(
            f"{name}: an AUDIO dict needs a positive 'sample_rate' to resample; "
            f"got {original_sr!r}"
        )

    mono = downmix_to_mono(extract_waveform(audio, name=name))

    if not torch.isfinite(mono).all():
        logging.error(
            "[Chatterbox TTS] %s contains NaN or Inf samples; replacing them with "
            "zeros before resampling",
            name,
        )
        mono = torch.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    if original_sr != int(target_sr):
        mono = resample_waveform(mono, original_sr, int(target_sr))

    samples = mono.detach().to(device="cpu", dtype=torch.float32).numpy()

    if not np.all(np.isfinite(samples)):
        logging.error(
            "[Chatterbox TTS] %s still contains NaN or Inf after resampling; "
            "replacing them with zeros",
            name,
        )
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 10.0:
        logging.warning(
            "[Chatterbox TTS] %s peak amplitude is %.2f, far outside the usual "
            "[-1, 1] range; scaling it down",
            name,
            peak,
        )
        samples = samples / peak

    return np.ascontiguousarray(samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Temporary files
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def temp_wav_path(audio: Union[dict, torch.Tensor]) -> Iterator[str]:
    """Write ``audio`` to a temporary ``.wav`` and hand back its path.

    Chatterbox's ``ChatterboxTTS.generate`` / ``ChatterboxVC.generate`` take a
    *file path* for the voice prompt and read it back with ``librosa`` -- they
    cannot be handed a tensor. This is the only place in the codebase that
    creates that file.

    Cleanup lives in a ``finally``, so an exception raised by the body removes
    the file exactly like the normal exit does. The V1 nodes had two copies of
    this cleanup by hand and the one on the error path never ran its ``os.remove``
    when the write itself failed.

    Args:
        audio: A ComfyUI ``AUDIO`` dict, or a bare ``[C, T]`` waveform tensor
            (the sample rate must then already be ``S3GEN_SAMPLE_RATE``).

    Yields:
        The absolute path of the written file.

    Raises:
        ValueError: The audio carries no samples.
        RuntimeError: ``soundfile`` could not be imported.
    """
    mono = downmix_to_mono(extract_waveform(audio, name="voice prompt"))
    if mono.numel() == 0:
        raise ValueError("voice prompt: the waveform is empty")

    sample_rate = S3GEN_SAMPLE_RATE
    if isinstance(audio, dict):
        sample_rate = int(audio.get("sample_rate") or S3GEN_SAMPLE_RATE)

    # Always .wav: the suffix is not cosmetic, it is how soundfile picks the
    # container and how librosa picks the decoder on the way back.
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = handle.name
    handle.close()

    try:
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - soundfile ships with ComfyUI
            raise RuntimeError(
                "Writing a voice-prompt file requires soundfile, which could not "
                f"be imported: {exc}"
            ) from exc

        samples = (
            mono.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)
        )
        sf.write(path, samples, sample_rate)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError as exc:
            logging.debug(
                "[Chatterbox TTS] Could not remove the temporary voice-prompt file "
                "'%s': %s",
                path,
                exc,
            )