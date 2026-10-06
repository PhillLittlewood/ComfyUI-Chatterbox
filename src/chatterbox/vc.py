from pathlib import Path

import logging

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# PATCHED (src/PATCHES.md P-024): `perth` (the optional watermarking library) is
# a hard import upstream, so `import chatterbox.vc` dies without it. It is only
# needed for `apply_watermark`, so a missing library degrades to no watermark.
try:
    import perth
except ImportError:
    perth = None

from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"

# PATCHED (src/PATCHES.md P-027): the file set a conversion-only pack has to
# hold, and the up-front check for it. Same shape and same rationale as
# `tts.REQUIRED_CHECKPOINT_FILES`; kept here rather than imported so a user who
# vendors only `vc.py` still gets the actionable message. Only the vocoder is
# required: `conds.pt` (the built-in voice) is loaded when present and skipped
# when not, and no T3 weights are involved at all in a conversion.
REQUIRED_CHECKPOINT_FILES = ("s3gen.safetensors",)


def _require_checkpoint_files(ckpt_dir, required):
    """Raise ``FileNotFoundError`` naming every file in ``required`` that is absent.

    Mirrors ``tts._require_checkpoint_files`` (P-027) rather than importing it,
    so each entry point keeps working when it is vendored on its own.
    """
    missing = [name for name in required if not (Path(ckpt_dir) / name).is_file()]
    if not missing:
        return
    raise FileNotFoundError(
        f"The Chatterbox checkpoint at '{ckpt_dir}' is incomplete. "
        f"Missing: {', '.join(missing)}. A complete pack holds "
        f"{', '.join(required)}."
    )


# PATCHED (src/PATCHES.md P-006): duration of one S3 speech token (0.04 s).
S3_TOKEN_DURATION_S = 1 / S3_TOKEN_RATE

# PATCHED (src/PATCHES.md P-024): one named logger for the "no watermarking
# library" warning, shared by every entry point so a node that loads several
# models still produces a single, greppable warning per entry point.
WATERMARK_LOGGER = logging.getLogger("chatterbox.watermark")

_watermark_warning_emitted = False


# PATCHED (src/PATCHES.md P-024)
def _apply_watermark(wav, sample_rate, watermarker, watermark=True):
    """Watermark ``wav`` unless the caller opted out or ``perth`` is absent.

    Upstream calls ``self.watermarker.apply_watermark(...)`` unconditionally.
    This keeps that behaviour when ``watermark=True`` and a watermarker exists,
    skips the call when the caller passed ``watermark=False``, and -- when
    watermarking was asked for but the optional library is not installed --
    warns once and returns the audio untouched instead of raising.
    """
    global _watermark_warning_emitted

    if not watermark:
        return wav
    if watermarker is None:
        if not _watermark_warning_emitted:
            _watermark_warning_emitted = True
            WATERMARK_LOGGER.warning(
                "watermark=True was requested but the optional 'resemble-perth' "
                "library is not installed; returning un-watermarked audio. "
                "Install it, or pass watermark=False to silence this."
            )
        return wav
    return watermarker.apply_watermark(wav, sample_rate=sample_rate)


# PATCHED (src/PATCHES.md P-006)
def _pad_wav_to_40ms_multiple(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Pad a waveform to a multiple of 40 ms.

    Keeps the mel spectrogram (20 ms hop) and the S3 speech tokenizer (40 ms hop)
    on the same grid so ``embed_ref`` does not have to truncate tokens.
    """
    samples_per_token = int(sr * S3_TOKEN_DURATION_S)
    remainder = wav.shape[-1] % samples_per_token
    if remainder == 0:
        return wav
    return F.pad(wav, (0, samples_per_token - remainder))


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        # PATCHED (src/PATCHES.md P-024): `perth` may be absent.
        self.watermarker = perth.PerthImplicitWatermarker() if perth is not None else None
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)

        # PATCHED (src/PATCHES.md P-027): an incomplete directory is refused
        # here, by name, instead of at the first `torch.load`.
        _require_checkpoint_files(ckpt_dir, REQUIRED_CHECKPOINT_FILES)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        # PATCHED (src/PATCHES.md P-007): compare `str(device)` -- ComfyUI passes
        # a `torch.device`, which never compares equal to the string "cpu".
        if str(device) in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        # PATCHED (src/PATCHES.md P-008): pad onto the 40 ms token grid before
        # truncating, so the mel/token length invariant survives short clips.
        s3gen_ref_wav = torch.from_numpy(s3gen_ref_wav).float().unsqueeze(0)
        s3gen_ref_wav = _pad_wav_to_40ms_multiple(s3gen_ref_wav, S3GEN_SR)
        s3gen_ref_wav_np = s3gen_ref_wav.squeeze(0).numpy()

        s3gen_ref_wav_np = s3gen_ref_wav_np[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav_np, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
        # PATCHED (src/PATCHES.md P-009): ComfyUI node surface.
        n_timesteps=10,
        pbar=None,
        temperature=1.0,
        flow_cfg_scale=0.7,
        # PATCHED (src/PATCHES.md P-024): opt out of watermarking entirely.
        watermark=True,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `set_target_voice` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
                # PATCHED (src/PATCHES.md P-009): upstream 0.1.7 renamed this
                # kwarg; the node surface keeps the 0.1.2 ``n_timesteps`` name.
                n_cfm_timesteps=n_timesteps,
                pbar=pbar,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = _apply_watermark(wav, self.sr, self.watermarker, watermark)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)