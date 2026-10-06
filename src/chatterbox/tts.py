from dataclasses import dataclass
from pathlib import Path

import logging

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# PATCHED (src/PATCHES.md P-023): `perth` (the optional watermarking library) is
# a hard import upstream, so `import chatterbox.tts` dies without it. It is only
# needed for `apply_watermark`, so a missing library degrades to no watermark.
try:
    import perth
except ImportError:
    perth = None

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# PATCHED (src/PATCHES.md P-027): the file set a classic pack has to hold, and
# an up-front check for it. Upstream reaches its first `load_file` on a path that
# does not exist and dies with `FileNotFoundError: .../ve.safetensors`, which
# names one file and leaves the user guessing about the other three. An
# arbitrary local directory -- a community finetune, a hand-assembled pack -- has
# to clear the same bar as an official one, and be told the same way when it does
# not. Used by `ChatterboxTTS.from_local`; the other three entry points carry the
# same check against their own file sets.
REQUIRED_CHECKPOINT_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
)


def _require_checkpoint_files(ckpt_dir, required):
    """Raise ``FileNotFoundError`` naming every file in ``required`` that is absent.

    Args:
        ckpt_dir: The checkpoint directory.
        required: Filenames the entry point cannot build a model without.

    Raises:
        FileNotFoundError: One or more files are missing. The message lists the
            missing names first, then the full set, because the first is what
            the user has to fix and the second is what tells them whether the
            directory is a Chatterbox pack at all.
    """
    missing = [name for name in required if not (Path(ckpt_dir) / name).is_file()]
    if not missing:
        return
    raise FileNotFoundError(
        f"The Chatterbox checkpoint at '{ckpt_dir}' is incomplete. "
        f"Missing: {', '.join(missing)}. A complete pack holds "
        f"{', '.join(required)}."
    )


# PATCHED (src/PATCHES.md P-002): duration of one S3 speech token (0.04 s).
S3_TOKEN_DURATION_S = 1 / S3_TOKEN_RATE

# PATCHED (src/PATCHES.md P-023): one named logger for the "no watermarking
# library" warning, shared by every entry point so a node that loads several
# models still produces a single, greppable warning per entry point.
WATERMARK_LOGGER = logging.getLogger("chatterbox.watermark")

_watermark_warning_emitted = False


def _apply_watermark(wav, sample_rate, watermarker, watermark=True):
    """Watermark ``wav`` unless the caller opted out or ``perth`` is absent.

    PATCHED (src/PATCHES.md P-023): upstream calls
    ``self.watermarker.apply_watermark(...)`` unconditionally. This keeps that
    behaviour when ``watermark=True`` and a watermarker exists, skips the call
    when the caller passed ``watermark=False``, and -- when watermarking was
    asked for but the optional library is not installed -- warns once and
    returns the audio untouched instead of raising.
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


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


# PATCHED (src/PATCHES.md P-002)
def _pad_wav_to_40ms_multiple(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Pad a waveform to a multiple of 40 ms.

    The mel spectrogram hops every 20 ms while the S3 speech tokenizer emits one
    token every 40 ms. Without this alignment the tokenizer rounds the clip
    length independently and the two drift apart, which S3Gen detects as
    ``mel length != 2 * token length`` and only repairs by truncating tokens.
    """
    samples_per_token = int(sr * S3_TOKEN_DURATION_S)
    remainder = wav.shape[-1] % samples_per_token
    if remainder == 0:
        return wav
    return F.pad(wav, (0, samples_per_token - remainder))


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # PATCHED (src/PATCHES.md P-023): `perth` may be absent.
        self.watermarker = perth.PerthImplicitWatermarker() if perth is not None else None

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # PATCHED (src/PATCHES.md P-027): an incomplete directory is refused
        # here, by name, instead of half way through the first `load_file`.
        _require_checkpoint_files(ckpt_dir, REQUIRED_CHECKPOINT_FILES)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        # PATCHED (src/PATCHES.md P-003): compare `str(device)`. ComfyUI passes a
        # `torch.device`, and `torch.device(...) in ["cpu", "mps"]` is False for
        # every device, so a CPU load would try to restore CUDA tensors.
        if str(device) in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        # PATCHED (src/PATCHES.md P-004): round-trip through torch so the clip can
        # be padded onto the 40 ms token grid before resampling. Resampling a
        # fractional number of tokens produces a mel/token length mismatch
        # downstream, which S3Gen silently repairs by dropping tokens.
        s3gen_ref_wav = torch.from_numpy(s3gen_ref_wav).float().unsqueeze(0)
        s3gen_ref_wav = _pad_wav_to_40ms_multiple(s3gen_ref_wav, S3GEN_SR)
        s3gen_ref_wav_np = s3gen_ref_wav.squeeze(0).numpy()

        ref_16k_wav = librosa.resample(s3gen_ref_wav_np, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav_np = s3gen_ref_wav_np[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav_np, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # PATCHED (src/PATCHES.md P-005): ComfyUI node surface.
        pbar=None,
        max_new_tokens=1000,
        flow_cfg_scale=0.7,
        # PATCHED (src/PATCHES.md P-023): opt out of watermarking entirely.
        watermark=True,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                pbar=pbar,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                flow_cfg_scale=flow_cfg_scale,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = _apply_watermark(wav, self.sr, self.watermarker, watermark)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
