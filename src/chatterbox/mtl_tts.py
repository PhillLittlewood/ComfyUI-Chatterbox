from dataclasses import dataclass
from pathlib import Path
import logging
import os

import librosa
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

# PATCHED (src/PATCHES.md P-025): `perth` (the optional watermarking library) is
# a hard import upstream, so `import chatterbox.mtl_tts` dies without it. It is
# only needed for `apply_watermark`, so a missing library degrades to no
# watermark.
try:
    import perth
except ImportError:
    perth = None

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
DEFAULT_MULTILINGUAL_T3_MODEL = "t3_mtl23ls_v2.safetensors"
MULTILINGUAL_T3_MODELS = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3": "t3_mtl23ls_v3.safetensors",
}

# PATCHED (src/PATCHES.md P-027): the file set a multilingual pack has to hold
# *besides* the resolved T3, voice-encoder and s3gen checkpoints, and the
# up-front check for it. The other three filenames are pack-dependent (P-036)
# and resolved by the caller, so none of them is in this tuple.
REQUIRED_CHECKPOINT_FILES = (
    "grapheme_mtl_merged_expanded_v1.json",
)


# PATCHED (src/PATCHES.md P-036): the Single Language Pack finetunes ship
# their S3Gen decoder as ``s3gen_v3.safetensors`` (plus a legacy ``.pt``
# twin), while the shared multilingual repo ships ``s3gen.safetensors``.
# Resolve whichever serialization the pack actually holds, preferring
# safetensors (no pickle, what every other entry point loads); the canonical
# ``s3gen.safetensors`` stays the fallback so the missing-file message names
# it when nothing exists.
def _resolve_s3gen_checkpoint(ckpt_dir):
    for name in (
        "s3gen_v3.safetensors",
        "s3gen_v3.pt",
        "s3gen.safetensors",
        "s3gen.pt",
    ):
        if (Path(ckpt_dir) / name).is_file():
            return name
    return "s3gen.safetensors"


def _resolve_ve_checkpoint(ckpt_dir):
    """The voice-encoder file the pack holds, safetensors preferred (P-036)."""
    for name in ("ve.safetensors", "ve.pt"):
        if (Path(ckpt_dir) / name).is_file():
            return name
    return "ve.safetensors"


def _load_checkpoint(path, map_location=None):
    """Read a state dict from either serialization, by extension (P-036)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location=map_location, weights_only=True)


def _require_checkpoint_files(ckpt_dir, required):
    """Raise ``FileNotFoundError`` naming every file in ``required`` that is absent.

    Mirrors ``tts._require_checkpoint_files`` (P-027) rather than importing it,
    so each entry point keeps working when it is vendored on its own.
    """
    missing = [name for name in required if not (Path(ckpt_dir) / name).is_file()]
    if not missing:
        return
    raise FileNotFoundError(
        f"The multilingual Chatterbox checkpoint at '{ckpt_dir}' is incomplete. "
        f"Missing: {', '.join(missing)}. A complete pack holds "
        f"{', '.join(required)}, plus the T3 checkpoint for the selected version."
    )

# PATCHED (src/PATCHES.md P-025): one named logger for the "no watermarking
# library" warning, shared by every entry point so a node that loads several
# models still produces a single, greppable warning per entry point.
WATERMARK_LOGGER = logging.getLogger("chatterbox.watermark")

_watermark_warning_emitted = False


# PATCHED (src/PATCHES.md P-025)
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

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def _resolve_multilingual_t3_model(t3_model: str | None) -> str:
    if t3_model is None:
        return DEFAULT_MULTILINGUAL_T3_MODEL
    if t3_model in MULTILINGUAL_T3_MODELS:
        return MULTILINGUAL_T3_MODELS[t3_model]
    if t3_model.endswith(".safetensors"):
        return t3_model
    raise ValueError(
        f"Unknown multilingual T3 model '{t3_model}'. "
        f"Expected one of {sorted(MULTILINGUAL_T3_MODELS)} or a .safetensors filename."
    )


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
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
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


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
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
        # PATCHED (src/PATCHES.md P-025): `perth` may be absent.
        self.watermarker = perth.PerthImplicitWatermarker() if perth is not None else None

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        t3_model: str | None = None,
    ) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)
        t3_model = _resolve_multilingual_t3_model(t3_model)

        # PATCHED (src/PATCHES.md P-027): an incomplete directory is refused
        # here, by name, instead of at the first `torch.load`.
        # PATCHED (src/PATCHES.md P-036): the s3gen and ve filenames follow
        # the pack (safetensors preferred); the resolved names join the
        # required set so the missing-file message names what is actually
        # expected here.
        ve_checkpoint = _resolve_ve_checkpoint(ckpt_dir)
        s3gen_checkpoint = _resolve_s3gen_checkpoint(ckpt_dir)
        _require_checkpoint_files(
            ckpt_dir,
            (*REQUIRED_CHECKPOINT_FILES, ve_checkpoint, s3gen_checkpoint, t3_model),
        )

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        # PATCHED (src/PATCHES.md P-025): same `str(device)` comparison as P-003
        # and P-007. ComfyUI passes a `torch.device`, and `torch.device(...) in
        # ["cpu", "mps"]` is False for every device, so a CPU load would try to
        # restore CUDA-saved tensors.
        if str(device) in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        # PATCHED (src/PATCHES.md P-036): resolve the voice-encoder file the
        # pack actually holds (safetensors preferred) instead of hard-coding
        # the legacy ``ve.pt`` name.
        ve.load_state_dict(
            _load_checkpoint(ckpt_dir / _resolve_ve_checkpoint(ckpt_dir), map_location)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / t3_model)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        # PATCHED (src/PATCHES.md P-036): the Single Language Pack s3gen
        # checkpoints predate the `tokenizer._mel_filters`/`tokenizer.window`
        # buffers (see S3Token2Wav.ignore_state_dict_missing) -- both are
        # recomputed in __init__, never learned. Load non-strict like the
        # classic path in tts.py does; a strict load refuses every SLP pack.
        s3gen.load_state_dict(
            _load_checkpoint(ckpt_dir / s3gen_checkpoint, map_location),
            strict=False,
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        t3_model: str | None = None,
    ) -> 'ChatterboxMultilingualTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        t3_model = _resolve_multilingual_t3_model(t3_model)
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                # PATCHED (src/PATCHES.md P-036): fetch the safetensors
                # serializations only -- the repo's ``.pt`` twins are the
                # same weights and never loaded.
                allow_patterns=["ve.safetensors", t3_model, "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, t3_model=t3_model)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
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
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        # PATCHED (src/PATCHES.md P-025): opt out of watermarking entirely.
        watermark=True,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            # Drop the final speech token's audio: it is emitted just before
            # EOS with degraded attention and decodes to ~40 ms of noise.
            n_tokens = int(speech_tokens.shape[-1])
            st_len = max(1, n_tokens - 1)
            wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]

            watermarked_wav = _apply_watermark(wav, self.sr, self.watermarker, watermark)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
