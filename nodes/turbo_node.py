"""``ChatterboxTurboTTS`` -- the Turbo node, on the V3 schema.

**New node id, additive only.** It sits beside ``ChatterboxTTS`` and
``ChatterboxMTLTTS`` and disturbs neither; see the module docstring of
:mod:`nodes.tts_node` for why a saved graph binds widgets by index.

Four things about Turbo that differ from the classic node, and that the schema
encodes rather than explains afterwards:

**``top_k`` exists here and only here.** Turbo's sampler is a top-k / top-p
pair; the classic and multilingual samplers have no top-k at all. Offering one on
those would be a knob with no effect, so ``ChatterboxMTLTTS`` has no ``top_k``
input and a test asserts it does not.

**``top_k`` is 1-5000, default 1000, and ``temperature`` stops at 2.0.** The
classic node's temperature runs to 5.0; Turbo was distilled for a narrower range
and the reference implementations both cap it at 2.0.

**Turbo's tag set is its own.** ``[laugh]``, ``[sigh]``, ``[gasp]``,
``[chuckle]``, ``[cough]``, ``[sniff]``, ``[groan]``, ``[shush]`` and
``[clear throat]`` are square-bracket tokens the Turbo tokenizer was trained
on. They are a different vocabulary from the multilingual v2 set, which is
angle-bracket (``<giggle>``, ``<whisper>``, ``<clear_throat>``) and which this
node does not offer at all. The two spellings are not interchangeable and the
distinction is passed to :func:`modules.text_processing.plan_text` rather than
being left to a comment.

**There is no Nano node.** Upstream ships no ``tts_nano.py``: Nano is the same
module with a second set of weights, reached through ``nano=True``. A separate
node for it would be a separate file to keep in step with a model that is one
flag away. The ``nano`` widget is the whole of Nano, and it changes the cache
key, so the two weight sets can never be served for one another.

``pyloudnorm`` is not installed everywhere. Loudness normalisation of the
*reference* clip is skipped with a warning when it is absent (PATCHES.md P-026);
it never fails a run.
"""

from __future__ import annotations

import logging
from typing import Optional

from comfy_api.latest import io, ui

from ..modules.audio_utils import interrupt_exception_class, to_comfy_audio
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import (
    FAMILY_TURBO,
    generate_turbo_audio,
    load_chatterbox_entry_point,
)
from ..modules.model_discovery import get_chatterbox_model_pack_names
from ..modules.text_processing import TURBO_TAGS, plan_text
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs

__all__ = ["ChatterboxTurboTTSNode", "TURBO_TAG_LIST"]


#: Turbo's paralinguistic tags, rendered for the ``text`` tooltip. The source of
#: truth is :data:`modules.text_processing.TURBO_TAGS`, so the tooltip cannot
#: drift from what the parser actually recognises.
TURBO_TAG_LIST: str = " ".join(f"[{tag}]" for tag in TURBO_TAGS)


class ChatterboxTurboTTSNode(io.ComfyNode):
    """Fast English speech synthesis with paralinguistic tags."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema. Combo options are resolved here, not at import."""
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxTurboTTS",
            display_name="Chatterbox Turbo TTS ⚡",
            category="audio/generation",
            description=(
                "Fast English speech synthesis with Chatterbox Turbo (350M) or "
                "Nano (110M). Lower VRAM, one-step decoding, paralinguistic "
                "tags."
            ),
            inputs=[
                io.Combo.Input(
                    "model_pack_name",
                    options=pack_names,
                    default=default_pack_name(pack_names),
                    tooltip=(
                        "Select the Chatterbox Turbo (or Nano) checkpoint "
                        "directory to load."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default=(
                        "Hello, this is a test. [laugh] Isn't that funny?"
                    ),
                    tooltip=(
                        "Text to be synthesized into speech.\n\n"
                        f"Turbo tags: {TURBO_TAG_LIST}\n\n"
                        "These are Turbo's own tags and are spelled in square "
                        "brackets. The multilingual v2 special tokens "
                        "(<giggle>, <whisper>, <clear_throat>) are a different "
                        "vocabulary and belong on Chatterbox Multilingual TTS."
                    ),
                ),
                io.Boolean.Input(
                    "nano",
                    default=False,
                    tooltip=(
                        "Load the 110M Nano weights instead of the 350M Turbo "
                        "ones.\n\n"
                        "There is no separate Nano node: upstream ships no "
                        "tts_nano module, Nano is a second set of weights in the "
                        "same one. Nano runs on CPU at close to realtime and "
                        "needs roughly a quarter of the VRAM, at some cost in "
                        "quality. Turning this on changes the cache key, so the "
                        "two weight sets are never served for one another."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.05,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls the randomness of the sampling process. Turbo's "
                        "range stops at 2.0 rather than the classic node's 5.0."
                    ),
                ),
                io.Int.Input(
                    "top_k",
                    default=1000,
                    min=1,
                    max=5000,
                    step=1,
                    tooltip=(
                        "Limit sampling to the K most likely tokens. 1 is fully "
                        "greedy, 5000 is effectively unrestricted. This control "
                        "exists only on the Turbo family: the classic and "
                        "multilingual samplers have no top-k."
                    ),
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Nucleus sampling (Top-P) parameter. The model samples "
                        "from the smallest set of tokens whose cumulative "
                        "probability exceeds this value."
                    ),
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=1.0,
                    max=3.0,
                    step=0.1,
                    tooltip=(
                        "Penalizes repeated tokens to discourage monotonous or "
                        "repetitive speech. A value of 1.0 means no penalty."
                    ),
                ),
                io.Float.Input(
                    "exaggeration",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Intensity of the cloned voice's expressiveness. Turbo "
                        "does not use this for sampling -- upstream logs a "
                        "warning for any non-zero value -- so it only colours "
                        "the reference conditioning."
                    ),
                ),
                io.Boolean.Input(
                    "normalize_loudness",
                    default=True,
                    tooltip=(
                        "Normalise the reference clip to a consistent loudness "
                        "before cloning it. Skipped with a warning when the "
                        "optional 'pyloudnorm' library is not installed."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "Seed for random number generation. A value of 0 will "
                        "use a random seed."
                    ),
                ),
                io.Boolean.Input(
                    "use_watermark",
                    default=False,
                    tooltip=(
                        "Enable or disable the audio watermark. Requires "
                        "'resemble-perth' to be installed."
                    ),
                ),
                # Appended after every model-specific widget, in the shared order.
                *runtime_widgets(),
                io.Audio.Input(
                    "audio_prompt",
                    optional=True,
                    tooltip=(
                        "Reference audio to clone the voice from. Turbo wants at "
                        "least 5 seconds. When left unconnected the checkpoint's "
                        "built-in voice is used."
                    ),
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
            is_output_node=False,
        )

    @classmethod
    def validate_inputs(
        cls,
        model_pack_name: Optional[str] = None,
        dtype: Optional[str] = None,
        attention_mode: Optional[str] = None,
    ) -> bool | str:
        """Queue-time checks. Returns a message to refuse the prompt, else ``True``.

        Args:
            model_pack_name: The pack the combo carried.
            dtype: Dtype widget value.
            attention_mode: Attention widget value, as the user picked it.
        """
        message = validate_common_inputs(model_pack_name, dtype, attention_mode)
        return True if message is None else message

    @classmethod
    def execute(
        cls,
        model_pack_name: str,
        text: str,
        nano: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        exaggeration: float,
        normalize_loudness: bool,
        seed: int,
        use_watermark: bool,
        device: str = "auto",
        dtype: str = DTYPE_AUTO,
        attention_mode: str = "sdpa",
        audio_prompt: Optional[dict] = None,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Synthesise ``text`` with Turbo (or Nano) and return it as ``AUDIO``."""
        if not text or not text.strip():
            logging.info("[Chatterbox TTS] Empty text; returning silence.")
            return io.NodeOutput(silent_audio())

        # Turbo's own tag set, recognised because this *is* a Turbo model. The
        # multilingual v2 angle-bracket tokens are not stripped here -- they are
        # not in this tokenizer's vocabulary and passing them through is more
        # honest than silently deleting them.
        plan = plan_text(text, turbo_tags=True)

        patcher, model = load_chatterbox_entry_point(
            model_pack_name,
            entry_point="turbo",
            variant="nano" if nano else "turbo",
            family=FAMILY_TURBO,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            client_id=client_id,
        )

        interrupt_error = interrupt_exception_class()
        try:
            waveform, sample_rate = generate_turbo_audio(
                model,
                plan.text,
                audio_prompt=audio_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                exaggeration=exaggeration,
                norm_loudness=normalize_loudness,
                seed=seed,
                use_watermark=use_watermark,
            )
        except interrupt_error:
            logging.info(
                "[Chatterbox TTS] Turbo synthesis was cancelled; returning silence."
            )
            return io.NodeOutput(silent_audio())

        audio = to_comfy_audio(waveform, sample_rate)
        logging.info(
            "[Chatterbox TTS] Pack '%s' (%s) produced %d Hz audio (cache key %r).",
            model_pack_name,
            "Nano" if nano else "Turbo",
            sample_rate,
            getattr(patcher, "cache_key", None),
        )
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio))
