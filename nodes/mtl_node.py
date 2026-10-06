"""``ChatterboxMTLTTS`` -- the multilingual node, on the V3 schema.

**New node id, additive only.** ``ChatterboxMTLTTS`` sits beside
``ChatterboxTTS``; it does not replace it, does not rename it, and shares no
index with it. ComfyUI binds a saved graph's widget values by index *within one
node*, so a brand-new node id cannot disturb a stored value on any other node.
The frozen contract in ``tests/frozen_widget_snapshot.json`` covers
``ChatterboxTTS`` and ``ChatterboxVC`` only and is untouched by this file.

Three decisions here are not obvious from the widget list:

**``model_version`` shows the resolved filename, not ``"v3"``.** Upstream still
resolves ``None`` to v2 (``DEFAULT_MULTILINGUAL_T3_MODEL``), and the two
third-party implementations disagree about what "v3" even is. A dropdown reading
``v3 | v2 | v1`` tells a user nothing they can check against a Hugging Face
listing; ``t3_mtl23ls_v3.safetensors | t3_mtl23ls_v2.safetensors |
t3_23lang.safetensors`` does. **The default is v3**, which is a deliberate
departure from upstream's v2 and is called out in the widget tooltip.

**``exaggeration`` is inert on v3 and this node says so.** The reference
corpus records that current v3 weights show little or no audible response to it,
and that the 0.0-5.0 range on this family exists *because* the control does
almost nothing. Widening the slider and staying silent about it would be the
worst of both: a control that looks powerful and is not. The tooltip repeats the
caveat verbatim so a user does not spend an afternoon tuning it.

**``language_id`` is a keyword, never spliced into the text.** The vendored
tokenizer prepends the language tag itself; a node that also inserted it would
emit a doubled prefix that fails deep inside the T3. See
:func:`modules.generation.generate_mtl_audio`.

``pykakasi`` (Japanese/Chinese word segmentation) and ``spacy-pkuseg`` (Korean)
are optional and lazily imported *inside* the vendored tokenizer, which already
degrades with a warning. This node degrades identically: it logs once and
continues. It never raises for a missing normaliser.
"""

from __future__ import annotations

import logging
from typing import Optional

from comfy_api.latest import io, ui

from ..modules.audio_utils import interrupt_exception_class, to_comfy_audio
from ..modules.constants import (
    MTL_DEFAULT_T3_MODEL,
    MTL_LANGUAGES,
    MTL_T3_MODELS,
)
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import (
    FAMILY_MTL,
    generate_mtl_audio,
    load_chatterbox_entry_point,
)
from ..modules.model_discovery import get_chatterbox_model_pack_names
from ..modules.text_processing import (
    CRASH_PROTECTION_TEMPLATE,
    plan_text,
)
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs

__all__ = ["ChatterboxMTLTTSNode", "MTL_MODEL_VERSION_OPTIONS", "MTL_MODEL_VERSION_DEFAULT"]


#: The ``model_version`` options, in the order the Combo offers them: newest
#: first, because that is the order a reader wants them in.
MTL_MODEL_VERSION_OPTIONS: tuple[str, ...] = tuple(
    MTL_T3_MODELS[version] for version in ("v3", "v2", "v1")
)

#: Resolved to a **filename**, so the Combo's default is literally the file the
#: first run will look for.
MTL_MODEL_VERSION_DEFAULT: str = MTL_T3_MODELS[MTL_DEFAULT_T3_MODEL]


class ChatterboxMTLTTSNode(io.ComfyNode):
    """Synthesise speech in any of Chatterbox's 23 languages."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema.

        The Combo options are resolved here rather than at import time, for the
        same reason as every other node: the models directory changes while
        ComfyUI runs.
        """
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxMTLTTS",
            display_name="Chatterbox Multilingual TTS 🌍",
            category="audio/generation",
            description=(
                "Synthesise speech in 23 languages with the Chatterbox "
                "multilingual model, in v1, v2 or v3."
            ),
            inputs=[
                io.Combo.Input(
                    "model_pack_name",
                    options=pack_names,
                    default=default_pack_name(pack_names),
                    tooltip=(
                        "Select the Chatterbox checkpoint directory to load the "
                        "multilingual T3 weights from."
                    ),
                ),
                io.Combo.Input(
                    "model_version",
                    options=list(MTL_MODEL_VERSION_OPTIONS),
                    default=MTL_MODEL_VERSION_DEFAULT,
                    tooltip=(
                        "Which multilingual T3 checkpoint to load. The options "
                        "are the resolved filenames, not labels.\n\n"
                        "t3_mtl23ls_v3.safetensors (default) is the newest "
                        "official checkpoint: better similarity, fewer "
                        "hallucinations. It skips the legacy alignment analyzer "
                        "and trims the final-token artifact.\n\n"
                        "t3_mtl23ls_v2.safetensors is upstream's own default "
                        "(DEFAULT_MULTILINGUAL_T3_MODEL). This node defaults to "
                        "v3 deliberately.\n\n"
                        "t3_23lang.safetensors is the original 23-language "
                        "model, with no special-token vocabulary.\n\n"
                        "Only v2 has the 30 documented special tokens "
                        "(<giggle>, <whisper>, <clear_throat> and the rest)."
                    ),
                ),
                io.Combo.Input(
                    "language",
                    options=sorted(MTL_LANGUAGES),
                    default="en",
                    tooltip=(
                        "Language of the text. 23 languages, shown as ISO 639-1 "
                        "codes: ar da de el en es fi fr he hi it ja ko ms nl no "
                        "pl pt ru sv sw tr zh. There is no Vietnamese -- the "
                        "Vietnamese and Egyptian-Arabic models are community "
                        "finetunes, and load like any other local checkpoint."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default=(
                        "Hello, this is a test of Chatterbox multilingual TTS "
                        "in ComfyUI."
                    ),
                    tooltip=(
                        "Text to be synthesized into speech."
                    ),
                ),
                io.Float.Input(
                    "exaggeration",
                    default=0.5,
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    tooltip=(
                        "Controls the expressiveness and emotional intensity.\n\n"
                        "Note: current v3 weights show little or no audible "
                        "response to this control, which is why the range here "
                        "is 0.0-5.0 rather than the classic node's 0.25-2.0. "
                        "v1 and v2 still respond. There is no fix for this; it "
                        "is a property of the released weights."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.05,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Controls the randomness of the sampling process. Higher "
                        "values produce more diverse speech, while lower values "
                        "are more deterministic."
                    ),
                ),
                io.Float.Input(
                    "cfg_weight",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Classifier-Free Guidance (CFG) weight. Controls how "
                        "strongly the model adheres to the text prompt. Higher "
                        "values may reduce naturalness."
                    ),
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=1.0,
                    max=5.0,
                    step=0.1,
                    tooltip=(
                        "Penalizes repeated tokens to discourage monotonous or "
                        "repetitive speech. A value of 1.0 means no penalty."
                    ),
                ),
                io.Float.Input(
                    "min_p",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Sets a minimum probability threshold for nucleus "
                        "sampling (Min-P). Filters out tokens with very low "
                        "probability."
                    ),
                ),
                io.Float.Input(
                    "top_p",
                    default=1.0,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Nucleus sampling (Top-P) parameter. The model samples "
                        "from the smallest set of tokens whose cumulative "
                        "probability exceeds this value."
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
                io.String.Input(
                    "crash_protection_template",
                    default=CRASH_PROTECTION_TEMPLATE,
                    tooltip=(
                        "Padding applied to any segment shorter than 21 "
                        "characters. Upstream Chatterbox fails on short text in "
                        "sequential generation; {seg} is replaced by the "
                        "segment. Set to '' to disable.\n\n"
                        "Presets: '...ummmmm {seg}' (hesitation), "
                        "'{seg}... yes... {seg}' (repetition), "
                        "'Well, {seg}' (natural prefix)."
                    ),
                ),
                io.Audio.Input(
                    "audio_prompt",
                    optional=True,
                    tooltip=(
                        "Reference audio to clone the voice from. 5-6 seconds of "
                        "clean speech is recommended. When left unconnected the "
                        "checkpoint's built-in voice is used."
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
        language: Optional[str] = None,
        model_version: Optional[str] = None,
        dtype: Optional[str] = None,
        attention_mode: Optional[str] = None,
    ) -> bool | str:
        """Queue-time checks. Returns a message to refuse the prompt, else ``True``.

        The language and version checks exist because both are silent failures
        otherwise: an unknown language raises deep inside the vendored tokenizer,
        and an unknown checkpoint filename only fails once the loader is looking
        for a file that does not exist.

        Args:
            model_pack_name: The pack the combo carried.
            language: ISO code from the language combo.
            model_version: Resolved T3 checkpoint filename.
            dtype: Dtype widget value.
            attention_mode: Attention widget value, as the user picked it.
        """
        if language is not None and language not in MTL_LANGUAGES:
            return (
                f"'{language}' is not one of Chatterbox's 23 languages. Pick a "
                f"code from the language dropdown."
            )
        if model_version is not None and model_version not in MTL_T3_MODELS.values():
            return (
                f"'{model_version}' is not a Chatterbox multilingual "
                f"checkpoint. Expected one of: "
                f"{', '.join(MTL_MODEL_VERSION_OPTIONS)}."
            )

        message = validate_common_inputs(model_pack_name, dtype, attention_mode)
        return True if message is None else message

    @classmethod
    def execute(
        cls,
        model_pack_name: str,
        model_version: str,
        language: str,
        text: str,
        exaggeration: float,
        temperature: float,
        cfg_weight: float,
        repetition_penalty: float,
        min_p: float,
        top_p: float,
        seed: int,
        use_watermark: bool,
        device: str = "auto",
        dtype: str = DTYPE_AUTO,
        attention_mode: str = "sdpa",
        crash_protection_template: str = CRASH_PROTECTION_TEMPLATE,
        audio_prompt: Optional[dict] = None,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Synthesise ``text`` in ``language`` and return it as ``AUDIO``.

        The vendored ``chatterbox`` package is never imported here: the model
        class is reached lazily through ``modules.loader``, so a frontend asking
        for ``/object_info`` before anything is loaded never drags the model
        stack in.
        """
        if not text or not text.strip():
            # Checked before the load: an empty box must not cost a 4 GB
            # download.
            logging.info("[Chatterbox TTS] Empty text; returning silence.")
            return io.NodeOutput(silent_audio())

        # Tags, then crash protection -- in that order, per segment.
        plan = plan_text(text, crash_protection_template=crash_protection_template)

        patcher, model = load_chatterbox_entry_point(
            model_pack_name,
            entry_point="multilingual",
            variant=model_version,
            family=FAMILY_MTL,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            client_id=client_id,
        )

        interrupt_error = interrupt_exception_class()
        try:
            waveform, sample_rate = generate_mtl_audio(
                model,
                plan.text,
                language,
                audio_prompt=audio_prompt,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                seed=seed,
                use_watermark=use_watermark,
            )
        except interrupt_error:
            # BaseException-derived: only this clause can catch a cancellation,
            # and a cancelled run is silence rather than a failure.
            logging.info(
                "[Chatterbox TTS] Multilingual synthesis was cancelled; "
                "returning silence."
            )
            return io.NodeOutput(silent_audio())

        audio = to_comfy_audio(waveform, sample_rate)
        logging.info(
            "[Chatterbox TTS] Pack '%s' (%s) spoke %d character(s) of %s at "
            "%d Hz (cache key %r).",
            model_pack_name,
            model_version,
            plan.char_count,
            language,
            sample_rate,
            getattr(patcher, "cache_key", None),
        )
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio))
