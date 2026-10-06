"""``ChatterboxDialog`` -- a two-to-four voice conversation in one node.

**New node id, additive only.** It sits beside ``ChatterboxTTS`` and shares no
index with it; see :mod:`nodes.tts_node` for why a saved graph binds widgets by
index.

Upstream Chatterbox has no dialog capability of any kind. A conversation here is
therefore **four independent syntheses** whose only connection is the text
between them, and every design decision below follows from that: one turn is
one ``generate`` call with one reference voice, the turns are laid out back to
back so no two share a sample, and the "dialog" track is computed *by summing
the stems* rather than assembled by a second code path that could drift from
them.

**Six outputs, five of them isolated.** ``dialog_audio`` is the conversation.
``speaker_a_audio`` through ``speaker_d_audio`` contain **only** that speaker's
turns -- not a downmix, not the other voices quietly mixed in at low level. C
and D are optional inputs and still get a stem: real silence of the dialog's
full length rather than an empty dict, because downstream nodes chain AUDIO
sockets without checking and a stem that changes length when it is left
unconnected is a graph that only works on the machine it was built on.

``message`` is a plain-text summary of who spoke and for how long. A dialog
that renders badly is diagnosed by which turn ran long, and the node can simply
report that rather than leaving the user to work it out from four stems.

The multilingual entry point is used, because it is the family with per-speaker
conditioning already in place -- the same ``audio_prompt``-per-call mechanism the
MTL node exposes, applied once per turn. ``model_version`` and ``language``
reuse the MTL node's constants rather than restating them: two lists of 23
language codes that could disagree is a bug waiting for someone to fix one of
them.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from comfy_api.latest import io, ui

from ..modules.audio_utils import interrupt_exception_class, to_comfy_audio
from ..modules.dialog import (
    REQUIRED_SPEAKERS,
    SPEAKER_SLOTS,
    assemble_stems,
    build_message,
    parse_dialog,
)
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import (
    FAMILY_MTL,
    generate_mtl_audio,
    load_chatterbox_entry_point,
)
from ..modules.constants import MTL_LANGUAGES, MTL_T3_MODELS
from ..modules.model_discovery import get_chatterbox_model_pack_names
from ..modules.text_processing import CRASH_PROTECTION_TEMPLATE, plan_text
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs
from .mtl_node import MTL_MODEL_VERSION_DEFAULT, MTL_MODEL_VERSION_OPTIONS

__all__ = ["ChatterboxDialogNode"]


_CATEGORY = "WMNodes/sound/tts"


class ChatterboxDialogNode(io.ComfyNode):
    """Speak a ``SPEAKER A:`` / ``SPEAKER B:`` script in separate voices."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema. Combo options are resolved here, not at import."""
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxDialog",
            display_name="Chatterbox Multi-Speaker Dialog 🎭",
            category=_CATEGORY,
            description=(
                "Speak a SPEAKER A: / SPEAKER B: conversation with a separate "
                "voice per speaker, returning the mixed dialog plus each "
                "speaker's isolated stem."
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
                        "are the resolved filenames, not labels; the default is "
                        "v3. One model serves all four speakers here -- the "
                        "voices come from the reference clips, not from "
                        "separate checkpoints."
                    ),
                ),
                io.Combo.Input(
                    "language",
                    options=sorted(MTL_LANGUAGES),
                    default="en",
                    tooltip=(
                        "Language of the whole conversation, as an ISO 639-1 "
                        "code. Each speaker uses the same one: Chatterbox has no "
                        "per-speaker language setting, and mixing them inside "
                        "one model call is not something this node can do."
                    ),
                ),
                io.String.Input(
                    "dialog_text",
                    multiline=True,
                    default="SPEAKER A: Test test\nSPEAKER B: 1 2 3",
                    tooltip=(
                        "The conversation, one turn per line:\n"
                        "SPEAKER A: Good evening.\n"
                        "SPEAKER B: Good evening to you.\n\n"
                        "Speakers A to D are recognised. A line that does not "
                        "start with a speaker prefix continues the turn above "
                        "it, so a wrapped line is still one turn.\n\n"
                        "Character tags ([Alice]), language tags ([de:Alice]), "
                        "pause tags ([pause:1.5]) and the v2 angle-bracket "
                        "tokens work inside a turn."
                    ),
                ),
                io.Audio.Input(
                    "speaker_A_Audio",
                    tooltip=(
                        "Speaker A's voice. 5-6 seconds of clean speech. "
                        "Required."
                    ),
                ),
                io.Audio.Input(
                    "speaker_B_Audio",
                    tooltip="Speaker B's voice. Required.",
                ),
                io.Float.Input(
                    "exaggeration",
                    default=0.5,
                    min=0.25,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls the expressiveness and emotional intensity, "
                        "applied to every speaker equally."
                    ),
                ),
                io.Float.Input(
                    "cfg_weight",
                    default=0.5,
                    min=0.2,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Classifier-Free Guidance (CFG) weight. Higher values "
                        "may reduce naturalness."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.05,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Controls the randomness of the sampling process."
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
                        "use a random seed. Every turn is seeded from this one "
                        "plus its own index, so one turn re-rolling does not "
                        "change the others."
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
                    "speaker_C_Audio",
                    optional=True,
                    tooltip=(
                        "Speaker C's voice. Leave unconnected if the "
                        "conversation has no third speaker; speaker_c_audio is "
                        "then silence of the dialog's length."
                    ),
                ),
                io.Audio.Input(
                    "speaker_D_Audio",
                    optional=True,
                    tooltip=(
                        "Speaker D's voice. Leave unconnected if the "
                        "conversation has no fourth speaker; speaker_d_audio is "
                        "then silence of the dialog's length."
                    ),
                ),
                io.String.Input(
                    "crash_protection_template",
                    default=CRASH_PROTECTION_TEMPLATE,
                    tooltip=(
                        "Padding applied to any turn shorter than 21 "
                        "characters. Conversational lines are short by nature "
                        "and upstream Chatterbox fails on short text in "
                        "sequential generation. {seg} is replaced by the turn's "
                        "text. Set to '' to disable."
                    ),
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="dialog_audio"),
                io.Audio.Output(display_name="speaker_a_audio"),
                io.Audio.Output(display_name="speaker_b_audio"),
                io.Audio.Output(display_name="speaker_c_audio"),
                io.Audio.Output(display_name="speaker_d_audio"),
                io.String.Output(display_name="message"),
            ],
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
        dialog_text: str,
        speaker_A_Audio: dict,
        speaker_B_Audio: dict,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: int = 0,
        use_watermark: bool = False,
        device: str = "auto",
        dtype: str = DTYPE_AUTO,
        attention_mode: str = "sdpa",
        speaker_C_Audio: Optional[dict] = None,
        speaker_D_Audio: Optional[dict] = None,
        crash_protection_template: str = CRASH_PROTECTION_TEMPLATE,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Speak the conversation and split it into isolated per-speaker stems.

        Returns:
            ``NodeOutput(dialog_audio, speaker_a_audio, speaker_b_audio,
            speaker_c_audio, speaker_d_audio, message)``.

        Raises:
            ValueError: The script has no turns, names a speaker with no
                reference clip, or has text before the first ``SPEAKER X:``.
        """
        turns = parse_dialog(dialog_text)
        if not turns:
            logging.info("[Chatterbox TTS] Empty dialog text; returning silence.")
            silence = silent_audio()
            return io.NodeOutput(
                silence, silence, silence, silence, silence, "turns: 0"
            )

        references = {
            "A": speaker_A_Audio,
            "B": speaker_B_Audio,
            "C": speaker_C_Audio,
            "D": speaker_D_Audio,
        }
        for slot in REQUIRED_SPEAKERS:
            if not references.get(slot):
                raise ValueError(
                    f"Speaker {slot} needs a reference voice. Connect "
                    f"speaker_{slot}_Audio; a conversation cannot be spoken "
                    f"without one."
                )
        used = {turn.speaker for turn in turns}
        for slot in sorted(used):
            if slot not in SPEAKER_SLOTS:
                raise ValueError(
                    f"'{slot}' is not a speaker this node knows. Expected one "
                    f"of: {', '.join(SPEAKER_SLOTS)}."
                )
            if not references.get(slot):
                raise ValueError(
                    f"Speaker {slot} has lines but no reference audio "
                    f"(speaker_{slot}_Audio). Connect one, or remove those "
                    f"lines from the dialog."
                )

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
        clips: list[torch.Tensor] = []
        try:
            for turn in turns:
                # Seeded from the widget plus the turn's own index: rolling the
                # seed must not re-roll every line in the conversation, and two
                # turns sharing a seed would sample two different speakers from
                # the same noise, which is the one thing that makes a dialog
                # sound like one person reading two parts.
                plan = plan_text(
                    turn.text, crash_protection_template=crash_protection_template
                )
                waveform, sample_rate = generate_mtl_audio(
                    model,
                    plan.text,
                    language,
                    audio_prompt=references[turn.speaker],
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=(int(seed) + turn.index + 1) if seed else 0,
                    use_watermark=use_watermark,
                )
                clips.append(waveform)
        except interrupt_error:
            logging.info(
                "[Chatterbox TTS] Dialog synthesis was cancelled; returning silence."
            )
            silence = silent_audio()
            return io.NodeOutput(silence, silence, silence, silence, silence, "")

        stems, dialog = assemble_stems(turns, clips, sample_rate=sample_rate)

        outputs = [to_comfy_audio(dialog, sample_rate)]
        for slot in SPEAKER_SLOTS:
            outputs.append(to_comfy_audio(stems[slot], sample_rate))
        message = build_message(turns, stems, sample_rate=sample_rate)

        logging.info(
            "[Chatterbox TTS] Pack '%s' spoke %d turn(s) across %d speaker(s) "
            "(cache key %r).",
            model_pack_name,
            len(turns),
            len(used),
            getattr(patcher, "cache_key", None),
        )
        return io.NodeOutput(*outputs, message, ui=ui.PreviewAudio(outputs[0]))