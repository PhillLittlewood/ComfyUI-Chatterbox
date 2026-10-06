"""``ChatterboxTTS`` -- the speech-synthesis node, on the V3 schema.

**The input order is a compatibility contract, not a style choice.** ComfyUI
binds a saved workflow's widget values *by index*: nothing in the protocol
records which widget a stored value belonged to. Reorder, rename or insert an
input and every existing graph keeps loading -- it just binds the wrong values
(``use_watermark`` lands on ``seed``, ``text`` lands on ``max_new_tokens``), and
nothing complains until the audio comes out wrong.

The block below marked ``FROZEN`` is therefore transcribed from
``tests/frozen_widget_snapshot.json``, captured from the V1 implementation
before ``nodes.py`` was deleted. ``device``, ``dtype`` and ``attention_mode``
are the only additions, and they are **appended after every frozen widget**
rather than grouped with their kin, precisely so that no frozen index moves.
``tests/test_node_schema.py`` asserts exact tuple equality against the
snapshot; ``tests/test_node_compat.py`` asserts the appendage is exactly those
three, in that order, and nothing else.

Two more frozen details:

* ``seed`` carries ``control_after_generate``. ComfyUI emits an extra companion
  widget right after any input with that flag, which is why the shipped example
  workflow stores 13 values for 12 required inputs: index 11 is ``"randomize"``
  and index 12 is ``use_watermark``.
* ``text`` is a multiline ``STRING``; the script a user pastes in is multi-line
  by nature.

``OUTPUT_NODE`` is ``False``. V1 set it ``True`` while also returning a real
socket, which made ComfyUI treat the node as a terminal: it executed on its own
even with nothing downstream, and its UI result was cached rather than passed
along. Returning a socket is the honest description of what this node does.
This is a behaviour flag, not a widget, so it does not disturb the frozen
binding.
"""

from __future__ import annotations

import logging
from typing import Optional

from comfy_api.latest import io, ui

from ..modules.audio_utils import interrupt_exception_class, to_comfy_audio
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import generate_audio, load_chatterbox_model
from ..modules.model_discovery import get_chatterbox_model_pack_names
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs

__all__ = ["ChatterboxTTSNode"]


_CATEGORY = "WMNodes/sound/tts"


class ChatterboxTTSNode(io.ComfyNode):
    """Synthesise speech from text with a Chatterbox voice pack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema.

        Called by the ComfyUI loader, by the frontend's ``/object_info`` request
        and by every test in this repository, so the Combo options are computed
        here -- they have to reflect the models directory *now*, not at import
        time.
        """
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxTTS",
            display_name="Chatterbox TTS 📢",
            category=_CATEGORY,
            description=(
                "Synthesise speech from text with a Chatterbox voice pack. "
                "Optionally clone a voice from a reference clip."
            ),
            inputs=[
                # ── FROZEN BLOCK, indices 0-11. Never reorder, never insert,
                #    never rename. New widgets go after this block.
                io.Combo.Input(
                    "model_pack_name",
                    options=pack_names,
                    default=default_pack_name(pack_names),
                    tooltip=(
                        "Select the Chatterbox voice model pack to use for "
                        "generation."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="Hello, this is a test of Chatterbox TTS in ComfyUI.",
                    tooltip="Text to be synthesized into speech.",
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=1000,
                    min=16,
                    max=4000,
                    step=8,
                    tooltip=(
                        "Maximum number of audio tokens to generate. 25 tokens "
                        "≈ 1 second. The hard limit is 4096 tokens "
                        "(≈ 163 seconds)."
                    ),
                ),
                io.Float.Input(
                    "flow_cfg_scale",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "CFG scale for the mel spectrogram decoder (flow "
                        "matching). Higher values increase adherence to content "
                        "and timbre but may reduce naturalness."
                    ),
                ),
                io.Float.Input(
                    "exaggeration",
                    default=0.5,
                    min=0.25,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls the expressiveness and emotional intensity. "
                        "Higher values lead to more exaggerated prosody."
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
                    min=0.2,
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
                    max=2.0,
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
                    min=0.0,
                    max=1.0,
                    step=0.01,
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
                    # One flagged input is what makes the shipped workflow's 13
                    # widget values out of 12 required inputs. Dropping it would
                    # shift every index after `seed` by one.
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
                # ── END FROZEN BLOCK ──
                # Appended in this fixed order, after every frozen widget.
                *runtime_widgets(),
                io.Audio.Input(
                    "audio_prompt",
                    optional=True,
                    tooltip=(
                        "Reference audio to clone the voice from. When left "
                        "unconnected the pack's built-in voice is used."
                    ),
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
            # V1 set this True while returning a real socket, which made ComfyUI
            # treat the node as a terminal. See the module docstring.
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
        max_new_tokens: int,
        flow_cfg_scale: float,
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
        audio_prompt: Optional[dict] = None,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Synthesise ``text`` and return it as a ComfyUI ``AUDIO`` socket.

        The vendored ``chatterbox`` package is never imported here: the model
        classes are reached lazily by ``modules.loader``, so importing this
        module on a machine without the vendor installed -- or a ComfyUI
        frontend asking for ``/object_info`` before anything is loaded -- never
        drags in torch weights.
        """
        if not text or not text.strip():
            # The one legitimate silence. A graph wired up before it is
            # finished runs with the text box still empty, and a node that
            # raises there turns every intermediate state into a red prompt.
            # Checked before the load so an empty box never costs a 2.7 GB
            # download.
            logging.info("[Chatterbox TTS] Empty text; returning silence.")
            return io.NodeOutput(silent_audio())

        patcher, tts_model, _vc_model = load_chatterbox_model(
            model_pack_name,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            client_id=client_id,
        )
        if tts_model is None:
            raise ValueError(
                f"The Chatterbox voice pack '{model_pack_name}' loaded without "
                f"a synthesis model. Use the Chatterbox Voice Conversion node "
                f"for a conversion-only pack."
            )

        interrupt_error = interrupt_exception_class()
        try:
            waveform, sample_rate = generate_audio(
                tts_model,
                text,
                audio_prompt=audio_prompt,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens,
                flow_cfg_scale=flow_cfg_scale,
                use_watermark=use_watermark,
            )
        except interrupt_error:
            # InterruptProcessingException derives from BaseException, so this
            # clause is the only one that can catch it. A cancelled run is not
            # a failed run: silence, and no traceback.
            logging.info("[Chatterbox TTS] Synthesis was cancelled; returning silence.")
            return io.NodeOutput(silent_audio())

        audio = to_comfy_audio(waveform, sample_rate)
        logging.info(
            "[Chatterbox TTS] Pack '%s' produced %d Hz audio (cache key %r).",
            model_pack_name,
            sample_rate,
            getattr(patcher, "cache_key", None),
        )
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio))