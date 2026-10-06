"""``ChatterboxVC`` -- the voice-conversion node, on the V3 schema.

The same compatibility rules as :mod:`nodes.tts_node` apply, and for the same
reason: ComfyUI binds a saved workflow's widget values by index, so the order of
the inputs below is transcribed from ``tests/frozen_widget_snapshot.json``
rather than chosen. ``device``, ``dtype`` and ``attention_mode`` are appended
after the frozen block and never before it.

``refinement_passes``, ``max_chunk_duration`` and ``chunk_method`` (S8.6) are
appended **after those three** and after the frozen block, never inside it. A
saved graph re-binds every stored value from the insertion point on, so the only
safe place for a new widget is the end. ``tests/test_node_schema.py`` and
``tests/test_node_compat.py`` pin the frozen prefix and the exact set of widgets
that follow it.

**The output socket is named ``converted_audio``, not ``audio``.** That name is
part of the frozen surface: a saved graph renames or highlights the output it
is linked to by its displayed name, and ``ChatterboxTTS`` already publishes a
socket called ``audio``. Renaming it would leave every existing VC link either
unresolved or silently pointed at the wrong socket. ``tests/test_node_schema.py``
and ``tests/test_workflow_compat.py`` both pin it.
"""

from __future__ import annotations

import logging
from typing import Optional

from comfy_api.latest import io, ui

from ..modules.audio_utils import S3GEN_SAMPLE_RATE, interrupt_exception_class, to_comfy_audio
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import (
    RefinementCache,
    combine_audio_chunks,
    convert_voice_audio,
    iterative_convert_voice,
    load_chatterbox_model,
    split_audio_for_conversion,
)
from ..modules.model_discovery import get_chatterbox_model_pack_names
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs

__all__ = ["ChatterboxVCNode", "CHUNK_METHODS", "CHUNK_METHOD_SMART", "VC_REFINEMENT_CACHE"]

#: The two ways ``max_chunk_duration`` may cut a source clip.
CHUNK_METHODS: tuple[str, ...] = ("smart", "fixed")

#: Cut near silence. The default, and the only one whose boundary a listener
#: does not hear.
CHUNK_METHOD_SMART = "smart"

#: Refinement iterations retained across a single run. See
#: :class:`modules.generation.RefinementCache` -- the cap is the documented
#: behaviour, not a memory budget.
VC_REFINEMENT_CACHE = RefinementCache()


_CATEGORY = "WMNodes/sound/vc"


class ChatterboxVCNode(io.ComfyNode):
    """Convert the voice in a speech clip while keeping its content."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema.

        As on the TTS node, the Combo options are resolved here rather than at
        import time, because the models directory can change while ComfyUI runs
        (a pack is downloaded by picking it in the dropdown).
        """
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxVC",
            display_name="Chatterbox Voice Conversion 🗣️",
            category=_CATEGORY,
            description=(
                "Convert recorded speech into a Chatterbox voice while keeping "
                "the words and the delivery."
            ),
            inputs=[
                # ── FROZEN BLOCK, indices 0-5. Never reorder or insert.
                io.Combo.Input(
                    "model_pack_name",
                    options=pack_names,
                    default=default_pack_name(pack_names),
                    tooltip=(
                        "Select the Chatterbox voice model pack to use for "
                        "conversion."
                    ),
                ),
                io.Audio.Input(
                    "source_audio",
                    tooltip=(
                        "The audio containing the speech content to be converted."
                    ),
                ),
                io.Int.Input(
                    "n_timesteps",
                    default=10,
                    min=2,
                    max=50,
                    step=1,
                    tooltip=(
                        "Number of diffusion steps for the flow matching "
                        "process. Higher values may improve quality at the cost "
                        "of speed."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls the randomness of the initial noise. 1.0 is "
                        "standard. Lower values are more deterministic."
                    ),
                ),
                io.Float.Input(
                    "flow_cfg_scale",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "CFG scale for the mel spectrogram decoder. Higher "
                        "values increase adherence to the target voice but may "
                        "reduce naturalness."
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
                io.Int.Input(
                    "refinement_passes",
                    default=1,
                    min=1,
                    max=30,
                    step=1,
                    tooltip=(
                        "Run the conversion this many times, feeding each "
                        "iteration's output back in as the next one's input.\n\n"
                        "Recommended maximum is 5; more passes cause audible "
                        "distortion. Each iteration is deterministic, so "
                        "raising this on a clip you already converted returns "
                        "the cached result rather than redoing the work -- the "
                        "cache keeps 5 iterations."
                    ),
                ),
                io.Float.Input(
                    "max_chunk_duration",
                    default=30.0,
                    min=0.0,
                    max=300.0,
                    step=5.0,
                    tooltip=(
                        "Split source audio longer than this many seconds before "
                        "converting it, and join the results afterwards. Bounds "
                        "the peak memory of one conversion, which is what "
                        "overflows on a long recording.\n\n"
                        "0 disables chunking and converts the clip in one pass. "
                        "5 seconds is the smallest useful value."
                    ),
                ),
                io.Combo.Input(
                    "chunk_method",
                    options=list(CHUNK_METHODS),
                    default=CHUNK_METHOD_SMART,
                    tooltip=(
                        "Where to cut when 'max_chunk_duration' splits the "
                        "source.\n\n"
                        "smart: cut as close to silence as possible near the "
                        "ideal boundary. A boundary in a pause is inaudible.\n"
                        "fixed: cut at exact intervals. Cheaper, and the joins "
                        "are more audible."
                    ),
                ),
                io.Audio.Input(
                    "target_voice_audio",
                    optional=True,
                    tooltip=(
                        "The audio file containing the target voice timbre. If "
                        "not provided, the default voice from the model pack "
                        "will be used."
                    ),
                ),
            ],
            outputs=[io.Audio.Output(display_name="converted_audio")],
            # Same correction as the TTS node: a real socket, so the node is
            # not a terminal.
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
        source_audio: dict,
        n_timesteps: int,
        temperature: float,
        flow_cfg_scale: float,
        use_watermark: bool,
        device: str = "auto",
        dtype: str = DTYPE_AUTO,
        attention_mode: str = "sdpa",
        refinement_passes: int = 1,
        max_chunk_duration: float = 0.0,
        chunk_method: str = CHUNK_METHOD_SMART,
        target_voice_audio: Optional[dict] = None,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Convert ``source_audio`` into the target voice and return it as ``AUDIO``.

        The vendored ``chatterbox`` package is never imported here; the model
        classes are reached lazily by ``modules.loader``.

        Three optional behaviours, in the order they apply:

        1. **Chunking** (``max_chunk_duration``) cuts the *source* before
           anything else, bounds the peak memory of one conversion, and joins
           the results afterwards. ``0`` skips it entirely and the run takes
           exactly the single-pass path it always did.
        2. **Refinement** (``refinement_passes``) re-converts, feeding each
           iteration's output back in. Results are cached, so asking for more
           passes than have already been computed costs only the difference.
        3. **Chunking again** is not applied to the refined result -- refining a
           chunk and then joining is the same as joining and refining, except
           that the second order re-runs every chunk on every pass. The source
           is cut once, each chunk is refined to completion, and the chunks are
           joined.

        Note the asymmetry with :meth:`nodes.tts_node.ChatterboxTTSNode.execute`:
        blank *text* is silence there, but unusable *audio* raises here. A
        disconnected input cannot reach this node -- ``source_audio`` is a
        required socket -- so empty audio is a genuine fault in the graph, not
        an unfinished one, and swallowing it would hide a mis-wired converter.
        """
        patcher, _tts_model, vc_model = load_chatterbox_model(
            model_pack_name,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            client_id=client_id,
        )
        if vc_model is None:
            raise ValueError(
                f"The Chatterbox voice pack '{model_pack_name}' loaded without "
                f"a conversion model. Use the Chatterbox TTS node for it."
            )

        interrupt_error = interrupt_exception_class()
        try:
            waveform, sample_rate = cls._convert_with_optional_chunking(
                vc_model,
                source_audio,
                target_voice_audio=target_voice_audio,
                n_timesteps=n_timesteps,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale,
                use_watermark=use_watermark,
                refinement_passes=refinement_passes,
                max_chunk_duration=max_chunk_duration,
                chunk_method=chunk_method,
                cache_key=getattr(patcher, "cache_key", None),
            )
        except interrupt_error:
            # BaseException-derived: only this clause can catch a cancellation,
            # and a cancelled run is silence rather than a failure.
            logging.info(
                "[Chatterbox TTS] Voice conversion was cancelled; returning silence."
            )
            return io.NodeOutput(silent_audio())

        audio = to_comfy_audio(waveform, sample_rate)
        logging.info(
            "[Chatterbox TTS] Voice pack '%s' converted a clip to %d Hz audio "
            "(cache key %r).",
            model_pack_name,
            sample_rate,
            getattr(patcher, "cache_key", None),
        )
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio))

    @staticmethod
    def _convert_with_optional_chunking(
        vc_model: object,
        source_audio: dict,
        *,
        target_voice_audio: Optional[dict],
        n_timesteps: int,
        temperature: float,
        flow_cfg_scale: float,
        use_watermark: bool,
        refinement_passes: int,
        max_chunk_duration: float,
        chunk_method: str,
        cache_key: Optional[str],
    ) -> tuple:
        """Run the conversion, chunking and refining where asked.

        Split out of :meth:`execute` so the ordering above is one readable block
        rather than five interleaved branches in the middle of a prompt
        handler. Returns ``(waveform, sample_rate)``.
        """
        if not max_chunk_duration or float(max_chunk_duration) <= 0:
            return iterative_convert_voice(
                vc_model,
                source_audio,
                target_voice_audio=target_voice_audio,
                n_timesteps=n_timesteps,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale,
                use_watermark=use_watermark,
                refinement_passes=refinement_passes,
                cache=VC_REFINEMENT_CACHE,
                cache_key=cache_key,
                single_pass=convert_voice_audio,
            )

        waveform, sample_rate = _source_waveform(source_audio)
        pieces = split_audio_for_conversion(
            waveform, sample_rate, max_chunk_duration, chunk_method
        )
        if len(pieces) == 1:
            return iterative_convert_voice(
                vc_model,
                source_audio,
                target_voice_audio=target_voice_audio,
                n_timesteps=n_timesteps,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale,
                use_watermark=use_watermark,
                refinement_passes=refinement_passes,
                cache=VC_REFINEMENT_CACHE,
                cache_key=cache_key,
                single_pass=convert_voice_audio,
            )

        logging.info(
            "[Chatterbox TTS] Source audio is %.1f s; splitting into %d chunks "
            "of at most %.1f s (%s).",
            waveform.shape[-1] / float(sample_rate),
            len(pieces),
            float(max_chunk_duration),
            chunk_method,
        )

        converted = []
        for position, piece in enumerate(pieces):
            converted.append(
                iterative_convert_voice(
                    vc_model,
                    {"waveform": piece.unsqueeze(0), "sample_rate": sample_rate},
                    target_voice_audio=target_voice_audio,
                    n_timesteps=n_timesteps,
                    temperature=temperature,
                    flow_cfg_scale=flow_cfg_scale,
                    use_watermark=use_watermark,
                    refinement_passes=refinement_passes,
                    cache=VC_REFINEMENT_CACHE,
                    cache_key=f"{cache_key or 'adhoc'}#chunk{position}",
                    single_pass=convert_voice_audio,
                )[0]
            )

        joined = combine_audio_chunks(converted, sample_rate, method="concatenate")
        return joined, sample_rate


def _source_waveform(source_audio: dict) -> tuple:
    """``(waveform[1, T], sample_rate)`` from a ComfyUI ``AUDIO`` dict.

    Raises:
        ValueError: The dict is missing, or carries no samples. Same contract as
            :func:`modules.generation.convert_voice_audio`, so the two paths
            refuse the same input the same way.
    """
    if (
        not source_audio
        or source_audio.get("waveform") is None
        or source_audio["waveform"].numel() == 0
    ):
        raise ValueError(
            "Chatterbox voice conversion needs source audio. Connect an AUDIO "
            "input carrying the speech to convert."
        )
    waveform = source_audio["waveform"]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, int(source_audio.get("sample_rate") or S3GEN_SAMPLE_RATE)