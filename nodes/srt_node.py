"""``ChatterboxSRT`` -- synthesise a script that arrives with its own timings.

**New node id, additive only.** It shares no index with ``ChatterboxTTS`` and
touches nothing in ``tests/frozen_widget_snapshot.json``; see the module
docstring of :mod:`nodes.tts_node` for why a saved graph binds widgets by index.

An SRT file is a *timed* script. This node reads the timings, speaks each cue,
and then has to reconcile what the model actually produced with what the file
said -- which is the whole job, and it is done in
:mod:`modules.subtitles` rather than here. This module is the surface: the
widgets, the four outputs, and the one place the per-cue synthesis happens.

**Four outputs, because the three interesting results are not the audio.**
``audio`` is the laid-out track. ``Adjusted_SRT`` is the subtitle file that
actually matches it -- which is *not* the input file except in ``concatenate``
mode, and is the reason this node exists. ``timing_report`` says, cue by cue,
what was asked for and what happened, including whether the overlap fallback
fired. ``generation_info`` carries the pack and cache key so a run can be traced
back to the weights that made it.

**Four timing modes, and the default is ``smart_natural``.** The corpus is
explicit that the other three each sacrifice something: ``stretch_to_fit``
time-distorts the voice to obey the file, ``pad_with_silence`` may let two cues
overlap, and ``concatenate`` throws the file away. ``smart_natural`` is the only
one that keeps the voice natural *and* guarantees no cue overlaps -- by
detecting the overlap and falling back, which the timing report records rather
than hides.

**``use_native_duration_targeting`` is deliberately absent.** The reference
documentation scopes that widget to engines that expose a native duration
parameter (OmniVoice, DramaBox). Chatterbox has no such parameter, so the
socket would be a control with no effect, which is worse than no control.

Per-cue synthesis goes through :func:`modules.generation.generate_audio` -- the
classic English entry point -- with the tag engine and crash protection applied
to each cue's text first, so a two-word subtitle is padded the same way a
two-word line in the main text box is.
"""

from __future__ import annotations

import logging
from typing import Optional

from comfy_api.latest import io, ui

from ..modules.audio_utils import interrupt_exception_class, to_comfy_audio
from ..modules.dtype_utils import DTYPE_AUTO
from ..modules.generation import generate_audio, load_chatterbox_model
from ..modules.model_discovery import get_chatterbox_model_pack_names
from ..modules.subtitles import (
    DEFAULT_FADE_FOR_STRETCH_TO_FIT,
    DEFAULT_MAX_STRETCH_RATIO,
    DEFAULT_MIN_STRETCH_RATIO,
    DEFAULT_TIMING_TOLERANCE,
    TIMING_MODE_DEFAULT,
    TIMING_MODES,
    SpokenCue,
    apply_timing_mode,
    build_generation_info,
    build_timing_report,
    parse_srt,
    render_srt,
)
from ..modules.text_processing import CRASH_PROTECTION_TEMPLATE, plan_text
from .common import default_pack_name, runtime_widgets, silent_audio, validate_common_inputs

__all__ = ["ChatterboxSRTNode", "SRT_TIMING_MODES", "SRT_TIMING_MODE_DEFAULT"]


#: Re-exported so tests and callers have one place to read the enum from.
SRT_TIMING_MODES: tuple[str, ...] = TIMING_MODES
SRT_TIMING_MODE_DEFAULT: str = TIMING_MODE_DEFAULT

#: A three-cue demo, so a user who drops the node into a graph immediately sees
#: the output shaped like a real subtitle file.
_DEMO_SRT = (
    "1\n"
    "00:00:00,000 --> 00:00:03,000\n"
    "The first line of the subtitle file.\n"
    "\n"
    "2\n"
    "00:00:03,000 --> 00:00:06,500\n"
    "The second line, three and a half seconds long.\n"
    "\n"
    "3\n"
    "00:00:06,500 --> 00:00:09,000\n"
    "And a third to finish with.\n"
)


_CATEGORY = "WMNodes/sound/util"


class ChatterboxSRTNode(io.ComfyNode):
    """Speak a timed script and emit a subtitle file that matches the audio."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Build the schema. Combo options are resolved here, not at import."""
        pack_names = list(get_chatterbox_model_pack_names())

        return io.Schema(
            node_id="ChatterboxSRT",
            display_name="Chatterbox SRT / Subtitles 📝",
            category=_CATEGORY,
            description=(
                "Speak a timed SRT script and reconcile the model's actual "
                "timings with the file's, returning an adjusted subtitle file "
                "that matches the audio."
            ),
            inputs=[
                io.Combo.Input(
                    "model_pack_name",
                    options=pack_names,
                    default=default_pack_name(pack_names),
                    tooltip=(
                        "Select the Chatterbox checkpoint directory to load."
                    ),
                ),
                io.String.Input(
                    "srt_content",
                    multiline=True,
                    default=_DEMO_SRT,
                    tooltip=(
                        "The subtitle file to speak. Standard SRT, with "
                        "'HH:MM:SS,mmm' timestamps.\n\n"
                        "Each cue is synthesised separately, so a cue's length "
                        "is whatever the model said, not what the file "
                        "promised -- which is the problem timing_mode exists to "
                        "solve.\n\n"
                        "Character tags ([Alice]), language tags ([de:Alice]), "
                        "pause tags ([pause:1.5]) and the v2 angle-bracket "
                        "tokens all work inside a cue, exactly as they do on "
                        "the main TTS node."
                    ),
                ),
                io.Combo.Input(
                    "timing_mode",
                    options=list(TIMING_MODES),
                    default=TIMING_MODE_DEFAULT,
                    tooltip=(
                        "How the model's timings are reconciled with the "
                        "file's.\n\n"
                        "stretch_to_fit - stretches/compresses the audio to "
                        "exactly match each SRT segment's duration. Obeys the "
                        "file; distorts the voice.\n\n"
                        "pad_with_silence - natural audio at the SRT start "
                        "times, gaps padded with silence. May result in "
                        "overlaps.\n\n"
                        "smart_natural (default) - keeps the audio natural, "
                        "shifts later segments when a line runs long, and "
                        "applies a bounded stretch only when the drift is "
                        "bigger than timing_tolerance. Never emits overlapping "
                        "cues: if honouring the timings would overlap, the "
                        "track is re-laid out and the fallback is recorded in "
                        "timing_report.\n\n"
                        "concatenate - ignores the original timings entirely, "
                        "speaks the cues back to back, and writes a new SRT "
                        "from the actual timings."
                    ),
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=1000,
                    min=1,
                    max=5000,
                    step=1,
                    tooltip=(
                        "Token budget per cue. 25 tokens is roughly one "
                        "second, so a cue written long needs more than the "
                        "default."
                    ),
                ),
                io.Float.Input(
                    "flow_cfg_scale",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Flow-matching CFG scale for the mel decoder."
                    ),
                ),
                io.Float.Input(
                    "exaggeration",
                    default=0.5,
                    min=0.25,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls the expressiveness and emotional intensity."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.05,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Controls the randomness of the sampling process. "
                        "Lower values are more deterministic, which matters "
                        "here: a cue whose length changes run to run changes "
                        "the subtitle file too."
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
                io.Float.Input(
                    "timing_tolerance",
                    default=DEFAULT_TIMING_TOLERANCE,
                    min=0.5,
                    max=10.0,
                    step=0.5,
                    tooltip=(
                        "How far a cue's realised length may miss its SRT slot "
                        "before a bounded stretch is applied, as a multiple of "
                        "that slot. 2.0 lets a line run to three times its "
                        "written length untouched; lower it toward 0.5 to keep "
                        "the subtitles tight."
                    ),
                ),
                io.Float.Input(
                    "max_stretch_ratio",
                    default=DEFAULT_MAX_STRETCH_RATIO,
                    min=0.5,
                    max=5.0,
                    step=0.1,
                    tooltip=(
                        "Upper bound on the time-stretch factor. The default of "
                        "1.0 never makes a cue longer -- only compression is "
                        "enabled out of the box, because slowing speech down is "
                        "the adjustment a listener always notices. Raise it "
                        "above 1.0 to allow expansion."
                    ),
                ),
                io.Float.Input(
                    "min_stretch_ratio",
                    default=DEFAULT_MIN_STRETCH_RATIO,
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    tooltip=(
                        "Lower bound on the time-stretch factor. 0.5 is twice "
                        "as fast as the original, about as fast as a "
                        "synthesised voice stays intelligible."
                    ),
                ),
                io.Float.Input(
                    "fade_for_StretchToFit",
                    default=DEFAULT_FADE_FOR_STRETCH_TO_FIT,
                    min=0.0,
                    max=0.1,
                    step=0.001,
                    tooltip=(
                        "Head/tail fade applied to a stretched clip, as a "
                        "fraction of its length. A resampled join is a step "
                        "discontinuity, and a step is a click."
                    ),
                ),
                # Appended after every model-specific widget, in the shared order.
                *runtime_widgets(),
                io.String.Input(
                    "crash_protection_template",
                    default=CRASH_PROTECTION_TEMPLATE,
                    tooltip=(
                        "Padding applied to any cue shorter than 21 characters. "
                        "Subtitle cues are short by nature -- 'Yes.' is a cue -- "
                        "and upstream Chatterbox fails on short text in "
                        "sequential generation. {seg} is replaced by the cue's "
                        "text. Set to '' to disable."
                    ),
                ),
                io.Audio.Input(
                    "audio_prompt",
                    optional=True,
                    tooltip=(
                        "Reference audio to clone the narrator's voice from. "
                        "When left unconnected the checkpoint's built-in voice "
                        "is used."
                    ),
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="generation_info"),
                io.String.Output(display_name="timing_report"),
                io.String.Output(display_name="Adjusted_SRT"),
            ],
            is_output_node=False,
        )

    @classmethod
    def validate_inputs(
        cls,
        model_pack_name: Optional[str] = None,
        timing_mode: Optional[str] = None,
        dtype: Optional[str] = None,
        attention_mode: Optional[str] = None,
    ) -> bool | str:
        """Queue-time checks. Returns a message to refuse the prompt, else ``True``.

        Args:
            model_pack_name: The pack the combo carried.
            timing_mode: The timing strategy the combo carried.
            dtype: Dtype widget value.
            attention_mode: Attention widget value, as the user picked it.
        """
        if timing_mode is not None and timing_mode not in TIMING_MODES:
            return (
                f"'{timing_mode}' is not a timing mode. Expected one of: "
                f"{', '.join(TIMING_MODES)}."
            )

        message = validate_common_inputs(model_pack_name, dtype, attention_mode)
        return True if message is None else message

    @classmethod
    def execute(
        cls,
        model_pack_name: str,
        srt_content: str,
        timing_mode: str = TIMING_MODE_DEFAULT,
        max_new_tokens: int = 1000,
        flow_cfg_scale: float = 0.7,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        seed: int = 0,
        use_watermark: bool = False,
        timing_tolerance: float = DEFAULT_TIMING_TOLERANCE,
        max_stretch_ratio: float = DEFAULT_MAX_STRETCH_RATIO,
        min_stretch_ratio: float = DEFAULT_MIN_STRETCH_RATIO,
        fade_for_StretchToFit: float = DEFAULT_FADE_FOR_STRETCH_TO_FIT,
        device: str = "auto",
        dtype: str = DTYPE_AUTO,
        attention_mode: str = "sdpa",
        crash_protection_template: str = CRASH_PROTECTION_TEMPLATE,
        audio_prompt: Optional[dict] = None,
        client_id: Optional[str] = None,
    ) -> io.NodeOutput:
        """Speak every cue and reconcile its length with the SRT slot.

        Returns:
            ``NodeOutput(audio, generation_info, timing_report,
            Adjusted_SRT)``. A blank script returns one second of silence and
            the three empty strings, rather than a failure: an empty text box
            is not an error on any other Chatterbox node either.
        """
        cues = parse_srt(srt_content)
        if not cues:
            # Checked before the load: an empty box must not cost a 4 GB
            # download. A non-empty box that parses to nothing is a different
            # thing and raises below.
            logging.info("[Chatterbox TTS] Empty SRT content; returning silence.")
            return io.NodeOutput(
                silent_audio(),
                "",
                "timing_mode: " + timing_mode + "\ncues: 0\ntotal: 1.000 s",
                "",
            )

        spoken = [cue for cue in cues if cue.text.strip()]
        if not spoken:
            raise ValueError(
                f"{len(cues)} cue(s) were found in the SRT content, but none of "
                "them has any text. Every cue needs a line under its timestamp."
            )

        patcher, model, _ = load_chatterbox_model(
            model_pack_name,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            client_id=client_id,
        )

        interrupt_error = interrupt_exception_class()
        clips: list[SpokenCue] = []
        try:
            for cue in spoken:
                # Tags first, then crash protection, per cue -- a subtitle cue
                # is short by construction, which is exactly the input upstream
                # segfaults on.
                plan = plan_text(cue.text, crash_protection_template=crash_protection_template)
                waveform, sample_rate = generate_audio(
                    model,
                    plan.text,
                    audio_prompt=audio_prompt,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    repetition_penalty=1.2,
                    min_p=0.05,
                    top_p=1.0,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    flow_cfg_scale=flow_cfg_scale,
                    use_watermark=use_watermark,
                )
                clips.append(
                    SpokenCue(
                        text=cue.text,
                        waveform=waveform[0],
                        slot_start=cue.start,
                        slot_end=cue.end,
                    )
                )
        except interrupt_error:
            # BaseException-derived: only this clause can catch a cancellation,
            # and a cancelled run is silence rather than a failure.
            logging.info(
                "[Chatterbox TTS] SRT synthesis was cancelled; returning silence."
            )
            return io.NodeOutput(silent_audio(), "", "", "")

        result = apply_timing_mode(
            clips,
            timing_mode=timing_mode,
            sample_rate=sample_rate,
            timing_tolerance=timing_tolerance,
            max_stretch_ratio=max_stretch_ratio,
            min_stretch_ratio=min_stretch_ratio,
            fade_for_stretch_to_fit=fade_for_StretchToFit,
        )

        audio = to_comfy_audio(result.waveform, sample_rate)
        generation_info = build_generation_info(
            model_pack_name=model_pack_name,
            timing_mode=timing_mode,
            sample_rate=sample_rate,
            cue_count=len(result.cues),
            total_seconds=result.total_seconds,
            cache_key=getattr(patcher, "cache_key", None),
        )
        timing_report = build_timing_report(
            clips, result, timing_mode=timing_mode, sample_rate=sample_rate
        )
        adjusted_srt = render_srt(result.cues)

        logging.info(
            "[Chatterbox TTS] Pack '%s' spoke %d SRT cue(s) in '%s' mode -> "
            "%.3f s (cache key %r)%s.",
            model_pack_name,
            len(result.cues),
            timing_mode,
            result.total_seconds,
            getattr(patcher, "cache_key", None),
            "; overlap fallback applied" if result.fallback_used else "",
        )
        return io.NodeOutput(
            audio,
            generation_info,
            timing_report,
            adjusted_srt,
            ui=ui.PreviewAudio(audio),
        )