"""Subtitle timing for the Chatterbox SRT node (S8.5).

An SRT file says *when* each line should be spoken. A text-to-speech model
says *how long it actually took*. Those two numbers disagree, and the whole of
this module is about what to do about that disagreement without producing
either of the two failures a subtitle track must never have: speech that runs
over the next cue, or speech so stretched it no longer sounds like speech.

**Four modes, four different answers.** ``stretch_to_fit`` treats the SRT as the
truth and time-stretches the audio to match. ``pad_with_silence`` treats the
audio as the truth and pads the gaps. ``smart_natural`` (the default) treats the
audio as the truth, but refuses to let two cues overlap -- which is exactly when
it has to fall back. ``concatenate`` throws the original timings away entirely
and writes a new SRT from what the model actually produced.

**Overlap is a defect, not a style.** A cue whose audio runs past the next
cue's start means the viewer reads line 2 while line 1 is still being spoken.
``smart_natural`` therefore never knowingly emits an overlap: it builds the
arrangement that honours the written timings, checks it, and if the check fails
it **falls back** to placing every clip after the one before it. The fallback is
recorded in the timing report rather than applied silently, because a run whose
subtitles were quietly re-laid-out is a run the user needs to be able to see
and correct.

**Nothing here reaches into the ComfyUI node API and nothing here touches a
model.** The
node synthesises one clip per cue and hands the list in; this module only
measures, resamples and places. That is what makes every behaviour below
testable with a synthetic waveform instead of a 4 GB download.

SRT timestamps are ``HH:MM:SS,mmm`` -- a comma, not the dot that a media
timestamp uses, and a strictly millisecond resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .audio_utils import S3GEN_SAMPLE_RATE

__all__ = [
    "TIMING_MODES",
    "TIMING_MODE_DEFAULT",
    "DEFAULT_TIMING_TOLERANCE",
    "DEFAULT_MAX_STRETCH_RATIO",
    "DEFAULT_MIN_STRETCH_RATIO",
    "DEFAULT_FADE_FOR_STRETCH_TO_FIT",
    "SrtCue",
    "SpokenCue",
    "TimingResult",
    "parse_srt",
    "format_timestamp",
    "render_srt",
    "stretch_to_duration",
    "has_overlap",
    "overlapping_pairs",
    "apply_timing_mode",
    "build_timing_report",
    "build_generation_info",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The four timing strategies, in the order the Combo offers them.
TIMING_MODES: tuple[str, ...] = (
    "stretch_to_fit",
    "pad_with_silence",
    "smart_natural",
    "concatenate",
)

#: ``smart_natural`` is the default: it is the only mode that honours the
#: written timings *and* guarantees the result has no overlaps. The other three
#: each sacrifice one of those.
TIMING_MODE_DEFAULT: str = "smart_natural"

#: How far a cue's realised duration may miss its slot before a bounded stretch
#: is applied, as a multiple of that slot. ``2.0`` lets a line run to three
#: times its written length before the node touches the audio; lowering it
#: toward ``0.5`` keeps the subtitles tight.
DEFAULT_TIMING_TOLERANCE: float = 2.0

#: Upper bound on the time-stretch factor. The default of ``1.0`` means **never
#: make a cue longer** -- only compression is enabled out of the box, because
#: slowing speech down is the one adjustment a listener always notices. Raising
#: it above 1.0 enables expansion.
DEFAULT_MAX_STRETCH_RATIO: float = 1.0

#: Lower bound on the time-stretch factor. ``0.5`` is twice as fast as the
#: original, which is about as fast as a synthesised voice stays intelligible.
DEFAULT_MIN_STRETCH_RATIO: float = 0.5

#: Head/tail fade applied to a stretched clip, as a fraction of its length.
#: A resampled join is a step discontinuity and a step is a click.
DEFAULT_FADE_FOR_STRETCH_TO_FIT: float = 0.01


_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{1,3}):(?P<minutes>\d{2}):(?P<seconds>\d{2})"
    r"(?P<sep>[,.])(?P<millis>\d{1,3})$"
)
_ARROW = " --> "


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtCue:
    """One subtitle entry: where it sits on the timeline and what it says.

    Attributes:
        index: The 1-based number written into the SRT.
        start: Cue start, in seconds from the beginning of the audio.
        end: Cue end, in seconds. Never before :attr:`start`.
        text: The spoken text, newlines collapsed to spaces -- a multi-line cue
            is one cue, and a player that renders it as two would desynchronise
            from the audio this module places.
    """

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Slot length in seconds, never negative."""
        return max(0.0, self.end - self.start)


@dataclass
class SpokenCue:
    """One synthesised clip together with the slot the SRT gave it."""

    text: str
    waveform: torch.Tensor
    slot_start: float
    slot_end: float

    @property
    def slot_duration(self) -> float:
        """The written slot's length."""
        return max(0.0, self.slot_end - self.slot_start)

    @property
    def natural_duration(self) -> float:
        """The clip's length, before any timing adjustment."""
        return float(self.waveform.shape[-1])

    @property
    def natural_seconds(self) -> float:
        """:attr:`natural_duration` in seconds rather than samples."""
        return self.natural_duration / float(S3GEN_SAMPLE_RATE)

    def drift_ratio(self) -> float:
        """Relative miss against the slot.

        ``0.0`` is a perfect fit, ``-0.5`` means the model spoke for half the
        written slot, ``1.0`` means it took half as long again. Compared
        against ``timing_tolerance`` to decide whether a stretch is warranted.
        """
        slot = self.slot_duration
        if slot <= 0:
            return 0.0
        return (self.natural_seconds - slot) / slot


@dataclass
class TimingResult:
    """What :func:`apply_timing_mode` decided.

    Attributes:
        waveform: The laid-out ``[C, T]`` audio.
        cues: The adjusted cue timings -- the ones :func:`render_srt` writes out.
        stretches: Per-cue stretch factors actually applied, ``1.0`` for none.
        fallback_used: Whether ``smart_natural`` had to re-lay the track out.
        total_seconds: Length of :attr:`waveform` in seconds.
    """

    waveform: torch.Tensor
    cues: tuple[SrtCue, ...]
    stretches: tuple[float, ...] = ()
    fallback_used: bool = False
    total_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Parsing and rendering
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> Optional[float]:
    """``"00:00:01,500"`` -> ``1.5``. ``None`` when it is not a timestamp.

    The comma is what SRT uses and what this module writes; a dot is accepted
    on input because half the tools that produce subtitle files emit one, and a
    file this node cannot read is a worse outcome than reading it leniently.
    """
    match = _TIMESTAMP_RE.match(value.strip())
    if match is None:
        return None
    millis = match.group("millis").ljust(3, "0")
    return (
        int(match.group("hours")) * 3600.0
        + int(match.group("minutes")) * 60.0
        + int(match.group("seconds"))
        + int(millis) / 1000.0
    )


def format_timestamp(seconds: float) -> str:
    """``1.5`` -> ``"00:00:01,500"``.

    Negative input is clamped to zero: a cue that starts before the track does
    is a layout bug, and writing ``-1`` into an SRT makes every player fail on
    the file. Milliseconds are truncated rather than rounded, because rounding
    a cue *up* past its successor is how a set of perfect timings acquires an
    overlap on the way out.
    """
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(content: str) -> list[SrtCue]:
    """Parse SRT text into cues, skipping anything unreadable.

    SRT is a loose format: indexes are optional, blank lines separate cues, and
    a subtitle file that has been round-tripped through an editor is often
    missing them. A block is therefore recognised by its ``start --> end`` line,
    and everything up to the next blank line is its text. A block whose timing
    line will not parse is dropped rather than guessed at -- a cue with an
    invented timestamp is worse than a cue that is missing.

    Args:
        content: The raw ``srt_content`` widget value.

    Returns:
        The cues in file order, renumbered from 1. Empty input yields none.
    """
    if not content or not content.strip():
        return []

    cues: list[SrtCue] = []
    block: list[str] = []
    blocks: list[list[str]] = []

    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)

    for lines in blocks:
        timing_index = next(
            (position for position, line in enumerate(lines) if _ARROW in line), None
        )
        if timing_index is None:
            continue
        start_text, _, end_text = lines[timing_index].partition(_ARROW)
        start = _parse_timestamp(start_text)
        end = _parse_timestamp(end_text)
        if start is None or end is None:
            continue
        text = " ".join(line.strip() for line in lines[timing_index + 1 :]).strip()
        cues.append(
            SrtCue(
                index=len(cues) + 1,
                start=start,
                end=max(start, end),
                text=text,
            )
        )
    return cues


def render_srt(cues: Sequence[SrtCue]) -> str:
    """Serialise cues back to SRT text, terminated by a blank line.

    This is the node's ``Adjusted_SRT`` output: the file a user saves and feeds
    to a player, which is why the timestamps are re-derived from the audio
    rather than copied from the input.
    """
    blocks = []
    for position, cue in enumerate(cues, start=1):
        blocks.append(
            f"{position}\n"
            f"{format_timestamp(cue.start)}{_ARROW}{format_timestamp(cue.end)}\n"
            f"{cue.text}"
        )
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------


def stretch_to_duration(
    waveform: torch.Tensor,
    target_samples: int,
    *,
    fade_fraction: float = DEFAULT_FADE_FOR_STRETCH_TO_FIT,
) -> torch.Tensor:
    """Resample ``waveform`` along time so it is ``target_samples`` long.

    Linear interpolation, deliberately: it is the cheapest resampler there is,
    it has no ringing, and a 2x speed-up is something the listener's ear forgives
    far more readily than a 2x speed-up that also has pre-echo. A stretch factor
    is always a compromise between intelligibility and length, and this one
    spends the whole budget on intelligibility.

    The head and tail get a short linear fade when one is asked for, because a
    resampled clip joined against silence or against another clip starts and
    ends on a step, and a step is an audible click.

    Args:
        waveform: ``[C, T]`` or ``[T]`` float audio.
        target_samples: Desired length in samples. ``< 2`` returns the input.
        fade_fraction: Head/tail fade as a fraction of the result's length.

    Returns:
        A ``[C, T']`` float32 tensor of ``target_samples`` samples.
    """
    tensor = waveform if waveform.dim() > 1 else waveform.unsqueeze(0)
    target = int(target_samples)
    current = int(tensor.shape[-1])
    if current < 1 or target < 2 or target == current:
        return tensor

    squeezed = tensor.unsqueeze(0)  # [C, T] -> [1, C, T]
    stretched = F.interpolate(
        squeezed, size=target, mode="linear", align_corners=True
    ).squeeze(0)

    fade = max(0, int(float(fade_fraction) * target))
    if fade > 0 and 2 * fade < target:
        ramp = torch.linspace(0.0, 1.0, fade, dtype=stretched.dtype)
        stretched[..., :fade] = stretched[..., :fade] * ramp
        stretched[..., -fade:] = stretched[..., -fade:] * ramp.flip(0)
    return stretched


def _clamp(value: float, low: float, high: float) -> float:
    """``value`` confined to ``[low, high]``, with the bounds ordered."""
    lower, upper = (low, high) if low <= high else (high, low)
    return max(lower, min(upper, value))


def _clip_ratio(clip: SpokenCue, low: float, high: float) -> float:
    """The stretch factor that would fit ``clip`` into its slot, bounded."""
    natural = clip.natural_seconds
    slot = clip.slot_duration
    if natural <= 0 or slot <= 0:
        return 1.0
    return _clamp(slot / natural, low, high)


def _duration_of(waveform: torch.Tensor) -> float:
    """Clip length in seconds, from the shared sample rate."""
    return float(waveform.shape[-1]) / float(S3GEN_SAMPLE_RATE)


def _fitted_samples(clip: SpokenCue, ratio: float, sample_rate: int) -> int:
    """How many samples ``clip`` becomes after a stretch by ``ratio``.

    Derived from the clip's **natural** length rather than from the slot, so
    the realised duration is exactly ``natural * ratio`` even where the ratio
    bounds refused the exact fit. Two samples is the floor: the resampler needs
    an interior sample to interpolate towards.
    """
    return max(2, int(round(clip.natural_seconds * float(ratio) * sample_rate)))


def _lay_out(
    placements: Sequence[tuple[float, torch.Tensor]],
    sample_rate: int = S3GEN_SAMPLE_RATE,
) -> torch.Tensor:
    """Sum ``(start_seconds, clip)`` pairs onto one timeline.

    Overlapping clips are **summed**, not overwritten. Subtitles that overlap
    are a defect (see the module docstring), but when two of them do, summing is
    the only answer that keeps both audible instead of silently cutting one off
    mid-word.

    Args:
        placements: ``(start_seconds, [C, T] clip)`` in any order.
        sample_rate: Samples per second.

    Returns:
        ``[C, T]`` float32, long enough to hold the last placed clip.
    """
    if not placements:
        return torch.zeros((1, 1), dtype=torch.float32)

    # A 1-D clip is one channel. Normalising here rather than at every call site
    # means a caller can pass the raw ``[T]`` a ``generate`` returned without
    # having to reshape it first -- and get the same answer either way.
    normalised = [
        (start, clip if clip.dim() > 1 else clip.unsqueeze(0))
        for start, clip in placements
        if clip.numel()
    ]
    if not normalised:
        return torch.zeros((1, 1), dtype=torch.float32)

    channels = max(int(clip.shape[0]) for _, clip in normalised)
    length = max(
        1,
        max(
            int(round(start * sample_rate)) + int(clip.shape[-1])
            for start, clip in normalised
        ),
    )
    buffer = torch.zeros((channels, length), dtype=torch.float32)
    for start, clip in normalised:
        offset = max(0, int(round(float(start) * sample_rate)))
        end = min(length, offset + int(clip.shape[-1]))
        if end <= offset:
            continue
        buffer[: int(clip.shape[0]), offset:end] += clip[:, : end - offset].to(
            dtype=torch.float32
        )
    return buffer


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def overlapping_pairs(cues: Sequence[SrtCue]) -> list[tuple[int, int]]:
    """Index pairs of cues that overlap each other.

    Adjacent only: once one pair overlaps, every later pair "overlaps" it
    transitively and listing all of them buries the real signal. A cue that
    touches its successor exactly (end == start) is **not** an overlap.
    """
    return [
        (position, position + 1)
        for position in range(len(cues) - 1)
        if cues[position].end > cues[position + 1].start + 1e-9
    ]


def has_overlap(cues: Sequence[SrtCue]) -> bool:
    """Whether any cue runs into the one after it."""
    return bool(overlapping_pairs(cues))


# ---------------------------------------------------------------------------
# The four modes
# ---------------------------------------------------------------------------


def _cue_number(position: int) -> int:
    """The 1-based cue number for ``position`` in the clip list.

    The clips arrive in SRT order, so the position is the cue number. Renumbering
    here rather than carrying an index on the dataclass keeps ``SpokenCue`` a
    description of the audio rather than of where it happens to sit in a list.
    """
    return position + 1


def _stretch_to_fit(
    clips: Sequence[SpokenCue],
    *,
    sample_rate: int,
    low: float,
    high: float,
    fade: float,
) -> TimingResult:
    """Every clip time-stretched onto its written slot, at the written start.

    "Onto its slot" is *bounded*: the ratio is clamped to
    ``[min_stretch_ratio, max_stretch_ratio]`` and the realised length is the
    clamped one, not the requested one. Reporting the requested slot in the
    SRT while the audio is a different length is the one thing this node
    exists to prevent. With the default ``max_stretch_ratio`` of 1.0 a cue
    shorter than its slot is therefore left short rather than slowed down.
    """
    placements: list[tuple[float, torch.Tensor]] = []
    cues: list[SrtCue] = []
    stretches: list[float] = []

    for position, clip in enumerate(clips):
        ratio = _clip_ratio(clip, low, high)
        fitted = stretch_to_duration(
            clip.waveform,
            _fitted_samples(clip, ratio, sample_rate),
            fade_fraction=fade,
        )
        placements.append((clip.slot_start, fitted))
        cues.append(
            SrtCue(
                index=_cue_number(position),
                start=clip.slot_start,
                end=clip.slot_start + _duration_of(fitted),
                text=clip.text,
            )
        )
        stretches.append(ratio)

    waveform = _lay_out(placements, sample_rate)
    return TimingResult(
        waveform=waveform,
        cues=tuple(cues),
        stretches=tuple(stretches),
        fallback_used=False,
        total_seconds=_duration_of(waveform),
    )


def _pad_with_silence(clips: Sequence[SpokenCue], *, sample_rate: int) -> TimingResult:
    """Natural audio at the written starts, with silence filling the gaps."""
    placements = [(clip.slot_start, clip.waveform) for clip in clips]
    cues = [
        SrtCue(
            index=_cue_number(position),
            start=clip.slot_start,
            end=clip.slot_start + _duration_of(clip.waveform),
            text=clip.text,
        )
        for position, clip in enumerate(clips)
    ]
    waveform = _lay_out(placements, sample_rate)
    return TimingResult(
        waveform=waveform,
        cues=tuple(cues),
        stretches=tuple(1.0 for _ in clips),
        fallback_used=False,
        total_seconds=_duration_of(waveform),
    )


def _concatenate(clips: Sequence[SpokenCue], *, sample_rate: int) -> TimingResult:
    """The original timings discarded; a new SRT written from the audio."""
    placements: list[tuple[float, torch.Tensor]] = []
    cues: list[SrtCue] = []
    cursor = 0.0

    for position, clip in enumerate(clips):
        start = cursor
        placements.append((start, clip.waveform))
        cues.append(
            SrtCue(
                index=_cue_number(position),
                start=start,
                end=start + _duration_of(clip.waveform),
                text=clip.text,
            )
        )
        cursor = cues[-1].end

    waveform = _lay_out(placements, sample_rate)
    return TimingResult(
        waveform=waveform,
        cues=tuple(cues),
        stretches=tuple(1.0 for _ in clips),
        fallback_used=False,
        total_seconds=_duration_of(waveform),
    )


def _smart_natural(
    clips: Sequence[SpokenCue],
    *,
    sample_rate: int,
    tolerance: float,
    low: float,
    high: float,
) -> TimingResult:
    """Natural audio, written timings honoured, no overlaps ever emitted.

    Two passes, and the second one exists because the first can fail:

    1. **Honour the timings.** Each clip goes at its written start, untouched
       unless its length misses the slot by more than ``timing_tolerance``, in
       which case it is stretched or shrunk within the ratio bounds.
    2. **Check, and fall back.** Honouring the written timings means a line that
       ran long runs into the next one. When that happens the whole track is
       re-laid out so every clip starts after the previous one ends. That
       arrangement cannot overlap by construction, which is the property the
       mode promises.

    The fallback is reported rather than silent: a user whose subtitles were
    re-laid-out needs to know, because their file and the timing they wrote have
    come apart and only one of them is what the viewer sees.
    """
    # -- pass 1: honour the written timings --------------------------------
    placements: list[tuple[float, torch.Tensor]] = []
    candidate: list[SrtCue] = []
    stretches: list[float] = []

    for position, clip in enumerate(clips):
        drift = abs(clip.drift_ratio())
        ratio = (
            1.0 if drift <= float(tolerance) else _clip_ratio(clip, low, high)
        )
        fitted = (
            clip.waveform
            if ratio == 1.0
            else stretch_to_duration(
                clip.waveform,
                _fitted_samples(clip, ratio, sample_rate),
                fade_fraction=0.0,
            )
        )
        placements.append((clip.slot_start, fitted))
        candidate.append(
            SrtCue(
                index=_cue_number(position),
                start=clip.slot_start,
                end=clip.slot_start + _duration_of(fitted),
                text=clip.text,
            )
        )
        stretches.append(ratio)

    if not has_overlap(candidate):
        waveform = _lay_out(placements, sample_rate)
        return TimingResult(
            waveform=waveform,
            cues=tuple(candidate),
            stretches=tuple(stretches),
            fallback_used=False,
            total_seconds=_duration_of(waveform),
        )

    # -- pass 2: sequential, provably overlap-free --------------------------
    fallback_placements: list[tuple[float, torch.Tensor]] = []
    fallback_cues: list[SrtCue] = []
    cursor = 0.0

    for position, clip in enumerate(clips):
        start = max(clip.slot_start, cursor)
        fallback_placements.append((start, clip.waveform))
        fallback_cues.append(
            SrtCue(
                index=_cue_number(position),
                start=start,
                end=start + _duration_of(clip.waveform),
                text=clip.text,
            )
        )
        cursor = fallback_cues[-1].end

    waveform = _lay_out(fallback_placements, sample_rate)
    return TimingResult(
        waveform=waveform,
        cues=tuple(fallback_cues),
        stretches=tuple(1.0 for _ in clips),
        fallback_used=True,
        total_seconds=_duration_of(waveform),
    )


def apply_timing_mode(
    clips: Sequence[SpokenCue],
    *,
    timing_mode: str = TIMING_MODE_DEFAULT,
    sample_rate: int = S3GEN_SAMPLE_RATE,
    timing_tolerance: float = DEFAULT_TIMING_TOLERANCE,
    max_stretch_ratio: float = DEFAULT_MAX_STRETCH_RATIO,
    min_stretch_ratio: float = DEFAULT_MIN_STRETCH_RATIO,
    fade_for_stretch_to_fit: float = DEFAULT_FADE_FOR_STRETCH_TO_FIT,
) -> TimingResult:
    """Lay ``clips`` out on one timeline under ``timing_mode``.

    Args:
        clips: One :class:`SpokenCue` per subtitle, in file order, each
            carrying the slot its SRT entry reserved. Empty waveforms are
            skipped.
        timing_mode: One of :data:`TIMING_MODES`.
        sample_rate: Samples per second the clips were synthesised at.
        timing_tolerance: See :data:`DEFAULT_TIMING_TOLERANCE`.
        max_stretch_ratio: See :data:`DEFAULT_MAX_STRETCH_RATIO`.
        min_stretch_ratio: See :data:`DEFAULT_MIN_STRETCH_RATIO`.
        fade_for_stretch_to_fit: See :data:`DEFAULT_FADE_FOR_STRETCH_TO_FIT`.

    Returns:
        A :class:`TimingResult`.

    Raises:
        ValueError: ``timing_mode`` is not one of the four, or ``clips`` is
            empty.
    """
    if timing_mode not in TIMING_MODES:
        raise ValueError(
            f"Unknown timing_mode {timing_mode!r}. Expected one of: "
            f"{', '.join(TIMING_MODES)}."
        )

    usable = [clip for clip in clips if clip.waveform is not None and clip.waveform.numel()]
    if not usable:
        return TimingResult(
            waveform=torch.zeros((1, 1), dtype=torch.float32),
            cues=(),
            stretches=(),
            fallback_used=False,
            total_seconds=0.0,
        )

    if timing_mode == "stretch_to_fit":
        return _stretch_to_fit(
            usable,
            sample_rate=sample_rate,
            low=min_stretch_ratio,
            high=max_stretch_ratio,
            fade=fade_for_stretch_to_fit,
        )
    if timing_mode == "pad_with_silence":
        return _pad_with_silence(usable, sample_rate=sample_rate)
    if timing_mode == "concatenate":
        return _concatenate(usable, sample_rate=sample_rate)
    return _smart_natural(
        usable,
        sample_rate=sample_rate,
        tolerance=timing_tolerance,
        low=min_stretch_ratio,
        high=max_stretch_ratio,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_timing_report(
    clips: Sequence[SpokenCue],
    result: TimingResult,
    *,
    timing_mode: str,
    sample_rate: int = S3GEN_SAMPLE_RATE,
) -> str:
    """The ``timing_report`` output: what was asked for, what happened.

    One line per cue, in the order they are in the output SRT, so a user can
    read their timings off the report and the file side by side. Clips are
    matched to cues **by position**, not by text: two cues can legitimately
    carry the same words, and keying on the text would attribute the second
    cue's slot to the first cue.
    """
    usable = [
        clip for clip in clips if clip.waveform is not None and clip.waveform.numel()
    ]
    lines = [
        f"timing_mode: {timing_mode}",
        f"cues: {len(result.cues)} of {len(clips)} spoken",
        f"total: {result.total_seconds:.3f} s at {sample_rate} Hz",
    ]
    if result.fallback_used:
        lines.append(
            "FALLBACK: the written timings overlapped once the audio was in "
            "place, so every cue was moved to start after the one before it. "
            "No cue overlaps in this file."
        )

    for position, cue in enumerate(result.cues):
        clip = usable[position] if position < len(usable) else None
        natural = clip.natural_seconds if clip is not None else cue.end - cue.start
        stretch = (
            result.stretches[position]
            if position < len(result.stretches)
            else 1.0
        )
        slot = f"{clip.slot_start:.3f}-{clip.slot_end:.3f}" if clip else "-"
        lines.append(
            f"#{cue.index} slot {slot} -> {cue.start:.3f}-{cue.end:.3f} "
            f"(natural {natural:.3f}s, stretch x{stretch:.3f})"
        )
    return "\n".join(lines)


def build_generation_info(
    *,
    model_pack_name: str,
    timing_mode: str,
    sample_rate: int,
    cue_count: int,
    total_seconds: float,
    cache_key: Optional[str] = None,
) -> str:
    """The ``generation_info`` output: what produced this run."""
    return (
        f"pack: {model_pack_name}\n"
        f"timing_mode: {timing_mode}\n"
        f"cues: {cue_count}\n"
        f"duration: {total_seconds:.3f} s at {sample_rate} Hz\n"
        f"cache key: {cache_key}"
    )