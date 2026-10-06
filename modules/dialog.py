"""Multi-speaker dialog parsing and stem assembly for ``ChatterboxDialog``.

Upstream Chatterbox has **no** dialog capability: there is one model, one
reference voice, one utterance at a time. A conversation is therefore four
independent syntheses whose only connection is the text between them, and the
whole of what this module provides is the bookkeeping that makes that usable:

* :func:`parse_dialog` turns ``SPEAKER A: ...\\nSPEAKER B: ...`` into turns,
  each bound to one speaker slot;
* :func:`assemble_stems` places every turn on one timeline **per speaker** and
  returns the isolated stems plus their sum.

**The stems are the contract.** ``speaker_a_audio`` contains speaker A's turns
and nothing else -- not a downmix, not the other speakers quietly mixed in at
low level, not silence-with-a-marker. Two consequences make that cheap to keep
honest:

* the dialog track is placed **sequentially**, one turn after another, so two
  turns never share a sample and the stems cannot disagree about who spoke;
* the mixed track is computed *by summing the stems*, not assembled
  independently. A track built by a second code path would be free to drift
  from the stems it is supposed to be the sum of; this one cannot, because it
  is the sum.

An unconfigured speaker (C or D with no reference clip) still gets a stem, and
that stem is real silence of the dialog's length rather than an empty dict --
downstream nodes chain AUDIO sockets without checking, and a stem that changes
length when it is unconnected is a graph that only works on the machine it was
built on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .audio_utils import S3GEN_SAMPLE_RATE

__all__ = [
    "SPEAKER_SLOTS",
    "REQUIRED_SPEAKERS",
    "DialogTurn",
    "parse_dialog",
    "assemble_stems",
    "build_message",
]


#: The four speaker slots, in the order the node's outputs declare them.
SPEAKER_SLOTS: tuple[str, ...] = ("A", "B", "C", "D")

#: Slots that must have a reference voice. The corpus node makes A and B
#: required sockets and C and D optional; a two-voice conversation is the
#: smallest thing this node is for.
REQUIRED_SPEAKERS: tuple[str, ...] = ("A", "B")

#: ``SPEAKER A:`` / ``Speaker_A:`` / ``speaker a:`` -- case and separator are
#: the only variation tolerated, because a dialog script gets retyped.
_SPEAKER_RE = re.compile(r"^\s*speaker[\s_-]*(?P<slot>[a-dA-D])\s*:\s*", re.IGNORECASE)


@dataclass(frozen=True)
class DialogTurn:
    """One utterance by one speaker.

    Attributes:
        speaker: Upper-case slot letter, one of :data:`SPEAKER_SLOTS`.
        text: What they say, tag-resolved and crash-protected by the caller.
        index: Position in the conversation, 0-based.
    """

    speaker: str
    text: str
    index: int = 0


def parse_dialog(text: str) -> list[DialogTurn]:
    """Parse a dialog script into turns.

    A line matching ``SPEAKER X:`` opens a turn. Any line that does not match
    continues the turn above it, so a line-wrapped script -- which is what an
    editor produces the moment the text is longer than one screen -- stays one
    turn instead of becoming one per line.

    Args:
        text: The ``dialog_text`` widget value.

    Returns:
        The turns in order. Empty input yields none.

    Raises:
        ValueError: The script contains text before the first ``SPEAKER X:``
            line. That text has no voice to be spoken in, and assigning it to
            speaker A silently would put words in someone's mouth.
    """
    if not text or not text.strip():
        return []

    turns: list[DialogTurn] = []
    current: Optional[DialogTurn] = None
    pending: list[str] = []
    preamble: list[str] = []

    def _flush() -> None:
        nonlocal current, pending
        if current is not None and pending:
            # The continuation lines *extend* what the speaker line already
            # said; they do not replace it. A script that wrapped its first
            # line onto a second screen must not lose the first screen.
            continued = " ".join(part for part in pending if part).strip()
            current = DialogTurn(
                speaker=current.speaker,
                text=f"{current.text} {continued}".strip(),
                index=current.index,
            )
        pending = []
        if current is not None and current.text:
            turns.append(current)
        current = None

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _SPEAKER_RE.match(line)
        if match is not None:
            _flush()
            current = DialogTurn(
                speaker=match.group("slot").upper(),
                text=line[match.end() :].strip(),
                index=len(turns),
            )
            continue
        if current is None:
            if line.strip():
                preamble.append(line.strip())
            continue
        pending.append(line.strip())

    _flush()

    if preamble:
        raise ValueError(
            "These lines come before the first 'SPEAKER A:' line, so there is "
            "no voice to speak them in: "
            + " / ".join(preamble[:3])
            + ". Start the script with 'SPEAKER A:'."
        )
    return turns


def assemble_stems(
    turns: Sequence[DialogTurn],
    clips: Sequence[torch.Tensor],
    *,
    sample_rate: int = S3GEN_SAMPLE_RATE,
    slots: Sequence[str] = SPEAKER_SLOTS,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Lay the turns out sequentially and split them into per-speaker stems.

    Args:
        turns: The conversation, in order.
        clips: One ``[C, T]`` waveform **per turn**, in the same order. One
            entry per turn rather than one per speaker: a speaker who talks
            three times has three separate clips, and keying by speaker would
            keep only the last one.
        sample_rate: Samples per second.
        slots: Which slots to return, in output order.

    Returns:
        ``(stems, dialog)`` where ``stems`` maps every slot in ``slots`` to a
        ``[C, T]`` tensor of the dialog's full length, and ``dialog`` is their
        sum -- computed *as* their sum, so the two cannot disagree.

    Raises:
        ValueError: The turns are empty, a turn has no clip, or ``clips`` is
            shorter than ``turns``.
    """
    if not turns:
        raise ValueError("There are no dialog turns to place.")
    if len(clips) < len(turns):
        raise ValueError(
            f"{len(turns)} dialog turn(s) but only {len(clips)} clip(s). Every "
            f"turn needs its own synthesised audio."
        )

    channels = 1
    for waveform in clips:
        if waveform is not None and waveform.numel():
            channels = max(channels, int(waveform.shape[0]) if waveform.dim() > 1 else 1)
            break

    placements: list[tuple[str, int, torch.Tensor]] = []
    cursor = 0
    for turn, waveform in zip(turns, clips):
        if waveform is None or not waveform.numel():
            raise ValueError(
                f"Speaker {turn.speaker}'s line produced no audio, so the "
                f"conversation cannot be assembled."
            )
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        placements.append((turn.speaker, cursor, waveform))
        cursor += int(waveform.shape[-1])

    stems = {
        slot: torch.zeros((channels, cursor), dtype=torch.float32) for slot in slots
    }
    for speaker, start, waveform in placements:
        target = stems[speaker]
        length = int(waveform.shape[-1])
        target[: int(waveform.shape[0]), start : start + length] += waveform.to(
            dtype=torch.float32
        )

    dialog = torch.zeros((channels, cursor), dtype=torch.float32)
    for stem in stems.values():
        dialog += stem
    return stems, dialog


def build_message(
    turns: Sequence[DialogTurn],
    stems: dict[str, torch.Tensor],
    *,
    sample_rate: int = S3GEN_SAMPLE_RATE,
) -> str:
    """The node's ``message`` output: who spoke, for how long, in what order.

    Reported because a dialog that renders badly is diagnosed by which turn ran
    long, and guessing that from four stems and a waveform viewer is work the
    node can just do.
    """
    lines = [f"turns: {len(turns)}"]
    for turn in turns:
        lines.append(f"  {turn.speaker}: {turn.text}")
    lines.append("")
    for slot in sorted(stems):
        stem = stems[slot]
        active = int(torch.count_nonzero(stem))
        seconds = float(stem.shape[-1]) / float(sample_rate)
        if active == 0:
            lines.append(f"speaker {slot}: silent ({seconds:.3f} s track)")
        else:
            lines.append(f"speaker {slot}: {seconds:.3f} s of track")
    return "\n".join(lines)