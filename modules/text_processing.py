"""The Chatterbox text-tag and chunking engine (S8.3, S8.4).

Everything a user can type into a Chatterbox text box is parsed here, in one
place, by one function: :func:`plan_text`. The nodes hand it a script and get
back a :class:`TextPlan` -- the segments, each one resolved to a character, a
language, a set of parameter overrides and a trailing pause, plus the chunking
plan that decides how those segments are cut and joined.

**The processing order is load-bearing** and is exactly the one the reference
corpus specifies:

1. character switching -- ``[Alice] Hi.``
2. language tags -- ``[de:Alice] Hallo.``
3. pause tags -- ``[pause:1.5]`` / ``[wait:2s]`` / ``[stop:500ms]``
4. per-segment parameter overrides -- ``[Alice|seed:42|temp:0.5]``
5. the v2 angle-bracket tokens -- ``<giggle>`` (re-spelled ``[giggle]`` on a
   Turbo pack, whose tokenizer knows the square spelling and not the angle one)

Stages 1-4 classify one ``[...]`` token, so they are tried **in that order**
against each token in turn: a token that is not a character is then tested as a
language tag, and so on. Stage 5 runs *after* the whole script has been split,
because a ``<giggle>`` is only rewritten once the character names around it have
resolved -- rewriting it first is what makes ``[Alice] <giggle>`` lose its
character.

The nine Turbo tag names (``[laugh]``, ``[clear throat]``, ...) are claimed
**before** stage 1, on every path. A Turbo pack needs them kept verbatim for
the tokenizer, and a pack that has no Turbo vocabulary still has to say the
word rather than swallow it as a voice name -- so the tag is never a
character, anywhere.

Three rules that follow from the syntax and are worth stating up front:

* **Square brackets are the Turbo tag set's home, and angle brackets are the v2
  token set's.** They are deliberately not interchangeable. ``<giggle>`` is a
  v2 special token; ``[laugh]`` is a Turbo paralinguistic tag, and it is only
  *recognised* as one when the caller says it is speaking to a Turbo model.
  On any other pack the nine Turbo names are kept as **literal text** and
  reported as a notice -- swallowing ``[laugh]`` as a voice name would take the
  word out of the speech without saying so. Every other bare bracket stays a
  *character name*, which is what upstream does.
* **Unknown parameters are ignored, not fatal.** A script carried over from
  another tool is full of knobs this node has no socket for; refusing the whole
  run over ``[Alice|batch:4]`` would be hostile.
* **Nothing here touches the model.** Parsing is pure text; the callers decide
  what to do with a :class:`TextPlan`, and none of it needs ComfyUI, which is why
  ``modules/`` can import it freely.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import MTL_LANGUAGE_LOOKUP

__all__ = [
    "CRASH_PROTECTION_TEMPLATE",
    "CRASH_PROTECTION_PRESETS",
    "MIN_SAFE_SEGMENT_CHARS",
    "CHUNK_COMBINATION_METHODS",
    "AUTO_CHUNK_THRESHOLDS",
    "PARAM_ALIASES",
    "TURBO_TAGS",
    "TURBO_TAG_V2_SPELLING",
    "V2_SPECIAL_TOKENS",
    "UPPERCASE_V2_TOKENS",
    "TextSegment",
    "TextPlan",
    "parse_duration",
    "try_parse_duration",
    "split_segments",
    "resolve_v2_tokens",
    "apply_crash_protection",
    "select_chunk_method",
    "chunk_text",
    "plan_text",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Upstream Chatterbox segfaults (CUDA tensor errors, not a Python exception) on
#: a sequential segment shorter than roughly this many characters. The padding is
#: therefore not a nicety: without it, every short line in a dialogue kills the
#: process instead of raising something a node could catch.
MIN_SAFE_SEGMENT_CHARS = 21

#: The documented default template. ``{seg}`` is replaced by the segment's text.
CRASH_PROTECTION_TEMPLATE = "hmm ,, {seg} hmm ,,"

#: The documented presets, offered in the node tooltips. ``""`` disables.
CRASH_PROTECTION_PRESETS: tuple[str, ...] = (
    CRASH_PROTECTION_TEMPLATE,
    "...ummmmm {seg}",
    "{seg}... yes... {seg}",
    "Well, {seg}",
    "",
)

#: How chunks are joined once each has been synthesised.
CHUNK_COMBINATION_METHODS: tuple[str, ...] = (
    "auto",
    "concatenate",
    "silence_padding",
    "crossfade",
)

#: ``auto``'s decision table, longest-text-first. Read literally: the comparison
#: is **strictly greater than** in both branches, so exactly 500 characters is
#: still ``concatenate`` and 501 is the first ``crossfade``. The corpus phrases
#: the same rule as "<500 -> concatenate"; the boundary case at 500 is spelled
#: here because a rule with an unstated boundary is a rule with a bug in it.
AUTO_CHUNK_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (1000, "silence_padding"),
    (500, "crossfade"),
)

#: Per-segment override names, and the aliases the corpus documents. Keys are
#: lower-cased; values are the widget names the caller maps onto.
PARAM_ALIASES: dict[str, str] = {
    "seed": "seed",
    "temp": "temperature",
    "temperature": "temperature",
    "cfg": "cfg_weight",
    "cfg_weight": "cfg_weight",
    "exag": "exaggeration",
    "exaggeration": "exaggeration",
    "min_p": "min_p",
    "top_p": "top_p",
    "top_k": "top_k",
}

#: Turbo's own paralinguistic tags, in square brackets. A *different* vocabulary
#: from the v2 angle-bracket set below; Turbo's tokenizer was trained on these
#: spellings and has never seen ``<giggle>``.
TURBO_TAGS: tuple[str, ...] = (
    "laugh",
    "sigh",
    "gasp",
    "chuckle",
    "cough",
    "sniff",
    "groan",
    "shush",
    "clear throat",
)

#: The v2 tokens whose Turbo spelling differs from their own name. Only one
#: does: the v2 set writes ``<clear_throat>`` with an underscore, Turbo's
#: tokenizer was trained on ``[clear throat]`` with a space. Every other token
#: keeps its own spelling on both sides of the translation.
TURBO_TAG_V2_SPELLING: dict[str, str] = {
    "clear_throat": "clear throat",
}

#: The 30 named v2 special tokens. The tokenizer upstream actually contains 118
#: special tokens; these are the 30 the reference documentation names, and they
#: are undocumented upstream, experimental, and tracked in upstream issue #186.
V2_SPECIAL_TOKENS: tuple[str, ...] = (
    # emotional
    "giggle",
    "laughter",
    "guffaw",
    "sigh",
    "cry",
    "gasp",
    "groan",
    # breathing / speech modifiers
    "inhale",
    "exhale",
    "whisper",
    "mumble",
    "UH",
    "UM",
    # vocal performances
    "singing",
    "music",
    "humming",
    "whistle",
    # body sounds
    "cough",
    "sneeze",
    "sniff",
    "snore",
    "clear_throat",
    "chew",
    "sip",
    "kiss",
    # animal sounds
    "bark",
    "howl",
    "meow",
    # other
    "shhh",
    "gibberish",
)

#: The two tokens whose casing is part of the vocabulary. ``<uh>`` is not
#: ``<UH>``, so they are normalised upward rather than down with the rest.
UPPERCASE_V2_TOKENS = frozenset({"UH", "UM"})

_V2_TOKEN_LOOKUP = frozenset(token.lower() for token in V2_SPECIAL_TOKENS)
_TURBO_TAG_LOOKUP = frozenset(TURBO_TAGS)

# A single ``[...]`` or ``<...>`` run. Angle brackets are matched separately
# because they are never tags -- only v2 special tokens.
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_ANGLE_RE = re.compile(r"<([^<>]*)>")
#: A sentence end: the mark that ends a sentence and the space that proves it.
_SENTENCE_END_RE = re.compile(r"[.!?] ")
#: The bracket pairs a chunk boundary must never land inside.
_BRACKET_PAIRS = (("[", "]"), ("<", ">"))
#: A Turbo tag as it appears in *text*: the square spelling, in any case.
_TURBO_TAG_IN_TEXT_RE = re.compile(
    r"\[\s*(" + "|".join(re.escape(tag) for tag in TURBO_TAGS) + r")\s*\]",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"^(pause|wait|stop)\s*:\s*(?P<value>.+)$", re.IGNORECASE)
_OVERRIDE_SPLIT_RE = re.compile(r"\|")
_DURATION_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)?|\.\d+)\s*(?P<unit>[a-z]*)\s*$", re.IGNORECASE)
_DURATION_UNITS = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 0.001,
    "msec": 0.001,
    "msecs": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TextSegment:
    """One span of the script, with everything its tags declared resolved.

    Attributes:
        text: The words to speak, with tags removed and v2 special tokens
            normalised.
        character: Voice name from ``[Alice]`` / ``[de:Alice]``, or ``None`` for
            the default voice.
        language: ISO code from ``[de:Alice]``, or ``None``.
        params: Per-segment overrides, keyed by their canonical widget name.
        pause_after: Seconds of silence to insert **after** this segment, from
            the ``[pause:…]`` / ``[wait:…]`` / ``[stop:…]`` tag that follows it.
            A pause with nothing before it in the script is dropped.
        turbo_tags: The Turbo paralinguistic tags found inside this segment.
            Recorded on a Turbo pack, where they are kept verbatim in ``text``
            and handed to the model, which is where they have an effect. On any
            other pack a Turbo tag is kept as literal text -- that pack has no
            such vocabulary -- and the plan reports a notice about it instead
            of recording it here.
        index: Position in the original script, preserved through chunking so a
            caller can still map a chunk back to where it came from.
    """

    text: str
    character: Optional[str] = None
    language: Optional[str] = None
    params: dict[str, str] = field(default_factory=dict)
    pause_after: float = 0.0
    turbo_tags: tuple[str, ...] = ()
    index: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether this segment has nothing left to say."""
        return not self.text.strip()


@dataclass
class TextPlan:
    """A parsed script: its segments and how to synthesise them.

    Attributes:
        segments: The script, split and resolved.
        text: Everything that will be spoken, as one string.
        chunks: ``text`` cut into the pieces that are synthesised separately.
        chunk_method: The concrete join method -- never ``"auto"``.
        char_count: Length of ``text``.
        notices: Things the engine could not honour, said out loud rather
            than dropped. A notice is never an error and never changes the
            audio: it is the engine admitting that a tag in the script is not
            going to do what it looks like it does -- a Turbo tag on a pack
            without the Turbo vocabulary, a pause that is parsed but never
            rendered as silence, a language that is parsed but never switched.
            The caller decides whether to log them.
    """

    segments: tuple[TextSegment, ...]
    text: str
    chunks: tuple[str, ...]
    chunk_method: str
    char_count: int
    notices: tuple[str, ...] = ()

    @property
    def characters(self) -> tuple[str, ...]:
        """Distinct voice names referenced, in first-appearance order."""
        seen: list[str] = []
        for segment in self.segments:
            if segment.character and segment.character not in seen:
                seen.append(segment.character)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------


def parse_duration(value: str) -> float:
    """Parse a pause-tag duration into seconds.

    Accepts a bare number (seconds) and the ``ms`` / ``s`` suffixes, in any
    case, with surrounding space: ``"1.5"``, ``"2s"``, ``"500ms"``, ``" 800 MS "``.

    Args:
        value: The text after the colon.

    Returns:
        Seconds, always non-negative.

    Raises:
        ValueError: The value is not a duration this engine can read. Callers on
            the hot path use :func:`try_parse_duration` instead, so a malformed
            tag degrades to literal text rather than failing the run.
    """
    match = _DURATION_RE.match(str(value))
    if match is None:
        raise ValueError(f"Cannot read {value!r} as a duration")
    number = float(match.group("number"))
    unit = match.group("unit").lower()
    if unit not in _DURATION_UNITS:
        raise ValueError(f"Unknown duration unit {unit!r} in {value!r}")
    seconds = number * _DURATION_UNITS[unit]
    if seconds < 0:
        raise ValueError(f"A pause cannot be negative ({value!r})")
    return seconds


def try_parse_duration(value: str) -> Optional[float]:
    """:func:`parse_duration`, or ``None`` when the value is unreadable."""
    try:
        return parse_duration(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Stage 1-4: classifying one [...] token
# ---------------------------------------------------------------------------


def _canonical_param_name(name: str) -> Optional[str]:
    """The widget name an override targets, or ``None`` when unknown."""
    return PARAM_ALIASES.get(name.strip().lower())


def _parse_overrides(body: str) -> tuple[Optional[str], dict[str, str], list[str]]:
    """``"Alice|seed:42|TEMP:0.5"`` -> ``("Alice", {"seed": "42", ...}, junk)``.

    Order-independent and case-insensitive, as documented: each ``|``-separated
    field is examined on its own, so ``[Alice|temp:0.5|seed:42]`` means the same
    as the other order. An unrecognised field -- no colon, or a parameter with
    no widget here -- is **silently dropped** and reported in the third element
    so a caller can log it. It is never an error.
    """
    parts = _OVERRIDE_SPLIT_RE.split(body)
    character: Optional[str] = None
    params: dict[str, str] = {}
    ignored: list[str] = []

    for position, part in enumerate(parts):
        field_text = part.strip()
        if not field_text:
            continue
        if ":" not in field_text:
            # The first bare field is the voice name; a bare field anywhere else
            # is a parameter this engine has no socket for.
            if position == 0:
                character = field_text
            else:
                ignored.append(field_text)
            continue
        name, _, value = field_text.partition(":")
        canonical = _canonical_param_name(name)
        if canonical is None:
            ignored.append(field_text)
            continue
        params[canonical] = value.strip()

    return character, params, ignored


def _classify_bracket(body: str) -> Optional[tuple[str, object]]:
    """Classify one ``[...]`` body, trying stages 1 to 4 **in that order**.

    Returns ``(kind, payload)`` where ``kind`` is one of ``"character"``,
    ``"language"``, ``"pause"``, ``"overrides"``, ``"turbo_tag"`` or ``"text"``
    (``text`` meaning: not a tag at all, keep it as written). ``None`` is never
    returned -- a body that matches nothing is literal text.

    The order is the whole point: ``[de:Alice]`` is not a character called
    ``"de:Alice"``, and ``[pause:1.5]`` is not a character called
    ``"pause:1.5"``. Because every stage declines a form it does not own and the
    next one is tried, the first match is always the *most specific* reading.
    """
    stripped = body.strip()
    if not stripped:
        return ("text", stripped)

    # -- stage 1: character switching ------------------------------------
    # ``[Alice]``, and only that: a colon or a pipe belongs to a later stage.
    if ":" not in stripped and "|" not in stripped:
        return ("character", stripped)

    # -- stage 2: language tags ------------------------------------------
    # ``[de:Alice]`` / ``[German:Alice]`` / ``[de:]`` -- code or display name,
    # the character optional.
    if ":" in stripped:
        head, _, tail = stripped.partition(":")
        code = MTL_LANGUAGE_LOOKUP.get(head.strip().lower())
        if code is not None:
            return ("language", (code, tail.strip() or None))

    # -- stage 3: pause tags ---------------------------------------------
    # ``[pause:1.5]`` -- ``pause`` / ``wait`` / ``stop`` are interchangeable.
    pause = _PAUSE_RE.match(stripped)
    if pause is not None:
        seconds = try_parse_duration(pause.group("value"))
        if seconds is not None:
            return ("pause", seconds)

    # -- stage 4: per-segment parameter overrides -------------------------
    if "|" in stripped:
        character, params, ignored = _parse_overrides(stripped)
        return ("overrides", (character, params, ignored))

    return ("text", stripped)


# ---------------------------------------------------------------------------
# Stage 5: the v2 angle-bracket tokens
# ---------------------------------------------------------------------------


def _normalise_v2_token(inner: str) -> Optional[str]:
    """The canonical spelling of a v2 special token, or ``None`` if unknown."""
    candidate = " ".join(inner.split()).lower()
    if candidate not in _V2_TOKEN_LOOKUP:
        return None
    # Two of the thirty are upper case *in the vocabulary* -- ``<UH>`` is not
    # ``<uh>`` -- so they are raised rather than lowered. The other 28 are
    # canonically lower case.
    return candidate.upper() if candidate.upper() in UPPERCASE_V2_TOKENS else candidate


def resolve_v2_tokens(
    text: str,
    *,
    turbo_tags: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Normalise the v2 ``<...>`` special tokens in ``text``.

    Runs **after** the script has been split into segments, which is what makes
    ``[Alice] <giggle>`` work: the character is resolved first, and only then is
    the token rewritten.

    A recognised token is re-spelled canonically -- ``<GIGGLE>`` becomes
    ``<giggle>``, ``< clear throat >``'s underscore spelling is kept, ``<UH>``
    stays upper case -- and reported. An unrecognised ``<something>`` is left
    exactly as written: it may be a real token this table does not name (the
    upstream tokenizer carries 118 and only 30 are documented), and silently
    deleting a word the model can speak is worse than passing it through.

    Args:
        text: One segment's text.
        turbo_tags: Whether the pack speaks Turbo. A Turbo checkpoint's
            tokenizer knows the **square** spelling and has never seen
            ``<giggle>``, so on a Turbo pack a recognised token is re-spelled
            into that vocabulary: ``<giggle>`` becomes ``[giggle]`` and
            ``<clear_throat>`` becomes ``[clear throat]``, the spelling Turbo
            was actually trained on. On every other pack the angle spelling is
            left exactly as it is -- those packs have no tag vocabulary at all,
            and rewriting a token into a spelling that means nothing there
            would be a change for its own sake.

    Returns:
        ``(resolved_text, tokens_seen)``. The second element lists the
        canonical token *names* (``"giggle"``, ``"UH"``, ``"clear_throat"``)
        whatever spelling was emitted for them.
    """
    seen: list[str] = []

    def _replace(match: "re.Match[str]") -> str:
        canonical = _normalise_v2_token(match.group(1))
        if canonical is None:
            return match.group(0)
        seen.append(canonical)
        if not turbo_tags:
            return f"<{canonical}>"
        spelling = TURBO_TAG_V2_SPELLING.get(canonical, canonical)
        return f"[{spelling}]"

    return _ANGLE_RE.sub(_replace, text), tuple(seen)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_segments(
    text: str,
    *,
    turbo_tags: bool = False,
) -> list[TextSegment]:
    """Split ``text`` into :class:`TextSegment` values, resolving stages 1-4.

    A character, language or pause tag **starts a new segment**: ``[Alice] One.
    Two. [Bob] Three.`` is two segments, not one with a mid-sentence switch.
    Consecutive untagged text stays in the segment it is already in, so a plain
    script comes back as exactly one segment.

    A pause tag does not open a segment of its own: its duration lands on
    ``pause_after`` of the segment it follows, which is the only place a
    duration can mean something. A pause with no text before it is dropped.

    Args:
        text: The script.
        turbo_tags: Recognise the square-bracket Turbo tag set.

            On a Turbo pack the tag is **kept exactly as written** and recorded
            in ``segment.turbo_tags``: upstream hands the tag to the tokenizer
            verbatim (``src/chatterbox/tts_turbo.py:401-403``) and never parses
            it, so a Turbo checkpoint only speaks the tag if the text still
            carries it. Deleting it here is what made the feature silent.

            On any other pack the nine Turbo names have no vocabulary to land
            in, so they are kept as literal text too -- recorded as text, not
            as a tag -- and reported, by :func:`plan_text`, as a notice. They
            are deliberately **not** treated as character names: ``[laugh]``
            would be swallowed as a voice called "laugh", and the word would
            leave the script without anyone being told. Every other bare
            bracket keeps today's character behaviour.

    Returns:
        The segments, in order. Empty input yields no segments.
    """
    if not text:
        return []

    segments: list[TextSegment] = []
    current = TextSegment(text="", index=0)
    cursor = 0
    # A pause belongs to the segment it *follows*: ``[Alice] Hi. [pause:0.5]``
    # means 0.5 s of silence after "Hi.". It is held here rather than written
    # straight onto ``current`` because the tag closes the segment it lands on.
    pending_pause = 0.0

    def _flush() -> None:
        nonlocal current, pending_pause
        current.text = current.text.strip()
        if pending_pause:
            current.pause_after += pending_pause
            pending_pause = 0.0
        if current.text or current.character or current.language:
            segments.append(current)

    for match in _BRACKET_RE.finditer(text):
        chunk = text[cursor : match.start()]
        current.text += chunk
        cursor = match.end()
        body = match.group(1)
        tag_name = body.strip().lower()

        if (
            tag_name in _TURBO_TAG_LOOKUP
            and ":" not in body
            and "|" not in body
        ):
            # A Turbo paralinguistic tag. The text is kept exactly as written
            # on both paths: upstream hands the tag to the tokenizer verbatim
            # and never parses it, so a Turbo checkpoint only speaks the tag
            # if the text still carries it -- and a pack that cannot speak it
            # says the word rather than swallowing it. What differs is
            # whether the tag is *recorded* as one, and that is what the
            # caller's ``turbo_tags`` flag means.
            if turbo_tags:
                current.turbo_tags = current.turbo_tags + (tag_name,)
            current.text += match.group(0)
            continue

        kind, payload = _classify_bracket(body)

        if kind == "text":
            # Not a tag: keep the user's brackets rather than eating them.
            current.text += match.group(0)
            continue

        if kind == "pause":
            # A pause with nothing before it has no segment to attach to and is
            # dropped: there is nothing to be silent *after*.
            pending_pause += float(payload)  # type: ignore[arg-type]
            _flush()
            current = TextSegment(text="", index=len(segments))
            continue

        if kind == "overrides":
            character, params, ignored = payload  # type: ignore[misc]
            if ignored:
                logging.debug(
                    "[Chatterbox TTS] Ignoring unsupported per-segment parameters "
                    "%s in a tag on this machine's Chatterbox nodes.",
                    ignored,
                )
            _flush()
            current = TextSegment(
                text="",
                character=character,
                params=dict(params),
                index=len(segments),
            )
            continue

        # ``character`` or ``language``: both open a new segment.
        if kind == "character":
            character, language = payload, None  # type: ignore[assignment]
        else:
            language, character = payload  # type: ignore[misc]
        _flush()
        current = TextSegment(
            text="", character=character, language=language, index=len(segments)
        )

    tail = text[cursor:]
    current.text += tail
    _flush()
    return segments


# ---------------------------------------------------------------------------
# Crash protection (S8.4)
# ---------------------------------------------------------------------------


def apply_crash_protection(
    text: str,
    template: str = CRASH_PROTECTION_TEMPLATE,
    min_chars: int = MIN_SAFE_SEGMENT_CHARS,
) -> str:
    """Pad ``text`` up to a length upstream can survive.

    Applied **per segment**, never per chunk: the upstream crash is a property of
    a single sequential generation call, and chunking does not create one. A
    five-line dialogue of ten-character lines is five short segments, and each
    needs its own padding.

    Args:
        text: The segment's text.
        template: A ``{seg}`` template, or ``""`` to disable.
        min_chars: Length below which padding applies.

    Returns:
        ``text`` unchanged when it is long enough or padding is disabled,
        otherwise the template with ``{seg}`` substituted.
    """
    if not template:
        return text
    if len(text.strip()) >= min_chars:
        return text
    # ``str.replace`` rather than ``str.format``: a user-authored template may
    # legitimately contain other braces, and a KeyError from the crash-protection
    # feature itself would be a spectacular piece of self-harm.
    return template.replace("{seg}", text).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def select_chunk_method(method: str, char_count: int) -> str:
    """Resolve ``chunk_combination_method`` against the script's length.

    ``auto`` picks by size, longest threshold first: **more than 1000**
    characters joins with ``silence_padding``, **more than 500** with
    ``crossfade``, anything shorter with a plain ``concatenate``. Both
    comparisons are strict, so exactly 500 is still ``concatenate``.

    Args:
        method: The widget value.
        char_count: Number of characters in the whole script.

    Returns:
        A concrete method name -- never ``"auto"``.
    """
    if method != "auto":
        return method
    for threshold, resolved in AUTO_CHUNK_THRESHOLDS:
        if char_count > threshold:
            return resolved
    return "concatenate"


def _last_sentence_end(text: str, limit: int) -> int:
    """The index of the last sentence end inside the window, or ``-1``.

    A sentence end is ``.`` / ``!`` / ``?`` followed by a space -- the three
    marks that end a sentence and the space that proves it. The window is
    ``text[:limit]`` plus one character, because a sentence end whose space
    falls just past the window is still a sentence end, and cutting there
    fills the chunk exactly.
    """
    found = -1
    for match in _SENTENCE_END_RE.finditer(text[: limit + 1]):
        if match.start() < limit:
            found = match.start()
    return found


def _bracket_run_at(text: str, position: int) -> Optional[tuple[int, int]]:
    """The ``[...]`` / ``<...>`` run that a cut at ``position`` lands inside.

    Returns ``(open_index, close_index)``, or ``None`` when the cut is not
    inside a run. The run is the one opened by the last opening bracket
    before the cut whose closing bracket is at or after it.
    """
    for opener, closer in _BRACKET_PAIRS:
        start = text.rfind(opener, 0, position)
        if start < 0:
            continue
        end = text.find(closer, start + 1)
        if end < 0:
            continue
        if end >= position:
            return (start, end)
    return None


def _snap_out_of_bracket_run(text: str, take: int, limit: int) -> int:
    """Move a cut that landed inside a bracket run out of it.

    Forward to the closing bracket when that still fits inside the limit,
    otherwise back to before the opening bracket. A run that is longer than
    the limit on its own is never split: the chunk grows past the limit
    instead, which is the one documented exception to the ceiling.
    """
    run = _bracket_run_at(text, take)
    if run is None:
        return take
    start, end = run
    forward = end + 1
    if forward <= limit:
        return forward
    if start > 0:
        return start
    return forward


def chunk_text(
    text: str,
    max_chars_per_chunk: int = 400,
    *,
    enabled: bool = True,
) -> list[str]:
    """Cut ``text`` into chunks of at most ``max_chars_per_chunk`` characters.

    The cut prefers a sentence end, then a line break, then a space, then the
    hard limit -- in that order. Hard-cutting mid-word is the last resort
    because it is the one that sounds wrong, and it only happens on a single
    unbroken run longer than the limit (a URL, a long number).

    A cut never lands **inside** a tag. Half a ``[chuckle]`` in one chunk and
    half in the next is not a tag at all: the model sees two pieces of
    punctuation and says neither. So a cut that falls inside a ``[...]`` or
    ``<...>`` run is moved out of it -- forward to the closing bracket when
    that still fits inside the limit, otherwise back to before the opening
    one. The single documented exception is a tag run that is longer than the
    limit by itself: it stays whole and the chunk grows past the ceiling,
    because splitting it would be worse than exceeding it.

    Args:
        text: The script to cut.
        max_chars_per_chunk: The soft ceiling. Values below 1 disable cutting.
        enabled: When ``False`` the whole script comes back as one chunk.

    Returns:
        The chunks, in order, with empty chunks dropped.
    """
    if not enabled or max_chars_per_chunk < 1 or len(text) <= max_chars_per_chunk:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars_per_chunk:
        limit = max_chars_per_chunk
        window = remaining[:limit]
        # Sentence end, then line break, then space: in that order of preference.
        cut = _last_sentence_end(remaining, limit)
        if cut < 0:
            cut = window.rfind("\n")
        if cut < 0:
            cut = window.rfind(" ")
        if cut <= 0:
            # No boundary inside the window -- one unbreakable run. Consume the
            # whole window, or the loop never advances.
            take = limit
        else:
            # The boundary character belongs to the chunk that ends on it.
            take = cut + 1
        # The cut may have landed inside a tag; move it out rather than split.
        take = _snap_out_of_bracket_run(remaining, take, limit)
        chunks.append(remaining[:take].strip())
        remaining = remaining[take:]

    tail = remaining.strip()
    if tail:
        chunks.append(tail)
    return [chunk for chunk in chunks if chunk]


# ---------------------------------------------------------------------------
# Notices (D8) -- what the engine could not honour
# ---------------------------------------------------------------------------


def _turbo_tag_notice(tag: str) -> str:
    """The notice a Turbo tag earns on a pack that has no Turbo vocabulary."""
    return (
        f"[{tag}] is a Turbo/Nano paralinguistic tag. This pack has no Turbo "
        f"tag vocabulary, so the tag is spoken as literal text -- use a Turbo "
        f"or Nano pack for it to have an effect."
    )


#: A pause is parsed onto its segment and never rendered: nothing in this
#: repository turns ``pause_after`` into silence.
_PAUSE_NOTICE = (
    "A pause tag ([pause:1.5]) is parsed, but nothing renders it as silence: "
    "the pause is not audible."
)

#: A language is parsed onto its segment and never switched: no pack here
#: changes voice or language halfway through a script.
_LANGUAGE_NOTICE = (
    "A language tag ([de:Alice]) is parsed, but no Chatterbox pack switches "
    "language per segment: the whole run speaks one language."
)


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


def _collect_notices(
    segments: list[TextSegment],
    spoken: str,
    *,
    turbo_tags: bool,
) -> tuple[str, ...]:
    """What the engine could not honour, said out loud rather than dropped.

    Three things can be parsed and then not mean anything, and all three are
    announced instead of swallowed:

    * a Turbo tag on a pack that has no Turbo vocabulary -- the word is
      spoken, but it will not do what it looks like it does;
    * a pause, which lands on ``pause_after`` where nothing renders it;
    * a language, which lands on ``segment.language`` where no pack switches.

    A notice is never an error and never changes the audio.
    """
    notices: list[str] = []

    if not turbo_tags:
        # The tags are still in the spoken text -- they were kept, not eaten --
        # so they are found there rather than remembered on the way through.
        for match in _TURBO_TAG_IN_TEXT_RE.finditer(spoken):
            notice = _turbo_tag_notice(match.group(1).lower())
            if notice not in notices:
                notices.append(notice)

    if any(segment.pause_after for segment in segments):
        notices.append(_PAUSE_NOTICE)

    if any(segment.language for segment in segments):
        notices.append(_LANGUAGE_NOTICE)

    return tuple(notices)


def plan_text(
    text: str,
    *,
    turbo_tags: bool = False,
    crash_protection_template: str = CRASH_PROTECTION_TEMPLATE,
    enable_chunking: bool = True,
    max_chars_per_chunk: int = 400,
    chunk_combination_method: str = "auto",
    silence_between_chunks_ms: int = 100,
) -> TextPlan:
    """Parse ``text`` end to end and decide how to synthesise it.

    The order of operations is the one this module's docstring promises, and it
    is the order that matters:

    1. split on character / language / pause / override tags (stages 1-4);
    2. resolve the v2 ``<...>`` tokens, per segment (stage 5);
    3. apply crash protection, **per segment**;
    4. decide the chunking method from the resulting length, then chunk.

    Args:
        text: The script.
        turbo_tags: Recognise the square-bracket Turbo tag set.
        crash_protection_template: ``{seg}`` template, or ``""`` to disable.
        enable_chunking: Whether to cut the script at all.
        max_chars_per_chunk: Soft chunk ceiling in characters.
        chunk_combination_method: ``auto`` / ``concatenate`` /
            ``silence_padding`` / ``crossfade``.
        silence_between_chunks_ms: Gap to insert between chunks when the method
            is ``silence_padding``. Carried on the plan; the audio join lives in
            :mod:`modules.generation`.

    Returns:
        A :class:`TextPlan`. Blank input yields an empty plan with one empty
        chunk, so a caller never has to special-case it. The plan's
        ``notices`` say what the engine could not honour -- a Turbo tag on a
        pack without the Turbo vocabulary, a pause that is never rendered, a
        language that is never switched.
    """
    segments = split_segments(text or "", turbo_tags=turbo_tags)

    resolved: list[TextSegment] = []
    for segment in segments:
        body, _tokens = resolve_v2_tokens(segment.text, turbo_tags=turbo_tags)
        segment.text = apply_crash_protection(body.strip(), crash_protection_template)
        resolved.append(segment)

    spoken = " ".join(segment.text for segment in resolved if segment.text).strip()
    method = select_chunk_method(chunk_combination_method, len(spoken))
    chunks = chunk_text(spoken, max_chars_per_chunk, enabled=enable_chunking)
    if not chunks:
        chunks = [spoken]

    return TextPlan(
        segments=tuple(resolved),
        text=spoken,
        chunks=tuple(chunks),
        chunk_method=method,
        char_count=len(spoken),
        notices=_collect_notices(resolved, spoken, turbo_tags=turbo_tags),
    )
