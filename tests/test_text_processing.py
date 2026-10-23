"""S8.3 -- the tag engine: five features, in one order, that order load-bearing.

A tag engine that runs its passes in the wrong order does not crash. It quietly
produces different speech: ``[Alice] <giggle>`` loses its character if the
angle-bracket pass runs first, ``[de:Alice]`` becomes a voice called
``"de:Alice"`` if the character pass claims it, and ``[pause:1.5]`` says the
word "pause" out loud. None of that is catchable except by asserting the *order*,
which is why the ordering tests below are about which stage produced a value and
not only about what the final text says.

Every documented tag form is parsed here, the three interchangeable pause words
are parsed with each documented unit, unsupported per-segment parameters are
shown to be ignored rather than fatal, and ``auto``'s chunk-method selection is
pinned at every boundary the corpus names -- including 500, where the strict
comparison makes the answer surprising and therefore worth a test.
"""

from __future__ import annotations

import pytest

from ComfyUI_Chatterbox.modules.text_processing import (
    AUTO_CHUNK_THRESHOLDS,
    CHUNK_COMBINATION_METHODS,
    CRASH_PROTECTION_TEMPLATE,
    MIN_SAFE_SEGMENT_CHARS,
    PARAM_ALIASES,
    TURBO_EMOTION_TAGS,
    TURBO_SOUND_TAGS,
    TURBO_TAGS,
    TURBO_TAG_V2_SPELLING,
    V2_SPECIAL_TOKENS,
    apply_crash_protection,
    chunk_text,
    parse_duration,
    plan_text,
    resolve_v2_tokens,
    select_chunk_method,
    split_segments,
)


# ---------------------------------------------------------------------------
# Durations and pause tags (stage 3)
# ---------------------------------------------------------------------------


class TestDurations:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1.5", 1.5),
            ("2s", 2.0),
            ("500ms", 0.5),
            ("800MS", 0.8),
            (" 1200 ms ", 1.2),
            ("3", 3.0),
            ("0.25s", 0.25),
            (".5", 0.5),
        ],
    )
    def test_documented_spellings_parse(self, value: str, expected: float) -> None:
        assert parse_duration(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "abc", "5 parsecs", "-1", "1.2.3"])
    def test_nonsense_is_refused(self, value: str) -> None:
        """A refusal is caught by the caller, which leaves the tag as literal text."""
        with pytest.raises(ValueError):
            parse_duration(value)


class TestPauseTagsAreInterchangeable:
    """``pause`` / ``wait`` / ``stop`` are three spellings of one tag."""

    @pytest.mark.parametrize("word", ["pause", "wait", "stop"])
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1.5", 1.5), ("2s", 2.0), ("500ms", 0.5)],
    )
    def test_all_three_words_read_all_three_units(
        self, word: str, value: str, expected: float
    ) -> None:
        segments = split_segments(f"Hello there. [{word}:{value}] And more words.")
        assert len(segments) == 2, "a pause opens a new segment"
        assert segments[0].pause_after == pytest.approx(expected)
        assert segments[0].text == "Hello there."
        assert segments[1].text == "And more words."
        assert segments[1].pause_after == 0.0

    def test_the_word_is_case_insensitive(self) -> None:
        assert split_segments("Hi. [PAUSE:2s] There.")[0].pause_after == 2.0

    def test_an_unreadable_duration_is_kept_as_text_not_fatal(self) -> None:
        segments = split_segments("Hi. [pause:soon] There.")
        assert len(segments) == 1
        assert "[pause:soon]" in segments[0].text


# ---------------------------------------------------------------------------
# Character switching (stage 1) and language tags (stage 2)
# ---------------------------------------------------------------------------


class TestCharacterTags:
    def test_a_bare_name_opens_a_segment(self) -> None:
        segments = split_segments("[Alice] Hi there, Bob.")
        assert len(segments) == 1
        assert segments[0].character == "Alice"
        assert segments[0].text == "Hi there, Bob."

    def test_an_untagged_run_stays_in_the_same_segment(self) -> None:
        assert len(split_segments("Just talking, with no tags at all.")) == 1

    def test_two_characters_are_two_segments(self) -> None:
        segments = split_segments("[Alice] One. [Bob] Two.")
        assert [s.character for s in segments] == ["Alice", "Bob"]
        assert [s.text for s in segments] == ["One.", "Two."]

    def test_a_language_tag_is_not_a_character_named_de_alice(self) -> None:
        """The regression the ordering exists to prevent."""
        segments = split_segments("[de:Alice] Hallo.")
        assert len(segments) == 1
        assert segments[0].character == "Alice"
        assert segments[0].language == "de"
        assert segments[0].text == "Hallo."

    def test_a_display_name_resolves_to_its_code(self) -> None:
        segments = split_segments("[German:Alice] Hallo.")
        assert segments[0].language == "de"
        assert segments[0].character == "Alice"

    def test_a_language_tag_may_carry_no_character(self) -> None:
        segments = split_segments("[de:] Hallo.")
        assert segments[0].language == "de"
        assert segments[0].character is None

    def test_a_pause_tag_is_not_a_character_named_pause(self) -> None:
        """It is a pause, and a leading one has nothing to be silent after."""
        segments = split_segments("[pause:1.5] Hallo.")
        assert len(segments) == 1
        assert segments[0].character is None
        assert segments[0].text == "Hallo."
        assert segments[0].pause_after == 0.0, (
            "a pause with no text before it has no segment to attach to"
        )


# ---------------------------------------------------------------------------
# Per-segment overrides (stage 4)
# ---------------------------------------------------------------------------


class TestPerSegmentOverrides:
    def test_order_does_not_matter(self) -> None:
        first = split_segments("[Alice|seed:42|temp:0.5] Hi.")[0]
        second = split_segments("[Alice|temp:0.5|seed:42] Hi.")[0]
        assert first.params == second.params
        assert first.params == {"seed": "42", "temperature": "0.5"}

    def test_case_does_not_matter(self) -> None:
        segment = split_segments("[Alice|SEED:42|Temp:0.5] Hi.")[0]
        assert segment.params == {"seed": "42", "temperature": "0.5"}

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("temp", "temperature"), ("cfg", "cfg_weight"),
         ("cfg_weight", "cfg_weight"), ("exag", "exaggeration")],
    )
    def test_documented_aliases_resolve(self, alias: str, canonical: str) -> None:
        segment = split_segments(f"[Alice|{alias}:0.5] Hi.")[0]
        assert segment.params == {canonical: "0.5"}
        assert canonical in PARAM_ALIASES.values()

    def test_an_unsupported_parameter_is_ignored_not_fatal(self) -> None:
        """A script carried over from another tool must still run."""
        segment = split_segments("[Alice|batch:4|temp:0.5] Hi.")[0]
        assert segment.params == {"temperature": "0.5"}
        assert segment.character == "Alice"
        assert segment.text == "Hi."

    def test_a_parameter_with_no_socket_at_all_is_ignored(self) -> None:
        segment = split_segments("[Alice|batch_size:8] Hi.")[0]
        assert segment.params == {}
        assert segment.text == "Hi."


# ---------------------------------------------------------------------------
# v2 angle-bracket tokens (stage 5) -- and the order it runs in
# ---------------------------------------------------------------------------


class TestV2Tokens:
    def test_all_thirty_documented_names_are_known(self) -> None:
        assert len(V2_SPECIAL_TOKENS) == 30
        assert len(set(V2_SPECIAL_TOKENS)) == 30

    @pytest.mark.parametrize("token", V2_SPECIAL_TOKENS)
    def test_every_documented_token_is_recognised(self, token: str) -> None:
        resolved, seen = resolve_v2_tokens(f"Hello <{token}> there.")
        assert seen == (token,), f"<{token}> was not recognised"
        assert resolved == f"Hello <{token}> there."

    def test_casing_is_normalised_except_where_it_is_significant(self) -> None:
        assert resolve_v2_tokens("<GIGGLE>")[0] == "<giggle>"
        assert resolve_v2_tokens("<UH>")[0] == "<UH>"
        assert resolve_v2_tokens("<um>")[0] == "<UM>"

    def test_an_unknown_angle_token_is_left_alone(self) -> None:
        """The upstream tokenizer carries 118 special tokens; only 30 are named.

        Deleting a word the model can still speak is worse than passing it on.
        """
        resolved, seen = resolve_v2_tokens("Hello <not_a_real_token> there.")
        assert seen == ()
        assert resolved == "Hello <not_a_real_token> there."

    def test_a_giggle_after_a_character_resolves_after_the_character(self) -> None:
        """The load-bearing ordering assertion.

        Stage 5 must not have run yet when the character is resolved, or
        ``[Alice] <giggle>`` loses its voice.
        """
        segments = split_segments("[Alice] <giggle> Hello there, nice to meet you.")
        assert len(segments) == 1
        assert segments[0].character == "Alice", (
            "the character was not resolved; stage 5 ran before stage 1"
        )
        assert "<giggle>" in segments[0].text, (
            "the v2 token was rewritten during the character stage; it must still "
            "be untouched text when the character resolves"
        )

        plan = plan_text("[Alice] <giggle> Hello there, nice to meet you.")
        assert plan.segments[0].character == "Alice"
        assert "<giggle>" in plan.text
        assert "[" not in plan.text

    def test_the_same_example_composes_with_a_pause(self) -> None:
        """The corpus's own worked example, end to end."""
        plan = plan_text(
            "[Alice] Hello! <giggle> Nice to meet you. [pause:0.5] How are you? "
            "<whisper> I have a secret."
        )
        assert [s.character for s in plan.segments] == ["Alice", None]
        assert plan.segments[0].pause_after == pytest.approx(0.5)
        assert plan.segments[1].pause_after == 0.0
        assert "<giggle>" in plan.text
        assert "<whisper>" in plan.text


class TestV2TokensOnTheTurboPath:
    """A Turbo checkpoint knows the square spelling and not the angle one.

    The v2 set writes ``<clear_throat>`` with an underscore and Turbo's
    tokenizer was trained on ``[clear throat]`` with a space, so the one
    spelling that differs is published rather than buried.
    """

    def test_v2_tokens_are_untouched_on_the_classic_path(self) -> None:
        assert resolve_v2_tokens("<giggle>")[0] == "<giggle>"

    def test_v2_tokens_become_square_brackets_on_the_turbo_path(self) -> None:
        assert resolve_v2_tokens("<giggle>", turbo_tags=True)[0] == "[giggle]"

    def test_clear_throat_uses_the_turbo_spelling(self) -> None:
        assert resolve_v2_tokens("<clear_throat>", turbo_tags=True)[0] == (
            "[clear throat]"
        )

    def test_uh_keeps_its_casing_in_both_spellings(self) -> None:
        assert resolve_v2_tokens("<UH>")[0] == "<UH>"
        assert resolve_v2_tokens("<UH>", turbo_tags=True)[0] == "[UH]"

    def test_an_unknown_angle_token_is_left_alone_on_the_turbo_path(self) -> None:
        resolved, seen = resolve_v2_tokens("<not_a_real_token>", turbo_tags=True)
        assert seen == ()
        assert resolved == "<not_a_real_token>"

    def test_a_v2_token_after_a_character_keeps_the_character(self) -> None:
        """The conversion happens after the character is resolved, so the
        character survives it."""
        segments = split_segments("[Alice] <giggle> Hello.", turbo_tags=True)
        assert segments[0].character == "Alice"
        assert "<giggle>" in segments[0].text

        plan = plan_text("[Alice] <giggle> Hello.", turbo_tags=True)
        assert plan.segments[0].character == "Alice"
        assert "[giggle]" in plan.text

    def test_the_turbo_plan_text_carries_the_square_spelling(self) -> None:
        # Crash protection is off here: it pads any segment shorter than
        # twenty-one characters, and a two-word script is shorter than that.
        plan = plan_text("<giggle> Hi.", turbo_tags=True, crash_protection_template="")
        assert plan.text == "[giggle] Hi."

    def test_the_only_differing_spelling_is_published(self) -> None:
        assert TURBO_TAG_V2_SPELLING == {"clear_throat": "clear throat"}


class TestTurboTagsAreADifferentVocabulary:
    """``[laugh]`` is a Turbo tag, and a Turbo tag is never eaten."""

    def test_a_turbo_tag_is_preserved_in_the_segment_text(self) -> None:
        """The tag reaches the model exactly as the user typed it.

        Upstream does not parse these tags; it hands the text to the
        tokenizer, which knows the square spelling. A tag that is deleted
        before the model sees it is a feature that looks implemented and is
        not.
        """
        segment = split_segments("[laugh] Hi.", turbo_tags=True)[0]
        assert segment.text == "[laugh] Hi."

    def test_a_turbo_tag_is_still_recorded(self) -> None:
        """Recognition is kept; only the deletion is gone."""
        segment = split_segments("[laugh] Hi.", turbo_tags=True)[0]
        assert segment.turbo_tags == ("laugh",)
        assert segment.character is None

    def test_a_turbo_tag_at_the_start_of_a_segment_is_preserved(self) -> None:
        segment = split_segments("[Alice] [chuckle] Hello.", turbo_tags=True)
        assert segment[0].text == "[chuckle] Hello."
        assert segment[0].turbo_tags == ("chuckle",)
        assert segment[0].character == "Alice"

    def test_two_turbo_tags_are_both_preserved_and_recorded(self) -> None:
        segment = split_segments("[laugh] and [chuckle] that", turbo_tags=True)[0]
        assert segment.text == "[laugh] and [chuckle] that"
        assert segment.turbo_tags == ("laugh", "chuckle")

    def test_the_same_text_is_a_tag_on_the_turbo_path(self) -> None:
        segments = split_segments("[laugh] Hello.", turbo_tags=True)
        assert segments[0].character is None
        assert segments[0].turbo_tags == ("laugh",)

    def test_turbo_tags_are_recognised_case_insensitively(self) -> None:
        assert split_segments("[Clear Throat] Hi.", turbo_tags=True)[0].turbo_tags == (
            "clear throat",
        )

    def test_the_turbo_set_is_not_the_v2_set(self) -> None:
        """Nineteen square-bracket tokens against thirty angle-bracket ones.

        Five names are shared between the two vocabularies (``sigh``, ``gasp``,
        ``cough``, ``sniff``, ``groan``); what differs is the *syntax* each
        vocabulary uses, which is the point of this module.
        """
        assert len(TURBO_TAGS) == 19
        assert len(V2_SPECIAL_TOKENS) == 30
        shared = set(TURBO_TAGS) & {token.lower() for token in V2_SPECIAL_TOKENS}
        assert shared == {"sigh", "gasp", "cough", "sniff", "groan"}
        assert "clear throat" not in V2_SPECIAL_TOKENS
        assert "clear_throat" not in TURBO_TAGS

    def test_an_angle_token_is_still_normalised_on_the_turbo_path(self) -> None:
        segments = split_segments("[laugh] <giggle> Hi.", turbo_tags=True)
        assert segments[0].turbo_tags == ("laugh",)
        assert "<giggle>" in segments[0].text


class TestATurboTagIsNotAVoiceName:
    """On a pack that cannot speak them, Turbo tags are text -- not a name.

    Treating ``[laugh]`` as a voice called "laugh" swallows the word: the
    script says nothing where the user wrote a direction, and the voice never
    changes either, because this repository ships no voice library.
    """

    def test_a_turbo_tag_name_on_the_classic_path_is_not_a_character(self) -> None:
        segment = split_segments("[laugh] Hello.")[0]
        assert segment.character is None
        assert "[laugh]" in segment.text

    def test_an_unknown_bracket_on_the_classic_path_is_still_a_character(self) -> None:
        """The unchanged behaviour: every other bare bracket is a name."""
        segment = split_segments("[Alice] Hello.")[0]
        assert segment.character == "Alice"
        assert segment.text == "Hello."

    def test_a_turbo_tag_name_on_the_classic_path_produces_a_notice(self) -> None:
        notices = plan_text("[laugh] Hi.").notices
        assert notices, "the tag was swallowed without a word"
        assert any("Turbo" in notice and "laugh" in notice for notice in notices)

    def test_the_classic_path_does_not_record_the_tag_as_a_turbo_tag(self) -> None:
        """It is not a Turbo tag here -- it is the word the user typed."""
        assert split_segments("[laugh] Hello.")[0].turbo_tags == ()

    @pytest.mark.parametrize("body", ["de:Alice", "Alice|seed:42", "Alice|temp:0.5"])
    def test_a_bracket_with_a_colon_or_pipe_is_never_treated_as_a_turbo_tag(
        self, body: str
    ) -> None:
        """Only the nineteen bare names are Turbo tags; the other forms keep
        their own meaning."""
        segment = split_segments(f"[{body}] Hello.")[0]
        assert segment.turbo_tags == ()

    def test_a_turbo_tag_is_not_noticed_on_the_turbo_path(self) -> None:
        """It is native there, so there is nothing to announce."""
        plan = plan_text("[laugh] Hi.", turbo_tags=True)
        assert not [notice for notice in plan.notices if "Turbo" in notice]


# ---------------------------------------------------------------------------
# The upstream Turbo/Nano vocabulary -- nineteen names, nine sounds and ten
# emotions
# ---------------------------------------------------------------------------

#: Upstream's own spellings, hard-coded. The module's table is checked against
#: these lists, never the other way round: the tokenizer's ``added_tokens.json``
#: is the truth, and a table that drifts away from it is the bug this section
#: exists to prevent.
UPSTREAM_SOUND_TAGS: tuple[str, ...] = (
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

UPSTREAM_EMOTION_TAGS: tuple[str, ...] = (
    "angry",
    "fear",
    "surprised",
    "whispering",
    "advertisement",
    "dramatic",
    "narration",
    "crying",
    "happy",
    "sarcastic",
)

UPSTREAM_TURBO_TAGS: tuple[str, ...] = (
    "laugh",
    "sigh",
    "gasp",
    "chuckle",
    "cough",
    "sniff",
    "groan",
    "shush",
    "clear throat",
    "angry",
    "fear",
    "surprised",
    "whispering",
    "advertisement",
    "dramatic",
    "narration",
    "crying",
    "happy",
    "sarcastic",
)


class TestTheUpstreamTagTable:
    """The table is the whole fix, so the table is pinned against upstream."""

    def test_the_table_holds_the_19_upstream_spellings(self) -> None:
        assert len(TURBO_TAGS) == 19
        assert set(TURBO_TAGS) == set(UPSTREAM_TURBO_TAGS)

    def test_the_two_groups_split_9_and_10(self) -> None:
        assert len(TURBO_SOUND_TAGS) == 9
        assert set(TURBO_SOUND_TAGS) == set(UPSTREAM_SOUND_TAGS)
        assert len(TURBO_EMOTION_TAGS) == 10
        assert set(TURBO_EMOTION_TAGS) == set(UPSTREAM_EMOTION_TAGS)

    def test_the_union_is_the_two_groups(self) -> None:
        """One list for the claim, the lookup and the notice regex."""
        assert TURBO_TAGS == TURBO_SOUND_TAGS + TURBO_EMOTION_TAGS

    def test_the_clear_throat_spelling_has_a_space_not_an_underscore(self) -> None:
        """The tokenizer was trained on ``[clear throat]``; the v2 set's
        ``<clear_throat>`` is a different token."""
        assert "clear throat" in TURBO_TAGS
        assert "clear_throat" not in TURBO_TAGS

    def test_no_tag_is_a_prefix_of_another(self) -> None:
        """The guard that keeps the notice regex's alternation safe.

        The alternation is built from the table in order, so a shorter name
        that is a prefix of a longer one would shadow it: ``[cough]`` would win
        over ``[coughing]`` and the longer tag would never be noticed.
        """
        for tag in TURBO_TAGS:
            for other in TURBO_TAGS:
                if tag == other:
                    continue
                assert not other.startswith(tag), f"{tag!r} shadows {other!r}"


@pytest.mark.parametrize("tag", UPSTREAM_TURBO_TAGS)
class TestEveryUpstreamTagBehavesLikeATurboTag:
    """What the nine sound tags already did, and the ten emotion tags now do.

    The claim, the recording and the notice are all name-driven, so widening
    the table widens every behaviour below: no new code path, no new state.
    """

    def test_it_survives_the_plan_verbatim(self, tag: str) -> None:
        """The tag reaches the tokenizer exactly as upstream spells it.

        Crash protection is off here: it pads any segment shorter than
        twenty-one characters, and a two-word script is shorter than that.
        """
        plan = plan_text(f"[{tag}] Hi.", turbo_tags=True, crash_protection_template="")
        assert plan.text == f"[{tag}] Hi."
        assert plan.segments[0].text == f"[{tag}] Hi."

    def test_it_is_recorded(self, tag: str) -> None:
        segment = split_segments(f"[{tag}] Hi.", turbo_tags=True)[0]
        assert segment.turbo_tags == (tag,)

    def test_it_is_not_a_character_name(self, tag: str) -> None:
        """Neither path: a tag is never swallowed as a voice called "happy"."""
        turbo = split_segments(f"[{tag}] Hi.", turbo_tags=True)[0]
        assert turbo.character is None
        classic = split_segments(f"[{tag}] Hi.")[0]
        assert classic.character is None

    def test_the_classic_path_keeps_the_tag(self, tag: str) -> None:
        """A pack with no Turbo vocabulary says the word rather than eating it."""
        plan = plan_text(f"[{tag}] Hi.", crash_protection_template="")
        assert plan.text == f"[{tag}] Hi."

    def test_the_classic_path_reports_one_notice(self, tag: str) -> None:
        """One notice, naming the tag and the family of packs that speak it."""
        plan = plan_text(f"[{tag}] Hi.", crash_protection_template="")
        assert len(plan.notices) == 1
        notice = plan.notices[0]
        assert f"[{tag}]" in notice
        assert "Turbo" in notice
        assert "Nano" in notice


class TestAMixedCaseTag:
    """Recognition folds the case; the text does not.

    Upstream's added tokens are exact strings, so ``[HAPPY]`` is not the token
    the model was trained on. Re-spelling it here would change what the user
    typed, so the tag is recognised -- and recorded as ``happy`` -- and handed
    on exactly as it was written.
    """

    def test_it_is_recognised_and_recorded_in_lower_case(self) -> None:
        segment = split_segments("[HAPPY] Hi.", turbo_tags=True)[0]
        assert segment.turbo_tags == ("happy",)
        assert segment.character is None

    def test_the_text_is_kept_as_typed(self) -> None:
        segment = split_segments("[HAPPY] Hi.", turbo_tags=True)[0]
        assert segment.text == "[HAPPY] Hi."
        plan = plan_text("[HAPPY] Hi.", turbo_tags=True, crash_protection_template="")
        assert plan.text == "[HAPPY] Hi."


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestAutoChunkMethodBoundaries:
    """``auto`` selects by length, and both comparisons are strict.

    The corpus phrases the rule as "<500 -> concatenate" while the thresholds are
    written as ">1000" and ">500". Those disagree at exactly 500, so the strict
    reading is implemented and 500 is pinned here -- a rule with an unstated
    boundary is a rule with a bug in it.
    """

    @pytest.mark.parametrize(
        ("chars", "expected"),
        [
            (499, "concatenate"),
            (500, "concatenate"),
            (501, "crossfade"),
            (1000, "crossfade"),
            (1001, "silence_padding"),
            (5000, "silence_padding"),
        ],
    )
    def test_the_documented_boundaries(self, chars: int, expected: str) -> None:
        assert select_chunk_method("auto", chars) == expected
        assert plan_text("x" * chars).chunk_method == expected

    def test_the_thresholds_are_published_not_implicit(self) -> None:
        assert AUTO_CHUNK_THRESHOLDS == ((1000, "silence_padding"), (500, "crossfade"))

    @pytest.mark.parametrize(
        "method", [name for name in CHUNK_COMBINATION_METHODS if name != "auto"]
    )
    def test_an_explicit_method_passes_through(self, method: str) -> None:
        """Whatever the user picked is what runs, at any length.

        ``auto`` is the only value that gets *replaced*; the other three are
        names, not hints, and a short script must not quietly switch the user
        off ``silence_padding`` because it was under a thousand characters.
        """
        assert select_chunk_method(method, 10) == method
        assert select_chunk_method(method, 5000) == method

    def test_the_resolved_method_is_never_auto(self) -> None:
        for chars in (0, 1, 500, 5000):
            assert select_chunk_method("auto", chars) != "auto"


class TestChunking:
    def test_a_short_script_is_one_chunk(self) -> None:
        assert chunk_text("Short enough.", max_chars_per_chunk=400) == ["Short enough."]

    def test_a_long_script_is_cut(self) -> None:
        text = "A sentence here. " * 100
        chunks = chunk_text(text, 200)
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
        # No word is lost or split in half on the way through.
        assert text.split() == " ".join(chunks).split()

    def test_cutting_is_disabled_by_the_flag(self) -> None:
        text = "A sentence here. " * 100
        assert chunk_text(text, 200, enabled=False) == [text]

    def test_a_crash_protected_short_script_is_still_short(self) -> None:
        plan = plan_text("Hi", enable_chunking=True, max_chars_per_chunk=400)
        assert len(plan.chunks) == 1


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


class TestPlanText:
    def test_a_plain_script_yields_one_segment_and_one_chunk(self) -> None:
        plan = plan_text("Hello there, this is a plain script.")
        assert len(plan.segments) == 1
        assert plan.chunks == ("Hello there, this is a plain script.",)
        assert plan.char_count == len(plan.text)

    def test_blank_input_is_an_empty_plan_not_an_error(self) -> None:
        plan = plan_text("   ")
        assert plan.segments == ()
        assert plan.chunks == ("",)

    def test_characters_are_listed_in_first_appearance_order(self) -> None:
        plan = plan_text("[Bob] One. [Alice] Two. [Bob] Three.")
        assert plan.characters == ("Bob", "Alice")

    def test_the_default_template_is_the_documented_one(self) -> None:
        assert CRASH_PROTECTION_TEMPLATE == "hmm ,, {seg} hmm ,,"
        assert MIN_SAFE_SEGMENT_CHARS == 21
        assert apply_crash_protection("Short.").startswith("hmm ,,")

    def test_auto_is_unchanged_by_the_new_boundary_rule(self) -> None:
        """The tag-safe cut changes *where* a script is cut, not *how long* it
        is: ``auto`` still resolves on the script's length alone."""
        assert plan_text("x" * 500).chunk_method == "concatenate"
        assert plan_text("x" * 501).chunk_method == "crossfade"
        assert plan_text("x" * 1001).chunk_method == "silence_padding"

    def test_a_script_with_tags_resolves_auto_on_its_length(self) -> None:
        script = "[laugh] " + ("x" * 600)
        assert plan_text(script).chunk_method == "crossfade"


# ---------------------------------------------------------------------------
# Notices -- what the engine could not honour
# ---------------------------------------------------------------------------


class TestNotices:
    """A tag that is parsed and then means nothing is announced, not dropped.

    The alternative is a script that quietly loses a word, and a user who has
    no idea why ``[laugh]`` did nothing.
    """

    def test_a_plain_script_has_no_notices(self) -> None:
        assert plan_text("Hello there, this is a plain script.").notices == ()

    def test_a_turbo_tag_on_a_non_turbo_pack_is_noticed(self) -> None:
        notices = plan_text("[laugh] Hi.").notices
        assert len(notices) == 1
        assert "[laugh]" in notices[0]
        assert "Turbo" in notices[0]

    def test_the_same_tag_is_not_noticed_twice(self) -> None:
        plan = plan_text("[laugh] Hi. [laugh] Bye.")
        assert len([notice for notice in plan.notices if "Turbo" in notice]) == 1

    def test_a_pause_is_noticed_rather_than_silently_dropped(self) -> None:
        notices = plan_text("Hi there. [pause:1.5] And more.").notices
        assert any("pause" in notice.lower() for notice in notices)

    def test_a_language_is_noticed_rather_than_silently_dropped(self) -> None:
        notices = plan_text("[de:Alice] Hallo.").notices
        assert any("language" in notice.lower() for notice in notices)

    def test_a_notice_never_changes_the_audio(self) -> None:
        """A notice is a note about the text, not a change to it: the same
        words are spoken on either path."""
        classic = plan_text("[laugh] Hi there, this is long enough to speak.")
        turbo = plan_text(
            "[laugh] Hi there, this is long enough to speak.", turbo_tags=True
        )
        assert classic.text == turbo.text
        assert classic.chunks == turbo.chunks
        assert classic.notices != ()
        assert turbo.notices == ()
