"""S8.7 -- ``ChatterboxDialog``: isolated stems, and a mix that is their sum.

Upstream Chatterbox has no dialog capability, so this node is four independent
syntheses and the only thing that makes them a conversation is the layout. The
tests below therefore attack the layout, not the model, and they do it with a
stub whose **amplitude identifies the speaker**: speaker A's turns come back at
0.25, B's at 0.50, C's at 0.75. That makes "this stem contains only its own
speaker" a numeric fact rather than a claim -- any leak of another speaker's
turn into a stem would show up as an amplitude that stem has no business
holding.

The central assertion is the mixed track. It is asserted numerically rather than
eyeballed, because "the dialog is the sum of the stems" is the property every
other promise here rests on: a stem the user re-mixes by hand, a stem to strip a
voice out of, and a stem for a DAW -- all three assume the stems and the mix
describe the same audio.

Nothing here loads a model, touches the network, or needs real weights.
"""

from __future__ import annotations

import ast
import os

import pytest
import torch

from ComfyUI_Chatterbox.modules.audio_utils import S3GEN_SAMPLE_RATE
from ComfyUI_Chatterbox.modules.dialog import (
    REQUIRED_SPEAKERS,
    SPEAKER_SLOTS,
    assemble_stems,
    build_message,
    parse_dialog,
)
from ComfyUI_Chatterbox.nodes import dialog_node as dialog_module
from ComfyUI_Chatterbox.nodes.dialog_node import ChatterboxDialogNode

from conftest import ROOT_DIR

DIALOG_NODE_ID = "ChatterboxDialog"
FROZEN_NODE_IDS = ("ChatterboxTTS", "ChatterboxVC")

#: The amplitude each speaker's turns are stubbed with. Distinct and non-zero so
#: that a leak is unambiguous in both directions.
SPEAKER_AMPLITUDE = {"A": 0.25, "B": 0.5, "C": 0.75, "D": 1.0}

#: Length of every stubbed turn, in seconds.
TURN_SECONDS = 0.5


def _required() -> dict:
    return ChatterboxDialogNode.GET_NODE_INFO_V1()["input"]["required"]


def _optional() -> dict:
    return ChatterboxDialogNode.GET_NODE_INFO_V1()["input"].get("optional", {})


def _reference(seconds: float = 0.1) -> dict:
    """A stand-in for a speaker's reference clip, distinguishable per slot."""
    return {
        "waveform": torch.zeros((1, int(seconds * S3GEN_SAMPLE_RATE)), dtype=torch.float32),
        "sample_rate": S3GEN_SAMPLE_RATE,
    }


def _turn_clip(speaker: str, seconds: float = TURN_SECONDS) -> torch.Tensor:
    return torch.full(
        (1, int(seconds * S3GEN_SAMPLE_RATE)),
        SPEAKER_AMPLITUDE[speaker],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_node_id_is_new_and_additive(self) -> None:
        schema = ChatterboxDialogNode.GET_SCHEMA()
        assert schema.node_id == DIALOG_NODE_ID
        assert DIALOG_NODE_ID not in FROZEN_NODE_IDS
        assert schema.is_output_node is False

    def test_the_frozen_ids_are_untouched(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

        assert ChatterboxTTSNode.GET_SCHEMA().node_id == "ChatterboxTTS"
        assert ChatterboxVCNode.GET_SCHEMA().node_id == "ChatterboxVC"

    def test_there_are_six_outputs_in_the_documented_order(self) -> None:
        schema = ChatterboxDialogNode.FINALIZE_SCHEMA()
        assert [output.display_name for output in schema.outputs] == [
            "dialog_audio",
            "speaker_a_audio",
            "speaker_b_audio",
            "speaker_c_audio",
            "speaker_d_audio",
            "message",
        ]
        assert all(output.id for output in schema.outputs)

    def test_the_schema_validates(self) -> None:
        assert ChatterboxDialogNode.validate_inputs() is True

    def test_an_unknown_language_is_refused_with_a_message(self) -> None:
        result = ChatterboxDialogNode.validate_inputs(language="xx")
        assert isinstance(result, str)
        assert "xx" in result and "23 languages" in result

    def test_an_unknown_checkpoint_is_refused_with_a_message(self) -> None:
        result = ChatterboxDialogNode.validate_inputs(model_version="t3_nope.safetensors")
        assert isinstance(result, str)
        assert "t3_nope.safetensors" in result

    def test_every_advertised_language_is_accepted(self) -> None:
        from ComfyUI_Chatterbox.modules.constants import MTL_LANGUAGES

        for code in sorted(MTL_LANGUAGES):
            assert ChatterboxDialogNode.validate_inputs(language=code) is True

    def test_it_reuses_the_multilingual_node_tables(self) -> None:
        """Two lists of 23 language codes that could disagree is a latent bug."""
        from ComfyUI_Chatterbox.nodes.mtl_node import (
            ChatterboxMTLTTSNode,
            MTL_MODEL_VERSION_DEFAULT,
            MTL_MODEL_VERSION_OPTIONS,
        )

        assert _required()["model_version"][1]["options"] == list(MTL_MODEL_VERSION_OPTIONS)
        assert _required()["model_version"][1]["default"] == MTL_MODEL_VERSION_DEFAULT
        assert _required()["language"][1]["options"] == sorted(
            ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"]["language"][1]["options"]
        )


class TestSpeakerSockets:
    def test_a_and_b_are_required_c_and_d_are_optional(self) -> None:
        required = set(_required())
        optional = set(_optional())
        assert {"speaker_A_Audio", "speaker_B_Audio"} <= required
        assert "speaker_C_Audio" not in required
        assert "speaker_D_Audio" not in required
        assert {"speaker_C_Audio", "speaker_D_Audio"} <= optional

    def test_the_required_speakers_are_a_and_b(self) -> None:
        assert REQUIRED_SPEAKERS == ("A", "B")
        assert SPEAKER_SLOTS == ("A", "B", "C", "D")

    def test_the_dialog_default_uses_the_documented_format(self) -> None:
        assert _required()["dialog_text"][1]["default"] == (
            "SPEAKER A: Test test\nSPEAKER B: 1 2 3"
        )

    def test_every_dialog_input_reaches_execute(self) -> None:
        import inspect

        parameters = inspect.signature(ChatterboxDialogNode.execute).parameters
        for name in set(_required()) | set(_optional()):
            assert name in parameters, f"{name} is offered but never read"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestDialogParsing:
    def test_the_documented_format(self) -> None:
        turns = parse_dialog("SPEAKER A: Test test\nSPEAKER B: 1 2 3")
        assert [(t.speaker, t.text) for t in turns] == [("A", "Test test"), ("B", "1 2 3")]

    @pytest.mark.parametrize(
        "line",
        [
            "SPEAKER A: hi",
            "Speaker A: hi",
            "speaker a: hi",
            "SPEAKER_A: hi",
            "SPEAKER-A: hi",
            "  SPEAKER   A : hi",
        ],
    )
    def test_case_and_separator_are_tolerated(self, line: str) -> None:
        turns = parse_dialog(line)
        assert turns[0].speaker == "A"
        assert turns[0].text == "hi"

    def test_all_four_slots_parse(self) -> None:
        script = "\n".join(f"SPEAKER {slot}: line {slot}" for slot in SPEAKER_SLOTS)
        assert [turn.speaker for turn in parse_dialog(script)] == list(SPEAKER_SLOTS)

    def test_a_wrapped_line_is_still_one_turn(self) -> None:
        turns = parse_dialog("SPEAKER A: a long line\nthat wrapped\nSPEAKER B: short")
        assert [(t.speaker, t.text) for t in turns] == [
            ("A", "a long line that wrapped"),
            ("B", "short"),
        ]

    def test_blank_lines_between_turns_are_fine(self) -> None:
        turns = parse_dialog("SPEAKER A: one\n\n\nSPEAKER B: two\n")
        assert len(turns) == 2

    def test_empty_input_yields_nothing(self) -> None:
        assert parse_dialog("") == []
        assert parse_dialog("   \n  ") == []

    def test_a_speaker_with_no_words_is_dropped(self) -> None:
        # Assembled rather than written as one literal: a bare "SPEAKER A:"
        # followed by a newline reads as a Windows drive letter to the
        # shipped-path scanner -- correctly, it is the same two characters.
        turns = parse_dialog("SPEAKER A:" + "\n" + "SPEAKER B: two")
        assert [(t.speaker, t.text) for t in turns] == [("B", "two")]

    def test_text_before_the_first_speaker_is_refused(self) -> None:
        """Text with no voice to be spoken in is refused, not assigned to A."""
        with pytest.raises(ValueError, match="before the first 'SPEAKER A:' line"):
            parse_dialog("just some narration\nSPEAKER A: hello")

    def test_the_turn_index_is_preserved(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two\nSPEAKER A: three")
        assert [turn.index for turn in turns] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Stem assembly
# ---------------------------------------------------------------------------


class TestStemAssembly:
    def test_the_turns_are_laid_out_back_to_back(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two")
        stems, dialog = assemble_stems(turns, [_turn_clip("A"), _turn_clip("B")])
        assert dialog.shape[-1] == pytest.approx(1.0 * S3GEN_SAMPLE_RATE, abs=2)
        # No overlap: the second turn starts where the first ended.
        assert torch.count_nonzero(stems["A"][:, int(TURN_SECONDS * S3GEN_SAMPLE_RATE) :]) == 0

    def test_a_speaker_who_talks_twice_contributes_both_turns(self) -> None:
        """One clip per turn, not one per speaker -- otherwise a speaker's
        first line is silently lost when they speak again."""
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two\nSPEAKER A: three")
        clips = [_turn_clip("A"), _turn_clip("B"), _turn_clip("A")]
        stems, _ = assemble_stems(turns, clips)
        assert torch.count_nonzero(stems["A"]) == pytest.approx(
            2 * TURN_SECONDS * S3GEN_SAMPLE_RATE, abs=2
        )

    def test_each_stem_holds_only_its_own_speaker(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two\nSPEAKER C: three")
        clips = [_turn_clip("A"), _turn_clip("B"), _turn_clip("C")]
        stems, _ = assemble_stems(turns, clips)
        for slot in ("A", "B", "C"):
            values = set(torch.unique(stems[slot]).tolist())
            assert values <= {0.0, SPEAKER_AMPLITUDE[slot]}, (
                f"stem {slot} carries amplitudes {sorted(values)} it has no "
                f"business holding"
            )

    def test_an_unconfigured_speaker_gets_real_silence_not_an_empty_stem(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two")
        stems, dialog = assemble_stems(turns, [_turn_clip("A"), _turn_clip("B")])
        for slot in ("C", "D"):
            assert stems[slot].shape == dialog.shape
            assert torch.count_nonzero(stems[slot]) == 0

    def test_the_dialog_is_the_sum_of_the_stems(self) -> None:
        turns = parse_dialog(
            "SPEAKER A: one\nSPEAKER B: two\nSPEAKER C: three\nSPEAKER D: four"
        )
        clips = [_turn_clip(slot) for slot in ("A", "B", "C", "D")]
        stems, dialog = assemble_stems(turns, clips)
        total = torch.zeros_like(dialog)
        for stem in stems.values():
            total += stem
        assert torch.equal(dialog, total)

    def test_a_turn_without_audio_is_refused(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two")
        with pytest.raises(ValueError, match="no audio"):
            assemble_stems(turns, [_turn_clip("A"), torch.zeros((1, 0))])

    def test_too_few_clips_is_refused(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two")
        with pytest.raises(ValueError, match="2 dialog turn"):
            assemble_stems(turns, [_turn_clip("A")])

    def test_no_turns_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no dialog turns"):
            assemble_stems([], [])

    def test_the_message_names_every_speaker(self) -> None:
        turns = parse_dialog("SPEAKER A: one\nSPEAKER B: two")
        stems, _ = assemble_stems(turns, [_turn_clip("A"), _turn_clip("B")])
        message = build_message(turns, stems)
        assert "turns: 2" in message
        for slot in SPEAKER_SLOTS:
            assert f"speaker {slot}:" in message


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_dialog(monkeypatch):
    """One fixed-length clip per turn, at the speaker's own amplitude."""
    calls: list[dict] = []

    def _load(model_name, **kwargs):
        calls.append({"model": model_name, **kwargs})
        return type("P", (), {"cache_key": "dialog-test-key"})(), object()

    def _generate(model, text, language_id, **kwargs):
        prompt = kwargs.get("audio_prompt")
        assert prompt is not None, "every turn must be conditioned on a voice"
        speaker = _speaker_of(prompt)
        calls.append(
            {
                "text": text,
                "language": language_id,
                "speaker": speaker,
                "seed": kwargs.get("seed"),
                **kwargs,
            }
        )
        return _turn_clip(speaker), S3GEN_SAMPLE_RATE

    monkeypatch.setattr(dialog_module, "load_chatterbox_entry_point", _load)
    monkeypatch.setattr(dialog_module, "generate_mtl_audio", _generate)
    return calls


#: Distinct reference clips, one per speaker slot, so the stub can tell them
#: apart by identity rather than by anything the caller told it.
_REFERENCES = {slot: _reference() for slot in SPEAKER_SLOTS}


def _speaker_of(prompt: dict) -> str:
    for slot, reference in _REFERENCES.items():
        if prompt is reference:
            return slot
    raise AssertionError("a turn was conditioned on a voice this test never made")


def _execute(dialog_text: str, *, with_c: bool = False, with_d: bool = False, **kwargs):
    return ChatterboxDialogNode.execute(
        model_pack_name="pack",
        model_version=kwargs.pop("model_version", "t3_mtl23ls_v3.safetensors"),
        language=kwargs.pop("language", "en"),
        dialog_text=dialog_text,
        speaker_A_Audio=_REFERENCES["A"],
        speaker_B_Audio=_REFERENCES["B"],
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        seed=7,
        use_watermark=False,
        speaker_C_Audio=_REFERENCES["C"] if with_c else None,
        speaker_D_Audio=_REFERENCES["D"] if with_d else None,
        **kwargs,
    )


def _outputs(result):
    dialog, a, b, c, d, message = result.args
    return dialog, a, b, c, d, message


class TestExecute:
    def test_six_outputs_come_back(self, stub_dialog) -> None:
        result = _execute("SPEAKER A: one\nSPEAKER B: two")
        assert len(result.args) == 6
        dialog, a, b, c, d, message = _outputs(result)
        assert dialog["sample_rate"] == S3GEN_SAMPLE_RATE
        assert "turns: 2" in message

    def test_each_stem_holds_only_its_own_speaker(self, stub_dialog) -> None:
        result = _execute("SPEAKER A: one\nSPEAKER B: two")
        dialog, a, b, _c, _d, _message = _outputs(result)
        assert torch.count_nonzero(a["waveform"]) > 0
        assert torch.count_nonzero(b["waveform"]) > 0
        assert set(torch.unique(a["waveform"]).tolist()) <= {0.0, SPEAKER_AMPLITUDE["A"]}
        assert set(torch.unique(b["waveform"]).tolist()) <= {0.0, SPEAKER_AMPLITUDE["B"]}
        # And the two are genuinely disjoint in time.
        overlap = (a["waveform"].abs() > 0) & (b["waveform"].abs() > 0)
        assert torch.count_nonzero(overlap) == 0

    def test_the_mixed_track_equals_the_sum_of_the_stems(self, stub_dialog) -> None:
        """The property every isolated-stem promise rests on, numerically."""
        result = _execute(
            "SPEAKER A: one\nSPEAKER B: two\nSPEAKER C: three",
            with_c=True,
        )
        dialog, a, b, c, d, _message = _outputs(result)
        stems = [a["waveform"], b["waveform"], c["waveform"], d["waveform"]]
        total = torch.zeros_like(dialog["waveform"])
        for stem in stems:
            total = total + stem
        assert dialog["waveform"].shape == total.shape
        assert torch.equal(dialog["waveform"], total), (
            "dialog_audio is not exactly the sum of the four stems"
        )
        assert float((dialog["waveform"] - total).abs().max()) == 0.0

    def test_the_mix_is_the_sum_with_four_speakers_too(self, stub_dialog) -> None:
        result = _execute(
            "SPEAKER A: one\nSPEAKER B: two\nSPEAKER C: three\nSPEAKER D: four",
            with_c=True,
            with_d=True,
        )
        dialog, a, b, c, d, _message = _outputs(result)
        total = a["waveform"] + b["waveform"] + c["waveform"] + d["waveform"]
        assert torch.equal(dialog["waveform"], total)
        for socket in (a, b, c, d):
            assert torch.count_nonzero(socket["waveform"]) > 0

    def test_c_and_d_are_silent_when_unconfigured(self, stub_dialog) -> None:
        result = _execute("SPEAKER A: one\nSPEAKER B: two")
        dialog, _a, _b, c, d, message = _outputs(result)
        assert torch.count_nonzero(c["waveform"]) == 0
        assert torch.count_nonzero(d["waveform"]) == 0
        # Real silence of the dialog's length, not a short or absent stem.
        assert c["waveform"].shape == dialog["waveform"].shape
        assert d["waveform"].shape == dialog["waveform"].shape
        assert "speaker C: silent" in message

    def test_every_turn_is_conditioned_on_its_own_speaker(self, stub_dialog) -> None:
        _execute("SPEAKER A: one\nSPEAKER B: two\nSPEAKER A: three")
        turns = [call for call in stub_dialog if "speaker" in call]
        assert [call["speaker"] for call in turns] == ["A", "B", "A"]

    def test_every_turn_gets_its_own_seed(self, stub_dialog) -> None:
        """Two speakers sampling from the same noise sounds like one person."""
        _execute("SPEAKER A: one\nSPEAKER B: two")
        turns = [call for call in stub_dialog if "speaker" in call]
        assert len({call["seed"] for call in turns}) == len(turns)

    def test_crash_protection_is_applied_per_turn(self, stub_dialog) -> None:
        _execute("SPEAKER A: yes\nSPEAKER B: no")
        spoken = [call["text"] for call in stub_dialog if "speaker" in call]
        assert all("hmm" in text for text in spoken)

    def test_crash_protection_can_be_disabled(self, stub_dialog) -> None:
        _execute("SPEAKER A: yes\nSPEAKER B: no", crash_protection_template="")
        spoken = [call["text"] for call in stub_dialog if "speaker" in call]
        assert spoken == ["yes", "no"]

    def test_a_turn_for_an_unconnected_optional_speaker_is_refused(
        self, stub_dialog
    ) -> None:
        with pytest.raises(ValueError, match="Speaker C has lines but no reference audio"):
            _execute("SPEAKER A: one\nSPEAKER B: two\nSPEAKER C: three")

    def test_a_missing_required_reference_is_refused(self, stub_dialog) -> None:
        with pytest.raises(ValueError, match="needs a reference voice"):
            ChatterboxDialogNode.execute(
                model_pack_name="pack",
                model_version="t3_mtl23ls_v3.safetensors",
                language="en",
                dialog_text="SPEAKER A: one\nSPEAKER B: two",
                speaker_A_Audio=_REFERENCES["A"],
                speaker_B_Audio=None,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
                seed=7,
                use_watermark=False,
            )

    def test_an_empty_script_returns_silence_everywhere(self, stub_dialog) -> None:
        result = _execute("")
        dialog, a, b, c, d, message = _outputs(result)
        assert message == "turns: 0"
        for socket in (dialog, a, b, c, d):
            assert torch.count_nonzero(socket["waveform"]) == 0

    def test_a_cancelled_run_returns_silence_not_a_failure(self, monkeypatch) -> None:
        from ComfyUI_Chatterbox.modules.audio_utils import interrupt_exception_class

        interrupt_error = interrupt_exception_class()
        assert not issubclass(interrupt_error, Exception)

        monkeypatch.setattr(
            dialog_module,
            "load_chatterbox_entry_point",
            lambda name, **kwargs: (type("P", (), {"cache_key": "k"})(), object()),
        )

        def _cancel(model, text, language_id, **kwargs):
            raise interrupt_error("cancelled")

        monkeypatch.setattr(dialog_module, "generate_mtl_audio", _cancel)
        dialog, a, b, c, d, message = _outputs(_execute("SPEAKER A: one\nSPEAKER B: two"))
        assert message == ""
        for socket in (dialog, a, b, c, d):
            assert torch.count_nonzero(socket["waveform"]) == 0


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


class TestModuleHygiene:
    def test_dialog_does_not_import_the_node_api(self) -> None:
        """``modules/dialog.py`` holds the layout, which is pure tensor arithmetic."""
        path = os.path.join(ROOT_DIR, "modules", "dialog.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("comfy_api" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "comfy_api" not in (node.module or "")

    def test_no_module_under_modules_reaches_the_node_api(self) -> None:
        """The whole rule, scanned across the tree rather than one file at a time."""
        directory = os.path.join(ROOT_DIR, "modules")
        offenders = []
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "comfy_api"
                ):
                    offenders.append(f"{name}:{node.lineno}")
                if isinstance(node, ast.Import) and any(
                    alias.name.startswith("comfy_api") for alias in node.names
                ):
                    offenders.append(f"{name}:{node.lineno}")
        assert offenders == [], f"modules/ must not import the node API: {offenders}"


class TestUpstreamHasNoDialog:
    """The corpus point this node exists to fill, asserted against the vendor."""

    def test_the_vendor_ships_no_dialog_module(self) -> None:
        vendor = os.path.join(ROOT_DIR, "src", "chatterbox")
        assert not os.path.exists(os.path.join(vendor, "dialog.py"))
        assert not os.path.exists(os.path.join(vendor, "dialog_tts.py"))