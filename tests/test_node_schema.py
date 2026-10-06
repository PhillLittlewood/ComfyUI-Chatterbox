"""S6.1/S6.2 -- the V3 schemas reproduce the frozen V1 widget surface exactly.

ComfyUI binds a saved workflow's widget values **by index**. Nothing in the
protocol records which widget a stored value belonged to, so a schema that
merely *contains* the right widgets in a different order silently re-binds
every graph a user already has: ``use_watermark`` lands on ``seed``, the text
script lands on ``max_new_tokens``, and nothing errors until the audio comes out
wrong.

That is why the assertions here are **exact tuple equality**, never a prefix
match. A prefix assertion -- ``schema_names[:12] == frozen_names`` on its own,
or ``frozen_names[0] in schema_names`` -- passes the moment a widget is
inserted in the middle of the frozen block or a fourth one is appended, which
is precisely the failure being guarded against.

The three widgets this rebuild legitimately adds (``device``, ``dtype``,
``attention_mode``) are therefore handled explicitly, not waved through: the
frozen window is compared for exact equality *and* the remainder is pinned to
exactly those three names, in order, so "we appended one more widget" is a
failing test and a deliberate, reviewed change.
"""

from __future__ import annotations

import pytest

from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

from .conftest_helpers import (
    load_frozen_snapshot,
    schema_signatures,
    snapshot_signatures,
)

# The node ids are frozen too: they are what a saved graph names.
TTS_NODE_ID = "ChatterboxTTS"
VC_NODE_ID = "ChatterboxVC"

# The widgets each schema appends after its frozen block, in the one order they
# may be appended. Per node, not a single tuple: S8.6 added three widgets to
# ``ChatterboxVC`` only, and a shared tuple would either force them onto the TTS
# node or force this test to stop filtering by name -- and a test that stops
# filtering by name stops being exact tuple equality, which is the whole point.
APPENDED_WIDGETS: dict[str, tuple[str, ...]] = {
    TTS_NODE_ID: ("device", "dtype", "attention_mode"),
    VC_NODE_ID: (
        "device",
        "dtype",
        "attention_mode",
        "refinement_passes",
        "max_chunk_duration",
        "chunk_method",
    ),
}

NODES = (
    pytest.param(ChatterboxTTSNode, TTS_NODE_ID, id="tts"),
    pytest.param(ChatterboxVCNode, VC_NODE_ID, id="vc"),
)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return load_frozen_snapshot()


def _schema(node_cls):
    """The finalized, validated schema core will register."""
    return node_cls.GET_SCHEMA()


class TestTTSSchema:
    """``ChatterboxTTS``: identity, frozen widgets, outputs."""

    def test_node_id_is_frozen(self) -> None:
        assert _schema(ChatterboxTTSNode).node_id == TTS_NODE_ID

    def test_display_name_and_category(self, snapshot: dict) -> None:
        node = _schema(ChatterboxTTSNode)
        assert node.display_name == snapshot["nodes"][TTS_NODE_ID]["display_name"]
        # Category was moved to the WMNodes/sound group at the owner's
        # direction; the snapshot keeps the V1 value as history only.
        assert node.category == "WMNodes/sound/tts"

    def test_required_inputs_equal_the_snapshot_exactly(self, snapshot: dict) -> None:
        """The load-bearing assertion. Exact tuple equality, twice over.

        Not a prefix match. The schema's required inputs are partitioned into
        the frozen set (compared for **whole-tuple equality** with the snapshot)
        and the three widgets S6.3 sanctioned (compared for whole-tuple equality
        with each other), and the total length is pinned. A reorder fails the
        first comparison; a fourth appended widget fails the third. A prefix
        assertion would pass on both, which is exactly the defect being
        guarded against.
        """
        expected = snapshot_signatures(snapshot, TTS_NODE_ID, "required")
        actual = schema_signatures(ChatterboxTTSNode, "required")

        appended_names = APPENDED_WIDGETS[TTS_NODE_ID]
        frozen = tuple(entry for entry in actual if entry[0] not in appended_names)
        appended = tuple(entry[0] for entry in actual if entry[0] in appended_names)

        assert frozen == expected
        assert appended == appended_names
        assert len(actual) == len(expected) + len(appended_names)

    def test_optional_inputs_equal_the_snapshot_exactly(self, snapshot: dict) -> None:
        expected = snapshot_signatures(snapshot, TTS_NODE_ID, "optional")
        assert schema_signatures(ChatterboxTTSNode, "optional") == expected

    def test_seed_carries_control_after_generate(self) -> None:
        """The flag that makes the shipped graph store 13 values, not 12."""
        seed = next(
            entry
            for entry in ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"].items()
            if entry[0] == "seed"
        )
        assert seed[1][0] == "INT"
        assert seed[1][1]["control_after_generate"] is True

    def test_seed_is_the_only_control_after_generate_input(self) -> None:
        flagged = [
            name
            for name, (_type, cfg) in ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"].items()
            if cfg.get("control_after_generate")
        ]
        assert flagged == ["seed"]

    def test_text_is_a_multiline_string(self) -> None:
        config = ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"]["text"]
        assert config[0] == "STRING"
        assert config[1]["multiline"] is True

    def test_audio_prompt_is_optional(self) -> None:
        optional = ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["optional"]
        assert list(optional) == ["audio_prompt"]
        assert optional["audio_prompt"][0] == "AUDIO"

    def test_output_is_a_real_audio_socket_named_audio(self) -> None:
        schema = _schema(ChatterboxTTSNode)
        assert [output.io_type for output in schema.outputs] == ["AUDIO"]
        assert [output.display_name for output in schema.outputs] == ["audio"]

    def test_output_node_is_false(self) -> None:
        """V1 declared ``OUTPUT_NODE = True`` *and* returned a socket.

        Both cannot be right. An output node is a terminal: ComfyUI executes it
        on its own even with nothing downstream, and caches its UI result
        instead of passing it along. Returning the socket is what this node
        actually does. The flag is behaviour, not a widget, so changing it does
        not move a single saved index.
        """
        assert _schema(ChatterboxTTSNode).is_output_node is False
        assert ChatterboxTTSNode.OUTPUT_NODE is False

    def test_schema_validates_and_finalizes(self) -> None:
        """``finalize`` gives every output an id and adds the hidden inputs."""
        schema = ChatterboxTTSNode.FINALIZE_SCHEMA()
        assert schema.outputs[0].id
        # Not an output node, so it must not have been handed a prompt.
        assert schema.hidden == []


class TestVCSchema:
    """``ChatterboxVC``: the mirror image, plus one frozen output name."""

    def test_node_id_is_frozen(self) -> None:
        assert _schema(ChatterboxVCNode).node_id == VC_NODE_ID

    def test_display_name_and_category(self, snapshot: dict) -> None:
        node = _schema(ChatterboxVCNode)
        assert node.display_name == snapshot["nodes"][VC_NODE_ID]["display_name"]
        # Category moved to WMNodes/sound; see the TTS note above.
        assert node.category == "WMNodes/sound/vc"

    def test_required_inputs_equal_the_snapshot_exactly(self, snapshot: dict) -> None:
        expected = snapshot_signatures(snapshot, VC_NODE_ID, "required")
        actual = schema_signatures(ChatterboxVCNode, "required")

        appended_names = APPENDED_WIDGETS[VC_NODE_ID]
        frozen = tuple(entry for entry in actual if entry[0] not in appended_names)
        appended = tuple(entry[0] for entry in actual if entry[0] in appended_names)

        assert frozen == expected
        assert appended == appended_names
        assert len(actual) == len(expected) + len(appended_names)

    def test_optional_inputs_equal_the_snapshot_exactly(self, snapshot: dict) -> None:
        expected = snapshot_signatures(snapshot, VC_NODE_ID, "optional")
        assert schema_signatures(ChatterboxVCNode, "optional") == expected

    def test_output_socket_is_named_converted_audio(self) -> None:
        """The name is part of the frozen surface; ``audio`` is the TTS node's.

        A saved graph refers to an output by its displayed name. Renaming this
        to ``audio`` would collide with the sibling node's socket and leave
        every existing VC link unresolved.
        """
        schema = _schema(ChatterboxVCNode)
        assert [output.io_type for output in schema.outputs] == ["AUDIO"]
        assert [output.display_name for output in schema.outputs] == ["converted_audio"]

    def test_output_node_is_false(self) -> None:
        assert _schema(ChatterboxVCNode).is_output_node is False
        assert ChatterboxVCNode.OUTPUT_NODE is False

    def test_schema_validates_and_finalizes(self) -> None:
        schema = ChatterboxVCNode.FINALIZE_SCHEMA()
        assert schema.outputs[0].id
        assert schema.hidden == []


class TestValidateInputs:
    """Queue-time checks must *refuse with a message*, never with a traceback."""

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_the_default_pack_is_accepted(self, node_cls) -> None:
        from ComfyUI_Chatterbox.modules.model_discovery import (
            get_chatterbox_model_pack_names,
        )

        for name in get_chatterbox_model_pack_names():
            assert node_cls.validate_inputs(model_pack_name=name) is True

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_an_unknown_pack_returns_a_readable_message(self, node_cls) -> None:
        result = node_cls.validate_inputs(model_pack_name="no_such_pack")
        assert isinstance(result, str), (
            "an unknown voice pack must produce a message the user reads in "
            "the queue panel, not an exception"
        )
        assert "no_such_pack" in result
        assert result.strip() == result and "\n" not in result

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_no_inputs_at_all_is_fine(self, node_cls) -> None:
        assert node_cls.validate_inputs() is True

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_every_advertised_combo_option_passes_validation(self, node_cls) -> None:
        """A dropdown must never offer a value its own validator refuses.

        Only the inputs the validator actually declares are swept: core calls
        ``validate_inputs`` with just those, so an undeclared widget is not a
        path this test can reach.
        """
        import inspect

        declared = set(inspect.signature(node_cls.validate_inputs).parameters)
        info = node_cls.GET_NODE_INFO_V1()["input"]
        for section in ("required", "optional"):
            for name, (_type, cfg) in info.get(section, {}).items():
                options = cfg.get("options")
                if not options or name not in declared:
                    continue
                for option in options:
                    kwargs = {name: option}
                    if name == "attention_mode":
                        # `auto` resolves to a half dtype on every GPU Sage
                        # supports, so it is exempt from the cross-check.
                        kwargs["dtype"] = "auto"
                    assert node_cls.validate_inputs(**kwargs) is True, (
                        f"{name}={option!r} is offered but refused"
                    )


class TestNoVendoredImportAtModuleScope:
    """Importing the node modules must never pull in the vendored model stack.

    ComfyUI imports every custom-node package at startup and the frontend asks
    ``/object_info`` before a single model is loaded. A module-scope
    ``from chatterbox.tts import ChatterboxTTS`` would drag torch weights,
    ``librosa`` and the whole transformer stack into that path.
    """

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_the_node_modules_never_import_chatterbox(self, node_cls) -> None:
        import inspect

        import ComfyUI_Chatterbox.nodes.tts_node as tts_module
        import ComfyUI_Chatterbox.nodes.vc_node as vc_module

        for module in (tts_module, vc_module):
            source = inspect.getsource(module)
            assert "import chatterbox" not in source
            assert "from chatterbox" not in source