"""S6.8 -- the node package loads against the **real** ComfyUI core.

Every other test in this repository mocks something. This one deliberately does
not: it drives the exact path ComfyUI's own loader drives -- await
``comfy_entrypoint()``, take ``get_node_list()``, call ``GET_SCHEMA()`` on each
class -- against the ``comfy_api`` installed in the target environment.

A mock proves the code agrees with the mock. Only this proves the schemas
finalize, that core's own validators accept them, and that the V1 projection
core builds for the frontend (``/object_info``) carries the widget order this
package depends on.

**This file must not be skipped.** A skip is what a core upgrade would produce
if ``comfy_api`` ever stopped being importable, and a skip is precisely the
outcome that must be loud: it would mean the package registers nothing and no
user finds out until they look for the node.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from comfy_api.latest import ComfyExtension, io

from ComfyUI_Chatterbox.chatterbox_nodes import (
    ChatterboxDialogNode,
    ChatterboxExtension,
    ChatterboxMTLTTSNode,
    ChatterboxSRTNode,
    ChatterboxTTSNode,
    ChatterboxTurboTTSNode,
    ChatterboxVCNode,
    comfy_entrypoint,
)

#: Menu order. This module exists so that *every* node the extension offers is
#: driven through the real core's schema finalizer, not only the two the
#: snapshot covers -- a schema that fails only against real core is exactly the
#: failure this file was written to catch.
NODES = (
    ChatterboxTTSNode,
    ChatterboxMTLTTSNode,
    ChatterboxTurboTTSNode,
    ChatterboxSRTNode,
    ChatterboxDialogNode,
    ChatterboxVCNode,
)

#: The V1 ``output`` projection each node must render. Spelled out per class
#: rather than derived from the schema: "every output is AUDIO" was true when
#: there were two nodes, and a check that reads ``["AUDIO"] * len(outputs)``
#: would quietly stop being a check the moment a node gained a STRING socket.
EXPECTED_OUTPUTS = {
    ChatterboxTTSNode: ["AUDIO"],
    ChatterboxMTLTTSNode: ["AUDIO"],
    ChatterboxTurboTTSNode: ["AUDIO"],
    ChatterboxSRTNode: ["AUDIO", "STRING", "STRING", "STRING"],
    ChatterboxDialogNode: ["AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING"],
    ChatterboxVCNode: ["AUDIO"],
}

EXPECTED_CATEGORIES = {
    ChatterboxTTSNode: "WMNodes/sound/tts",
    ChatterboxMTLTTSNode: "WMNodes/sound/tts",
    ChatterboxTurboTTSNode: "WMNodes/sound/tts",
    ChatterboxSRTNode: "WMNodes/sound/util",
    ChatterboxDialogNode: "WMNodes/sound/tts",
    ChatterboxVCNode: "WMNodes/sound/vc",
}


def run(coro):
    """Drive one coroutine from a synchronous test."""
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def extension() -> ChatterboxExtension:
    return run(comfy_entrypoint())


def test_comfy_api_is_the_real_core() -> None:
    """Guard the guard: a mocked ``comfy_api`` would void every test below."""
    import comfy_api.latest as comfy_api_latest

    assert comfy_api_latest.__file__ is not None
    assert "mock" not in type(comfy_api_latest).__name__.lower()
    assert io.ComfyNode.__module__.startswith("comfy_api")


def test_the_core_version_is_the_one_this_package_targets() -> None:
    """The snapshot was captured against 0.37.0's widget protocol.

    ``comfy_api`` publishes no version of its own, so the check is on the core
    checkout this suite resolved -- the one whose ``Schema.finalize`` and
    ``create_input_dict_v1`` produced the numbers the snapshot recorded.
    """
    from conftest import COMFYUI_ROOT

    import comfy_api.latest as comfy_api_latest

    resolved = os.path.realpath(comfy_api_latest.__file__)
    expected = os.path.realpath(os.path.join(COMFYUI_ROOT, "comfy_api"))
    assert resolved.startswith(expected), (
        f"comfy_api resolved to {resolved!r}, outside the pinned ComfyUI "
        f"checkout {expected!r}; this suite is only meaningful against the "
        f"core the snapshot was captured from"
    )
    assert os.path.isfile(os.path.join(COMFYUI_ROOT, "nodes.py"))


def test_the_entrypoint_produces_a_real_extension(extension) -> None:
    assert isinstance(extension, ComfyExtension)
    assert isinstance(extension, ChatterboxExtension)


def test_the_loader_receives_the_two_node_classes(extension) -> None:
    assert run(extension.get_node_list()) == list(NODES)


def test_the_frozen_two_are_still_the_outermost_classes(extension) -> None:
    """``ChatterboxTTS`` first, ``ChatterboxVC`` last, after four additions."""
    node_ids = [cls.GET_SCHEMA().node_id for cls in run(extension.get_node_list())]
    assert node_ids[0] == "ChatterboxTTS"
    assert node_ids[-1] == "ChatterboxVC"


@pytest.mark.parametrize("node_cls", NODES, ids=lambda cls: cls.__name__)
class TestSchemaFinalizesUnderTheRealCore:
    def test_the_schema_builds_validates_and_is_stored(self, node_cls) -> None:
        schema = node_cls.GET_SCHEMA()
        assert schema is node_cls.SCHEMA
        assert schema.node_id
        assert schema.inputs

    def test_outputs_get_ids_from_finalize(self, node_cls) -> None:
        """``finalize`` is what gives an unnamed output socket an id.

        Skipped silently, the id stays ``None``, core builds a link record with
        a null slot, and every output fails to connect -- at prompt time, on
        the user's machine.
        """
        schema = node_cls.FINALIZE_SCHEMA()
        assert schema.outputs
        for output in schema.outputs:
            assert output.id

    def test_input_ids_are_unique(self, node_cls) -> None:
        ids = [item.id for item in node_cls.GET_SCHEMA().inputs]
        assert len(ids) == len(set(ids))

    def test_the_v1_projection_renders(self, node_cls) -> None:
        """``/object_info`` is built from exactly this dictionary."""
        info = node_cls.INPUT_TYPES()
        assert set(info) >= {"required"}
        assert info["required"], f"{node_cls.__name__} rendered no required inputs"

    def test_node_info_v1_carries_the_identity(self, node_cls) -> None:
        info = node_cls.GET_NODE_INFO_V1()
        assert info["name"] == node_cls.GET_SCHEMA().node_id
        assert info["output"] == EXPECTED_OUTPUTS[node_cls]
        assert info["category"] == EXPECTED_CATEGORIES[node_cls]

    def test_relative_python_module_is_settable(self, node_cls) -> None:
        """ComfyUI stamps this at load time (``nodes.py``).

        It has to be a writable class attribute on a node class -- not a
        property, not a slot -- or the loader's assignment raises and the node
        never registers.
        """
        previous = node_cls.RELATIVE_PYTHON_MODULE
        try:
            node_cls.RELATIVE_PYTHON_MODULE = "ComfyUI-Chatterbox.chatterbox_nodes"
            assert (
                node_cls.GET_NODE_INFO_V1()["python_module"]
                == "ComfyUI-Chatterbox.chatterbox_nodes"
            )
        finally:
            node_cls.RELATIVE_PYTHON_MODULE = previous


def test_node_output_round_trips_through_from_dict() -> None:
    """``EXECUTE_NORMALIZED`` accepts a plain dict and rebuilds it.

    ComfyUI's own V1 compatibility path returns ``{"result": (...)}`` for a
    node that predates ``io.NodeOutput``. If ``from_dict`` could not rebuild
    what ``to_dict``-shaped payloads contain, that compatibility path would
    break -- and nothing else in this repository would notice.
    """
    payload = {"result": ({"waveform": [0.0], "sample_rate": 24000},)}
    rebuilt = io.NodeOutput.from_dict(payload)
    assert isinstance(rebuilt, io.NodeOutput)
    assert rebuilt[0] == payload["result"][0]

    with_ui = io.NodeOutput.from_dict(
        {"result": ({"waveform": [0.0], "sample_rate": 24000},), "ui": None}
    )
    assert with_ui.ui is None
    assert len(with_ui.args) == 1


def test_the_real_normalizer_accepts_a_node_output() -> None:
    """What ``execute`` returns is routed through core's own normaliser.

    ``EXECUTE_NORMALIZED`` is what ComfyUI actually calls, and it is what turns
    a bare ``dict`` / ``tuple`` return into a ``NodeOutput``. Exercised here on
    a throwaway node so the real ones are never run -- no model, no vendor, no
    weights.
    """

    class _Probe(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id="ChatterboxNormalizerProbe",
                inputs=[],
                outputs=[io.Audio.Output(display_name="audio")],
            )

        @classmethod
        def execute(cls, **kwargs):
            # The shape a node returns: a one-tuple matching the output list.
            return ({"waveform": [0.0], "sample_rate": 24000},)

    normalised = _Probe.EXECUTE_NORMALIZED()
    assert isinstance(normalised, io.NodeOutput)
    assert normalised.args == ({"waveform": [0.0], "sample_rate": 24000},)
    # And the V1 compatibility shape rebuilds to the same thing.
    assert io.NodeOutput.from_dict(
        {"result": normalised.args}
    ).args == normalised.args


@pytest.mark.parametrize("node_cls", NODES, ids=lambda cls: cls.__name__)
def test_the_class_itself_validates(node_cls) -> None:
    """``VALIDATE_CLASS`` rejects a node without both abstract methods."""
    node_cls.VALIDATE_CLASS()


def test_the_core_knows_both_node_ids() -> None:
    """A node whose id core has never seen is a node no graph can name."""
    from comfy_api.latest import ComfyExtension as CoreExtension

    assert CoreExtension is ComfyExtension
    ids = {node_cls.GET_SCHEMA().node_id for node_cls in NODES}
    assert {"ChatterboxTTS", "ChatterboxVC"} <= ids
    assert ids == {
        "ChatterboxTTS",
        "ChatterboxMTLTTS",
        "ChatterboxTurboTTS",
        "ChatterboxSRT",
        "ChatterboxDialog",
        "ChatterboxVC",
    }