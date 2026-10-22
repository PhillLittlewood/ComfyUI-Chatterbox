"""S6.3 -- the three appended widgets, and what they do to the frozen prefix.

Adding a widget is the one change to a node schema that is *allowed* to alter a
saved graph, and it is only safe in one shape: appended after every frozen
input. Insert ``device`` next to a related setting and every stored value from
that point on shifts by one -- a graph saved on a machine with a second GPU
silently re-binds ``dtype`` to whatever now occupies the ``device`` index.

So this module checks three separate things, and the first one is the reason the
other two exist:

1. the frozen prefix is byte-for-byte what the snapshot recorded;
2. what sits behind that prefix is *closed* -- exactly ``device``, ``dtype``,
   ``attention_mode``, in that order;
3. each new widget's option list is derived from the live machine, and an option
   that cannot work here is **removed from the schema** rather than offered and
   then refused.

Point 3 is the interesting one. An attention backend that is unavailable must
not appear in the dropdown, because a dropdown is a promise. But a mode string
inside a saved workflow is not a promise: it is a record of the machine that
saved the graph, and the graph has to keep opening on a machine that cannot run
it. The node therefore drops the unavailable mode from its schema while
*accepting* it in ``validate_inputs`` and downgrading it at load time through
``resolve_attention_mode``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ComfyUI_Chatterbox.modules.attention_utils import resolve_attention_mode
from ComfyUI_Chatterbox.modules.device_utils import get_device_options
from ComfyUI_Chatterbox.modules.dtype_utils import get_dtype_options
from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

from .conftest_helpers import (
    empty_pack_root,
    load_frozen_snapshot,
    schema_signatures,
    snapshot_signatures,
)


@pytest.fixture(autouse=True)
def _pin_pack_discovery_to_an_empty_root(empty_pack_root):
    """The frozen prefix records an empty models dir; keep it that way."""

TTS_NODE_ID = "ChatterboxTTS"
VC_NODE_ID = "ChatterboxVC"

# The appended widgets, and the option source each one is built from. Naming
# the source here is the point: a dropdown whose options are hard-coded here
# would drift from the machine it is rendered on.
APPENDED_WIDGETS = {
    "device": get_device_options,
    "dtype": get_dtype_options,
}

# Everything each schema appends *after* its frozen block, in the one order it
# may be appended. Per node, because S8.6 added three widgets to VC only: the
# assertion below is that this set is closed, and deriving it from "whatever
# happens to follow the frozen block" would make it true by construction.
APPENDED_WIDGET_ORDER: dict[str, tuple[str, ...]] = {
    TTS_NODE_ID: (
        "device",
        "dtype",
        "attention_mode",
        "top_k",  # Turbo-family controls (Turbo node merged into TTS node).
        "normalize_loudness",
        # Flash controls (Single Language Pack round, 2026-10-07).
        "num_steps",
        "cfg_scale",
        # Long-script controls (emotion-tag and chunking round, 2026-10-22).
        "enable_chunking",
        "max_chars_per_chunk",
        "chunk_combination_method",
        "silence_between_chunks_ms",
    ),
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


def _required_info(node_cls) -> dict:
    return node_cls.GET_NODE_INFO_V1()["input"]["required"]


class TestFrozenPrefixSurvived:
    """The appendage must not have moved a single frozen index."""

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_the_frozen_prefix_is_unchanged(self, node_cls, snapshot: dict) -> None:
        node_id = node_cls.GET_SCHEMA().node_id
        expected = snapshot_signatures(snapshot, node_id, "required")
        assert schema_signatures(node_cls, "required")[: len(expected)] == expected

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_the_appendage_is_closed(self, node_cls, snapshot: dict) -> None:
        """The set of post-frozen widgets is spelled out, per node.

        The point is that it is *closed*: "we appended one more widget" has to
        be a failing test and a reviewed change, which it cannot be if the
        expectation is derived from whatever happens to follow the frozen block.
        """
        node_id = node_cls.GET_SCHEMA().node_id
        frozen_count = len(snapshot["nodes"][node_id]["required"])
        actual = schema_signatures(node_cls, "required")
        assert tuple(entry[0] for entry in actual[frozen_count:]) == APPENDED_WIDGET_ORDER[
            node_id
        ]

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_optional_inputs_moved_no_widget_index(self, node_cls) -> None:
        """The optional inputs are AUDIO sockets, so they consume no index."""
        info = node_cls.GET_NODE_INFO_V1()["input"]
        for name, (_type, cfg) in info["optional"].items():
            assert "options" not in cfg, (
                f"optional input {name!r} is a widget; an appended socket-free "
                f"widget there would shift nothing today and everything the "
                f"day it is reordered"
            )


class TestAppendedWidgetOptions:
    """Each dropdown is built from the live machine, not from a literal."""

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    @pytest.mark.parametrize("widget", sorted(APPENDED_WIDGETS))
    def test_options_come_from_the_live_source(
        self, node_cls, widget: str
    ) -> None:
        info = _required_info(node_cls)
        assert widget in info
        assert info[widget][0] == "COMBO"
        assert info[widget][1]["options"] == APPENDED_WIDGETS[widget](), (
            f"{widget} options must be read from "
            f"{APPENDED_WIDGETS[widget].__module__} at schema-build time"
        )

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_every_advertised_dtype_is_accepted_by_the_validator(
        self, node_cls
    ) -> None:
        """Offering a value the node would refuse is a UI lie."""
        for option in _required_info(node_cls)["dtype"][1]["options"]:
            assert node_cls.validate_inputs(
                model_pack_name=None, dtype=option, attention_mode="auto"
            ) is True, f"dtype={option!r} is offered but refused"

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_every_advertised_device_resolves_to_a_torch_device(self, node_cls) -> None:
        """A device the loader cannot resolve is a device that fails at runtime."""
        from ComfyUI_Chatterbox.modules.device_utils import get_torch_device

        for option in _required_info(node_cls)["device"][1]["options"]:
            assert isinstance(get_torch_device(option), torch.device), (
                f"device={option!r} is offered but does not resolve"
            )

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_auto_is_the_last_device_option(self, node_cls) -> None:
        """Why it is last, in one line.

        A saved workflow binds ``device`` **by index**. Inserting ``"auto"`` at
        the front -- the natural place for a default -- would re-bind every
        stored device value to whatever shifted up into its slot. ``modules/
        device_utils.get_device_options`` appends it for exactly this reason and
        the schema must not reorder it afterwards.
        """
        options = _required_info(node_cls)["device"][1]["options"]
        assert options[-1] == "auto"
        assert options.count("auto") == 1

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_dtype_leads_with_auto(self, node_cls) -> None:
        options = _required_info(node_cls)["dtype"][1]["options"]
        assert options[0] == "auto"
        assert _required_info(node_cls)["dtype"][1]["default"] == "auto"


class TestUnavailableAttentionModesAreRemovedAtSchemaBuildTime:
    """A backend this machine cannot run must not be in the dropdown.

    ``get_available_attention_modes`` already returns only the modes that can be
    loaded; ``nodes.common.runtime_widgets`` passes that list straight into the
    Combo, so an unavailable ``sage`` / ``flash_attention_2`` is *removed while
    the schema is built* and never reaches the frontend's range check in the
    first place.
    """

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_an_unavailable_mode_is_dropped_from_the_schema(self, node_cls) -> None:
        with patch(
            "ComfyUI_Chatterbox.nodes.common.get_available_attention_modes",
            return_value=["eager", "sdpa"],
        ):
            options = _required_info(node_cls)["attention_mode"][1]["options"]
        assert options == ["eager", "sdpa"]
        assert "sage" not in options
        assert "flash_attention_2" not in options

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_a_saved_graph_naming_sage_is_not_refused(self, node_cls) -> None:
        """The graph has to keep opening on a machine without SageAttention.

        A mode string in a saved workflow describes the machine that saved it.
        Refusing the prompt here would mean a user who shared a graph cannot
        run it at all; downgrading it means it runs, on a different backend, and
        says so in the log.
        """
        with patch(
            "ComfyUI_Chatterbox.nodes.common.get_available_attention_modes",
            return_value=["eager", "sdpa"],
        ):
            # Schema side: the option is gone...
            options = _required_info(node_cls)["attention_mode"][1]["options"]
            assert "sage" not in options
            # ...but the validator side: the request is honoured, not rejected.
            assert node_cls.validate_inputs(
                model_pack_name=None, dtype="auto", attention_mode="sage"
            ) is True

    def test_an_unavailable_mode_still_reaches_the_loader_intact(self) -> None:
        """The fallback happens in the loader, where the hardware is known."""
        with patch(
            "ComfyUI_Chatterbox.modules.attention_utils.check_sage_attention_compatible",
            return_value=False,
        ):
            assert resolve_attention_mode("sage") == "sdpa"

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_the_live_options_are_offered_when_they_are_available(
        self, node_cls
    ) -> None:
        """Sanity check against the real machine, no mocks involved."""
        from ComfyUI_Chatterbox.modules.attention_utils import (
            get_available_attention_modes,
        )

        assert (
            _required_info(node_cls)["attention_mode"][1]["options"]
            == get_available_attention_modes()
        )


class TestDtypeAttentionCrossCheck:
    """The pair that crashes inside a CUDA kernel must be caught at queue time."""

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_fp32_with_sage_is_refused_with_a_message(self, node_cls) -> None:
        result = node_cls.validate_inputs(
            model_pack_name=None, dtype="fp32", attention_mode="sage"
        )
        assert isinstance(result, str)
        assert "fp32" in result and "sage" in result
        assert "bf16" in result and "sdpa" in result

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    @pytest.mark.parametrize("dtype", ["auto", "bf16", "fp16"])
    def test_sage_with_a_half_dtype_is_accepted(self, node_cls, dtype: str) -> None:
        assert node_cls.validate_inputs(
            model_pack_name=None, dtype=dtype, attention_mode="sage"
        ) is True

    @pytest.mark.parametrize("node_cls", [ChatterboxTTSNode, ChatterboxVCNode])
    def test_fp32_with_sdpa_is_accepted(self, node_cls) -> None:
        assert node_cls.validate_inputs(
            model_pack_name=None, dtype="fp32", attention_mode="sdpa"
        ) is True