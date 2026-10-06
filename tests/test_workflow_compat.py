"""S6.7 -- the shipped example workflows still bind the right values.

The snapshot (``tests/frozen_widget_snapshot.json``) says what the *schema*
declares. This file checks the other half: what a real, shipped graph actually
stores. A saved workflow is a flat list of widget values, positional, with no
names -- so the only thing tying index 12 of ``widgets_values`` to
``use_watermark`` is the order in the schema. The shipped graphs are the
user-visible proof of that order, and they are the thing a rename or a reorder
would silently invalidate.

Two independent traps are covered:

* **TTS stores 13 values for 12 required inputs.** Index 11 is ``"randomize"``
  -- the companion widget ComfyUI emits for the ``seed`` input's
  ``control_after_generate`` flag -- and index 12 is ``use_watermark``. Drop the
  flag and ``use_watermark`` slides onto the ``randomize`` slot.
* **VC stores 5 values for 6 required inputs.** ``source_audio`` is an AUDIO
  socket, not a widget, so it consumes no index at all.

Both directories are globbed -- ``example_workflows/`` (the current name, from
the S9.1 rename) and ``workflow-examples/`` (the pre-2.0 spelling, retained so a
branch that predates the rename still finds something to check). A test that
pinned one path would either break on the rename or, worse, quietly stop finding
anything to check.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

from .conftest_helpers import load_frozen_snapshot

from conftest import ROOT_DIR

# Both spellings, because the directory is renamed during the rebuild and this
# file must keep guarding the contents either way.
WORKFLOW_DIRS = ("example_workflows", "workflow-examples")

# The node ids a shipped graph may contain, and the class that owns each.
NODE_CLASSES = {
    "ChatterboxTTS": ChatterboxTTSNode,
    "ChatterboxVC": ChatterboxVCNode,
}

# The one control-after-generate companion widget, and where it lands.
RANDOMIZE = "randomize"
SEED_COMPANION_INDEX = 11


# Input types that are sockets. A socket renders as a connection point and
# stores no value, so it consumes no index in ``widgets_values``. This is why
# ChatterboxVC's six required inputs store only five values.
SOCKET_TYPES = frozenset(
    {"AUDIO", "IMAGE", "LATENT", "MASK", "MODEL", "CLIP", "CONDITIONING"}
)


def _iter_workflow_files() -> list[str]:
    """Every shipped ``.json`` graph, from whichever directory holds them."""
    found: list[str] = []
    for directory in WORKFLOW_DIRS:
        pattern = os.path.join(ROOT_DIR, directory, "*.json")
        found.extend(sorted(glob.glob(pattern)))
    return found


def _nodes_by_type(document: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for node in document.get("nodes", []):
        if node.get("type") in NODE_CLASSES:
            grouped.setdefault(node["type"], []).append(node)
    return grouped


WORKFLOW_FILES = _iter_workflow_files()
WORKFLOW_IDS = [
    os.path.relpath(path, ROOT_DIR).replace(os.sep, "/") for path in WORKFLOW_FILES
]


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _widget_names(node_cls) -> tuple[str, ...]:
    """The widget inputs of a node, in the order a graph stores their values.

    A socket input (``io.Audio.Input``) is not a widget and consumes no index,
    which is exactly why VC's six required inputs store only five values.
    """
    info = node_cls.GET_NODE_INFO_V1()["input"]
    ordered = list(info.get("required", {})) + list(info.get("optional", {}))
    return tuple(
        name
        for name in ordered
        if info["required"].get(name, info["optional"].get(name))[0] not in SOCKET_TYPES
    )


def _frozen_widget_names(snapshot: dict, node_id: str) -> tuple[str, ...]:
    """The same projection, taken from the frozen snapshot."""
    entries = snapshot["nodes"][node_id]["required"]
    return tuple(entry["name"] for entry in entries if entry["type"] not in SOCKET_TYPES)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return load_frozen_snapshot()


class TestShippedWorkflowsWereFound:
    def test_at_least_one_workflow_exists(self) -> None:
        assert WORKFLOW_FILES, (
            "no example workflow was found under any of "
            f"{WORKFLOW_DIRS}; this file has nothing to guard"
        )

    def test_the_known_example_is_present(self) -> None:
        assert "example_workflows/ChatterboxTTS-workflow.json" in WORKFLOW_IDS

    def test_every_file_parses(self) -> None:
        for path in WORKFLOW_FILES:
            document = _load(path)
            assert isinstance(document, dict)
            assert "nodes" in document


class TestTTSWorkflowBindsByIndex:
    """13 stored values, ``randomize`` at 11, ``use_watermark`` at 12."""

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_widget_values_are_13_long(self, path: str) -> None:
        tts_nodes = _nodes_by_type(_load(path)).get("ChatterboxTTS", [])
        for node in tts_nodes:
            assert len(node["widgets_values"]) == 13, (
                f"{path}: ChatterboxTTS stores {len(node['widgets_values'])} "
                f"values, expected 13 (12 required inputs + one seed companion)"
            )

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_the_seed_companion_sits_at_index_11(self, path: str) -> None:
        for node in _nodes_by_type(_load(path)).get("ChatterboxTTS", []):
            assert node["widgets_values"][SEED_COMPANION_INDEX] == RANDOMIZE

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_use_watermark_sits_at_index_12(self, path: str) -> None:
        for node in _nodes_by_type(_load(path)).get("ChatterboxTTS", []):
            assert node["widgets_values"][12] is False

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_every_stored_value_type_matches_its_schema_slot(
        self, path: str, snapshot: dict
    ) -> None:
        """The load-bearing check: value *k* belongs to widget *k*.

        A graph stores positions, not names. Reading the stored values back
        through the current schema and requiring each one to be acceptable to
        that slot is what catches a reorder: insert a widget at index 4 and the
        string script lands in an INT slot, which fails here immediately
        instead of producing 40 seconds of noise three minutes later.
        """
        expected = snapshot["nodes"]["ChatterboxTTS"]["required"]
        for node in _nodes_by_type(_load(path)).get("ChatterboxTTS", []):
            values = node["widgets_values"]
            # Skip the seed companion: it is the `randomize` control, not a
            # value of the `seed` input.
            schema_values = values[:SEED_COMPANION_INDEX] + values[SEED_COMPANION_INDEX + 1 :]
            assert len(schema_values) == len(expected)
            for index, (entry, value) in enumerate(zip(expected, schema_values)):
                _assert_value_matches(entry, value, f"{path}:{index}")

    def test_the_vc_workflow_binds_by_index(self, snapshot: dict) -> None:
        """5 stored values for 6 required inputs: ``source_audio`` is a socket."""
        frozen = _frozen_widget_names(snapshot, "ChatterboxVC")
        assert frozen == (
            "model_pack_name",
            "n_timesteps",
            "temperature",
            "flow_cfg_scale",
            "use_watermark",
        )
        assert _widget_names(ChatterboxVCNode)[: len(frozen)] == frozen

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_vc_stores_five_values(self, path: str) -> None:
        for node in _nodes_by_type(_load(path)).get("ChatterboxVC", []):
            assert len(node["widgets_values"]) == 5, (
                f"{path}: ChatterboxVC stores {len(node['widgets_values'])} "
                f"values, expected 5 (6 required inputs, one of them a socket)"
            )


def _assert_value_matches(entry: dict, value, where: str) -> None:
    """One stored value against the frozen schema entry it claims to belong to."""
    kind = entry["type"]
    if kind == "COMBO":
        # Dynamic: the option list is the models directory, read at load time.
        assert isinstance(value, str), f"{where}: {entry['name']} should be a string"
    elif kind == "STRING":
        assert isinstance(value, str), f"{where}: {entry['name']} should be a string"
    elif kind == "BOOLEAN":
        assert isinstance(value, bool), f"{where}: {entry['name']} should be a bool"
    elif kind == "INT":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{where}: {entry['name']} should be an int, got {value!r}"
        )
        if entry["min"] is not None:
            assert value >= entry["min"], f"{where}: {entry['name']} below min"
        if entry["max"] is not None:
            assert value <= entry["max"], f"{where}: {entry['name']} above max"
    elif kind == "FLOAT":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{where}: {entry['name']} should be a number, got {value!r}"
        )
        if entry["min"] is not None:
            assert value >= entry["min"], f"{where}: {entry['name']} below min"
        if entry["max"] is not None:
            assert value <= entry["max"], f"{where}: {entry['name']} above max"
    else:  # pragma: no cover - a type the snapshot does not know about
        raise AssertionError(f"{where}: unhandled frozen input type {kind!r}")


class TestShippedWidgetValuesAgainstTheLiveSchema:
    """The stored values must also be accepted by the schema as it stands now."""

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_tts_combo_values_are_offered_today(self, path: str) -> None:
        from ComfyUI_Chatterbox.modules.model_discovery import (
            get_chatterbox_model_pack_names,
        )

        packs = get_chatterbox_model_pack_names()
        for node in _nodes_by_type(_load(path)).get("ChatterboxTTS", []):
            assert node["widgets_values"][0] in packs

    @pytest.mark.parametrize("path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
    def test_tts_widgets_still_land_on_the_same_names(self, path: str, snapshot: dict) -> None:
        frozen = _frozen_widget_names(snapshot, "ChatterboxTTS")
        widgets = _widget_names(ChatterboxTTSNode)
        assert widgets[: len(frozen)] == frozen
        # Exactly three widgets were appended behind the frozen window.
        assert widgets[len(frozen) :] == ("device", "dtype", "attention_mode")
        assert len(widgets) == len(frozen) + 3

    def test_the_tts_node_validates_the_shipped_pack(self) -> None:
        path = os.path.join(ROOT_DIR, "example_workflows", "ChatterboxTTS-workflow.json")
        for node in _nodes_by_type(_load(path)).get("ChatterboxTTS", []):
            assert ChatterboxTTSNode.validate_inputs(
                model_pack_name=node["widgets_values"][0]
            ) is True


class TestOutputNamesInTheShippedGraphs:
    """``converted_audio`` is referenced by name, not by position."""

    def test_the_vc_output_socket_name_is_the_frozen_one(self) -> None:
        schema = ChatterboxVCNode.GET_SCHEMA()
        assert [output.display_name for output in schema.outputs] == ["converted_audio"]

    def test_the_tts_output_socket_name_is_the_frozen_one(self) -> None:
        schema = ChatterboxTTSNode.GET_SCHEMA()
        assert [output.display_name for output in schema.outputs] == ["audio"]