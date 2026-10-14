"""The frontend node filter has to name exactly the Python schema's nodes.

``js/extension.js`` listens for ``execution_error`` and toasts only the events
whose ``node_type`` it recognises. That list is hand-maintained, so it can drift:
a node renamed in ``nodes/`` leaves every Python test green -- the ids are pinned
there, against the frozen snapshot -- and the browser simply stops toasting for
that node. Nothing reports it. The graph keeps running, the node keeps failing,
and the operator sees nothing.

So the ids are parsed back out of the file itself and compared against the live
schema: the same registry ComfyUI reads the node list from. A match here is a
fact, not a comment promising the two sides agree.
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest

from conftest import ROOT_DIR

from ComfyUI_Chatterbox.chatterbox_nodes import comfy_entrypoint

EXTENSION_JS = os.path.join(ROOT_DIR, "js", "extension.js")

#: The opening expression of the filter's declaration. Anchoring on the whole
#: thing, not on the bare const name, is what keeps the slice off the comment
#: above it -- that comment names this very test, and a name-only anchor would
#: slice starting at the wrong place.
SET_ANCHOR = 'CHATTERBOX_NODE_TYPES = new Set(['


def frontend_node_types() -> list[str]:
    """Every node id the frontend filters on, in the order the JS lists them.

    The span is sliced between the anchor and its ``])`` rather than matched
    across the whole file: the extension names node ids in prose comments too,
    and a file-wide pattern would quietly compare the documentation against
    the schema instead of the code.
    """
    with open(EXTENSION_JS, "r", encoding="utf-8") as handle:
        source = handle.read()

    if SET_ANCHOR not in source:
        raise AssertionError(
            f"js/extension.js no longer declares {SET_ANCHOR!r}; the node "
            "filter this module compares against cannot be found"
        )

    start = source.index(SET_ANCHOR) + len(SET_ANCHOR)
    span = source[start : source.index("])", start)]
    return re.findall(r'"([^"]*)"', span)


@pytest.fixture(scope="module")
def schema_node_ids() -> set[str]:
    """The ids the live Python schema exposes, straight from the registry."""
    extension = asyncio.run(comfy_entrypoint())
    return {
        node_cls.GET_SCHEMA().node_id
        for node_cls in asyncio.run(extension.get_node_list())
    }


class TestFrontendNodeSet:
    def test_the_js_list_is_found_and_not_empty(self) -> None:
        """A renamed anchor must fail loudly, not compare two empty sets."""
        frontend = frontend_node_types()
        assert frontend, (
            "no node ids were parsed out of js/extension.js, so the comparison "
            "below could not tell a drift from an unreadable file"
        )

    def test_the_js_list_matches_the_live_schema(
        self, schema_node_ids: set[str]
    ) -> None:
        """Both directions: an id dropped from the JS and one invented in it.

        The direction that matters most is the second. A JS id the schema does
        not define toasts for a node type no graph can ever produce, which is
        the sound of a node having been renamed on the Python side and the
        filter left behind -- and the first direction, the silent one this
        guard exists for.
        """
        frontend = set(frontend_node_types())
        assert frontend == schema_node_ids
        assert schema_node_ids, (
            "the schema exposes no nodes at all; both sides of the comparison "
            "would be empty and nothing would be guarded"
        )
