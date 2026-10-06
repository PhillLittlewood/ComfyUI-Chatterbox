"""The ComfyUI extension object: what ComfyUI loads this package to get nodes.

ComfyUI imports the custom-node package and awaits ``comfy_entrypoint()``,
receiving a :class:`~comfy_api.latest.ComfyExtension`. It then calls
``get_node_list()`` and registers every returned class.

**The order of the returned list is the order the nodes appear in the Add Node
menu**, so it is part of the surface a user sees: every synthesis node first,
voice conversion last.

This module lives inside the package and every import in it is relative: the
node modules reach their own dependencies through ``..modules``, which only
resolves when the whole package has been loaded under a name. Loading this file
directly (a packaging smoke test) therefore has to go through the package root,
never around it.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .modules.vendor_loader import ensure_vendor_on_path
from .nodes import (
    ChatterboxDialogNode,
    ChatterboxMTLTTSNode,
    ChatterboxSRTNode,
    ChatterboxTTSNode,
    ChatterboxTurboTTSNode,
    ChatterboxVCNode,
)

__all__ = ["ChatterboxExtension", "comfy_entrypoint"]


class ChatterboxExtension(ComfyExtension):
    """Registers the Chatterbox V3 nodes with ComfyUI."""

    async def on_load(self) -> None:
        """Prepare the process for the first node execution.

        Idempotent, and deliberately minimal: the only thing that must happen
        before a node runs is the vendored package being importable, and
        ``ensure_vendor_on_path`` appends the repository's ``src/`` at most once.
        ComfyUI may call this more than once (a reload, a second server), so it
        must stay free of side effects that only make sense once.
        """
        ensure_vendor_on_path()
        logging.debug("[Chatterbox TTS] Extension loaded.")

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Every node this package provides, in Add-Node-menu order.

        Speech synthesis first, in the order a user reaches for them -- the
        classic English node, then the multilingual and Turbo families, then
        the two scripted nodes -- and voice conversion last, because it converts
        audio a graph already has rather than producing it.
        """
        return [
            ChatterboxTTSNode,
            ChatterboxMTLTTSNode,
            ChatterboxTurboTTSNode,
            ChatterboxSRTNode,
            ChatterboxDialogNode,
            ChatterboxVCNode,
        ]


async def comfy_entrypoint() -> ChatterboxExtension:
    """Entry point ComfyUI awaits to discover this package's nodes."""
    return ChatterboxExtension()