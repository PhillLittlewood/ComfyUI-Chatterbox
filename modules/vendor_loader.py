"""S2.1 -- make the vendored ``src/`` importable, exactly once, at the tail.

The vendored ``chatterbox`` tree lives under this repository's own ``src/``,
which is not a package ComfyUI puts on ``sys.path``. Rather than hide the tree
under a private alias or install it, the directory is appended to ``sys.path``
so ``import chatterbox`` resolves to the vendored files.

``append``, never ``insert(0, ...)``: anything already importable -- most
importantly a pip-installed ``chatterbox`` the user chose -- must keep winning.
A prepend would silently shadow it.

Idempotent by construction: the path is only added when it is not already
present, so repeated calls (ComfyUI reloads a custom node on every startup)
cannot grow the list.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ensure_vendor_on_path"]


def ensure_vendor_on_path() -> None:
    """Append this repository's ``src/`` to ``sys.path`` unless it is there.

    Returns ``None``; the effect is on :data:`sys.path`. Safe to call any number
    of times and from any entry point.
    """
    src_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    if src_dir not in sys.path:
        sys.path.append(src_dir)