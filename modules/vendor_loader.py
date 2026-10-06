"""S2.1 -- make the vendored ``src/`` importable, exactly once, at the tail.

The vendored ``chatterbox`` tree lives under this repository's own ``src/``,
which is not a package ComfyUI puts on ``sys.path``. Rather than hide the tree
under a private alias or install it, the directory is appended to ``sys.path``
so ``import chatterbox`` resolves to the vendored files. The Flash sampler
(``chatterbox-flash``, vendored at ``src/chatterbox-flash/``) is appended the
same way, so ``import chatterbox_flash`` needs no pip install either.

``append``, never ``insert(0, ...)``: anything already importable -- most
importantly a pip-installed ``chatterbox`` the user chose -- must keep winning.
A prepend would silently shadow it.

Idempotent by construction: a path is only added when it is not already
present, so repeated calls (ComfyUI reloads a custom node on every startup)
cannot grow the list.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ensure_vendor_on_path"]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_vendor_on_path() -> None:
    """Append this repository's vendored roots to ``sys.path`` unless present.

    Appends ``src/`` (the vendored ``chatterbox`` tree) and, when it exists,
    ``src/chatterbox-flash/`` (the vendored Flash sampler package). Returns
    ``None``; the effect is on :data:`sys.path`. Safe to call any number of
    times and from any entry point.
    """
    src_dir = os.path.join(_repo_root(), "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    # The Flash package is a project root *inside* src/ (its import name,
    # ``chatterbox_flash``, is one level down), so it needs its own entry.
    flash_dir = os.path.join(src_dir, "chatterbox-flash")
    if os.path.isdir(flash_dir) and flash_dir not in sys.path:
        sys.path.append(flash_dir)