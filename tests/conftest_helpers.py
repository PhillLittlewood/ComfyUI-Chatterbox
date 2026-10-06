"""Helpers shared by the test modules.

Kept separate from the root ``conftest.py`` because pytest imports that one
eagerly, before anything else; anything here is opt-in.

Two things live here:

* the real-vendor loader used by the env-gated e2e smoke test;
* the readers that project a V3 node schema and the frozen snapshot onto the
  same tuple shape, so the schema tests can compare them for **exact**
  equality rather than by eyeball.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Iterable

from conftest import PKG_ALIAS

__all__ = [
    "load_real_vendor",
    "unload_vendor_mocks",
    "real_vendor_modules",
    "FROZEN_SNAPSHOT_PATH",
    "load_frozen_snapshot",
    "signature",
    "schema_signatures",
    "snapshot_signatures",
]

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_SNAPSHOT_PATH = os.path.join(TESTS_DIR, "frozen_widget_snapshot.json")


def unload_vendor_mocks() -> int:
    """Drop every ``MagicMock`` the harness registered for the vendored tree.

    Returns the number of entries removed. Only the two prefixes the harness
    registers are touched -- ``chatterbox.*`` and the aliased
    ``ComfyUI_Chatterbox.src.chatterbox.*`` -- so nothing else in the process is
    disturbed.
    """
    prefixes = ("chatterbox.", "ComfyUI_Chatterbox.src.chatterbox.")
    doomed = [
        name
        for name in sys.modules
        if name == "chatterbox" or name.startswith(prefixes)
    ]
    for name in doomed:
        sys.modules.pop(name, None)
    return len(doomed)


def load_real_vendor() -> None:
    """Import the real vendored ``chatterbox`` package, undoing the harness mock.

    ``conftest.py`` mocks every name under ``src/chatterbox`` at collection time
    so the model stack never loads during a normal run. The e2e smoke test is
    the one place that must be undone, which is why it is an explicit call and
    not a fixture the whole suite inherits.

    The package is appended to ``sys.path``, never prepended, so a
    pip-installed ``chatterbox`` still takes precedence -- the same rule the
    vendor loader in the node follows.
    """
    unload_vendor_mocks()
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)

    import chatterbox  # noqa: PLC0415 -- deliberate late import

    assert hasattr(chatterbox, "__file__"), (
        "the vendored chatterbox resolved to a stub, not the real package"
    )


@contextlib.contextmanager
def real_vendor_modules(names: Iterable[str]):
    """Import real vendored modules, then put the harness mocks back exactly.

    A handful of tests are about code that exists only in the real source files
    -- a ``from_local`` guard, an attention kernel, a ``dtype`` property. The
    harness mocks the whole tree so collection never loads the model stack, so
    those tests lift the mocks for their duration.

    The restore is the part that matters. ``unload_vendor_mocks`` removes **both**
    prefixes the harness registers (``chatterbox.*`` and
    ``ComfyUI_Chatterbox.src.chatterbox.*``); a restore that only puts the first
    back leaves the suite quietly un-mocked for every test that runs afterwards,
    and ``tests/test_conftest.py`` finds it. So both are snapshotted here.

    Args:
        names: Dotted names relative to ``chatterbox``, e.g.
            ``["models.t3.modules.perceiver"]``.

    Yields:
        The imported modules, keyed by the dotted name that was asked for.
    """
    import importlib

    prefixes = ("chatterbox", f"{PKG_ALIAS}.src.chatterbox")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    # ``sys.path`` is restored too: ``tests/test_conftest.py`` asserts that
    # importing the node package leaves it untouched, and leaving this
    # repository's own ``src/`` behind would be a leak of exactly the kind that
    # test exists to catch.
    saved_path = list(sys.path)
    try:
        unload_vendor_mocks()
        src_dir = SRC_DIR
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        for name in [n for n in sys.modules if n.startswith(prefixes)]:
            sys.modules.pop(name, None)
        yield {name: importlib.import_module(f"chatterbox.{name}") for name in names}
    finally:
        for name in [n for n in sys.modules if n.startswith(prefixes)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
        importlib.invalidate_caches()

# ---------------------------------------------------------------------------
# Reading a node schema and the frozen snapshot in the same shape
# ---------------------------------------------------------------------------
#
# A V3 schema and the V1-era snapshot describe the same widgets, but not in the
# same vocabulary. ``GET_NODE_INFO_V1()`` is the bridge: it renders the V3
# ``io.Schema`` through core's own ``create_input_dict_v1``, producing exactly
# the ``{"required": {name: (TYPE, {...})}}`` layout the snapshot was captured
# as. Comparing the two therefore needs no hand-written translation table, and
# a divergence shows up as a plain tuple mismatch.

# The keys a re-bind would hit: the widget's identity, its type, the value a
# fresh graph starts on, its range, and the flag that inserts a companion
# widget into the saved value list.
SIGNATURE_KEYS = ("name", "type", "default", "min", "max", "step", "control_after_generate")


def load_frozen_snapshot() -> dict:
    """The frozen compatibility record, read from disk.

    Raises:
        FileNotFoundError: The snapshot is missing. It is captured *before* the
            V1 implementation is deleted and must never be reconstructed after
            the fact, so its absence is a hard failure rather than a skip.
    """
    with open(FROZEN_SNAPSHOT_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def signature(name: str, input_type: str, config: dict) -> tuple:
    """The comparable tuple for one input, from either vocabulary.

    Absent keys read as ``None``: core omits ``min``/``max``/``step`` when an
    input does not constrain them, and the snapshot records them as JSON null.
    """
    return (
        name,
        input_type,
        config.get("default"),
        config.get("min"),
        config.get("max"),
        config.get("step"),
        bool(config.get("control_after_generate", False)),
    )


def schema_signatures(node_cls, section: str) -> tuple[tuple, ...]:
    """One signature tuple per input of ``node_cls``, in declared order.

    Args:
        node_cls: A V3 node class.
        section: ``"required"`` or ``"optional"``.
    """
    info = node_cls.GET_NODE_INFO_V1()
    names = info["input_order"].get(section, [])
    return tuple(
        signature(name, *info["input"][section][name]) for name in names
    )


def snapshot_signatures(snapshot: dict, node_id: str, section: str) -> tuple[tuple, ...]:
    """The same shape, read from the frozen snapshot."""
    entries = snapshot["nodes"][node_id][section]
    return tuple(
        signature(entry["name"], entry["type"], entry) for entry in entries
    )
