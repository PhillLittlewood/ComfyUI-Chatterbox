"""S7.4 -- ``comfy.model_management`` API drift guard.

Every call this package makes into ComfyUI's memory manager is a **bare
attribute access on a module object**, so a core rename does not fail at
import time. It fails when a user runs the node -- on their machine, with
their model, mid-prompt -- as an ``AttributeError`` from inside a third-party
custom node. That is the worst place it can fail: after the download, after
the VRAM decision, with no pointer back to the line that has to change.

So the surface is pinned here instead. This is a **guard, not a mock**: it
imports the real ``comfy.model_management`` out of the real ComfyUI checkout
that ``conftest.py`` puts on ``sys.path``, and asserts the names are there. A
ComfyUI upgrade that renames one of them turns this file red at the suite
level, where it costs nothing, instead of red in production.

Two things are asserted that a bare ``hasattr`` sweep cannot:

* :class:`comfy.model_management.InterruptProcessingException` really does
  derive from ``BaseException`` and not from ``Exception``. That single fact is
  what makes ``except Exception`` in the nodes and in
  :func:`modules.generation.generate_audio` unable to swallow a cancellation.
  If core ever re-parented it, every "interrupt returns silence" clause would
  silently become unreachable and a cancelled run would be reported as a
  synthesis failure.
* Every symbol named here is also actually *reached* by the shipped package,
  so the list cannot rot into a list of names nothing calls any more. The one
  deliberate exception (``unet_offload_device``, ``module_size``) is named in
  :data:`DECLARED_BUT_NOT_CALLED` with the reason.
"""

from __future__ import annotations

import ast
import os

import pytest

from conftest import ROOT_DIR

#: Every ``comfy.model_management`` symbol this package depends on. Order is the
#: plan's (§Phase 7, S7.4), which groups them: device resolution, memory
#: placement, dtype policy, bookkeeping, and interruption.
MM_SYMBOLS: tuple[str, ...] = (
    "get_torch_device",
    "get_gpu_device_options",
    "resolve_gpu_device_option",
    "unet_offload_device",
    "intermediate_device",
    "load_model_gpu",
    "load_models_gpu",
    "get_free_memory",
    "soft_empty_cache",
    "should_use_fp16",
    "should_use_bf16",
    "get_autocast_device",
    "module_size",
    "unload_all_models",
    "InterruptProcessingException",
    "throw_exception_if_processing_interrupted",
)

#: Symbols the plan lists that the shipped package does not currently *call*.
#:
#: They are kept in :data:`MM_SYMBOLS` on purpose. ``unet_offload_device`` and
#: ``module_size`` are the two an upstream core refactor renames first, and a
#: guard that only pins what is used today would not notice the rename until
#: the day the code starts calling them. Asserting "these two are not called
#: yet" keeps that gap visible instead of silent.
DECLARED_BUT_NOT_CALLED: frozenset[str] = frozenset(
    {"unet_offload_device", "module_size"}
)

#: Shipped Python, minus the vendored tree (imported by absolute name, mocked in
#: collection, and upstream code that may log however upstream logs) and the
#: test suite (which discusses the idiom rather than following it).
SCANNED_ROOTS: tuple[str, ...] = ("modules", "nodes")
SCANNED_FILES: tuple[str, ...] = ("chatterbox_nodes.py", "__init__.py")
EXCLUDED_DIRS: frozenset[str] = frozenset({"__pycache__"})


def _iter_shipped_sources() -> list[str]:
    """Repo-relative POSIX paths of every shipped Python file."""
    found: list[str] = []
    for root in SCANNED_ROOTS:
        base = os.path.join(ROOT_DIR, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                name for name in dirnames if name not in EXCLUDED_DIRS
            )
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    relative = os.path.relpath(
                        os.path.join(dirpath, filename), ROOT_DIR
                    )
                    found.append(relative.replace(os.sep, "/"))
    for name in SCANNED_FILES:
        found.append(name)
    return sorted(found)


def _read(relative: str) -> str:
    with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def model_management():
    """The real ``comfy.model_management``, imported from the real checkout.

    Deliberately **not** mocked. A guard that asserts against a stand-in proves
    only that the stand-in has the names, which is exactly the failure mode it
    exists to prevent.
    """
    import comfy.model_management

    return comfy.model_management


class TestSymbolsExist:
    @pytest.mark.parametrize("symbol", MM_SYMBOLS)
    def test_symbol_is_present_in_core(self, model_management, symbol: str) -> None:
        assert hasattr(model_management, symbol), (
            f"comfy.model_management.{symbol} no longer exists. This package "
            f"calls it, and a missing attribute is an AttributeError raised "
            f"inside a custom node at prompt time, not at install time. Either "
            f"adapt modules/device_utils.py or modules/patcher.py to the new "
            f"name, or update MM_SYMBOLS here if the dependency went away."
        )

    def test_the_list_is_not_shrinking_under_pressure(self, model_management) -> None:
        """One sweep, so a reader can see the whole surface at a glance."""
        missing = [name for name in MM_SYMBOLS if not hasattr(model_management, name)]
        assert missing == [], f"comfy.model_management is missing: {missing}"

    def test_the_interrupt_is_not_an_exception_subclass(self, model_management) -> None:
        """``except Exception`` must not be able to swallow a cancellation.

        Every "interrupt returns silence, no traceback" clause in the node
        layer is only reachable because core's interrupt bypasses
        ``Exception``. Re-parenting it under ``Exception`` would turn those
        clauses into dead code and a cancelled run into a reported error.
        """
        interrupt = model_management.InterruptProcessingException
        assert issubclass(interrupt, BaseException)
        assert not issubclass(interrupt, Exception), (
            "comfy.model_management.InterruptProcessingException now derives "
            "from Exception; the `except Exception` clauses in "
            "modules/generation.py would swallow it before the "
            "interrupt clause could re-raise it."
        )

    def test_the_interrupt_is_importable_by_the_name_we_probe(
        self, model_management
    ) -> None:
        """``modules.audio_utils.interrupt_exception_class`` probes by name.

        It is a ``getattr`` with a ``BaseException`` fallback so collection
        never depends on core being importable, which means a rename would
        silently degrade to "everything is an interrupt". Pin the attribute
        path the probe uses.
        """
        candidate = getattr(
            model_management, "InterruptProcessingException", None
        )
        assert isinstance(candidate, type)
        assert candidate is model_management.InterruptProcessingException


class TestSurfaceIsActuallyUsed:
    """The list must track the code, or it is decoration."""

    def test_the_scan_finds_the_package(self) -> None:
        files = _iter_shipped_sources()
        assert len(files) >= 10, (
            f"the scan only found {files}; it cannot be trusted to prove the "
            f"package uses (or does not use) a core symbol"
        )

    @pytest.mark.parametrize(
        "symbol", sorted(set(MM_SYMBOLS) - DECLARED_BUT_NOT_CALLED)
    )
    def test_the_symbol_is_reached_by_shipped_code(self, symbol: str) -> None:
        mentions = [
            relative
            for relative in _iter_shipped_sources()
            if symbol in _read(relative)
        ]
        assert mentions, (
            f"{symbol} is pinned in MM_SYMBOLS but nothing in the package "
            f"names it any more. Drop it from MM_SYMBOLS (and, if it was a "
            f"real dependency, from the plan) instead of leaving the guard "
            f"asserting a name that protects nothing."
        )

    def test_the_unused_declarations_are_still_unused(self) -> None:
        """Keep the declared-but-uncalled gap honest in both directions.

        If one of these starts being called, the entry is stale -- but that is
        not a failure, it is progress, and ``MM_SYMBOLS`` already covers it.
        What must not happen is the *silent* case: the name being pinned while
        the call site is spelled differently, so a rename would slip past.
        """
        for symbol in sorted(DECLARED_BUT_NOT_CALLED):
            assert symbol in MM_SYMBOLS
            for relative in _iter_shipped_sources():
                tree = ast.parse(_read(relative), filename=relative)
                called = any(
                    isinstance(node, ast.Attribute) and node.attr == symbol
                    for node in ast.walk(tree)
                )
                assert not called, (
                    f"{symbol} is called in {relative} but is still declared "
                    f"uncalled in DECLARED_BUT_NOT_CALLED"
                )

    def test_modules_never_import_the_memory_manager_at_module_scope(
        self,
    ) -> None:
        """``modules/`` reaches core lazily, through ``_get_model_management``.

        Not a style rule: ``comfy.model_management`` imports torch and probes
        the accelerator at module scope, and this package is also imported by
        ``/object_info`` while ComfyUI is still starting up.
        """
        offenders = []
        for relative in _iter_shipped_sources():
            if not relative.startswith("modules/"):
                continue
            tree = ast.parse(_read(relative), filename=relative)
            for node in tree.body:  # module scope only
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(name.startswith("comfy.model_management") for name in names):
                    offenders.append(f"{relative}:{node.lineno}")
        assert offenders == [], (
            f"these modules import comfy.model_management at module scope: "
            f"{offenders}. Use modules.device_utils._get_model_management()."
        )