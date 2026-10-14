"""S6.5 -- the repository-root entry point.

A V3 custom-node package has exactly one public name: ``comfy_entrypoint``.
Everything else in ``__init__.py`` is bootstrap -- register the folders, scan the
models, wire the routes -- and has to survive being imported in a process that
has no ComfyUI server, because that is what a headless run and this test suite
look like.

The pytest guard is the other half. pytest imports every ``__init__.py`` it
meets during collection, and imports it *parentless*: the relative imports
below would fail with "attempted relative import with no known parent package"
before a single test ran. Under pytest the module therefore exports nothing and
returns. That is why the functional half of this file re-executes the source
with pytest temporarily out of ``sys.modules``, against fake ``folder_paths`` and
``server`` modules, rather than trusting the guarded import.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import conftest
from conftest import PKG_ALIAS, ROOT_DIR

INIT_PATH = os.path.join(ROOT_DIR, "__init__.py")


def read_init_source() -> str:
    with open(INIT_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def source() -> str:
    return read_init_source()


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source, filename=INIT_PATH)


def _assigned_names(tree: ast.Module) -> dict[str, ast.AST]:
    """Every ``NAME = ...`` assignment anywhere in the module, by name."""
    assigned: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value
    return assigned


def _referenced_names(tree: ast.Module) -> set[str]:
    """Every identifier the module binds or reads, as code.

    Deliberately an AST walk rather than a substring search: this file's own
    docstrings *name* the artifacts it refuses to declare, so a text scan
    reports the documentation as the defect.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _first_statement(tree: ast.Module) -> ast.stmt:
    """The first *executable* statement, skipping the docstring and imports.

    Importing ``sys`` before the guard is fine -- it is the guard's own
    subject. What must not precede it is any bootstrap side effect.
    """
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        return node
    raise AssertionError("the entry point has no executable statements")


class TestExports:
    def test_all_is_exactly_the_entrypoint(self, tree: ast.Module) -> None:
        """One export. A V1 package exported two mapping dicts instead."""
        value = _assigned_names(tree)["__all__"]
        exported = [ast.literal_eval(element) for element in value.elts]
        assert exported == ["comfy_entrypoint"]

    def test_the_v1_mappings_are_gone(self, tree: ast.Module) -> None:
        referenced = _referenced_names(tree)
        assert "NODE_CLASS_MAPPINGS" not in referenced
        assert "NODE_DISPLAY_NAME_MAPPINGS" not in referenced

    def test_web_directory_is_declared_and_shipped(self, tree: ast.Module) -> None:
        """The web directory the host is told to serve is there and holds the JS.

        The guard this replaces was its mirror image: the package deliberately
        declared no ``WEB_DIRECTORY``, because pointing the host at a directory
        that does not exist warns on every boot and makes "we ship a frontend"
        a claim nothing backs up. Both halves still have to hold -- declared
        **and** present -- so the check stays, inverted. Resolving the way the
        host does, against this module's own directory rather than the process
        working directory, is what makes the ``isdir`` half a real assertion
        instead of a guess about where the server was started.

        The attribute belongs to the module, not to the extension object;
        ``tests/test_extension.py`` is what keeps the object clean of it.
        """
        assigned = _assigned_names(tree)
        assert "WEB_DIRECTORY" in assigned, (
            "the entry point declares no web directory, so js/extension.js is "
            "never served and no node error is ever toasted"
        )
        web_dir = ast.literal_eval(assigned["WEB_DIRECTORY"])
        assert web_dir == "./js"

        resolved = os.path.abspath(os.path.join(os.path.dirname(INIT_PATH), web_dir))
        assert os.path.isdir(resolved), f"{web_dir} does not resolve to a directory: {resolved}"
        assert os.path.isfile(os.path.join(resolved, "extension.js")), (
            "the served directory has no extension.js, so the browser imports "
            "nothing and the toast listener never registers"
        )


class TestPytestGuard:
    def test_under_pytest_the_module_exports_nothing(self) -> None:
        module = sys.modules[PKG_ALIAS]
        assert module.__all__ == []
        assert not hasattr(module, "comfy_entrypoint"), (
            "the entry point ran its bootstrap during collection; under pytest "
            "it must return before touching folder_paths or the server"
        )

    def test_the_guard_is_the_first_thing_the_module_does(self, tree: ast.Module) -> None:
        first = _first_statement(tree)
        assert isinstance(first, ast.If)
        rendered = ast.unparse(first.test)
        assert "pytest" in rendered and "sys.modules" in rendered

    def test_the_guard_has_an_empty_exports_branch(self, tree: ast.Module) -> None:
        first = _first_statement(tree)
        assert isinstance(first, ast.If)
        guarded = first.body
        assert len(guarded) == 1
        statement = guarded[0]
        assert isinstance(statement, (ast.Assign, ast.AnnAssign))
        assert ast.literal_eval(statement.value) == []


class TestBootstrapRunsForReal:
    """Execute the module the way ComfyUI does, minus the server.

    pytest is removed from ``sys.modules`` for the duration so the guard opens,
    and ``folder_paths`` / ``server`` are replaced by stand-ins so nothing in
    this process's real ComfyUI registry is mutated. ``sys.path`` is snapshotted
    and restored because the vendor loader appends to it by design.
    """

    @pytest.fixture
    def loaded(self, monkeypatch):
        fake_folder_paths = SimpleNamespace(
            models_dir=os.path.join(ROOT_DIR, ".test-models"),
            folder_names_and_paths={},
            supported_pt_extensions={".safetensors"},
            get_folder_paths=lambda name: [],
            add_model_folder_path=lambda *a, **k: None,
            get_user_directory=lambda: os.path.join(ROOT_DIR, ".test-user"),
        )
        fake_server = SimpleNamespace(
            PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=MagicMock()))
        )
        monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
        monkeypatch.setitem(sys.modules, "server", fake_server)
        # The guard keys off pytest being importable; hide it for this exec only.
        monkeypatch.delitem(sys.modules, "pytest")

        spec = importlib.util.spec_from_file_location(
            f"{PKG_ALIAS}._entrypoint_probe",
            INIT_PATH,
            submodule_search_locations=[ROOT_DIR],
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        path_before = list(sys.path)
        try:
            spec.loader.exec_module(module)
            yield module, fake_folder_paths, fake_server
        finally:
            for entry in list(sys.path):
                if entry not in path_before:
                    sys.path.remove(entry)

    def test_the_entrypoint_is_exported(self, loaded) -> None:
        module, _folder_paths, _server = loaded
        assert module.__all__ == ["comfy_entrypoint"]
        assert callable(module.comfy_entrypoint)

    def test_folders_are_registered(self, loaded) -> None:
        _module, folder_paths_module, _server = loaded
        registered = folder_paths_module.folder_names_and_paths
        assert "tts" in registered
        assert "chatterbox" in registered

    def test_the_chatterbox_root_is_under_tts(self, loaded) -> None:
        _module, folder_paths_module, _server = loaded
        registered = folder_paths_module.folder_names_and_paths
        assert registered["tts"][0][-1].endswith(os.path.join("tts"))

    def test_routes_are_registered_when_a_server_exists(self, loaded) -> None:
        _module, _folder_paths, server = loaded
        assert server.PromptServer.instance.routes is not None

    def test_module_source_does_not_mutate_sys_modules(self, source: str) -> None:
        assert "sys.modules[" not in source
        assert "sys.path.insert" not in source


class TestStartupIsQuiet:
    """The bootstrap says nothing at INFO or above.

    The pack used to log ``[Chatterbox TTS] Registering the Chatterbox custom
    node.`` on every boot. Other extensions chat just as loudly, so the host's
    own startup output was buried under a wall of "look at me" -- and the note
    told the operator nothing: a silent import succeeded exactly as well as a
    chatty one. What survives is one ``debug`` pack-count line, invisible at the
    default level.

    The re-exec is the same one :class:`TestBootstrapRunsForReal` performs:
    pytest out of ``sys.modules`` so the guard opens, fake ``folder_paths`` /
    ``server`` so nothing real is registered. ``caplog`` is attached to the root
    logger first, so the records this fixture collects are exactly the ones the
    bootstrap emitted while being imported -- and they are captured explicitly,
    because ``caplog`` resets its handler when the *call* phase begins, after
    this fixture has already run.
    """

    @pytest.fixture
    def loaded(self, monkeypatch, caplog):
        fake_folder_paths = SimpleNamespace(
            models_dir=os.path.join(ROOT_DIR, ".test-models"),
            folder_names_and_paths={},
            supported_pt_extensions={".safetensors"},
            get_folder_paths=lambda name: [],
            add_model_folder_path=lambda *a, **k: None,
            get_user_directory=lambda: os.path.join(ROOT_DIR, ".test-user"),
        )
        fake_server = SimpleNamespace(
            PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=MagicMock()))
        )
        monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
        monkeypatch.setitem(sys.modules, "server", fake_server)
        # The guard keys off pytest being importable; hide it for this exec only.
        monkeypatch.delitem(sys.modules, "pytest")
        # Before the exec, so the caller can recount what the bootstrap emitted.
        caplog.set_level(logging.INFO)

        spec = importlib.util.spec_from_file_location(
            f"{PKG_ALIAS}._entrypoint_probe_quiet",
            INIT_PATH,
            submodule_search_locations=[ROOT_DIR],
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        path_before = list(sys.path)
        try:
            spec.loader.exec_module(module)
            captured = list(caplog.get_records("setup"))
            yield module, fake_folder_paths, fake_server, captured
        finally:
            for entry in list(sys.path):
                if entry not in path_before:
                    sys.path.remove(entry)

    def test_importing_the_entry_point_emits_no_info_or_above(self, loaded) -> None:
        """A startup note is a regression; this is where it comes back to fail.

        The silence has to be pinned by an assertion that can fail. The removed
        ``logging.info`` call was one line in this module, and nothing else in
        the suite looks at what the bootstrap says on the console.
        """
        _module, _folder_paths, _server, captured = loaded
        offenders = [
            f"{record.name}:{record.levelname}: {record.getMessage()}"
            for record in captured
            if record.levelno >= logging.INFO
        ]
        assert offenders == [], (
            "the bootstrap still speaks at INFO or above; an install-time note "
            "is exactly what the quiet start removed:\n  " + "\n  ".join(offenders)
        )

    def test_the_bootstrap_still_does_its_job(self, loaded) -> None:
        """Silence is not the goal; a silent, working import is.

        The quiet fixture replays the real bootstrap, so the registration it
        performs has to be as real as in :class:`TestBootstrapRunsForReal` --
        otherwise the silence above could be the sound of a bootstrap that broke
        early and logged nothing because it did nothing.
        """
        _module, folder_paths_module, _server, _captured = loaded
        registered = folder_paths_module.folder_names_and_paths
        assert "tts" in registered
        assert "chatterbox" in registered


class TestAbsentServerDegrades:
    """A process with no ``PromptServer`` still has to import cleanly."""

    @pytest.fixture
    def loaded_without_server(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(
            models_dir=os.path.join(ROOT_DIR, ".test-models"),
            folder_names_and_paths={},
            supported_pt_extensions={".safetensors"},
            get_folder_paths=lambda name: [],
            add_model_folder_path=lambda *a, **k: None,
            get_user_directory=lambda: os.path.join(ROOT_DIR, ".test-user"),
        ))
        monkeypatch.setitem(sys.modules, "server", None)
        monkeypatch.delitem(sys.modules, "pytest")

        spec = importlib.util.spec_from_file_location(
            f"{PKG_ALIAS}._entrypoint_probe_noserver",
            INIT_PATH,
            submodule_search_locations=[ROOT_DIR],
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        path_before = list(sys.path)
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            for entry in list(sys.path):
                if entry not in path_before:
                    sys.path.remove(entry)

    def test_the_entrypoint_is_still_exported(self, loaded_without_server) -> None:
        module = loaded_without_server
        assert module.__all__ == ["comfy_entrypoint"]
        assert callable(module.comfy_entrypoint)


class TestSourceHygiene:
    def test_the_module_owns_no_logging_handler(self, source: str) -> None:
        """ComfyUI installs its own ``ColoredFormatter`` on the root logger."""
        for forbidden in (
            "addHandler",
            "setLevel",
            "logging.Formatter",
            "logging.StreamHandler",
            "logging.Logger(",
            "LoggerAdapter",
        ):
            assert forbidden not in source, (
                f"{forbidden} is a custom-logging artifact; the package owns "
                f"only its message prefix"
            )

    def test_the_module_is_reachable_as_the_alias(self) -> None:
        assert conftest.PKG_ALIAS in sys.modules
        assert sys.modules[PKG_ALIAS].__file__ in (INIT_PATH, os.path.abspath(INIT_PATH))