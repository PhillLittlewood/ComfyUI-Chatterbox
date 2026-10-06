"""S6.6 -- the V1 node surface is gone, and stays gone.

ComfyUI kept the V1 API for backwards compatibility, so a V3 rewrite can
quietly grow one back: a leftover ``CATEGORY =`` on a node class, an
``INPUT_TYPES`` someone left beside ``define_schema``, a ``NODE_CLASS_MAPPINGS``
dict reappearing in the entry point so an old loader path still resolves. None
of that raises. The package keeps working, the schemas keep validating, and the
duplicated declaration starts drifting from the real one.

So this file scans. **AST, not regex.** A node's ``CATEGORY = "..."`` and a
sentence mentioning ``INPUT_TYPES`` in a docstring are the same text; only one
of them is a defect. Regex over source text cannot tell them apart, and a
multi-line f-string makes a regex pass vacuously against a deleted line.

What it asserts, in two halves:

* **nothing declares the V1 surface** -- no mapping dict, no ``INPUT_TYPES`` /
  ``RETURN_TYPES`` / ``RETURN_NAMES`` assignment, no ``CATEGORY =`` /
  ``FUNCTION =`` class attribute;
* **everything under ``nodes/`` is a V3 node** -- every file subclasses
  ``io.ComfyNode`` and defines ``define_schema``.
"""

from __future__ import annotations

import ast
import os

import pytest

from conftest import ROOT_DIR

# Never scanned: the vendored upstream tree (not ours to restyle) and this test
# suite (which names the very artifacts it forbids).
EXCLUDED_PREFIXES: tuple[str, ...] = ("src/", "tests/")

# Attribute / assignment names that make up the V1 registration API.
V1_NAMES: tuple[str, ...] = (
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "INPUT_TYPES",
    "RETURN_TYPES",
    "RETURN_NAMES",
)

# Class-body attributes a V1 node declares and a V3 node must not.
V1_CLASS_ATTRIBUTES: tuple[str, ...] = ("CATEGORY", "FUNCTION")


def _iter_python_files() -> list[str]:
    """Repo-relative POSIX paths of every scannable ``.py`` file.

    Walked from the filesystem rather than ``git ls-files``: the files under
    construction are not committed yet, and a scan that can only see committed
    history cannot see the thing it exists to catch.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__pycache__"))]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            absolute = os.path.join(dirpath, filename)
            relative = os.path.relpath(absolute, ROOT_DIR).replace(os.sep, "/")
            if relative.startswith(EXCLUDED_PREFIXES):
                continue
            found.append(relative)
    return sorted(found)


def _parse(relative: str) -> ast.Module:
    with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=relative)


def _assigned_names(tree: ast.Module) -> list[tuple[str, int]]:
    """``(name, lineno)`` for every assignment to a bare name in the module."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.append((target.id, node.lineno))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append((node.target.id, node.lineno))
    return found


def _class_attribute_names(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(class name, attribute, lineno)`` for every class-body assignment."""
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            targets: list[ast.expr] = []
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found.append((node.name, target.id, statement.lineno))
    return found


def _read_names(tree: ast.Module) -> list[tuple[str, int]]:
    """``(name, lineno)`` for every identifier *read* as a bare name or attribute.

    Reading ``self.INPUT_TYPES`` or importing somebody else's
    ``NODE_CLASS_MAPPINGS`` would re-expose the V1 surface just as effectively
    as declaring it.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            found.append((node.attr, node.lineno))
    return found


def _artifacts(relative: str) -> list[str]:
    """Every V1 artifact in one file, as human-readable strings.

    Returns ``[]`` when the file is clean; the count is what the tests assert
    against, and the messages name the file and line so a failure is actionable.
    """
    tree = _parse(relative)
    found: list[str] = []

    assigned = dict(_assigned_names(tree))
    for name in V1_NAMES:
        if name in assigned:
            found.append(f"{relative}:{assigned[name]}: assigns {name}")

    for class_name, attribute, lineno in _class_attribute_names(tree):
        if attribute in V1_CLASS_ATTRIBUTES:
            found.append(
                f"{relative}:{lineno}: class {class_name} declares "
                f"{attribute} = (a V1 node attribute)"
            )

    for name, lineno in _read_names(tree):
        if name in V1_NAMES:
            found.append(f"{relative}:{lineno}: reads {name}")

    return found


ALL_FILES = _iter_python_files()
ARTIFACTS: list[str] = [item for relative in ALL_FILES for item in _artifacts(relative)]
NODE_FILES = [relative for relative in ALL_FILES if relative.startswith("nodes/")]


class TestTheScanFoundFiles:
    """A scan that matched nothing would pass everything below it."""

    def test_the_scan_covers_this_repository(self) -> None:
        assert "__init__.py" in ALL_FILES
        assert "chatterbox_nodes.py" in ALL_FILES
        assert any(path.startswith("modules/") for path in ALL_FILES)

    def test_the_vendored_tree_is_excluded_not_missed(self) -> None:
        assert not any(path.startswith("src/") for path in ALL_FILES)
        assert not any(path.startswith("tests/") for path in ALL_FILES)

    def test_the_node_package_is_covered(self) -> None:
        assert "nodes/__init__.py" in NODE_FILES
        assert "nodes/tts_node.py" in NODE_FILES
        assert "nodes/vc_node.py" in NODE_FILES


class TestNoV1Artifacts:
    def test_the_repository_declares_no_v1_node_surface(self) -> None:
        assert ARTIFACTS == [], (
            "V1 node artifacts found:\n  "
            + "\n  ".join(ARTIFACTS)
            + "\n\nA V3 node declares its widgets in define_schema(); declaring "
            "them a second time in the V1 shape is how the two drift apart."
        )

    def test_the_artifact_count_is_zero(self) -> None:
        assert len(ARTIFACTS) == 0

    @pytest.mark.parametrize("relative", ALL_FILES)
    def test_file_is_clean(self, relative: str) -> None:
        found = _artifacts(relative)
        assert found == [], "\n".join(found)


def _defines_a_node_class(relative: str) -> bool:
    """Whether a file declares a class inheriting ``io.ComfyNode``."""
    tree = _parse(relative)
    return any(
        isinstance(node, ast.ClassDef)
        and any("io.ComfyNode" in ast.unparse(base) for base in node.bases)
        for node in tree.body
    )


class TestNodeModulesAreV3:
    """Every module under ``nodes/`` is a V3 node, not a V1 one in a hat."""

    @pytest.mark.parametrize("relative", NODE_FILES)
    def test_every_node_module_imports_comfy_api(self, relative: str) -> None:
        """``nodes/__init__.py`` is a re-export shim and declares no node, so
        it is exempt: the modules it re-exports are checked individually."""
        if not _defines_a_node_class(relative):
            assert not relative.endswith(("tts_node.py", "vc_node.py"))
            return
        source = open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8").read()
        assert "comfy_api" in source, (
            f"{relative} declares a node but never imports comfy_api; a V3 "
            f"node cannot be built without it"
        )

    def test_the_node_package_reexports_both_nodes(self) -> None:
        tree = _parse("nodes/__init__.py")
        exported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert {"ChatterboxTTSNode", "ChatterboxVCNode"} <= exported

    def test_the_node_classes_subclass_ComfyNode(self) -> None:
        for relative in ("nodes/tts_node.py", "nodes/vc_node.py"):
            tree = _parse(relative)
            bases = [
                ast.unparse(base)
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                for base in node.bases
            ]
            assert any("io.ComfyNode" in base for base in bases), (
                f"{relative} declares no io.ComfyNode subclass"
            )

    def test_every_node_class_defines_a_schema(self) -> None:
        for relative in ("nodes/tts_node.py", "nodes/vc_node.py"):
            tree = _parse(relative)
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            node_classes = [
                node
                for node in classes
                if any("io.ComfyNode" in ast.unparse(base) for base in node.bases)
            ]
            assert node_classes, f"{relative} declares no node class"
            for class_node in node_classes:
                methods = [
                    statement.name
                    for statement in class_node.body
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                assert "define_schema" in methods, (
                    f"{relative}:{class_node.name} has no define_schema"
                )
                assert "execute" in methods, (
                    f"{relative}:{class_node.name} has no execute"
                )

    def test_comfy_api_stays_out_of_the_modules_layer(self) -> None:
        """``modules/`` must be importable without a ComfyUI API in sight.

        That is what lets the download stack, the loader and the registry be
        unit-tested against fakes -- and what keeps a core API rename from
        breaking the half of the package that has nothing to do with nodes.
        """
        offenders = []
        for relative in ALL_FILES:
            if not relative.startswith("modules/"):
                continue
            source = open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8").read()
            if "comfy_api" in source:
                offenders.append(relative)
        assert offenders == [], (
            f"comfy_api leaked into {offenders}; it belongs to nodes/*.py, "
            f"chatterbox_nodes.py and the repository entry point only"
        )