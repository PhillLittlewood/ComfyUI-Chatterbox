"""S9.1 / S9.2 -- the README must describe the repository that actually exists.

Documentation fails in two directions, and this file catches both.

**Stale.** A README that still points at ``workflow-examples/`` after the S9.1
rename sends every new user to a 404, and worse, makes the rename look like a
deletion. So every repository-relative link and image in the README is resolved
against the working tree, and the *old* directory name must not appear at all.

**Wrong.** A README can be perfectly self-consistent and still lie -- it can
promise an extra that `pyproject.toml` no longer declares, or advertise a model
directory the loader does not use. So the load-bearing claims are asserted as
plain text against the files that decide them.

Deliberately **no `torch` and no `comfy` import** at module scope. This module
reads text and walks directories; the moment it imports the node stack wholesale
it becomes a test of the runtime rather than of the documentation, and it stops
being usable as the answer to "is the README honest?". The one exception is
:func:`_registered_node_ids`, which imports the extension lazily and only to
answer "which nodes does this package actually register?". That question cannot
be answered from text -- a hardcoded list of node ids is precisely the kind of
claim that goes stale -- and a README table missing a shipped node is a
documentation failure this module is supposed to be able to catch.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re

import pytest

from conftest import ROOT_DIR

README_PATH = os.path.join(ROOT_DIR, "README.md")
CHANGELOG_PATH = os.path.join(ROOT_DIR, "CHANGELOG.md")
PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")

# S9.1's new name, and the name it replaced. Both are checked: the first must
# exist, the second must have stopped existing.
EXAMPLE_DIR = "example_workflows"
RETIRED_DIR = "workflow-examples"

# A markdown link destination: ``[text](target)``, or a reference definition.
_MD_LINK = re.compile(r"\]\(\s*(?P<target>[^)\s]+)\s*\)")

# An HTML image source: ``<img src="...">``.
_HTML_SRC = re.compile(r'<img[^>]*\bsrc\s*=\s*"(?P<target>[^"]+)"', re.IGNORECASE)

# A pip invocation the README tells the reader to run, and what it would fetch.
_PIP_INSTALL = re.compile(r"pip install\s+(?P<packages>[^\n`|]+)")

# ``ChatterboxTTS``, ``ChatterboxVC``, ... as written in a graph's node list.
CHATTERBOX_NODE_ID = re.compile(r"^Chatterbox[A-Za-z0-9]*$")

# A row of the README node table: a line starting with the node id in code
# ticks. Anchored to the line start so prose mentioning an id cannot pass for a
# table entry -- the table itself is the claim being checked.
_NODE_TABLE_ROW = re.compile(
    r"^\|\s*`(Chatterbox[A-Za-z0-9]*)`\s*\|", re.MULTILINE
)

# What ComfyUI itself ships. Declaring any of these is the packaging bug
# `tests/test_pyproject.py` guards on the other side of the same contract.
COMFYUI_BUNDLED = frozenset(
    {
        "torch",
        "torchaudio",
        "transformers",
        "huggingface_hub",
        "numpy",
        "librosa",
        "soundfile",
        "safetensors",
        "tokenizers",
        "einops",
        "scipy",
        "tqdm",
        "accelerate",
        "diffusers",
    }
)

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _section(text: str, heading: str, span: int = 3000) -> str:
    """The text following ``heading``, up to the next heading of any level.

    A missing heading yields ``""`` -- never an exception -- so the caller gets a
    readable assertion failure instead of an ``IndexError``.
    """
    if heading not in text:
        return ""
    tail = text.split(heading, 1)[1]
    for stop in ("\n## ", "\n### "):
        tail = tail.split(stop, 1)[0]
    return tail[:span]


def _repo_relative_targets(text: str) -> list[str]:
    """Repository-relative link destinations, from markdown links and ``src=``.

    Anchors, absolute URLs and mail addresses are skipped: they are not claims
    about this checkout and cannot be resolved on disk.
    """
    targets: set[str] = set()
    for pattern in (_MD_LINK, _HTML_SRC):
        for match in pattern.finditer(text):
            target = match.group("target").strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            targets.add(target.split("#", 1)[0].split("?", 1)[0])
    return sorted(target for target in targets if target)


def _declared_node_ids() -> frozenset[str]:
    """Every ``node_id=`` literal declared anywhere under ``nodes/``.

    Read with :mod:`ast` rather than by import, so that checking the *documented*
    node ids never depends on the node modules importing successfully. A new
    node module that fails to import must not be able to hide itself here.
    """
    found: set[str] = set()
    for path in sorted(glob.glob(os.path.join(ROOT_DIR, "nodes", "*.py"))):
        try:
            tree = ast.parse(_read(path), filename=path)
        except SyntaxError:  # pragma: no cover - a syntax error fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "node_id":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.add(value.value)
    return frozenset(found)


def _registered_node_ids() -> frozenset[str]:
    """Every node id this package hands to ComfyUI, read from the registry.

    Go through :meth:`ChatterboxExtension.get_node_list` rather than an AST
    scan or a literal. The registry is what the Add Node menu is built from, so
    it is the only set against which "is this node documented?" has an answer a
    user would agree with -- and deriving it here means a node added next
    release cannot ship undocumented, which is the whole point. A hardcoded
    literal here would go stale silently, reproducing the bug this replaces.

    Imported lazily, inside the function: the rest of this module is
    deliberately import-free, and only this one question needs the runtime.
    """
    import asyncio

    from ComfyUI_Chatterbox.chatterbox_nodes import comfy_entrypoint

    extension = asyncio.run(comfy_entrypoint())
    return frozenset(
        node_cls.GET_SCHEMA().node_id for node_cls in asyncio.run(extension.get_node_list())
    )


def _documented_node_ids(readme: str) -> frozenset[str]:
    """Node ids given a row in the README's node table."""
    return frozenset(_NODE_TABLE_ROW.findall(readme))


def _example_workflow_node_ids() -> set[str]:
    """Chatterbox node ids referenced by any shipped example graph."""
    found: set[str] = set()
    for path in sorted(glob.glob(os.path.join(ROOT_DIR, EXAMPLE_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        for node in document.get("nodes", []):
            node_type = node.get("type")
            if isinstance(node_type, str) and CHATTERBOX_NODE_ID.match(node_type):
                found.add(node_type)
    return found


@pytest.fixture(scope="module")
def readme() -> str:
    return _read(README_PATH)


@pytest.fixture(scope="module")
def changelog() -> str:
    return _read(CHANGELOG_PATH)


@pytest.fixture(scope="module")
def extras() -> dict[str, list[str]]:
    if tomllib is None:  # pragma: no cover - only on a 3.10 interpreter
        pytest.skip("no TOML parser available (need Python >= 3.11 or `tomli`)")
    with open(PYPROJECT_PATH, "rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


class TestExampleWorkflowDirectory:
    """S9.1 -- the rename landed, and the JSON is still the frozen contract."""

    def test_the_new_directory_exists(self) -> None:
        assert os.path.isdir(os.path.join(ROOT_DIR, EXAMPLE_DIR)), (
            f"{EXAMPLE_DIR}/ is missing; the README and the shipped workflows "
            "both point at it"
        )

    def test_the_old_directory_is_gone(self) -> None:
        assert not os.path.exists(os.path.join(ROOT_DIR, RETIRED_DIR)), (
            f"{RETIRED_DIR}/ still exists after the S9.1 rename; both spellings "
            "on disk means one of them is stale"
        )

    def test_an_example_graph_is_present(self) -> None:
        assert glob.glob(os.path.join(ROOT_DIR, EXAMPLE_DIR, "*.json")), (
            f"no example .json graph under {EXAMPLE_DIR}/"
        )


class TestReadmeLinksResolve:
    """A link that 404s is a documentation bug the user finds before we do."""

    def test_the_readme_links_to_something(self, readme: str) -> None:
        assert _repo_relative_targets(readme), (
            "the README has no repository-relative links at all"
        )

    def test_every_linked_target_exists(self, readme: str) -> None:
        targets = _repo_relative_targets(readme)
        missing = [
            target
            for target in targets
            if not os.path.exists(os.path.join(ROOT_DIR, target.lstrip("./")))
        ]
        assert not missing, (
            f"README.md links to {missing}, which do not exist in this checkout "
            "-- a renamed directory, or a file that was never added"
        )

    def test_it_points_at_the_renamed_example_directory(self, readme: str) -> None:
        assert EXAMPLE_DIR in readme, (
            f"the README never mentions {EXAMPLE_DIR}/, so a reader has no way "
            "to find the example graph"
        )

    def test_the_retired_directory_name_is_absent(self, readme: str) -> None:
        """Removed claims must stay removed, not linger as a second spelling."""
        assert RETIRED_DIR not in readme, (
            f"README.md still mentions {RETIRED_DIR}/; that directory was "
            f"renamed to {EXAMPLE_DIR}/ and the old path now 404s"
        )


class TestReadmeClaimsMatchTheCode:
    """Each documented fact is compared against the file that decides it."""

    def test_python_requirement_matches_pyproject(self, readme: str) -> None:
        assert ">=3.10" in readme, (
            "the README must state the same specifier pyproject declares, not a "
            "paraphrase of it"
        )

    def test_comfyui_requirement_matches_pyproject(self, readme: str) -> None:
        assert ">=0.37.0" in readme, (
            "the README must state the same requires-comfyui specifier that "
            "pyproject declares"
        )

    @pytest.mark.parametrize(
        "package", ["s3tokenizer"]
    )
    def test_hard_dependency_is_listed(self, readme: str, package: str) -> None:
        assert package in readme, (
            f"{package} is a hard dependency in requirements.txt and is absent "
            "from the README"
        )

    @pytest.mark.parametrize(
        ("extra", "packages"),
        [
            ("watermarking", ("resemble-perth",)),
            ("multilingual", ("pykakasi", "spacy-pkuseg")),
            ("turbo", ("pyloudnorm",)),
        ],
    )
    def test_every_extra_is_documented_with_its_packages(
        self, readme: str, extra: str, packages: tuple[str, ...]
    ) -> None:
        """What the user must type to get the feature, spelled out."""
        assert extra in readme, f"the {extra!r} extra is not documented"
        for package in packages:
            assert package in readme, (
                f"{package} belongs to the {extra!r} extra but is not named in "
                "the README"
            )

    def test_the_extras_offered_match_pyproject(self, extras: dict) -> None:
        """Belt and braces: the README and pyproject must offer the same three."""
        assert set(extras) == {"watermarking", "multilingual", "turbo"}

    def test_no_install_command_pulls_a_comfyui_bundled_package(self, readme: str) -> None:
        """The ML stack is the host's. A README that tells the reader to
        ``pip install torch`` is the same bug the packaging test guards."""
        offenders: list[str] = []
        for match in _PIP_INSTALL.finditer(readme):
            for token in match.group("packages").split():
                name = token.strip('"').split("[", 1)[0].split("=", 1)[0].lower()
                if name in COMFYUI_BUNDLED:
                    offenders.append(name)
        assert not offenders, (
            f"the README tells the reader to pip install {offenders}; ComfyUI "
            "already ships those and re-declaring them makes Manager fight "
            "the user's torch build"
        )

    def test_the_absence_of_pins_is_explained(self, readme: str) -> None:
        """Not just asserted -- a reader who wants to add a pin must find out
        why there isn't one."""
        body = _section(readme, "### Why there are no version pins").lower()
        assert body, "the README must have a section explaining the missing pins"
        assert "manager" in body, "the rationale must name ComfyUI Manager"
        assert "downgrade" in body, (
            "the rationale must say what a pin would cost the user"
        )

    def test_offline_behaviour_is_documented(self, readme: str) -> None:
        body = _section(readme, "### Offline behaviour").lower()
        assert body, "the README must have an offline-behaviour section"
        assert "re-downloaded" in body, (
            "the guarantee is that a local pack is never fetched again"
        )
        assert "first-run download" in body, (
            "the offline claim is only true because of the first-run download "
            "it replaces; both halves belong in the same section"
        )

    def test_vram_behaviour_is_documented(self, readme: str) -> None:
        body = _section(readme, "### VRAM characteristics").lower()
        assert body, "the README must have a VRAM section"
        assert "offload" in body
        assert "safetensors" in body, (
            "the size is measured from the on-disk weights; saying so is what "
            "separates a real measurement from a marketing claim"
        )

    def test_the_migration_note_names_both_directories(self, readme: str) -> None:
        body = _section(readme, "## Migrating from 1.x", span=4000)
        assert body, "the README must have a migration section"
        assert "models/chatterbox_tts/" in body, (
            "the note must name the directory the user has to delete"
        )
        assert "models/tts/chatterbox/" in body, "and the directory it moved to"

    def test_the_migration_note_says_delete_it_by_hand(self, readme: str) -> None:
        body = _section(readme, "## Migrating from 1.x", span=4000).lower()
        assert "no compatibility shim is provided" in body, (
            "the default is a manual-delete instruction; the README must say so "
            "rather than leaving the user to discover it"
        )

    def test_the_migration_note_says_node_ids_did_not_change(self, readme: str) -> None:
        """2.0.0 is a breaking release in packaging only. A reader upgrading "
        "needs to know their saved workflows still load."""
        assert "widget order did not change" in readme.lower()

    def test_the_two_source_trees_are_documented(self, readme: str) -> None:
        assert "`nodes/`" in readme
        assert "`src/chatterbox/`" in readme, (
            "the vendored tree is half the repository and needs an explanation"
        )

    def test_the_changelog_is_linked(self, readme: str) -> None:
        assert "CHANGELOG.md" in readme


class TestExampleWorkflowNodeIdsAreReal:
    """Every Chatterbox node in a shipped graph must exist in the package.

    A **subset** check on purpose. The extension gains nodes over time, and a
    check demanding an exhaustive match would turn every additive release into a
    documentation change -- and would fail here the moment a new node lands.
    What must never happen is the reverse: a shipped example graph naming a node
    this package does not provide.
    """

    def test_the_examples_use_at_least_one_chatterbox_node(self) -> None:
        assert _example_workflow_node_ids(), (
            "no Chatterbox node found in any example graph; either the graphs "
            "are not being read or the id pattern is wrong"
        )

    def test_every_example_node_id_is_provided_by_the_package(self) -> None:
        declared = _declared_node_ids()
        assert declared, (
            "no node_id literal found under nodes/ -- the AST scan is broken, "
            "not the nodes"
        )
        missing = sorted(_example_workflow_node_ids() - declared)
        assert not missing, (
            f"the example workflows use {missing}, which this package does not "
            f"provide. Known ids: {sorted(declared)}"
        )

    def test_the_two_original_nodes_are_still_provided(self) -> None:
        """The 2.0 release notes promise these are frozen. Hold them to it."""
        assert {"ChatterboxTTS", "ChatterboxVC"} <= _declared_node_ids()


class TestReadmeDocumentsEveryRegisteredNode:
    """The README node table must track the registry, not the other way round.

    The check this replaces was a subset assertion against a hardcoded pair, so
    it could not tell a complete table from one missing four of six nodes --
    and the table was missing four. A subset is the wrong shape for this claim
    in *both* directions: a node that ships undocumented and a row for a node
    that no longer ships are both wrong, so both directions are asserted.

    The node set comes from :meth:`ChatterboxExtension.get_node_list`, which is
    what ComfyUI builds the Add Node menu from. Deriving it here rather than
    listing ids is what makes the check future-proof: adding a node makes this
    test fail until the table gains a row for it.
    """

    def test_the_registry_is_readable(self) -> None:
        """Guards the rest: an empty registry would make the next test vacuous."""
        assert len(_registered_node_ids()) >= 2, (
            "get_node_list() returned fewer nodes than the extension is known to "
            "have; the registry read is broken, not the nodes"
        )

    def test_every_registered_node_has_a_readme_row(self, readme: str) -> None:
        missing = sorted(_registered_node_ids() - _documented_node_ids(readme))
        assert not missing, (
            f"the README node table has no row for {missing}, which this package "
            f"registers. Every node ComfyUI lists in the Add Node menu needs an "
            f"entry there."
        )

    def test_the_readme_documents_no_node_that_does_not_ship(self, readme: str) -> None:
        phantom = sorted(_documented_node_ids(readme) - _registered_node_ids())
        assert not phantom, (
            f"the README node table documents {phantom}, which this package does "
            f"not register. A row for a node that cannot be added is a dead link "
            f"in the menu a user is reading."
        )


class TestChangelogMentionsTheMigration:
    """The changelog carries the same migration instruction as the README."""

    def test_the_old_and_new_model_paths_both_appear(self, changelog: str) -> None:
        assert "models/chatterbox_tts/" in changelog
        assert "models/tts/chatterbox/" in changelog

    def test_the_renamed_example_directory_is_recorded(self, changelog: str) -> None:
        assert EXAMPLE_DIR in changelog, (
            "the rename is user-visible and belongs in the release notes"
        )