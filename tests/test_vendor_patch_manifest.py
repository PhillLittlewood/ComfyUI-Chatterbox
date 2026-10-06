"""Contract tests for ``src/PATCHES.md``.

The vendored ``chatterbox`` tree is re-synced from upstream by taking upstream
first and re-applying the local patch layer. ``src/PATCHES.md`` is the record of
that layer, so it has to be checked mechanically rather than by reading:

* every patch entry must name a symbol that still exists in the vendored source;
* every category must be one of the three defined ones;
* the file must not leak a path outside the repository -- no scratch-directory
  reference, no absolute path, no drive letter;
* and, in the other direction, every vendored file that differs from upstream is
  recorded here -- by a row, or by an inline ``PATCHED (src/PATCHES.md ...)``
  marker -- so a local edit nobody wrote down cannot ship.

Symbol lookup is AST-first (a class, a function, a module-level assignment) with
a text fallback, because the patch layer legitimately touches things an AST
cannot name (an argument inside a signature, an attribute on ``self``).

The suite mocks the whole vendored tree at collection time, so these tests read
the source from disk and never import it.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass

import pytest

from conftest import ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")
VENDOR_DIR = os.path.join(SRC_DIR, "chatterbox")
MANIFEST_PATH = os.path.join(SRC_DIR, "PATCHES.md")

VALID_CATEGORIES = frozenset({"ComfyUI-API patch", "compat patch", "feature addition"})

# A table row: | ID | file | symbol | category | rationale |
ROW_RE = re.compile(r"^\|\s*(?P<id>[PU]-\d{3})\s*\|(?P<rest>.+)\|\s*$")
HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")

PATCH_SECTION = "Patch manifest"
UPSTREAM_SECTION = "Upstream changes adopted verbatim"

# Number of cells after the ID cell, per section. The upstream table carries no
# category column: nothing was edited locally, so there is nothing to categorise.
SECTION_CELLS = {PATCH_SECTION: 4, UPSTREAM_SECTION: 3}


@dataclass(frozen=True)
class Entry:
    """One row of either manifest table."""

    entry_id: str
    file: str
    symbol: str
    category: str
    rationale: str


def _strip_cell(cell: str) -> str:
    """Drop the Markdown code ticks and surrounding whitespace from a cell."""
    return cell.strip().strip("`").strip()


def _read_manifest() -> str:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _parse_entries(text: str) -> list[Entry]:
    """Every row of either manifest table, in file order.

    Section-aware: the category column only exists in the patch table, and a row
    outside a recognised section is an error rather than a silently skipped line.
    """
    entries: list[Entry] = []
    section: str | None = None
    for line in text.splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading is not None:
            section = heading.group("title")
            continue

        match = ROW_RE.match(line.strip())
        if match is None:
            continue

        assert section in SECTION_CELLS, (
            f"manifest row {match.group('id')} is outside a recognised section"
        )
        cells = [_strip_cell(cell) for cell in match.group("rest").split("|")]
        expected = SECTION_CELLS[section]
        assert len(cells) == expected, (
            f"row {match.group('id')} in '{section}' has {len(cells)} cells "
            f"after the ID, expected {expected}"
        )

        if section == PATCH_SECTION:
            file_cell, symbol, category, rationale = cells
        else:
            file_cell, symbol, rationale = cells
            category = ""

        entries.append(
            Entry(
                entry_id=match.group("id"),
                file=file_cell,
                symbol=symbol,
                category=category,
                rationale=rationale,
            )
        )
    return entries


@pytest.fixture(scope="module")
def manifest_text() -> str:
    return _read_manifest()


@pytest.fixture(scope="module")
def entries(manifest_text: str) -> list[Entry]:
    return _parse_entries(manifest_text)


@pytest.fixture(scope="module")
def patch_entries(entries: list[Entry]) -> list[Entry]:
    return [entry for entry in entries if entry.entry_id.startswith("P-")]


@pytest.fixture(scope="module")
def upstream_entries(entries: list[Entry]) -> list[Entry]:
    return [entry for entry in entries if entry.entry_id.startswith("U-")]


# ---------------------------------------------------------------------------
# Parsing / self-consistency
# ---------------------------------------------------------------------------


class TestManifestShape:
    def test_manifest_exists_and_is_not_empty(self, manifest_text: str) -> None:
        assert os.path.isfile(MANIFEST_PATH)
        assert manifest_text.strip(), "src/PATCHES.md is empty"

    def test_at_least_one_patch_entry(self, patch_entries: list[Entry]) -> None:
        # Guards the rest of this module: an empty parse would make every other
        # assertion vacuously true.
        assert len(patch_entries) >= 10

    def test_at_least_one_upstream_entry(self, upstream_entries: list[Entry]) -> None:
        assert len(upstream_entries) >= 5

    def test_ids_are_unique(self, entries: list[Entry]) -> None:
        ids = [entry.entry_id for entry in entries]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        assert not duplicates, f"duplicate manifest ids: {duplicates}"

    def test_every_patch_entry_has_a_category(self, patch_entries: list[Entry]) -> None:
        missing = [e.entry_id for e in patch_entries if not e.category]
        assert not missing, f"patch entries without a category: {missing}"

    def test_every_category_is_defined(self, patch_entries: list[Entry]) -> None:
        unknown = sorted({e.category for e in patch_entries} - VALID_CATEGORIES)
        assert not unknown, f"unknown categories {unknown}; allowed: {sorted(VALID_CATEGORIES)}"

    def test_all_three_categories_are_used(self, patch_entries: list[Entry]) -> None:
        used = {e.category for e in patch_entries}
        assert used == VALID_CATEGORIES, f"unused categories: {sorted(VALID_CATEGORIES - used)}"

    def test_every_entry_has_a_rationale(self, entries: list[Entry]) -> None:
        thin = [e.entry_id for e in entries if len(e.rationale) < 20]
        assert not thin, f"entries with no real rationale: {thin}"

    def test_every_entry_names_a_file_inside_the_vendor_tree(
        self, entries: list[Entry]
    ) -> None:
        bad = [
            e.entry_id
            for e in entries
            if not e.file.startswith("chatterbox/") or not e.file.endswith(".py")
        ]
        assert not bad, f"entries not naming a vendored source file: {bad}"

    def test_entries_are_grouped_per_table(
        self, patch_entries: list[Entry], upstream_entries: list[Entry]
    ) -> None:
        """Append-friendly: a new batch appends whole groups, never interleaves.

        Checked per table -- the same file may legitimately appear in both, since
        a file can carry a local edit *and* upstream improvements.
        """
        for label, group in (("patch", patch_entries), ("upstream", upstream_entries)):
            runs: list[str] = []
            for entry in group:
                if not runs or runs[-1] != entry.file:
                    runs.append(entry.file)
            repeated = sorted({f for f in runs if runs.count(f) > 1})
            assert not repeated, f"{label} table splits entries for {repeated}"


# ---------------------------------------------------------------------------
# Symbol existence
# ---------------------------------------------------------------------------


def _vendor_path(rel_file: str) -> str:
    return os.path.join(SRC_DIR, rel_file.replace("/", os.sep))


def _self_attributes(cls: ast.ClassDef) -> set[str]:
    """Names assigned as ``self.<name>`` anywhere in a class body.

    The patch layer sets flags (``self.fp16``, ``self.lock``) that never appear
    as a class-level definition, so a class-member check that only looks at
    ``def``/``Assign`` in the class body would report them missing.
    """
    names: set[str] = set()
    for node in ast.walk(cls):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return names


def _module_symbols(path: str) -> tuple[set[str], dict[str, set[str]]]:
    """Return ``(module_level_names, {class_name: its_members})`` for a module."""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    module_names: set[str] = set()
    class_members: dict[str, set[str]] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
            if isinstance(node, ast.ClassDef):
                members = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                for child in node.body:
                    if isinstance(child, ast.Assign):
                        members.update(
                            target.id for target in child.targets if isinstance(target, ast.Name)
                        )
                members |= _self_attributes(node)
                class_members[node.name] = members
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            module_names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.Import):
            module_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Try):
            # try/except ImportError blocks (optional deps) hide names from the
            # module-level walk above; descend so they are still discoverable.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    module_names.add(sub.name)

    return module_names, class_members


def _symbol_exists(rel_file: str, symbol: str, cache: dict[str, tuple[set[str], dict[str, set[str]]]]) -> bool:
    path = _vendor_path(rel_file)
    if not os.path.isfile(path):
        return False
    if path not in cache:
        cache[path] = _module_symbols(path)
    module_names, class_members = cache[path]

    if "." in symbol:
        class_name, member = symbol.split(".", 1)
        return member in class_members.get(class_name, set())

    if symbol in module_names:
        return True

    # Text fallback: the symbol is a real, named thing in the source but not a
    # module-level or class-level definition (a keyword argument, a local flag).
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return re.search(rf"\b{re.escape(symbol)}\b", source) is not None


class TestSymbolsExist:
    def test_every_patch_symbol_exists(self, patch_entries: list[Entry]) -> None:
        cache: dict[str, tuple[set[str], dict[str, set[str]]]] = {}
        missing = [
            (entry.entry_id, entry.file, entry.symbol)
            for entry in patch_entries
            if not _symbol_exists(entry.file, entry.symbol, cache)
        ]
        assert not missing, f"manifest symbols absent from the vendored source: {missing}"

    def test_every_upstream_symbol_exists(self, upstream_entries: list[Entry]) -> None:
        cache: dict[str, tuple[set[str], dict[str, set[str]]]] = {}
        missing = [
            (entry.entry_id, entry.file, entry.symbol)
            for entry in upstream_entries
            if not _symbol_exists(entry.file, entry.symbol, cache)
        ]
        assert not missing, f"upstream-fix symbols absent from the vendored source: {missing}"

    def test_every_patched_file_is_inside_the_vendor_tree(
        self, patch_entries: list[Entry]
    ) -> None:
        for entry in patch_entries:
            assert os.path.isfile(_vendor_path(entry.file)), (
                f"{entry.entry_id} names {entry.file}, which is not in src/chatterbox/"
            )

    def test_manifest_covers_every_patched_vendor_file(self, patch_entries: list[Entry]) -> None:
        """Every row names a file that really is vendored.

        This is the *forward* direction only: manifest row -> file. It cannot
        detect a local edit that nobody recorded, because that shows up as a
        vendored file with no row, which a subset assertion never sees. The
        reverse direction -- vendored file -> row or inline marker, by diffing
        against upstream -- is :class:`TestVendorTreeIsFullyClassified`, which
        needs an upstream checkout pointed at by ``CHATTERBOX_UPSTREAM_SRC``.
        """
        patched_files = {entry.file for entry in patch_entries}
        # Files that only ever receive a version-neutral re-sync carry no entry.
        assert patched_files <= set(vendor_files())


def vendor_files() -> list[str]:
    """Every vendored source file, as a repo-relative ``src/``-rooted path."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(VENDOR_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), SRC_DIR)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


# ---------------------------------------------------------------------------
# Reverse direction -- the manifest must not be able to miss a local edit
# ---------------------------------------------------------------------------

#: Points at the ``chatterbox/`` package directory of an upstream checkout (or
#: its parent). Supplied by the environment, never hardcoded: nothing shipped may
#: name where upstream was read from (see ``tests/test_no_dev_references.py`` and
#: the containment tests below), and a path baked into this file would be a
#: machine-specific pointer into the very provenance the manifest must hide.
UPSTREAM_ENV_VAR = "CHATTERBOX_UPSTREAM_SRC"

#: An inline marker, e.g. ``# PATCHED (src/PATCHES.md P-011, P-020, P-021): ...``
PATCH_MARKER_RE = re.compile(
    r"PATCHED \(src/PATCHES\.md ((?:[PU]-\d{3})(?:,\s*[PU]-\d{3})*)\)"
)

#: The three buckets a vendored file can fall into, as produced by classify().
IDENTICAL = "identical-to-upstream"
DIFFERS = "differs-from-upstream"
VENDORED_ONLY = "no-upstream-counterpart"


@dataclass(frozen=True)
class Classification:
    """How one vendored file stands against upstream and against the manifest."""

    rel_file: str
    state: str
    marker_ids: tuple[str, ...]
    recorded: bool


@pytest.fixture(scope="module")
def upstream_root() -> str:
    """The upstream ``chatterbox/`` package directory, or a skip that says why.

    Set ``CHATTERBOX_UPSTREAM_SRC`` to that directory in an upstream checkout,
    or to its parent. Unset, this skips with the variable named: the reverse
    check needs a real upstream tree to diff against, and silently passing
    without one would be exactly the vacuous guarantee this section exists to
    remove.
    """
    configured = os.environ.get(UPSTREAM_ENV_VAR, "").strip()
    if not configured:
        pytest.skip(
            f"{UPSTREAM_ENV_VAR} is not set: no upstream checkout to diff the "
            f"vendored tree against, so unrecorded local edits cannot be "
            f"detected. Point it at the upstream `chatterbox/` package "
            f"directory (or its parent) to run the reverse-direction check."
        )
    root = os.path.abspath(configured)
    # Accept either shape, because both are natural to type and guessing wrong
    # would make a correct invocation look like a broken one:
    #   .../chatterbox/src/chatterbox   (the package directory)
    #   .../chatterbox/src             (its parent)
    for candidate in (root, os.path.join(root, "chatterbox")):
        if os.path.isfile(os.path.join(candidate, "__init__.py")):
            return candidate
    # Set-but-wrong is a mistake in the invocation, not an absent feature:
    # skipping here would report the check as unavailable and let the gate go
    # green with the coverage silently off.
    pytest.fail(
        f"{UPSTREAM_ENV_VAR}={root} is neither the upstream chatterbox package "
        f"directory nor its parent: no __init__.py at {root} or "
        f"{os.path.join(root, 'chatterbox')}."
    )


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def classify(upstream_root: str, patch_entries: list[Entry]) -> dict[str, Classification]:
    """Diff every vendored file against upstream and against the manifest.

    Byte equality, not a text diff: a whitespace-only or line-ending change is
    still a divergence from upstream and has to be recorded either way.
    """
    recorded = {entry.file for entry in patch_entries}
    out: dict[str, Classification] = {}
    for rel_file in vendor_files():
        vendored_path = _vendor_path(rel_file)
        upstream_path = os.path.join(
            upstream_root, rel_file[len("chatterbox/"):].replace("/", os.sep)
        )
        if not os.path.isfile(upstream_path):
            state = VENDORED_ONLY
        elif _read_bytes(vendored_path) == _read_bytes(upstream_path):
            state = IDENTICAL
        else:
            state = DIFFERS
        with open(vendored_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        marker_ids = tuple(
            dict.fromkeys(
                part.strip()
                for match in PATCH_MARKER_RE.finditer(source)
                for part in match.group(1).split(",")
            )
        )
        out[rel_file] = Classification(
            rel_file=rel_file,
            state=state,
            marker_ids=marker_ids,
            recorded=rel_file in recorded,
        )
    return out


@pytest.fixture(scope="module")
def classified(
    upstream_root: str, patch_entries: list[Entry]
) -> dict[str, Classification]:
    return classify(upstream_root, patch_entries)


class TestVendorTreeIsFullyClassified:
    """The guarantee the forward-direction test cannot give.

    Every vendored file is sorted into exactly one bucket against upstream, and
    each bucket has one legal shape:

    * **differs from upstream** -- a local edit. It must be recorded: a row in
      the patch table, or an inline ``PATCHED (src/PATCHES.md ...)`` marker. An
      edit that is neither is the failure this catches.
    * **identical to upstream** -- a version-neutral re-sync carry. It must carry
      *no* patch marker: a marker here is a stale entry pointing at a difference
      that no longer exists, and would send the next re-sync looking for it.
    * **no upstream counterpart** -- a file this repository added. Same rule as
      a differing file: it has to be recorded somewhere or its origin is lost.

    Upstream rows (``U-``) are deliberately *not* consulted for the "recorded"
    half of that rule. They record upstream's own changes adopted verbatim --
    precisely the identical files -- so treating one as evidence of a local edit
    would flag all six of them.
    """

    def test_every_vendored_file_is_classified(
        self, classified: dict[str, Classification]
    ) -> None:
        """No file escapes the diff: an unknown bucket means the walk is broken."""
        assert set(classified) == set(vendor_files())
        unknown = sorted(
            rel
            for rel, item in classified.items()
            if item.state not in (IDENTICAL, DIFFERS, VENDORED_ONLY)
        )
        assert not unknown, f"files with no classification: {unknown}"

    def test_a_local_edit_is_always_recorded(
        self, classified: dict[str, Classification]
    ) -> None:
        """The planted-violation check: an unrecorded diff fails here."""
        unrecorded = sorted(
            item.rel_file
            for item in classified.values()
            if item.state in (DIFFERS, VENDORED_ONLY)
            and not item.recorded
            and not item.marker_ids
        )
        assert not unrecorded, (
            f"vendored files that differ from upstream but appear in neither "
            f"the patch table nor an inline PATCHED marker: {unrecorded}. Record "
            f"them in src/PATCHES.md, or revert them to upstream."
        )

    def test_an_identical_file_is_not_a_recorded_patch(
        self, classified: dict[str, Classification]
    ) -> None:
        """No stale manifest row left behind by a re-sync that undid a patch.

        Only the **table** can go stale here, and the asymmetry is structural
        rather than incidental: an inline ``PATCHED`` marker is itself a
        difference from upstream, so a file carrying one can never be
        byte-identical to upstream. A row, by contrast, survives its patch being
        reverted -- which is exactly this regression, and it is the failure the
        upstream-fix tests describe from the other side.
        """
        stale = sorted(
            item.rel_file
            for item in classified.values()
            if item.state == IDENTICAL and item.recorded
        )
        assert not stale, (
            f"files byte-identical to upstream that the patch table still "
            f"claims as locally edited ({stale}): those rows are stale"
        )

    def test_inline_marker_ids_exist_in_the_manifest(
        self, classified: dict[str, Classification], entries: list[Entry]
    ) -> None:
        """A marker pointing at a row that does not exist is no record at all."""
        known = {entry.entry_id for entry in entries}
        dangling = sorted(
            (item.rel_file, marker_id)
            for item in classified.values()
            for marker_id in item.marker_ids
            if marker_id not in known
        )
        assert not dangling, (
            f"inline PATCHED markers naming a manifest id that does not exist: "
            f"{dangling}"
        )


# ---------------------------------------------------------------------------
# Repository containment -- the manifest must not leak provenance by pointer
# ---------------------------------------------------------------------------


class TestManifestStaysInsideTheRepo:
    def test_no_dot_dev_reference(self, manifest_text: str) -> None:
        # The trailing `\b` is what keeps the English word "device." from
        # tripping the check; what we forbid is the scratch *directory*.
        leak = re.search(r"\.dev\b", manifest_text)
        assert leak is None, (
            "src/PATCHES.md names the scratch directory at offset "
            f"{leak.start() if leak else -1}; provenance must be recorded by "
            "rationale, not by pointer"
        )

    def test_no_reference_corpus_pointer(self, manifest_text: str) -> None:
        for needle in ("reference/repos", "reference/docs", "plans/", ".git/"):
            assert needle not in manifest_text, f"manifest leaks {needle!r}"

    def test_no_absolute_path(self, manifest_text: str) -> None:
        windows_drive = re.search(r"[A-Za-z]:[\\/]", manifest_text)
        assert windows_drive is None, f"absolute path in manifest: {windows_drive.group(0)}"
        assert "C:/" not in manifest_text and "C:\\" not in manifest_text

    def test_no_unc_or_posix_absolute_path(self, manifest_text: str) -> None:
        assert not re.search(r"(?<![\w.])/(?:home|Users|mnt|opt|var|tmp)/", manifest_text)

    def test_every_manifest_path_is_relative(self, patch_entries: list[Entry]) -> None:
        absolute = [e.entry_id for e in patch_entries if e.file.startswith(("/", "\\"))]
        assert not absolute, f"absolute paths in the file column: {absolute}"

    def test_no_http_link_to_upstream(self, manifest_text: str) -> None:
        """Provenance is by rationale. A URL to the upstream tree would also
        become a maintenance tripwire; upstream is identified by name only."""
        assert "http://" not in manifest_text and "https://" not in manifest_text


def test_manifest_is_tracked_by_git_not_ignored(repo_root: str) -> None:
    """The manifest is the contract, so an ignore rule would silently void it."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "src/PATCHES.md"],
        cwd=repo_root,
        capture_output=True,
    )
    assert result.returncode != 0, "src/PATCHES.md is matched by a .gitignore rule"