"""S7.3 -- the logging idiom, asserted on the AST rather than on the text.

ComfyUI installs its own ``ColoredFormatter`` on the **root** logger before any
custom node is imported, and it owns the console, the level and the colouring.
A custom node pack that builds its own ``Logger``, its own ``Formatter``, its
own ``LoggerAdapter`` or its own handlers does not add information: it *fights*
the host. Two handlers on one record means the message is printed twice, and a
``setLevel`` on the root logger silences every other extension in the process.

So the package owns exactly one thing: the ``[Chatterbox TTS]`` prefix. Every
call is a bare ``logging.<level>("[Chatterbox TTS] …")`` on the root logger.

**Why this file parses the AST instead of matching the source text.** A regex
over the file is worthless here, in both directions:

* it passes vacuously against a message split over several lines
  (``logging.info(\n    "[Chatterbox TTS] …",\n)`` has no single line holding
  both the call and its message), and
* it keeps passing after the line it was written for has been *deleted*,
  because a pattern that matches nothing is not a failure unless something
  else also fails.

Walking the tree instead means a call that exists is checked whether it is on
one line or nine, and a call that has been deleted is simply not visited --
which is why :func:`_scan` also counts what it found, and the count itself is
asserted. A guard that could pass by finding nothing is not a guard.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest

from conftest import ROOT_DIR
from ComfyUI_Chatterbox.modules import downloader as _downloader

#: The prefix the package owns. Everything else about a log record belongs to
#: ComfyUI.
LOG_PREFIX = "[Chatterbox TTS] "

#: Shipped Python. The vendored ``src/chatterbox`` tree is upstream code that
#: logs however upstream logs; ``tests/`` *describes* the idiom rather than
#: following it (this file's own ``getLogger`` mention would trip the scan).
SCANNED_ROOTS: tuple[str, ...] = ("modules", "nodes")
SCANNED_FILES: tuple[str, ...] = ("chatterbox_nodes.py", "__init__.py")

#: Files in the shipped tree that are *not* under the idiom, each with the
#: reason it is not. Empty.
#:
#: Shrink-only, and pinned by :meth:`TestExemptions.test_no_file_is_exempt`:
#: adding an entry is a deliberate act that has to come with the reason, and the
#: guard stops being a guard the moment one appears without one.
#:
#: ``modules/downloader.py`` used to be the sole entry -- a named logger of its
#: own plus ``setLevel`` calls on four third-party loggers. Both were removed in
#: B12; see the comment block above ``_suppress_hf_http_logs`` in that file for
#: what was dropped and why, and
#: :class:`TestDownloaderLoggingDecision` below for the test that keeps it from
#: creeping back.
EXEMPT_FILES: dict[str, str] = {}

#: Attribute / name spellings that mean "this package owns a logger or its
#: output". Each entry is (kind, spelling, why-it-is-banned).
FORBIDDEN_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("attribute", "getLogger", "builds a named logger; the root logger is the target"),
    ("attribute", "Logger", "a custom Logger subclass fights ComfyUI's formatter"),
    ("attribute", "Formatter", "a custom Formatter overrides the host's colouring"),
    ("attribute", "LoggerAdapter", "an adapter is a second place to configure output"),
    ("attribute", "addHandler", "a second handler on one record prints it twice"),
    ("attribute", "setLevel", "silences every other extension sharing the logger"),
    ("attribute", "basicConfig", "installs handlers on the root logger at import"),
    ("attribute", "fileConfig", "dictConfig replaces the host's whole logging tree"),
)

#: ``logging.<level>`` is the only call shape allowed to emit a record.
LOG_LEVELS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception"}
)


def _iter_shipped_sources() -> list[str]:
    """Repo-relative POSIX paths of every shipped Python file."""
    found: list[str] = []
    for root in SCANNED_ROOTS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT_DIR, root)):
            dirnames[:] = sorted(
                name for name in dirnames if name != "__pycache__"
            )
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    relative = os.path.relpath(
                        os.path.join(dirpath, filename), ROOT_DIR
                    )
                    found.append(relative.replace(os.sep, "/"))
    found.extend(SCANNED_FILES)
    return sorted(found)


def _scanned() -> list[str]:
    return [
        relative
        for relative in _iter_shipped_sources()
        if relative not in EXEMPT_FILES
    ]


def _parse(relative: str) -> ast.Module:
    with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=relative)


def _message_prefix(node: ast.Call) -> str | None:
    """The literal prefix of a call's first argument, or ``None`` if dynamic.

    ``None`` means the message is built at runtime (an f-string, a ``%``-format
    or a variable), which is reported separately rather than silently accepted:
    the caller has to prove those on purpose.
    """
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        # An implicitly concatenated literal is already folded into one
        # Constant by the parser, so this covers the multi-line form too.
        return first.value[: len(LOG_PREFIX)]
    return None


def _artifact_hits(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Every banned ``(lineno, kind, spelling)`` in one parsed module.

    Both spellings are matched: ``logging.getLogger(...)`` is an ``ast.Attribute``
    and ``from logging import getLogger`` then ``getLogger(...)`` is an
    ``ast.Name``. A package that bans the attribute but not the name has a hole
    exactly one import statement wide.
    """
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            for kind, spelling, _reason in FORBIDDEN_ARTIFACTS:
                if node.attr == spelling:
                    hits.append((node.lineno, kind, spelling))
        elif isinstance(node, ast.Name):
            for kind, spelling, _reason in FORBIDDEN_ARTIFACTS:
                if node.id == spelling:
                    hits.append((node.lineno, kind, spelling))
    return hits


def _scan() -> tuple[list[str], list[tuple[str, int, str, str]], list[str]]:
    """Return ``(files, artifacts, unprefixed)`` over the scanned tree.

    ``artifacts`` entries are ``(relative, lineno, kind, spelling)``.
    ``unprefixed`` entries are ``(relative, lineno, detail)``.
    """
    artifacts: list[tuple[str, int, str, str]] = []
    unprefixed: list[tuple[str, int, str]] = []
    files = _scanned()

    for relative in files:
        tree = _parse(relative)
        for lineno, kind, spelling in _artifact_hits(tree):
            artifacts.append((relative, lineno, kind, spelling))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func = node.func
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr in LOG_LEVELS
                ):
                    prefix = _message_prefix(node)
                    if prefix != LOG_PREFIX:
                        detail = (
                            "dynamic message, prefix unverifiable"
                            if prefix is None
                            else f"starts with {prefix!r}"
                        )
                        unprefixed.append((relative, node.lineno, detail))
    return files, artifacts, unprefixed


@pytest.fixture(scope="module")
def scan():
    return _scan()


class TestNoOwnLoggerOrHandler:
    """The package configures nothing about logging."""

    def test_the_scan_covers_the_shipped_tree(self, scan) -> None:
        files, _artifacts, _unprefixed = scan
        assert len(files) >= 10, (
            f"the scan only reached {files}; a guard that cannot see the code "
            f"is not a guard"
        )
        assert "modules/patcher.py" in files
        assert "nodes/tts_node.py" in files
        assert "modules/downloader.py" in files, (
            "modules/downloader.py was the last exempted file and has been "
            "converted; if it fell out of the scan, _scanned() is broken rather "
            "than the file being clean"
        )

    def test_artifact_count_is_zero(self, scan) -> None:
        """The number, so "found nothing" cannot be mistaken for "looked".

        A guard that reports a count of zero because it parsed zero files is
        exactly the failure this file exists to prevent.
        """
        files, artifacts, _unprefixed = scan
        assert files, "no files were scanned"
        assert len(artifacts) == 0, "logging artifacts found:\n" + "\n".join(
            f"  {relative}:{lineno} {kind} '{spelling}'"
            for relative, lineno, kind, spelling in artifacts
        )

    @pytest.mark.parametrize(
        ("kind", "spelling", "reason"), FORBIDDEN_ARTIFACTS,
        ids=[spelling for _k, spelling, _r in FORBIDDEN_ARTIFACTS],
    )
    def test_each_spelling_is_rejected_alone(self, kind, spelling, reason) -> None:
        """Every banned spelling is rejected on its own, not just as a set.

        Proves the table is live *and* that both spellings are caught: a
        spelling nobody writes would otherwise sit in the table contributing
        nothing while looking covered, and a table matched on ``ast.Attribute``
        only would let ``from logging import getLogger`` through.
        """
        assert reason, "every banned spelling needs a reason to be worth banning"

        as_attribute = ast.parse(f"logging.{spelling}\n")
        assert any(
            found == spelling for _lineno, _kind, found in _artifact_hits(as_attribute)
        ), f"logging.{spelling} was not detected"

        as_name = ast.parse(f"from logging import {spelling}\n{spelling}\n")
        assert any(
            found == spelling for _lineno, _kind, found in _artifact_hits(as_name)
        ), f"an imported {spelling} was not detected"

    def test_a_clean_module_produces_no_hits(self) -> None:
        """Negative control for the table, so it cannot be trivially broad."""
        clean = ast.parse(
            'import logging\n\n\ndef f():\n'
            '    logging.info("[Chatterbox TTS] fine")\n'
        )
        assert _artifact_hits(clean) == []


class TestEveryMessageCarriesThePrefix:
    def test_unprefixed_count_is_zero(self, scan) -> None:
        files, _artifacts, unprefixed = scan
        assert files, "no files were scanned"
        assert unprefixed == [], "log messages without the package prefix:\n" + "\n".join(
            f"  {relative}:{lineno} {detail}"
            for relative, lineno, detail in unprefixed
        )

    def test_the_scanner_can_actually_see_a_clean_call(self, scan) -> None:
        """Positive control.

        A scanner that reported "no violations" because it recognised no calls
        at all would pass every test above. This proves the shape is matched.
        """
        good = ast.parse('logging.info("[Chatterbox TTS] hello")\n')
        call = next(
            n for n in ast.walk(good)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        )
        assert _message_prefix(call) == LOG_PREFIX

    def test_the_scanner_rejects_a_prefix_on_the_next_line(self) -> None:
        """Multi-line calls are the reason this is not a regex.

        The message lives on line two and the ``logging.`` on line one; a
        line-based check sees neither. The AST sees both.
        """
        source = (
            "logging.info(\n"
            '    "[Chatterbox TTS] spread over lines",\n'
            "    1,\n"
            ")\n"
        )
        call = next(
            n for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        )
        assert _message_prefix(call) == LOG_PREFIX

    def test_the_scanner_rejects_a_bare_message(self) -> None:
        call = next(
            n for n in ast.walk(ast.parse('logging.warning("no prefix here")\n'))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        )
        assert _message_prefix(call) != LOG_PREFIX

    def test_the_scanner_reports_a_dynamic_message_as_unverifiable(self) -> None:
        """An f-string cannot be checked statically, so it is reported.

        Deliberately *not* waved through: a dynamic message is the one shape
        that can drop the prefix without anything noticing.
        """
        call = next(
            n
            for n in ast.walk(ast.parse('logging.info(f"hello {name}")\n'))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        )
        assert _message_prefix(call) is None


class TestExemptions:
    """The exemption list is the escape hatch, and it is shut."""

    def test_no_file_is_exempt(self) -> None:
        """Not a parametrised sweep: an empty sweep would pass vacuously.

        The three tests this replaces were parametrised over ``EXEMPT_FILES``,
        which is exactly how a retirement map turns into a graveyard -- with the
        last entry deleted they quietly assert nothing at all. One direct
        equality cannot pass by finding nothing.
        """
        assert EXEMPT_FILES == {}, (
            f"EXEMPT_FILES must be empty; it names {sorted(EXEMPT_FILES)}. Every "
            f"entry is a hole in the S7.3 guard, and an entry without a file "
            f"behind it hides real debt instead of recording it."
        )


class TestDownloaderLoggingDecision:
    """The one deliberate call this package makes about someone else's logging.

    ``modules/downloader.py`` used to raise the level of ``httpx``,
    ``urllib3.connectionpool``, ``filelock`` and the handlers on
    ``huggingface_hub``'s own logger for the duration of a download, then reset
    all four to ``NOTSET``. That is now gone: ComfyUI owns the root logger, and
    "restore to NOTSET" is not a restore -- it clobbers whatever level another
    extension had chosen. ``huggingface_hub`` is still asked to be quiet, through
    the verbosity API it publishes for exactly that purpose.

    These assertions exist so the decision is a fact rather than an accident:
    a well-meaning later patch that re-adds a global ``setLevel`` fails here.
    """

    def test_the_file_is_under_the_ordinary_guard(self) -> None:
        files, _artifacts, _unprefixed = _scan()
        assert "modules/downloader.py" in files

    def test_it_mutates_no_third_party_logger_level(self) -> None:
        tree = _parse("modules/downloader.py")
        offenders = sorted(
            f"{node.lineno}:{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in ("setLevel", "getLogger")
        )
        assert offenders == [], (
            "modules/downloader.py reaches into another package's logging tree:\n  "
            + "\n  ".join(offenders)
            + "\nComfyUI owns the root logger. If third-party noise has to be "
            "reduced, reduce it through that package's own API."
        )

    def test_huggingface_hub_is_still_asked_to_be_quiet(self) -> None:
        """The quieting that *is* deliberate, through the documented entry point."""
        source = inspect.getsource(_downloader)
        assert "set_verbosity_error" in source
        assert "disable_propagation" in source

    def test_the_decision_is_written_down_where_the_code_is(self) -> None:
        """A decision with no comment is indistinguishable from an oversight."""
        source = inspect.getsource(_downloader)
        assert "setLevel" in source, (
            "the comment explaining why the third-party setLevel calls were "
            "removed has gone with them"
        )