"""S2.2 -- ``perth`` must be optional in every vendored entry point.

Upstream hard-imports the watermarking library in four files and constructs
``perth.PerthImplicitWatermarker()`` unconditionally, so ``import chatterbox``
fails outright without it. These tests assert the shape of the fix from the AST
(never from text, so a comment or a docstring cannot satisfy them), then execute
the patched helper for real to check the three runtime outcomes.

``conftest.py`` mocks the vendor tree at collection time, so nothing here
imports the vendored modules. Where real behaviour is needed the relevant module
statements are extracted and executed verbatim in a fresh namespace -- the code
under test is the shipped code, not a paraphrase of it.
"""

from __future__ import annotations

import ast
import logging
import os

import pytest

from conftest import ROOT_DIR

VENDOR_DIR = os.path.join(ROOT_DIR, "src", "chatterbox")

#: The four entry points that upstream hard-imports ``perth`` in.
ENTRY_POINTS: tuple[str, ...] = (
    "chatterbox/tts.py",
    "chatterbox/vc.py",
    "chatterbox/mtl_tts.py",
    "chatterbox/tts_turbo.py",
)

WATERMARK_SYMBOLS = frozenset(
    {"_apply_watermark", "WATERMARK_LOGGER", "_watermark_warning_emitted"}
)


# ---------------------------------------------------------------------------
# Source access
# ---------------------------------------------------------------------------


def vendor_files() -> list[str]:
    """Every vendored source file, as ``chatterbox/``-rooted paths."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(VENDOR_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), VENDOR_DIR)
                out.append("chatterbox/" + rel.replace(os.sep, "/"))
    return sorted(out)


def vendor_path(rel_file: str) -> str:
    return os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))


def read(rel_file: str) -> str:
    with open(vendor_path(rel_file), "r", encoding="utf-8") as handle:
        return handle.read()


def parse(rel_file: str) -> ast.Module:
    return ast.parse(read(rel_file), filename=vendor_path(rel_file))


def parents(module: ast.Module) -> dict[ast.AST, ast.AST]:
    """Parent links for the whole tree, so "is this guarded?" is answerable."""
    links: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            links[child] = node
    return links


def enclosing(node: ast.AST, links: dict[ast.AST, ast.AST], kind: type) -> list[ast.AST]:
    """Every ancestor of ``node`` that is an instance of ``kind``, innermost first."""
    found: list[ast.AST] = []
    current = links.get(node)
    while current is not None:
        if isinstance(current, kind):
            found.append(current)
        current = links.get(current)
    return found


# ---------------------------------------------------------------------------
# The isolated watermark helper
# ---------------------------------------------------------------------------


def load_watermark_helper(rel_file: str) -> dict:
    """Execute an entry point's watermark helper in a fresh namespace.

    Only four things run: ``import logging``, the ``chatterbox.watermark``
    logger, the once-emitted flag and ``_apply_watermark`` itself. Nothing else
    in the module is touched, so librosa, transformers and the model stack stay
    out of the test process.

    A fresh namespace per call also resets the once-flag, so each test observes
    the warning sequence from the start.
    """
    module = parse(rel_file)
    keep: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.Import) and any(
            alias.name == "logging" for alias in node.names
        ):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WATERMARK_SYMBOLS
            for target in node.targets
        ):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WATERMARK_SYMBOLS:
            keep.append(node)

    namespace: dict = {"__name__": f"_isolated_{rel_file}"}
    exec(  # noqa: S102 -- executing the shipped source, by design
        compile(ast.Module(body=keep, type_ignores=[]), vendor_path(rel_file), "exec"),
        namespace,
    )
    assert "_apply_watermark" in namespace, (
        f"{rel_file} no longer defines _apply_watermark; the watermark patch is gone"
    )
    return namespace


class Recorder:
    """A stand-in for ``perth.PerthImplicitWatermarker``."""

    def __init__(self, result: str = "watermarked") -> None:
        self.calls: list[tuple[object, int]] = []
        self.result = result

    def apply_watermark(self, wav, sample_rate):
        self.calls.append((wav, sample_rate))
        return self.result


@pytest.fixture(params=ENTRY_POINTS, ids=lambda p: os.path.basename(p))
def entry_point(request) -> str:
    return request.param


# ---------------------------------------------------------------------------
# AST: the import is guarded everywhere
# ---------------------------------------------------------------------------


class TestPerthImportIsGuarded:
    def test_perth_is_imported_somewhere(self) -> None:
        """Guards the vacuous case: no ``import perth`` at all would pass below."""
        importers = [
            rel_file
            for rel_file in ENTRY_POINTS
            if any(
                alias.name == "perth" or alias.name.startswith("perth.")
                for node in ast.walk(parse(rel_file))
                if isinstance(node, ast.Import)
                for alias in node.names
            )
        ]
        assert set(importers) == set(ENTRY_POINTS), (
            f"these entry points no longer import perth at all: "
            f"{sorted(set(ENTRY_POINTS) - set(importers))}"
        )

    def test_no_import_perth_outside_a_try_except_importerror(self, entry_point: str) -> None:
        module = parse(entry_point)
        links = parents(module)
        bare: list[int] = []
        for node in ast.walk(module):
            if not isinstance(node, ast.Import):
                continue
            if not any(
                alias.name == "perth" or alias.name.startswith("perth.")
                for alias in node.names
            ):
                continue
            if not enclosing(node, links, ast.Try):
                bare.append(node.lineno)
        assert not bare, f"{entry_point}: unguarded `import perth` at lines {bare}"

    def test_except_importerror_assigns_perth_none(self, entry_point: str) -> None:
        """Without the ``perth = None`` fallback the name would be undefined."""
        module = parse(entry_point)
        links = parents(module)
        guarded = False
        for node in ast.walk(module):
            if not isinstance(node, ast.Import):
                continue
            if not any(alias.name == "perth" for alias in node.names):
                continue
            for block in enclosing(node, links, ast.Try):
                for handler in block.handlers:
                    if not (
                        isinstance(handler.type, ast.Name)
                        and handler.type.id == "ImportError"
                    ):
                        continue
                    for stmt in handler.body:
                        if (
                            isinstance(stmt, ast.Assign)
                            and any(
                                isinstance(t, ast.Name) and t.id == "perth"
                                for t in stmt.targets
                            )
                            and isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is None
                        ):
                            guarded = True
        assert guarded, (
            f"{entry_point}: the ImportError branch does not assign `perth = None`; "
            "every later `perth.PerthImplicitWatermarker()` would raise NameError"
        )

    def test_no_vendored_file_elsewhere_imports_perth(self) -> None:
        """The four entry points are the whole surface."""
        offenders: list[str] = []
        for rel_file in vendor_files():
            if rel_file in ENTRY_POINTS:
                continue
            for node in ast.walk(parse(rel_file)):
                if isinstance(node, ast.Import) and any(
                    alias.name == "perth" or alias.name.startswith("perth.")
                    for alias in node.names
                ):
                    offenders.append(rel_file)
                    break
        assert not offenders, f"perth imported outside the entry points: {offenders}"


# ---------------------------------------------------------------------------
# AST: construction and the call site are guarded
# ---------------------------------------------------------------------------


def _watermarker_constructions(module: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PerthImplicitWatermarker"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "perth"
    ]


class TestWatermarkerConstructionIsGuarded:
    def test_construction_exists_and_is_guarded(self, entry_point: str) -> None:
        source = read(entry_point)
        module = parse(entry_point)
        links = parents(module)
        calls = _watermarker_constructions(module)
        assert calls, f"{entry_point} never constructs PerthImplicitWatermarker"
        for call in calls:
            guards = enclosing(call, links, ast.If) + enclosing(call, links, ast.IfExp)
            assert guards, (
                f"{entry_point}: PerthImplicitWatermarker() at line {call.lineno} "
                "is unguarded; a missing perth would raise AttributeError"
            )
            assert "perth is not None" in source, (
                f"{entry_point}: the guard does not test `perth is not None`"
            )

    def test_watermarker_attribute_defaults_to_none(self, entry_point: str) -> None:
        source = read(entry_point)
        assert "self.watermarker = perth.PerthImplicitWatermarker() if perth is not None else None" in source


class TestApplyWatermarkCallSite:
    def test_apply_watermark_is_only_called_from_the_helper(self, entry_point: str) -> None:
        """AST, not text: the helper's docstring names ``apply_watermark`` too."""
        module = parse(entry_point)
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_watermark"
        ]
        assert len(calls) == 1, (
            f"{entry_point}: apply_watermark is called from {len(calls)} sites; "
            "every one of them has to be the guarded helper"
        )
        helper = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_watermark"
        )
        assert calls[0] in set(ast.walk(helper)), (
            f"{entry_point}: apply_watermark is called from outside _apply_watermark"
        )

    def test_generate_delegates_to_the_helper(self, entry_point: str) -> None:
        source = read(entry_point)
        assert "_apply_watermark(wav, self.sr, self.watermarker, watermark)" in source

    def test_generate_accepts_a_watermark_kwarg_defaulting_to_true(self, entry_point: str) -> None:
        module = parse(entry_point)
        functions = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        ]
        assert functions, f"{entry_point} has no generate()"
        for fn in functions:
            args = fn.args.args
            defaults = {
                arg.arg: default
                for arg, default in zip(
                    args[len(args) - len(fn.args.defaults):], fn.args.defaults
                )
            }
            assert "watermark" in defaults, (
                f"{entry_point}.generate has no `watermark` argument; the opt-out "
                "upstream lacks (S2.3) is missing"
            )
            value = defaults["watermark"]
            assert isinstance(value, ast.Constant) and value.value is True, (
                f"{entry_point}.generate: the default must be True so existing "
                "callers keep upstream's watermarked behaviour"
            )


# ---------------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------------


class TestWatermarkRuntime:
    def test_default_true_still_watermarks(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        recorder = Recorder()
        wav = object()
        assert apply(wav, 24000, recorder) == "watermarked"
        assert recorder.calls == [(wav, 24000)], (
            "the default must still route through the watermarker"
        )

    def test_watermark_false_skips_apply_watermark(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        recorder = Recorder()
        wav = object()
        assert apply(wav, 24000, recorder, watermark=False) is wav, (
            "watermark=False must return the audio untouched"
        )
        assert recorder.calls == []

    def test_watermark_false_is_silent(self, entry_point: str, caplog) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        with caplog.at_level(logging.WARNING, logger="chatterbox.watermark"):
            apply(object(), 24000, None, watermark=False)
        assert caplog.records == [], (
            "an explicit opt-out is not a degraded run; it must not warn"
        )

    def test_missing_library_warns_once_and_returns_the_tensor(
        self, entry_point: str, caplog
    ) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        wav = object()
        with caplog.at_level(logging.WARNING, logger="chatterbox.watermark"):
            first = apply(wav, 24000, None, watermark=True)
            second = apply(wav, 24000, None, watermark=True)

        assert first is wav and second is wav, (
            "with no watermarking library the audio must come back untouched, "
            "not as an exception"
        )
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1, (
            f"expected exactly one warning for two calls, got {len(messages)}: "
            f"{messages}"
        )
        assert caplog.records[0].name == "chatterbox.watermark", (
            "the warning must land on a named logger so it can be found or "
            "silenced by name"
        )
        assert "perth" in messages[0]

    def test_missing_library_does_not_raise(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        assert apply(object(), 24000, None) is not None


# ---------------------------------------------------------------------------
# Reversibility
# ---------------------------------------------------------------------------

#: The guarded-import block, exactly as the patch writes it.
_GUARD_IMPORT = """try:
    import perth
except ImportError:
    perth = None"""

_GUARDED_INIT = (
    "self.watermarker = perth.PerthImplicitWatermarker() if perth is not None else None"
)
_UPSTREAM_INIT = "self.watermarker = perth.PerthImplicitWatermarker()"
_UPSTREAM_CALL = "self.watermarker.apply_watermark(wav, sample_rate=self.sr)"
_PATCHED_CALL = "_apply_watermark(wav, self.sr, self.watermarker, watermark)"


def reverse_patch(source: str) -> str:
    """Undo the perth patch, restoring upstream's shape."""
    return (
        source.replace(_GUARD_IMPORT, "import perth")
        .replace(_GUARDED_INIT, _UPSTREAM_INIT)
        .replace(_PATCHED_CALL, _UPSTREAM_CALL)
    )


def forward_patch(source: str) -> str:
    """Re-apply the perth patch."""
    return (
        source.replace("import perth", _GUARD_IMPORT, 1)
        .replace(_UPSTREAM_INIT, _GUARDED_INIT)
        .replace(_UPSTREAM_CALL, _PATCHED_CALL)
    )


class TestPatchIsReversible:
    """A re-sync takes upstream first and re-applies this layer; it must be able to."""

    def test_reverse_restores_the_upstream_shape(self, entry_point: str) -> None:
        reversed_source = reverse_patch(read(entry_point))
        assert reversed_source != read(entry_point), (
            f"{entry_point}: the reverse edit changed nothing; the patch shapes "
            "have drifted and can no longer be undone"
        )

        module = ast.parse(reversed_source)
        bare = [
            node.lineno
            for node in ast.walk(module)
            if isinstance(node, ast.Import)
            and any(alias.name == "perth" for alias in node.names)
        ]
        assert bare, f"{entry_point}: the reverse edit left no plain `import perth`"

        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "PerthImplicitWatermarker"
        ]
        assert calls and not enclosing(calls[0], parents(module), ast.If), (
            f"{entry_point}: the reverse edit left the construction guarded"
        )
        assert _UPSTREAM_CALL in reversed_source

    def test_forward_after_reverse_is_byte_identical(self, entry_point: str) -> None:
        source = read(entry_point)
        assert forward_patch(reverse_patch(source)) == source, (
            f"{entry_point}: reverse-then-forward is not the identity; the patch "
            "and its documented inverse have drifted apart"
        )