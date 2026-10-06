"""S2.1 -- ``ensure_vendor_on_path`` appends ``src/``, once, at the tail.

The vendored ``chatterbox`` tree is not a package ComfyUI puts on ``sys.path``,
so the loader appends this repository's ``src/``. Two properties matter and both
are easy to get wrong:

* **append, never insert** -- a prepend would silently shadow a pip-installed
  ``chatterbox`` the user chose, and would shadow site-packages generally;
* **idempotent** -- ComfyUI re-imports a custom node on every startup, so a
  loader that appends unconditionally grows ``sys.path`` forever.

The in-process tests run the real function. The end-to-end check needs a clean
interpreter (the harness mocks the whole vendor tree), so it runs in a
subprocess and *skips with a named reason* when a dependency is genuinely
missing -- never silently, and never by swallowing the error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")

MODULES_DIR = os.path.join(ROOT_DIR, "modules")


@pytest.fixture
def vendor_loader():
    """The loader module, imported by file path and restored afterwards.

    ``modules`` is a real package in the repository, but under the pytest alias
    the harness registers it as a namespace stand-in; loading the file directly
    keeps this test honest about which function it exercises.
    """
    import importlib.util

    path = os.path.join(MODULES_DIR, "vendor_loader.py")
    spec = importlib.util.spec_from_file_location("_vendor_loader_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_sys_path():
    """``sys.path`` without our ``src/``, restored however the test ends."""
    original = list(sys.path)
    try:
        while SRC_DIR in sys.path:
            sys.path.remove(SRC_DIR)
        yield
    finally:
        sys.path[:] = original


class TestAppendSemantics:
    def test_src_lands_at_the_tail(self, vendor_loader, clean_sys_path) -> None:
        vendor_loader.ensure_vendor_on_path()
        assert sys.path[-1] == SRC_DIR, (
            "the vendored src/ must be appended, so anything already importable "
            "keeps winning"
        )

    def test_nothing_is_inserted_ahead_of_the_existing_entries(
        self, vendor_loader, clean_sys_path
    ) -> None:
        before = list(sys.path)
        vendor_loader.ensure_vendor_on_path()
        assert sys.path[: len(before)] == before, (
            "an existing entry was reordered; a pip-installed chatterbox or "
            "site-packages entry has to keep winning"
        )
        assert sys.path[len(before):] == [SRC_DIR], (
            f"expected exactly one appended entry, got {sys.path[len(before):]}"
        )

    def test_idempotent_across_repeated_calls(self, vendor_loader, clean_sys_path) -> None:
        vendor_loader.ensure_vendor_on_path()
        after_first = list(sys.path)
        for _ in range(5):
            vendor_loader.ensure_vendor_on_path()
        assert sys.path == after_first, (
            f"repeated calls grew sys.path: {len(after_first)} -> {len(sys.path)}"
        )
        assert sys.path.count(SRC_DIR) == 1

    def test_call_on_an_already_present_path_is_a_no_op(
        self, vendor_loader, clean_sys_path
    ) -> None:
        vendor_loader.ensure_vendor_on_path()
        # A pip-installed chatterbox sits in front of us; that must stay true.
        sys.path.insert(0, "/some/site-packages")
        before = list(sys.path)
        vendor_loader.ensure_vendor_on_path()
        assert sys.path == before
        assert sys.path[0] == "/some/site-packages"

    def test_returns_none(self, vendor_loader, clean_sys_path) -> None:
        assert vendor_loader.ensure_vendor_on_path() is None


class TestLoaderPackage:
    def test_modules_is_a_real_package(self) -> None:
        init = os.path.join(MODULES_DIR, "__init__.py")
        assert os.path.isfile(init), (
            "modules/ has no __init__.py; `from .modules.vendor_loader import ...` "
            "cannot resolve without it"
        )

    def test_modules_init_is_empty(self) -> None:
        """An empty ``__init__.py``: re-exporting would drag the handler in."""
        init = os.path.join(MODULES_DIR, "__init__.py")
        with open(init, "r", encoding="utf-8") as handle:
            assert handle.read().strip() == "", "modules/__init__.py must stay empty"

    def test_loader_exports_exactly_one_function(self, vendor_loader) -> None:
        assert vendor_loader.__all__ == ["ensure_vendor_on_path"]
        assert callable(vendor_loader.ensure_vendor_on_path)

    def test_no_insert_call_anywhere_in_the_loader(self, vendor_loader) -> None:
        with open(
            os.path.join(MODULES_DIR, "vendor_loader.py"), "r", encoding="utf-8"
        ) as handle:
            source = handle.read()
        assert "sys.path.insert" not in source, (
            "the loader must never prepend; a pip-installed chatterbox has to win"
        )

    def test_repo_entry_point_no_longer_inserts_src(self) -> None:
        """S2.4 removed the ``sys.path.insert(0, src_dir)`` from ``__init__.py``."""
        with open(os.path.join(ROOT_DIR, "__init__.py"), "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "sys.path.insert" not in source, (
            "the entry point still prepends src/ to sys.path"
        )
        assert "ensure_vendor_on_path()" in source


# ---------------------------------------------------------------------------
# End to end, in a clean interpreter
# ---------------------------------------------------------------------------

PROBE = textwrap.dedent(
    """
    import json, sys

    sys.path.append({modules_dir!r})
    from vendor_loader import ensure_vendor_on_path

    before = list(sys.path)
    ensure_vendor_on_path()
    delta = [entry for entry in sys.path if entry not in before]

    result = {{
        'added': delta,
        'is_tail': sys.path[-1] == {src_dir!r},
        'index': sys.path.index({src_dir!r}),
        'count': sys.path.count({src_dir!r}),
    }}

    try:
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except BaseException as exc:
        result['import_kind'] = type(exc).__name__
        result['import_detail'] = str(exc)[:300]
        result['missing'] = getattr(exc, 'name', None)
    else:
        result['import_kind'] = 'ok'
        result['classes'] = [
            ChatterboxTTS.__name__, ChatterboxVC.__name__,
            ChatterboxMultilingualTTS.__name__, ChatterboxTurboTTS.__name__,
        ]
        result['perth_imported'] = 'perth' in sys.modules
        result['diffusers_imported'] = 'diffusers' in sys.modules

    print('__PROBE__' + json.dumps(result))
    """
)


def run_probe() -> dict:
    script = PROBE.format(modules_dir=MODULES_DIR, src_dir=SRC_DIR)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__PROBE__"):
            return json.loads(line[len("__PROBE__"):])
    pytest.fail(
        "the vendor-loader probe produced no result\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}\n"
        f"--- stderr ---\n{completed.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def probe() -> dict:
    return run_probe()


#: Third-party modules the vendored tree legitimately needs. If one of these is
#: absent the probe skips -- by name, so "why did this not run?" is answerable.
KNOWN_EXTERNAL_REQUIREMENTS = (
    "librosa",
    "torchaudio",
    "transformers",
    "s3tokenizer",
    "conformer",
    "huggingface_hub",
    "safetensors",
    "einops",
    "pyloudnorm",
)


class TestRealVendorResolvesAfterTheLoaderRuns:
    def test_loader_appends_exactly_one_entry_at_the_tail(self, probe: dict) -> None:
        assert probe["added"] == [SRC_DIR], (
            f"the loader added {probe['added']!r}; it must add src/ and nothing else"
        )
        assert probe["is_tail"] is True
        assert probe["count"] == 1

    def test_vendor_entry_points_import(self, probe: dict) -> None:
        if probe["import_kind"] != "ok":
            pytest.skip(
                "vendored chatterbox does not import in this environment: "
                f"{probe['import_kind']} on {probe.get('missing')!r} "
                f"({probe.get('import_detail')}); install "
                f"{', '.join(KNOWN_EXTERNAL_REQUIREMENTS)} to run this check"
            )
        assert set(probe["classes"]) == {
            "ChatterboxTTS",
            "ChatterboxVC",
            "ChatterboxMultilingualTTS",
            "ChatterboxTurboTTS",
        }

    def test_import_does_not_drag_perth_or_diffusers_in(self, probe: dict) -> None:
        if probe["import_kind"] != "ok":
            pytest.skip(
                "vendored chatterbox does not import in this environment "
                f"({probe['import_kind']}: {probe.get('missing')})"
            )
        assert probe["perth_imported"] is False, (
            "perth is an optional dependency; importing chatterbox must not need it"
        )
        assert probe["diffusers_imported"] is False, (
            "diffusers is no longer a dependency of the vendored tree"
        )

    def test_a_missing_dependency_is_never_reported_as_success(self, probe: dict) -> None:
        """A vacuous pass is the failure mode this guards against."""
        if probe["import_kind"] != "ok":
            assert probe["import_kind"] in {"ModuleNotFoundError", "ImportError"}, (
                f"import chatterbox failed on {probe['import_kind']}: "
                f"{probe.get('import_detail')}"
            )
            assert probe.get("missing"), "the failure names no missing module"