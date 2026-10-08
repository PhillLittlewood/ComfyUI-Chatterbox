"""Baseline environment smoke tests (plan S0.6).

These assert the environment the modernization plan assumes. They are the
canary for the ComfyUI environment drifting underneath the repository: a
ComfyUI upgrade or a dependency bump that breaks the vendored chatterbox should
show up here first, as one named failure, not as a dozen behavioural failures.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


class TestBaselineEnvironment:
    """The versions and packages the rest of the plan is written against."""

    def test_comfyui_version_at_least_0_37(self):
        """The node API this repo targets (INPUT_TYPES, IS_CHANGED) landed in 0.37."""
        from comfyui_version import __version__ as comfyui_version

        parts = [int(p) for p in re_version(comfyui_version)]
        assert tuple(parts[:2]) >= (0, 37), (
            f"ComfyUI {comfyui_version} is older than the 0.37 baseline"
        )

    def test_comfy_core_modules_import(self, conftest_module):
        """``comfy`` and ``folder_paths`` come from the resolved checkout."""
        comfy = importlib.import_module("comfy")
        folder_paths = importlib.import_module("folder_paths")
        # ``comfy`` is a namespace package in current ComfyUI, so it has no
        # __file__; what matters is that it resolved from the chosen root.
        assert comfy.__path__
        assert any(
            os.path.realpath(entry).startswith(
                os.path.realpath(conftest_module.COMFYUI_ROOT)
            )
            for entry in comfy.__path__
        )
        assert folder_paths.__file__ is not None
        assert folder_paths.models_dir is not None

    def test_comfyui_root_resolvable(self, conftest_module):
        """The resolved root exists and carries ComfyUI's marker modules."""
        root = conftest_module.COMFYUI_ROOT
        assert root == conftest_module._resolve_comfyui_root()
        assert os.path.isdir(root)
        assert os.path.isfile(os.path.join(root, "nodes.py"))
        assert os.path.isfile(os.path.join(root, "execution.py"))

    def test_huggingface_hub_at_least_1_0(self):
        """transformers 5.x requires the 1.x hub API (no more cached_download)."""
        import huggingface_hub

        version = getattr(huggingface_hub, "__version__", None)
        assert version is not None, "huggingface_hub does not expose __version__"
        parts = [int(p) for p in re_version(version)]
        assert tuple(parts[:2]) >= (1, 0), (
            f"huggingface_hub {version} is older than the 1.0 baseline"
        )

    @pytest.mark.parametrize("package", ["s3tokenizer"])
    def test_newly_installed_packages_import(self, package):
        """The one addition Phase 0 installs for the vendored package."""
        assert package not in sys.modules or sys.modules[package] is not None
        module = importlib.import_module(package)
        assert module.__file__ is not None or hasattr(module, "__path__")

    def test_diffusers_is_not_imported(self):
        """diffusers is installed but broken here; nothing may pull it in."""
        assert "diffusers" not in sys.modules


@pytest.fixture(scope="module")
def conftest_module():
    """The imported root ``conftest`` module (path-independent access)."""
    module = sys.modules.get("conftest")
    if module is None:  # pragma: no cover - pytest always imports it under this name
        raise RuntimeError("root conftest.py is not in sys.modules")
    return module


def re_version(version: str) -> list[str]:
    """Extract the leading dotted-numeric part of a version string."""
    import re

    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(version))
    assert match, f"unparsable version string: {version!r}"
    return match.group(1).split(".")