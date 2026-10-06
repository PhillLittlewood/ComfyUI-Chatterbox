"""S4.3 -- ``modules/settings.py``, the read side of ``comfy.settings.json``.

The module exists so the frontend's ``storeSetting("chatterbox.*", value)``
reaches the backend. What is asserted here is the read pattern: the file is
found, filtered to our prefix, re-read on ``reload()``, and every failure mode
degrades to defaults rather than raising into a node.
"""

from __future__ import annotations

import json
import os

import pytest

from ComfyUI_Chatterbox.modules import settings as settings_module
from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_MODEL_SUBDIR, DEFAULT_MODEL_PACK_NAME
from ComfyUI_Chatterbox.modules.settings import (
    SETTINGS_PREFIX,
    ChatterboxSettings,
    get_settings,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """The settings singleton is module state; every test starts from none."""
    settings_module._settings_instance = None
    yield
    settings_module._settings_instance = None


@pytest.fixture()
def user_dir(tmp_path, monkeypatch):
    """Point ``folder_paths.get_user_directory()`` at a temporary directory."""
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path))
    return tmp_path


def write_settings(user_dir, payload: dict) -> str:
    target = user_dir / "default" / "comfy.settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(target)


def make_pack(parent, name: str, with_weights: bool = True) -> str:
    pack = parent / name
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
    if with_weights:
        (pack / "t3_cfg.safetensors").write_bytes(b"weights")
    return str(pack)


class TestPrefix:
    def test_prefix_is_namespaced_to_chatterbox(self) -> None:
        assert SETTINGS_PREFIX == "chatterbox."

    def test_the_borrowed_voxcpm_prefix_is_gone(self) -> None:
        import inspect

        source = inspect.getsource(settings_module)
        assert "voxcpm." not in source.replace("``voxcpm.``", "")


class TestLoading:
    def test_reads_and_filters_the_chatterbox_keys(self, user_dir) -> None:
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": "/models/mine",
                "comfy.something_else": "ignored",
                "voxcpm.use_custom_path": "stale, from the other project",
            },
        )
        settings = ChatterboxSettings()
        assert settings.use_custom_path is True
        assert settings.custom_model_path == "/models/mine"
        assert "comfy.something_else" not in settings._settings
        assert "voxcpm.use_custom_path" not in settings._settings

    def test_missing_user_directory_degrades(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "gone"))
        settings = ChatterboxSettings()
        assert settings.use_custom_path is False
        assert settings.custom_model_path is None

    def test_missing_settings_file_degrades(self, user_dir) -> None:
        settings = ChatterboxSettings()
        assert settings._settings == {}

    def test_corrupt_json_degrades_without_raising(self, user_dir) -> None:
        target = user_dir / "default" / "comfy.settings.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{not json", encoding="utf-8")
        settings = ChatterboxSettings()
        assert settings._settings == {}

    def test_no_folder_paths_degrades_without_raising(self, monkeypatch) -> None:
        """The module must import and run with no ComfyUI at all."""
        import sys

        monkeypatch.setitem(sys.modules, "folder_paths", None)
        settings = ChatterboxSettings()
        assert settings.use_custom_path is False
        assert settings.get_effective_model_path().endswith(CHATTERBOX_MODEL_SUBDIR)


class TestReload:
    def test_reload_picks_up_a_new_write(self, user_dir) -> None:
        settings = ChatterboxSettings()
        assert settings.use_custom_path is False

        write_settings(user_dir, {"chatterbox.use_custom_path": True})
        settings.reload()

        assert settings.use_custom_path is True

    def test_reload_picks_up_a_changed_value(self, user_dir) -> None:
        write_settings(user_dir, {"chatterbox.custom_model_path": "/first"})
        settings = ChatterboxSettings()
        assert settings.custom_model_path == "/first"

        write_settings(user_dir, {"chatterbox.custom_model_path": "/second"})
        settings.reload()
        assert settings.custom_model_path == "/second"


class TestSingleton:
    def test_get_settings_is_a_singleton(self, user_dir) -> None:
        assert get_settings() is get_settings()

    def test_the_singleton_reads_the_file(self, user_dir) -> None:
        write_settings(user_dir, {"chatterbox.use_custom_path": True})
        assert get_settings().use_custom_path is True


class TestEffectivePath:
    def test_custom_path_wins_when_enabled(self, user_dir, tmp_path) -> None:
        custom = tmp_path / "custom"
        custom.mkdir()
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": str(custom),
            },
        )
        assert get_settings().get_effective_model_path() == str(custom)

    def test_custom_path_is_ignored_when_disabled(self, user_dir, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(
            folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "models" / "tts")]
        )
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": False,
                "chatterbox.custom_model_path": str(tmp_path / "elsewhere"),
            },
        )
        effective = get_settings().get_effective_model_path()
        assert "elsewhere" not in effective
        assert effective == os.path.join(str(tmp_path / "models" / "tts"), "chatterbox")

    def test_defaults_to_the_models_root(self, user_dir, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(
            folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "models" / "tts")]
        )
        expected = os.path.join(str(tmp_path / "models" / "tts"), "chatterbox")
        assert get_settings().get_effective_model_path() == expected

    def test_falls_back_to_models_dir_without_a_tts_folder(self, user_dir, tmp_path, monkeypatch) -> None:
        import folder_paths

        def boom(name):
            raise KeyError(name)

        monkeypatch.setattr(folder_paths, "get_folder_paths", boom)
        monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
        expected = os.path.join(str(tmp_path / "models"), CHATTERBOX_MODEL_SUBDIR)
        assert get_settings().get_effective_model_path() == expected


class TestValidation:
    def test_a_directory_holding_a_pack_is_valid(self, user_dir, tmp_path) -> None:
        root = tmp_path / "root"
        make_pack(root, "resembleai_default_voice")
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": str(root),
            },
        )
        assert get_settings().is_effective_path_valid() is True

    def test_an_empty_directory_is_invalid(self, user_dir, tmp_path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": str(root),
            },
        )
        assert get_settings().is_effective_path_valid() is False

    def test_a_missing_directory_is_invalid(self, user_dir, tmp_path) -> None:
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": str(tmp_path / "nope"),
            },
        )
        assert get_settings().is_effective_path_valid() is False

    def test_pointing_straight_at_a_pack_is_valid(self, user_dir, tmp_path) -> None:
        pack = make_pack(tmp_path, "resembleai_default_voice")
        write_settings(
            user_dir,
            {"chatterbox.use_custom_path": True, "chatterbox.custom_model_path": pack},
        )
        assert get_settings().is_effective_path_valid() is True

    def test_a_pack_without_weights_is_not_a_valid_effective_path(self, user_dir, tmp_path) -> None:
        pack = make_pack(tmp_path, "half_downloaded", with_weights=False)
        write_settings(
            user_dir,
            {"chatterbox.use_custom_path": True, "chatterbox.custom_model_path": pack},
        )
        # A tokenizer alone is enough to be worth keeping, but a directory with
        # nothing at all is not.
        empty = tmp_path / "nothing"
        empty.mkdir()
        write_settings(
            user_dir,
            {"chatterbox.use_custom_path": True, "chatterbox.custom_model_path": str(empty)},
        )
        assert get_settings().is_effective_path_valid() is False


class TestToDict:
    def test_carries_the_frontend_shape(self, user_dir, tmp_path) -> None:
        root = tmp_path / "root"
        make_pack(root, "pack")
        write_settings(
            user_dir,
            {
                "chatterbox.use_custom_path": True,
                "chatterbox.custom_model_path": str(root),
            },
        )
        payload = get_settings().to_dict()
        assert payload["use_custom_path"] is True
        assert payload["custom_model_path"] == str(root)
        assert payload["effective_path"] == str(root)
        assert payload["effective_path_valid"] is True

    def test_names_the_default_pack(self, user_dir) -> None:
        assert get_settings().to_dict()["default_model_pack_name"] == DEFAULT_MODEL_PACK_NAME

    def test_effective_path_valid_is_always_a_bool(self, user_dir) -> None:
        assert isinstance(get_settings().to_dict()["effective_path_valid"], bool)


class TestModuleStandalone:
    def test_no_comfy_api_import(self) -> None:
        with open(settings_module.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()

    def test_folder_paths_is_only_imported_lazily(self) -> None:
        """A module-scope ``import folder_paths`` would break standalone use,
        so the only import must live inside a function body."""
        import ast

        with open(settings_module.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name != "folder_paths" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "folder_paths"