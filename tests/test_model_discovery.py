"""S4.7 -- ``modules/model_discovery.py``.

This module is a cache. The node schemas rebuild their pack combo on every
frontend refresh, and the naive implementation ``listdir``s the models root
each time. So the behaviour under test is mostly about *not* touching the disk:
two calls against an unchanged tree must cost exactly one ``os.scandir``, and
touching the tree must cost a second one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ComfyUI_Chatterbox.modules import model_discovery
from ComfyUI_Chatterbox.modules.constants import (
    CHATTERBOX_MODEL_SUBDIR,
    DEFAULT_MODEL_PACK_NAME,
)
from ComfyUI_Chatterbox.modules.model_discovery import (
    get_chatterbox_model_pack_names,
    get_model_pack_entries,
    get_model_pack_path,
    get_models_root,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    """A models root at ``<tmp>/models/tts/chatterbox``, wired into folder_paths."""
    import folder_paths

    root = tmp_path / "models" / "tts" / "chatterbox"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "models" / "tts")]
    )
    return root


def touch_root(root: Path) -> None:
    """Change the root's mtime the way adding a pack would.

    A real write bumps ``st_mtime_ns`` by well over a millisecond; sleeping
    keeps this honest on filesystems with coarse timestamps.
    """
    import time

    time.sleep(0.01)
    (root / "touched.txt").write_text("x", encoding="utf-8")


class TestModelsRoot:
    def test_resolves_under_the_tts_folder(self, models_root) -> None:
        assert get_models_root() == str(models_root)

    def test_falls_back_to_models_dir(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        def unregistered(name):
            raise KeyError(name)

        monkeypatch.setattr(folder_paths, "get_folder_paths", unregistered)
        monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
        assert get_models_root() == os.path.join(str(tmp_path / "models"), CHATTERBOX_MODEL_SUBDIR)

    def test_falls_back_to_a_relative_path_without_comfy(self, monkeypatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "folder_paths", None)
        assert get_models_root() == os.path.join("models", CHATTERBOX_MODEL_SUBDIR)


class TestPackNames:
    def test_lists_every_subdirectory(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "beta").mkdir()
        assert get_chatterbox_model_pack_names() == ["alpha", "beta"]

    def test_ignores_loose_files(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "stray.txt").write_text("x", encoding="utf-8")
        assert get_chatterbox_model_pack_names() == ["alpha"]

    def test_a_partially_downloaded_pack_is_still_offered(self, models_root) -> None:
        """The pack must stay selectable, or the download can never finish it."""
        half = models_root / "half"
        half.mkdir()
        (half / "ve.safetensors.incomplete").write_bytes(b"x")
        assert get_chatterbox_model_pack_names() == ["half"]

    def test_the_default_pack_is_hoisted_to_index_zero(self, models_root) -> None:
        for name in ("aaa", "zzz", DEFAULT_MODEL_PACK_NAME, "mmm"):
            (models_root / name).mkdir()
        names = get_chatterbox_model_pack_names()
        assert names[0] == DEFAULT_MODEL_PACK_NAME
        assert sorted(names) == sorted(["aaa", "zzz", DEFAULT_MODEL_PACK_NAME, "mmm"])

    def test_an_empty_root_still_offers_the_default(self, models_root) -> None:
        assert get_chatterbox_model_pack_names() == [DEFAULT_MODEL_PACK_NAME]

    def test_a_missing_root_still_offers_the_default(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        assert get_chatterbox_model_pack_names() == [DEFAULT_MODEL_PACK_NAME]

    def test_the_names_are_the_frozen_default(self) -> None:
        assert DEFAULT_MODEL_PACK_NAME == "resembleai_default_voice"


class TestPackEntries:
    def test_entries_pair_names_with_paths(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        entries = get_model_pack_entries()
        assert entries == [("alpha", str(models_root / "alpha"))]

    def test_the_returned_list_is_a_copy(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        first = get_model_pack_entries()
        first.append(("bogus", "/nowhere"))
        assert get_model_pack_entries() == [("alpha", str(models_root / "alpha"))]

    def test_a_missing_root_yields_nothing(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        assert get_model_pack_entries() == []


class TestPackPath:
    def test_resolves_a_name(self, models_root) -> None:
        assert get_model_pack_path("alpha") == str(models_root / "alpha")

    @pytest.mark.parametrize("blank", ["", None])
    def test_a_blank_name_returns_none(self, blank) -> None:
        assert get_model_pack_path(blank) is None

    def test_a_blank_name_does_not_resolve_to_the_models_root(self, models_root) -> None:
        """A saved workflow can carry an empty combo value; resolving it to the
        models root would load the wrong directory."""
        assert get_model_pack_path("") != str(models_root)


class TestCaching:
    """The whole point of the module."""

    @staticmethod
    def _count_scandir(monkeypatch) -> list[int]:
        calls = []
        real_scandir = os.scandir

        def counting_scandir(path=".", *args, **kwargs):
            calls.append(1)
            return real_scandir(path, *args, **kwargs)

        monkeypatch.setattr(model_discovery.os, "scandir", counting_scandir)
        return calls

    def test_two_calls_on_an_unchanged_tree_scan_once(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "beta").mkdir()
        calls = self._count_scandir(monkeypatch)

        first = get_chatterbox_model_pack_names()
        second = get_chatterbox_model_pack_names()

        assert first == second == ["alpha", "beta"]
        assert len(calls) == 1, (
            f"the models root was scanned {len(calls)} times for an unchanged tree; "
            "the cache is not working"
        )

    def test_touching_the_root_invalidates_the_cache(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        calls = self._count_scandir(monkeypatch)

        assert get_chatterbox_model_pack_names() == ["alpha"]
        assert len(calls) == 1

        (models_root / "beta").mkdir()
        touch_root(models_root)

        assert get_chatterbox_model_pack_names() == ["alpha", "beta"]
        assert len(calls) == 2

    def test_the_cache_key_is_the_root_mtime(self, models_root) -> None:
        first = model_discovery._root_key(str(models_root))
        touch_root(models_root)
        assert model_discovery._root_key(str(models_root)) != first

    def test_a_missing_root_is_cached_too(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        calls = self._count_scandir(monkeypatch)

        assert get_chatterbox_model_pack_names() == [DEFAULT_MODEL_PACK_NAME]
        assert get_chatterbox_model_pack_names() == [DEFAULT_MODEL_PACK_NAME]
        assert len(calls) == 1, (
            "a models root that does not exist must be cached too, otherwise "
            "every schema rebuild pays a failed directory open"
        )

    def test_invalidate_cache_forces_a_rescan(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        calls = self._count_scandir(monkeypatch)

        get_chatterbox_model_pack_names()
        invalidate_cache()
        get_chatterbox_model_pack_names()

        assert len(calls) == 2

    def test_changing_the_root_resolves_different_trees(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        first_root = tmp_path / "one" / "chatterbox"
        second_root = tmp_path / "two" / "chatterbox"
        first_root.mkdir(parents=True)
        second_root.mkdir(parents=True)
        (first_root / "from_one").mkdir()
        (second_root / "from_two").mkdir()

        roots = {"value": str(tmp_path / "one")}
        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [roots["value"]])
        assert get_chatterbox_model_pack_names() == ["from_one"]

        roots["value"] = str(tmp_path / "two")
        assert get_chatterbox_model_pack_names() == ["from_two"]


class TestFrozenConstants:
    """``modules/constants.py`` against the recorded snapshot.

    ``tests/test_move_parity.py`` checks the snapshot against a hardcoded copy
    of the values. This checks the snapshot against the module that actually
    ships, so the two cannot drift apart unnoticed.
    """

    @pytest.fixture(scope="class")
    def recorded(self) -> dict:
        from conftest import ROOT_DIR

        import json

        with open(
            os.path.join(ROOT_DIR, "tests", "frozen_widget_snapshot.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)["constants"]

    def test_the_module_subdir_matches_the_snapshot(self, recorded: dict) -> None:
        assert CHATTERBOX_MODEL_SUBDIR == os.path.join(*recorded["CHATTERBOX_MODEL_SUBDIR"]["parts"])
        assert recorded["CHATTERBOX_MODEL_SUBDIR"]["value"] == "tts/chatterbox"

    def test_the_repo_id_matches_the_snapshot(self, recorded: dict) -> None:
        from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_REPO_ID

        assert CHATTERBOX_REPO_ID == recorded["CHATTERBOX_REPO_ID"]["value"]
        assert CHATTERBOX_REPO_ID == "".join(recorded["CHATTERBOX_REPO_ID"]["parts"])

    def test_the_file_list_matches_the_snapshot(self, recorded: dict) -> None:
        from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_FILES_TO_DOWNLOAD

        assert CHATTERBOX_FILES_TO_DOWNLOAD == recorded["CHATTERBOX_FILES_TO_DOWNLOAD"]["value"]
        assert CHATTERBOX_FILES_TO_DOWNLOAD == list(
            recorded["CHATTERBOX_FILES_TO_DOWNLOAD"]["parts"]
        )

    def test_the_default_pack_name_matches_the_snapshot(self, recorded: dict) -> None:
        assert DEFAULT_MODEL_PACK_NAME == recorded["DEFAULT_MODEL_PACK_NAME"]["value"]
        assert DEFAULT_MODEL_PACK_NAME == "".join(recorded["DEFAULT_MODEL_PACK_NAME"]["parts"])


class TestModuleHygiene:
    def test_no_comfy_api_import(self) -> None:
        with open(model_discovery.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()

    def test_the_scan_uses_scandir_not_listdir(self) -> None:
        """``os.listdir`` would defeat the point of counting directory reads,
        and cannot distinguish a file from a subdirectory in one pass."""
        import ast

        with open(model_discovery.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "scandir" in called
        assert "listdir" not in called