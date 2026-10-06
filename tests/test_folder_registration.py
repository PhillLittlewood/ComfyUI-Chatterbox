"""S5.1 -- ``modules/folder_registration.py``.

Registration is exercised against a **fake** ``folder_paths`` module. That is the
whole reason the production signature takes ``folder_paths_module`` as an
injected parameter: a real ``folder_paths`` cannot be reset between tests, and a
test against it would mutate the running ComfyUI's global
``folder_names_and_paths`` table.

The behaviours pinned here are the ones that were actually wrong before:

* registering only the *first* TTS root, which made a model installed under a
  secondary ``extra_model_paths.yaml`` root invisible;
* overwriting a ``tts`` entry another node had already registered;
* an unwritable models tree taking the node down at import.
"""

from __future__ import annotations

import logging
import os

import pytest

from modules.folder_registration import (
    CHATTERBOX_FOLDER_KEY,
    CHATTERBOX_SUBDIR,
    TTS_FOLDER_KEY,
    ensure_chatterbox_directories,
    ensure_directory,
    register_chatterbox_folders,
)


class FakeFolderPaths:
    """The slice of ComfyUI's ``folder_paths`` that registration touches."""

    def __init__(self, models_dir: str, supported_pt_extensions=(".safetensors", ".ckpt")):
        self.models_dir = models_dir
        self.supported_pt_extensions = list(supported_pt_extensions)
        self.folder_names_and_paths: dict = {}


@pytest.fixture
def fake(tmp_path) -> FakeFolderPaths:
    return FakeFolderPaths(str(tmp_path / "ComfyUI" / "models"))


class TestRegistersBothKeys:
    def test_tts_and_chatterbox_keys_exist(self, fake: FakeFolderPaths) -> None:
        roots = register_chatterbox_folders(fake)

        assert TTS_FOLDER_KEY in fake.folder_names_and_paths
        assert CHATTERBOX_FOLDER_KEY in fake.folder_names_and_paths
        assert roots, "register_chatterbox_folders returned no roots at all"

    def test_primary_tts_root_is_under_models_dir(self, fake: FakeFolderPaths) -> None:
        roots = register_chatterbox_folders(fake)
        expected = os.path.join(fake.models_dir, TTS_FOLDER_KEY)

        assert os.path.abspath(expected) in roots

    def test_chatterbox_dir_is_registered_under_the_tts_root(self, fake: FakeFolderPaths) -> None:
        roots = register_chatterbox_folders(fake)
        chatterbox_paths, _ = fake.folder_names_and_paths[CHATTERBOX_FOLDER_KEY]

        assert os.path.join(roots[0], CHATTERBOX_SUBDIR) in [
            os.path.abspath(p) for p in chatterbox_paths
        ]

    def test_tts_entry_keeps_comfyui_weight_extensions(self, fake: FakeFolderPaths) -> None:
        register_chatterbox_folders(fake)
        _, extensions = fake.folder_names_and_paths[TTS_FOLDER_KEY]

        assert ".safetensors" in extensions
        assert ".ckpt" in extensions
        assert ".json" in extensions, "config files must be discoverable too"


class TestEveryRootIsRegistered:
    """The ``extra_model_paths.yaml`` case: several TTS roots at once."""

    def test_all_existing_tts_roots_get_a_chatterbox_subdir(self, fake: FakeFolderPaths) -> None:
        secondary = os.path.join(os.path.dirname(fake.models_dir), "extra_models", "tts")
        other_node_paths = [secondary]
        fake.folder_names_and_paths[TTS_FOLDER_KEY] = (other_node_paths, {".safetensors"})

        roots = register_chatterbox_folders(fake)
        chatterbox_paths, _ = fake.folder_names_and_paths[CHATTERBOX_FOLDER_KEY]
        registered = {os.path.abspath(p) for p in chatterbox_paths}

        assert len(roots) == 2, f"expected both roots, got {roots}"
        for root in roots:
            assert os.path.abspath(os.path.join(root, CHATTERBOX_SUBDIR)) in registered

    def test_a_foreign_root_is_not_dropped_from_the_tts_key(self, fake: FakeFolderPaths) -> None:
        secondary = os.path.join(os.path.dirname(fake.models_dir), "extra_models", "tts")
        fake.folder_names_and_paths[TTS_FOLDER_KEY] = ([secondary], {".safetensors"})

        roots = register_chatterbox_folders(fake)

        assert os.path.abspath(secondary) in [os.path.abspath(r) for r in roots]


class TestIdempotent:
    def test_repeated_calls_add_no_duplicate_paths(self, fake: FakeFolderPaths) -> None:
        first = register_chatterbox_folders(fake)
        second = register_chatterbox_folders(fake)
        third = register_chatterbox_folders(fake)

        assert first == second == third
        assert len(set(second)) == len(second), "a TTS root was registered twice"

    def test_repeated_calls_add_no_duplicate_chatterbox_paths(self, fake: FakeFolderPaths) -> None:
        register_chatterbox_folders(fake)
        register_chatterbox_folders(fake)
        paths, _ = fake.folder_names_and_paths[CHATTERBOX_FOLDER_KEY]

        assert len(paths) == len(set(paths))

    def test_relative_and_absolute_spellings_deduplicate(self, fake: FakeFolderPaths) -> None:
        """ComfyUI hands out abspaths, but a caller may not."""
        register_chatterbox_folders(fake)
        chatterbox_paths, _ = fake.folder_names_and_paths[CHATTERBOX_FOLDER_KEY]
        before = len(chatterbox_paths)

        # Re-register the same roots spelled with redundant segments.
        for root in list(register_chatterbox_folders(fake)):
            messy = os.path.join(root, "..", os.path.basename(root))
            fake.folder_names_and_paths[TTS_FOLDER_KEY][0].append(messy)
        register_chatterbox_folders(fake)
        after, _ = fake.folder_names_and_paths[CHATTERBOX_FOLDER_KEY]

        assert len(after) == before


class TestNeverClobbersAnotherNode:
    def test_existing_paths_and_extensions_survive(self, fake: FakeFolderPaths) -> None:
        mine = os.path.join(os.path.dirname(fake.models_dir), "other_node", "tts")
        fake.folder_names_and_paths[TTS_FOLDER_KEY] = ([mine], {".ckpt"})

        register_chatterbox_folders(fake)
        paths, extensions = fake.folder_names_and_paths[TTS_FOLDER_KEY]

        assert os.path.abspath(mine) in [os.path.abspath(p) for p in paths]
        assert ".ckpt" in extensions, "another node's extension was dropped"
        assert ".safetensors" in extensions, "our own extension was not added"

    def test_the_registered_list_object_is_the_one_comfyui_holds(self, fake: FakeFolderPaths) -> None:
        """Rebinding a fresh list would detach ComfyUI's entry silently."""
        foreign_paths = []
        fake.folder_names_and_paths[TTS_FOLDER_KEY] = (foreign_paths, {".ckpt"})

        register_chatterbox_folders(fake)

        assert foreign_paths is fake.folder_names_and_paths[TTS_FOLDER_KEY][0]
        assert foreign_paths, "the list ComfyUI holds was replaced, not extended"


class TestDirectoryCreationIsBestEffort:
    def test_ensure_directory_creates_a_missing_tree(self, tmp_path) -> None:
        target = tmp_path / "a" / "b" / "chatterbox"

        assert ensure_directory(str(target)) is True
        assert target.is_dir()

    def test_ensure_directory_is_idempotent(self, tmp_path) -> None:
        target = tmp_path / "chatterbox"
        ensure_directory(str(target))

        assert ensure_directory(str(target)) is True

    def test_makedirs_failure_is_logged_not_raised(self, tmp_path, caplog) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        target = blocker / "chatterbox"

        with caplog.at_level(logging.WARNING):
            result = ensure_directory(str(target))

        assert result is False
        assert any("chatterbox" in record.getMessage() for record in caplog.records), (
            "a swallowed OSError must still be visible in the log"
        )

    def test_ensure_chatterbox_directories_reports_only_what_worked(
        self, tmp_path, caplog
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        good_root = tmp_path / "models" / "tts"
        good_root.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            created = ensure_chatterbox_directories([str(good_root), str(blocker)])

        assert created == [os.path.abspath(str(good_root / CHATTERBOX_SUBDIR))]
        assert (good_root / CHATTERBOX_SUBDIR).is_dir()
        assert caplog.records, "the failing root was not reported"

    def test_blank_roots_are_skipped(self) -> None:
        assert ensure_chatterbox_directories(["", None]) == []


class TestModuleSurface:
    def test_module_imports_nothing_from_comfy(self) -> None:
        """No ``comfy`` / ``folder_paths`` import anywhere in the source."""
        import modules.folder_registration as module

        source_path = module.__file__
        with open(source_path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert "import comfy" not in source
        assert "import folder_paths" not in source

    def test_public_names_are_exported(self) -> None:
        import modules.folder_registration as module

        for name in ("TTS_FOLDER_KEY", "CHATTERBOX_FOLDER_KEY", "register_chatterbox_folders"):
            assert name in module.__all__, f"{name} is part of the batch contract"

    def test_folder_key_values_are_the_shared_comfyui_keys(self) -> None:
        assert TTS_FOLDER_KEY == "tts"
        assert CHATTERBOX_FOLDER_KEY == "chatterbox"