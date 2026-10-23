"""The pack alias: the old spelling still resolves, the dropdown shows the new one.

Upstream renamed a pack. A saved workflow carries the name it was saved with,
so the old spelling has to keep working -- but the dropdown may only offer the
current name, or the user is handed a spelling nothing else in the repository
uses.

That means the alias has to be resolved at **every** seam a pack name reaches:
the loader's path resolution, the registry entry, the Turbo-variant answer,
the API's presence check and the nodes' queue-time validation. A seam that
forgets the alias turns a renamed pack into a broken one exactly where the
user notices: a workflow that saved fine yesterday and refuses to queue today.

The old spelling is written out here on purpose: this file and the alias
table itself are the only two places allowed to name it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ComfyUI_Chatterbox.modules import api_routes, model_info
from ComfyUI_Chatterbox.modules import generation as gen
from ComfyUI_Chatterbox.modules.folder_registration import TTS_FOLDER_KEY
from ComfyUI_Chatterbox.modules.generation import FAMILY_TURBO, _resolve_turbo_variant
from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader, _registry_size_estimate
from ComfyUI_Chatterbox.modules.model_discovery import (
    get_chatterbox_model_pack_names,
    invalidate_cache,
)
from ComfyUI_Chatterbox.modules.model_info import (
    AVAILABLE_CHATTERBOX_MODELS,
    MODEL_CONFIGS,
    canonical_pack_name,
)

#: The spelling upstream used before the rename.
OLD_NAME = "chatterbox-turbo-nano"
#: The spelling upstream uses now.
NEW_NAME = "chatterbox-nano"


# ── Doubles ──────────────────────────────────────────────────────────


class FakeFolderPaths:
    """``folder_paths`` stand-in with the one attribute the presence check uses."""

    def __init__(self, tts_roots=()) -> None:
        self._roots = [str(root) for root in tts_roots]

    def get_folder_paths(self, key: str) -> list[str]:
        if key != TTS_FOLDER_KEY:
            raise ValueError(f"Folder {key!r} is not registered with ComfyUI")
        return list(self._roots)


# ── Fixtures ─────────────────────────────────────────────────────────


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


@pytest.fixture()
def registry(monkeypatch):
    """The official registry, with no local entries another test left behind.

    ``AVAILABLE_CHATTERBOX_MODELS`` is a module-level dict that the local scan
    mutates in place, so a test that reads it cannot assume it is pristine.
    The loader holds its own reference to it (imported by value), hence the
    two patch targets -- and the *same* dict for both, so a scan that registers
    a local pack is visible to the loader as well.
    """
    official = {
        name: {"type": "official", **config}
        for name, config in MODEL_CONFIGS.items()
    }
    monkeypatch.setattr(model_info, "AVAILABLE_CHATTERBOX_MODELS", official)
    monkeypatch.setattr(
        "ComfyUI_Chatterbox.modules.loader.AVAILABLE_CHATTERBOX_MODELS", official
    )
    return official


def make_pack(directory, files):
    """A pack directory holding every file the registry declares."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).write_bytes(b"")
    return directory


# ── The alias table ──────────────────────────────────────────────────


class TestTheAliasTable:
    def test_the_old_name_maps_to_the_current_one(self) -> None:
        assert canonical_pack_name(OLD_NAME) == NEW_NAME

    def test_the_current_name_is_left_alone(self) -> None:
        assert canonical_pack_name(NEW_NAME) == NEW_NAME

    def test_an_unknown_name_is_left_alone(self) -> None:
        assert canonical_pack_name("whoever") == "whoever"

    def test_the_alias_is_not_a_registry_table_entry(self) -> None:
        assert OLD_NAME not in MODEL_CONFIGS

    def test_the_alias_is_not_an_official_registry_entry(self) -> None:
        official = [
            name
            for name, entry in AVAILABLE_CHATTERBOX_MODELS.items()
            if entry.get("type") == "official"
        ]
        assert OLD_NAME not in official


# ── The dropdown ─────────────────────────────────────────────────────


class TestTheDropdown:
    """The alias is accepted everywhere and offered nowhere."""

    def test_the_current_name_is_offered(self, models_root) -> None:
        assert NEW_NAME in get_chatterbox_model_pack_names()

    def test_the_old_name_is_not_offered(self, models_root) -> None:
        assert OLD_NAME not in get_chatterbox_model_pack_names()

    def test_the_shared_folder_is_offered_under_the_current_name(
        self, models_root
    ) -> None:
        """One folder, two members -- and the Nano member carries the name
        upstream gives it, not the one it used to."""
        family = models_root / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)

        names = get_chatterbox_model_pack_names()

        assert NEW_NAME in names
        assert OLD_NAME not in names


# ── Seam: the loader path ────────────────────────────────────────────


class TestTheLoaderPath:
    def test_the_alias_resolves_to_the_same_directory(
        self, models_root, registry
    ) -> None:
        assert ChatterboxLoader.resolve_pack_dir(OLD_NAME) == (
            ChatterboxLoader.resolve_pack_dir(NEW_NAME)
        )

    def test_the_alias_resolves_to_the_shared_turbo_folder(
        self, models_root, registry
    ) -> None:
        """The Nano member downloads into the shared ``chatterbox-turbo``
        folder -- the alias must not send the download anywhere else."""
        assert ChatterboxLoader.resolve_pack_dir(OLD_NAME) == str(
            models_root / "chatterbox-turbo"
        )

    def test_the_alias_follows_a_local_pack_of_the_current_name(
        self, models_root, registry
    ) -> None:
        """A pack extracted into a folder of its own is found under either
        name -- the alias names the pack, not a directory."""
        local = make_pack(
            models_root / NEW_NAME,
            ("tokenizer.json", "t3_nano_v1.safetensors"),
        )
        model_info.refresh_local_models()

        assert ChatterboxLoader.resolve_pack_dir(OLD_NAME) == str(local)

    def test_a_blank_name_is_still_refused(self, models_root, registry) -> None:
        with pytest.raises(ValueError, match="No Chatterbox model"):
            ChatterboxLoader.resolve_pack_dir("")


# ── Seam: the registry entry ─────────────────────────────────────────


class TestTheRegistryEntry:
    def test_the_alias_answers_with_the_nano_entry(self, models_root, registry) -> None:
        entry = ChatterboxLoader._registry_entry(OLD_NAME)

        assert entry is not None
        assert entry["architecture"] == "chatterbox_nano"
        assert entry["repo_id"] == "ResembleAI/chatterbox-nano"
        assert entry["folder"] == "chatterbox-turbo"

    def test_the_alias_is_measured_against_the_canonical_entry(
        self, models_root, registry
    ) -> None:
        assert _registry_size_estimate(OLD_NAME) == _registry_size_estimate(NEW_NAME)
        assert _registry_size_estimate(OLD_NAME) > 0


# ── Seam: the Turbo variant ──────────────────────────────────────────


class TestTheTurboVariant:
    def test_the_alias_answers_nano(self, models_root, registry) -> None:
        assert _resolve_turbo_variant(OLD_NAME) == "nano"

    def test_the_alias_answers_nano_with_only_turbo_weights_on_disk(
        self, models_root, registry
    ) -> None:
        """The registry knows which member the name means, so the answer
        never degrades to what the folder happens to hold."""
        family = models_root / "chatterbox-turbo"
        family.mkdir()
        (family / "t3_turbo_v1.safetensors").write_bytes(b"\x00" * 4)

        assert _resolve_turbo_variant(OLD_NAME) == "nano"

    def test_the_current_name_still_answers_nano(self, models_root, registry) -> None:
        assert _resolve_turbo_variant(NEW_NAME) == "nano"


# ── Seam: the API presence check ─────────────────────────────────────


class TestTheApiPresenceCheck:
    def test_the_alias_asks_about_the_same_directory(
        self, models_root, registry
    ) -> None:
        config = MODEL_CONFIGS[NEW_NAME]
        folder_paths = FakeFolderPaths([str(models_root.parent)])

        assert api_routes._is_official_pack_present(
            OLD_NAME, config, folder_paths
        ) == api_routes._is_official_pack_present(NEW_NAME, config, folder_paths)

    def test_the_alias_finds_the_shared_folder(self, models_root, registry) -> None:
        config = MODEL_CONFIGS[NEW_NAME]
        make_pack(models_root / "chatterbox-turbo", config["files"])
        folder_paths = FakeFolderPaths([str(models_root.parent)])

        assert api_routes._is_official_pack_present(
            OLD_NAME, config, folder_paths
        ) is True

    def test_the_alias_does_not_look_for_a_folder_of_its_own(
        self, models_root, registry
    ) -> None:
        """The alias names the canonical pack, so it is present exactly when
        that pack is -- not when a folder happens to carry the old name."""
        config = MODEL_CONFIGS[NEW_NAME]
        make_pack(models_root / OLD_NAME, config["files"])
        folder_paths = FakeFolderPaths([str(models_root.parent)])

        assert api_routes._is_official_pack_present(
            OLD_NAME, config, folder_paths
        ) is False


# ── Seam: the cache identity ─────────────────────────────────────────


class TestTheCacheIdentity:
    """One pack, one identity.

    The cache key is the pack's process-wide identity, and it is built from
    the pack's name. Two spellings of one pack must not be two identities --
    the second load would re-read the same weights from disk for nothing.
    """

    @pytest.fixture()
    def entry_point(self, tmp_path, monkeypatch):
        """An entry point that records the cache it was given."""
        pack = tmp_path / "chatterbox-turbo"
        pack.mkdir()
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        monkeypatch.setattr(gen, "resolve_attention_mode", lambda mode, _f: mode)
        monkeypatch.setattr(
            gen, "get_torch_device", lambda _d: torch.device("cpu")
        )
        monkeypatch.setattr(gen, "get_offload_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(
            gen, "resolve_dtype", lambda _dtype, _dev: torch.float32
        )
        # A no-op, so a second *identity* would show up as a second cache
        # entry instead of being evicted before the assertion runs.
        monkeypatch.setattr(gen, "evict_if_changed", lambda *a, **k: [])

        names: list[str] = []

        class RecordingHandler:
            size = 0

            def __init__(self, name, **kwargs):
                self.model_pack_name = name
                names.append(name)

        class RecordingPatcher:
            def __init__(self, handler, **kwargs):
                self.model = type("M", (), {"tts_model": object()})()
                self.is_loaded = False

        monkeypatch.setattr(gen, "ChatterboxModelHandler", RecordingHandler)
        monkeypatch.setattr(
            gen, "select_patcher_class", lambda *a, **k: RecordingPatcher
        )
        monkeypatch.setattr(gen, "load_to_device", lambda patcher: None)
        cache: dict = {}
        monkeypatch.setattr(gen, "CHATTERBOX_PATCHER_CACHE", cache)
        return SimpleNamespace(cache=cache, names=names)

    def _load(self, name: str) -> None:
        gen.load_chatterbox_entry_point(
            name, entry_point="turbo", variant="nano", family=FAMILY_TURBO
        )

    def test_the_alias_and_the_current_name_share_one_cache_entry(
        self, entry_point
    ) -> None:
        self._load(OLD_NAME)
        self._load(NEW_NAME)

        assert len(entry_point.cache) == 1

    def test_the_loaded_pack_is_named_after_the_canonical_pack(
        self, entry_point
    ) -> None:
        self._load(OLD_NAME)

        assert entry_point.names == [NEW_NAME]


# ── Seam: the queue-time validation ──────────────────────────────────


class TestTheQueueTimeValidation:
    def test_the_alias_passes_the_shared_check(self, models_root) -> None:
        from ComfyUI_Chatterbox.nodes.common import validate_common_inputs

        assert validate_common_inputs(OLD_NAME) is None

    def test_the_alias_passes_the_node_gate(self, models_root) -> None:
        from ComfyUI_Chatterbox.nodes.tts_node import ChatterboxTTSNode

        assert ChatterboxTTSNode.validate_inputs(model_pack_name=OLD_NAME) is True

    def test_an_unknown_pack_is_still_refused(self, models_root) -> None:
        from ComfyUI_Chatterbox.nodes.common import validate_common_inputs

        message = validate_common_inputs("no_such_pack")

        assert message is not None
        assert "no_such_pack" in message
