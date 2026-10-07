"""The shared-weight store (2026-10-07 dedupe).

``ve.safetensors``, ``s3gen.safetensors``, ``s3gen_v3.safetensors`` and the
grapheme tokenizer are byte-identical across every repo that ships them, so
they are fetched at most once into the reserved ``_shared`` directory and
hardlinked into each pack. Downloads carry only the pack-unique files.
"""

from __future__ import annotations

import os

import pytest

from ComfyUI_Chatterbox.modules import model_discovery
from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader
from ComfyUI_Chatterbox.modules.model_info import (
    CHATTERBOX_WEIGHT_FILES,
    MTL_TOKENIZER_FILE,
    SHARED_ASSET_DIR_NAME,
    SHARED_ASSET_FILES,
)

SHARED_STORE = ("ve.safetensors", "s3gen.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE)


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    """A private Chatterbox models root per test."""
    root = tmp_path / "models" / "tts" / "chatterbox"
    monkeypatch.setattr(model_discovery, "get_models_root", lambda: str(root))
    return root


class TestTheScanSkipsTheSharedStore:
    def test_the_reserved_store_is_never_a_pack(self, models_root) -> None:
        store = models_root / SHARED_ASSET_DIR_NAME
        pack = models_root / "some-pack"
        store.mkdir(parents=True)
        pack.mkdir(parents=True)
        assert model_discovery._scan(str(models_root)) == [("some-pack", str(pack))]


class TestSharedAssetMaterialization:
    def test_a_shared_file_is_linked_from_the_store(self, models_root) -> None:
        store = models_root / SHARED_ASSET_DIR_NAME
        store.mkdir(parents=True)
        (store / "ve.safetensors").write_bytes(b"ve-weights")
        pack = models_root / "fresh-pack"
        pack.mkdir(parents=True)

        ChatterboxLoader._ensure_shared_assets(pack, ["ve.safetensors"])

        linked = pack / "ve.safetensors"
        assert linked.read_bytes() == b"ve-weights"
        assert os.stat(store / "ve.safetensors").st_nlink == 2

    def test_a_legacy_full_repo_folder_is_used_before_any_fetch(
        self, models_root, monkeypatch
    ) -> None:
        legacy = models_root / "chatterbox"
        legacy.mkdir(parents=True)
        (legacy / "s3gen_v3.safetensors").write_bytes(b"v3-decoder")
        pack = models_root / "slp"
        pack.mkdir(parents=True)

        fetched: list = []

        def refuse_fetch(**kwargs):
            fetched.append(kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse_fetch)

        ChatterboxLoader._ensure_shared_assets(pack, ["s3gen_v3.safetensors"])

        assert (pack / "s3gen_v3.safetensors").read_bytes() == b"v3-decoder"
        assert not fetched, "a byte-identical legacy copy must not be re-fetched"

    def test_a_missing_shared_file_is_fetched_into_the_store(
        self, models_root, monkeypatch
    ) -> None:
        pack = models_root / "slp"
        pack.mkdir(parents=True)

        def fake_fetch(*, repo_id, filename, local_dir):
            path = os.path.join(local_dir, filename)
            with open(path, "wb") as handle:
                handle.write(b"fresh")
            return path

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_fetch)

        ChatterboxLoader._ensure_shared_assets(pack, ["ve.safetensors"])

        assert (pack / "ve.safetensors").read_bytes() == b"fresh"
        assert (
            models_root / SHARED_ASSET_DIR_NAME / "ve.safetensors"
        ).read_bytes() == b"fresh"

    def test_a_pack_that_already_holds_the_file_is_left_alone(self, models_root) -> None:
        pack = models_root / "slp"
        pack.mkdir(parents=True)
        (pack / "ve.safetensors").write_bytes(b"pack-own")

        ChatterboxLoader._ensure_shared_assets(pack, ["ve.safetensors"])

        assert (pack / "ve.safetensors").read_bytes() == b"pack-own"


class TestFilteredDownloads:
    """The fetch carries pack-unique files only; shared ones never download."""

    def _run(self, monkeypatch, models_root, model_name, required, entry):
        from ComfyUI_Chatterbox.modules import model_info

        monkeypatch.setitem(model_info.MODEL_CONFIGS, model_name, entry)
        seen: dict = {}

        def fake_download(model_name, repo_id, local_dir, client_id=None, allow_patterns=None):
            seen["patterns"] = allow_patterns
            seen["repo_id"] = repo_id

        monkeypatch.setattr(ChatterboxLoader, "download_pack", fake_download)

        def fake_fetch(*, repo_id, filename, local_dir):
            path = os.path.join(local_dir, filename)
            with open(path, "wb") as handle:
                handle.write(b"shared")
            return path

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_fetch)

        ChatterboxLoader._ensure_entry_point_dir(
            str(models_root / "pack-dir"),
            entry["repo_id"],
            required,
            model_name,
        )
        return seen

    def test_an_slp_download_carries_only_its_unique_files(
        self, models_root, monkeypatch
    ) -> None:
        entry = {
            "repo_id": "ResembleAI/Chatterbox-Multilingual-zh-cmn",
            "files": [
                "t3_zh_cmn.safetensors",
                "s3gen_v3.safetensors",
                MTL_TOKENIZER_FILE,
                "conds.pt",
            ],
        }
        seen = self._run(
            monkeypatch,
            models_root,
            "chatterbox-multilingual-zh-cmn",
            ("grapheme_mtl_merged_expanded_v1.json", "ve.safetensors",
             "s3gen_v3.safetensors", "t3_zh_cmn.safetensors"),
            entry,
        )
        assert seen["patterns"] == ["t3_zh_cmn.safetensors", "conds.pt"]

    def test_flash_keeps_its_own_s3gen_decoder(self, models_root, monkeypatch) -> None:
        from ComfyUI_Chatterbox.modules import model_info

        entry = dict(model_info.MODEL_CONFIGS["chatterbox-flash"])
        seen = self._run(
            monkeypatch,
            models_root,
            "chatterbox-flash",
            ("ve.safetensors", "s3gen.safetensors", "t3_flash.safetensors",
             "tokenizer.json"),
            entry,
        )
        assert "s3gen.safetensors" in seen["patterns"]
        assert "ve.safetensors" not in seen["patterns"]

    def test_the_turbo_fetch_keeps_the_meanflow_decoder(
        self, models_root, monkeypatch
    ) -> None:
        from ComfyUI_Chatterbox.modules import model_info

        entry = dict(model_info.MODEL_CONFIGS["chatterbox-turbo"])
        seen = self._run(
            monkeypatch,
            models_root,
            "chatterbox-turbo",
            ("ve.safetensors", "s3gen.safetensors", "s3gen_meanflow.safetensors",
             "t3_turbo_v1.safetensors"),
            entry,
        )
        assert "s3gen_meanflow.safetensors" in seen["patterns"]
        assert "ve.safetensors" not in seen["patterns"]
        assert "s3gen.safetensors" not in seen["patterns"]


class TestTheSharedSetIsRealWeights:
    def test_every_shared_asset_is_a_recognised_weight(self) -> None:
        for name in SHARED_ASSET_FILES:
            assert name in CHATTERBOX_WEIGHT_FILES or name == MTL_TOKENIZER_FILE
