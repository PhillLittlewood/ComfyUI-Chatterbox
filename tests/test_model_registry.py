"""S5.2 -- ``modules/model_registry.py``.

The behaviours pinned here are the ones that decide whether a machine survives
loading a second model:

* a cache key that does not notice a re-downloaded checkpoint;
* an entry left in ComfyUI's ``current_loaded_models`` after an "eviction", so
  the freed VRAM was never actually free;
* a wrapper reference the eviction forgot, so the model object stayed alive.

Nothing here touches the network and no real weights are loaded: the patchers
are recording doubles, and ``comfy.model_management`` is monkeypatched with a
stand-in whose ``current_loaded_models`` list is rebuilt per test.
"""

from __future__ import annotations

import ast
import os

import pytest

import comfy.model_management as model_management

from modules import model_registry
from modules.model_registry import (
    FAMILY_TTS,
    clear_active_keys,
    clear_bundle_registry,
    evict_if_changed,
    evict_patcher,
    get_active,
    get_live_bundle,
    identity_for_external,
    register_model_bundle,
    release_model_bundles,
    set_active,
    unregister_from_comfy,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class RecordingFinalizer:
    def __init__(self) -> None:
        self.detach_calls = 0

    def detach(self) -> None:
        self.detach_calls += 1


class LoadedEntry:
    """A stand-in for one entry of ``model_management.current_loaded_models``."""

    def __init__(self, patcher, finalizer=None):
        self.model = patcher
        self.model_finalizer = finalizer
        self.real_model = object()


class Wrapper:
    """The wrapped handler; ``tts_model`` / ``vc_model`` live here."""

    def __init__(self) -> None:
        self.tts_model = object()
        self.vc_model = object()
        self.processor = object()


class Patcher:
    """A patcher double recording the eviction calls made against it."""

    def __init__(self, key: str, wrapper=None) -> None:
        self.cache_key = key
        self.model = wrapper if wrapper is not None else Wrapper()
        self.unpatch_calls: list[tuple] = []

    def unpatch_model(self, **kwargs) -> None:
        self.unpatch_calls.append(kwargs)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Give every test an empty registry and a fresh ComfyUI loaded-models list."""
    clear_active_keys()
    clear_bundle_registry()
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    soft_empty = []
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: soft_empty.append(True))
    yield
    clear_active_keys()
    clear_bundle_registry()


def _stat_key(path: str, **kwargs) -> str:
    return identity_for_external(path, "s3gen", "sdpa", **kwargs)


# ---------------------------------------------------------------------------
# Cache-key identity
# ---------------------------------------------------------------------------


class TestIdentityKey:
    def test_key_carries_basename_mtime_and_size(self, tmp_path) -> None:
        weights = tmp_path / "t3_mtl23ls_v2.safetensors"
        weights.write_bytes(b"x" * 128)

        key = _stat_key(str(weights))
        stat_result = os.stat(weights)

        assert "t3_mtl23ls_v2.safetensors" in key
        assert str(stat_result.st_mtime_ns) in key
        assert str(stat_result.st_size) in key

    def test_touching_the_weight_file_changes_the_key(self, tmp_path) -> None:
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"x" * 128)
        before = _stat_key(str(weights))

        # Bump mtime explicitly: a same-second rewrite would otherwise keep
        # st_mtime_ns identical on filesystems with coarse timestamps.
        stat_result = os.stat(weights)
        os.utime(weights, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000))
        after = _stat_key(str(weights))

        assert before != after, "a rewritten checkpoint reused the stale cache entry"

    def test_rewriting_with_different_content_changes_the_key(self, tmp_path) -> None:
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"x" * 128)
        before = _stat_key(str(weights))

        weights.write_bytes(b"x" * 4096)
        os.utime(weights)
        after = _stat_key(str(weights))

        assert before != after

    def test_distinct_files_produce_distinct_keys(self, tmp_path) -> None:
        first = tmp_path / "a.safetensors"
        second = tmp_path / "b.safetensors"
        first.write_bytes(b"same")
        second.write_bytes(b"same")

        assert _stat_key(str(first)) != _stat_key(str(second))

    def test_key_defaults_to_the_chatterbox_prefix(self, tmp_path) -> None:
        weights = tmp_path / "m.safetensors"
        weights.write_bytes(b"x")

        assert identity_for_external(str(weights), "s3gen", "sdpa").startswith("chatterbox_")

    def test_every_input_is_part_of_the_key(self, tmp_path) -> None:
        weights = tmp_path / "m.safetensors"
        weights.write_bytes(b"x")
        base = _stat_key(str(weights))

        assert base != _stat_key(str(weights), use_llm_4bit=True)
        assert base != _stat_key(str(weights), dtype_str="bf16")
        assert base != identity_for_external(str(weights), "s3gen", "sage")
        assert base != identity_for_external(str(weights), "cosyvoice", "sdpa")
        assert base != identity_for_external(str(weights), "s3gen", "sdpa", prefix="vc")

    def test_missing_file_degrades_instead_of_raising(self, tmp_path) -> None:
        key = _stat_key(str(tmp_path / "not-there.safetensors"))

        assert "not-there.safetensors@0@0" in key

    def test_empty_path_does_not_raise(self) -> None:
        assert _stat_key("").startswith("chatterbox_")


# ---------------------------------------------------------------------------
# ComfyUI registry
# ---------------------------------------------------------------------------


class TestUnregisterFromComfy:
    def test_removes_only_the_matching_entry(self, monkeypatch) -> None:
        target, other = Patcher("a"), Patcher("b")
        survivor, doomed = LoadedEntry(other), LoadedEntry(target)
        # The registry list is mutated in place, so the expectations are captured
        # as objects, not as indices into the list that is about to shrink.
        loaded = [survivor, doomed]
        monkeypatch.setattr(model_management, "current_loaded_models", loaded)

        removed = unregister_from_comfy(target)

        assert removed == [doomed]
        assert model_management.current_loaded_models == [survivor]

    def test_detaches_finalizers_of_the_removed_entry(self, monkeypatch) -> None:
        target = Patcher("a")
        finalizer = RecordingFinalizer()
        entry = LoadedEntry(target, finalizer)
        monkeypatch.setattr(model_management, "current_loaded_models", [entry])

        unregister_from_comfy(target)

        assert finalizer.detach_calls == 1
        assert entry.model_finalizer is None
        assert entry.real_model is None

    def test_a_foreign_finalizer_is_left_alone(self, monkeypatch) -> None:
        target, other = Patcher("a"), Patcher("b")
        other_finalizer = RecordingFinalizer()
        doomed = LoadedEntry(target, RecordingFinalizer())
        survivor = LoadedEntry(other, other_finalizer)
        monkeypatch.setattr(model_management, "current_loaded_models", [doomed, survivor])

        unregister_from_comfy(target)

        assert other_finalizer.detach_calls == 0
        assert survivor.model_finalizer is other_finalizer
        assert model_management.current_loaded_models == [survivor]

    def test_no_match_returns_empty_and_keeps_the_list(self, monkeypatch) -> None:
        entries = [LoadedEntry(Patcher("a"))]
        monkeypatch.setattr(model_management, "current_loaded_models", entries)

        removed = unregister_from_comfy(Patcher("never-loaded"))

        assert removed == []
        assert model_management.current_loaded_models == entries

    def test_an_entry_that_cannot_be_interrogated_is_kept(self, monkeypatch) -> None:
        class Hostile:
            @property
            def model(self):
                raise RuntimeError("boom")

        hostile = Hostile()
        monkeypatch.setattr(model_management, "current_loaded_models", [hostile])

        assert unregister_from_comfy(Patcher("a")) == []
        assert model_management.current_loaded_models == [hostile]


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEvictPatcher:
    def test_nulls_tts_and_vc_models_on_the_wrapper(self) -> None:
        """The D3 regression: the patcher's own attribute was never the holder."""
        patcher = Patcher("a")

        evict_patcher(patcher, {"a": patcher}, "a")

        assert patcher.model.tts_model is None
        assert patcher.model.vc_model is None

    def test_calls_unpatch_model_destructively(self) -> None:
        patcher = Patcher("a")

        evict_patcher(patcher, {"a": patcher}, "a")

        assert patcher.unpatch_calls == [{"unpatch_weights": True, "destroy": True}]

    def test_removes_the_cache_entry(self) -> None:
        patcher = Patcher("a")
        cache = {"a": patcher, "b": Patcher("b")}

        evict_patcher(patcher, cache, "a")

        assert cache == {"b": cache["b"]}

    def test_releases_the_registered_bundle(self) -> None:
        patcher = Patcher("a")
        bundle = {"model": object(), "processor": object()}
        register_model_bundle("a", bundle)

        evict_patcher(patcher, {"a": patcher}, "a")

        assert bundle == {"model": None, "processor": None}
        assert get_live_bundle("a") is None

    def test_collects_errors_instead_of_aborting_midway(self) -> None:
        class Exploding(Patcher):
            def unpatch_model(self, **kwargs):
                raise RuntimeError("cannot destroy")

        patcher = Exploding("a")
        cache = {"a": patcher}

        errors = evict_patcher(patcher, cache, "a")

        assert any("unpatch_model" in err for err in errors)
        assert cache == {}, "the cache entry was not popped despite the failure"
        assert patcher.model.tts_model is None, "wrapper release was skipped by the failure"

    def test_a_patcher_without_a_wrapper_is_not_fatal(self) -> None:
        class Bare:
            cache_key = "a"

            def unpatch_model(self, **kwargs):
                return None

        assert evict_patcher(Bare(), None, "a") == []

    def test_returns_empty_list_on_success(self) -> None:
        patcher = Patcher("a")

        assert evict_patcher(patcher, {"a": patcher}, "a") == []


class TestEvictIfChanged:
    def test_no_op_for_an_unchanged_key(self) -> None:
        set_active(FAMILY_TTS, "k1")
        patcher = Patcher("k1")
        cache = {"k1": patcher}

        evicted = evict_if_changed(FAMILY_TTS, "k1", [cache])

        assert evicted == []
        assert cache == {"k1": patcher}
        assert patcher.unpatch_calls == [], "an unchanged key must not destroy the model"

    def test_changed_key_evicts_the_previous_one(self) -> None:
        old, new = Patcher("old"), Patcher("new")
        cache = {"old": old, "new": new}
        set_active(FAMILY_TTS, "old")

        evicted = evict_if_changed(FAMILY_TTS, "new", [cache])

        assert evicted == ["old"]
        assert cache == {"new": new}
        assert old.unpatch_calls == [{"unpatch_weights": True, "destroy": True}]
        assert new.unpatch_calls == []

    def test_every_other_key_in_every_cache_is_evicted(self) -> None:
        set_active(FAMILY_TTS, "gone")
        cache_a = {"keep": Patcher("keep"), "gone": Patcher("gone")}
        cache_b = {"other": Patcher("other")}

        evicted = evict_if_changed(FAMILY_TTS, "keep", [cache_a, cache_b])

        assert sorted(evicted) == ["gone", "other"]
        assert list(cache_a) == ["keep"]
        assert cache_b == {}

    def test_records_the_new_active_key(self) -> None:
        evict_if_changed(FAMILY_TTS, "k2", [])

        assert get_active(FAMILY_TTS) == "k2"

    def test_first_load_records_the_key_without_evicting(self) -> None:
        assert evict_if_changed(FAMILY_TTS, "k1", [{"other": Patcher("other")}]) == [
            "other"
        ]
        assert get_active(FAMILY_TTS) == "k1"

    def test_none_caches_are_skipped(self) -> None:
        assert evict_if_changed(FAMILY_TTS, "k1", [None, {}]) == []


# ---------------------------------------------------------------------------
# Bundle registry
# ---------------------------------------------------------------------------


class TestBundleRegistry:
    def test_register_and_fetch(self) -> None:
        bundle = {"model": object(), "processor": object()}
        register_model_bundle("k", bundle)

        assert get_live_bundle("k") is bundle

    def test_release_neutralizes_and_returns_the_count(self) -> None:
        bundle = {"model": object(), "processor": object()}
        register_model_bundle("k", bundle)

        assert release_model_bundles("k") == 1
        assert bundle["model"] is None and bundle["processor"] is None

    def test_release_of_an_unknown_key_is_zero(self) -> None:
        assert release_model_bundles("nope") == 0

    def test_a_neutralized_bundle_is_not_handed_back(self) -> None:
        register_model_bundle("k", {"model": None, "processor": object()})

        assert get_live_bundle("k") is None

    def test_reregistering_the_same_key_neutralizes_the_previous_bundle(self) -> None:
        first = {"model": object(), "processor": object()}
        second = {"model": object(), "processor": object()}
        register_model_bundle("k", first)
        register_model_bundle("k", second)

        assert first["model"] is None
        assert get_live_bundle("k") is second

    def test_clear_forgets_everything(self) -> None:
        register_model_bundle("k", {"model": object(), "processor": object()})
        clear_bundle_registry()

        assert get_live_bundle("k") is None

    def test_a_bundle_that_is_not_a_mapping_is_handled(self) -> None:
        register_model_bundle("k", object())

        assert get_live_bundle("k") is None


class TestActiveKeys:
    def test_round_trip(self) -> None:
        set_active(FAMILY_TTS, "k")

        assert get_active(FAMILY_TTS) == "k"

    def test_unknown_family_is_none(self) -> None:
        assert get_active("nothing-here") is None

    def test_family_constant_value(self) -> None:
        assert FAMILY_TTS == "tts"


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


class TestNoComfyImportsAtModuleScope:
    """The reason this module is importable without ComfyUI on ``sys.path``."""

    @staticmethod
    def _module_level_imports(path: str) -> set[str]:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        names: set[str] = set()
        for node in tree.body:  # top level only -- nested imports are the point
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_module_scope_imports_no_comfy_or_folder_paths(self) -> None:
        imported = self._module_level_imports(model_registry.__file__)

        assert "comfy" not in imported, "comfy.model_management must stay a lazy import"
        assert "comfy_api" not in imported
        assert "folder_paths" not in imported

    def test_comfy_is_still_reached_lazily(self) -> None:
        with open(model_registry.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert "import comfy.model_management" in source, (
            "the lazy import the AST audit permits must actually be there"
        )