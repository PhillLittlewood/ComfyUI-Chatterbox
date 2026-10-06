"""Turbo-family routing through the main TTS node (2026-10-06 merge).

There is no Turbo node any more: the classic ``ChatterboxTTS`` node serves the
whole family. The pieces that make that safe:

* the pack -- not the folder name and not a widget -- decides the entry point,
  via the t3 checkpoint it carries;
* the variant-folded cache stem stays a *cache key* and never names a folder;
* ``generate_audio`` dispatches on the loaded model's class, so Turbo-family
  packs get the top-k sampler and the classic ones keep theirs.
"""

from __future__ import annotations

import pytest
import torch

from ComfyUI_Chatterbox.modules import generation as gen
from ComfyUI_Chatterbox.modules.generation import (
    FAMILY_TTS,
    FAMILY_TURBO,
    build_cache_key,
    generate_audio,
    load_chatterbox_model,
    variant_cache_stem,
)
from ComfyUI_Chatterbox.modules.model_registry import evict_if_changed


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class ChatterboxTurboTTS:
    """A stand-in matching the vendored pack's *class name* -- that is the
    whole dispatch key, deliberately spelled exactly like the vendor's."""

    def __init__(self, samples: int = 24000) -> None:
        self.samples = samples
        self.calls: list[dict] = []

    def generate(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append({"args": list(args), **kwargs})
        return torch.zeros((1, self.samples))


# ---------------------------------------------------------------------------
# Cache keys: two weight sets, two keys
# ---------------------------------------------------------------------------


class TestVariantStaysACacheKey:
    def test_the_variant_stem_differs(self) -> None:
        assert variant_cache_stem("chatterbox-turbo", "turbo") != variant_cache_stem(
            "chatterbox-turbo", "nano"
        )
        assert variant_cache_stem("chatterbox-turbo", "nano") == (
            "chatterbox-turbo__nano"
        )

    def test_the_cache_keys_differ(self) -> None:
        turbo = build_cache_key(
            variant_cache_stem("chatterbox-turbo", "turbo"),
            "sdpa", "fp32", torch.device("cpu"),
        )
        nano = build_cache_key(
            variant_cache_stem("chatterbox-turbo", "nano"),
            "sdpa", "fp32", torch.device("cpu"),
        )
        assert turbo != nano

    def test_multilingual_versions_also_differ(self) -> None:
        v2 = build_cache_key(
            variant_cache_stem("pack", "t3_mtl23ls_v2.safetensors"),
            "sdpa", "fp32", torch.device("cpu"),
        )
        v3 = build_cache_key(
            variant_cache_stem("pack", "t3_mtl23ls_v3.safetensors"),
            "sdpa", "fp32", torch.device("cpu"),
        )
        assert v2 != v3

    def test_an_empty_variant_leaves_the_name_alone(self) -> None:
        assert variant_cache_stem("pack", "") == "pack"


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------


class TestResolveTurboVariant:
    @pytest.fixture()
    def pack_root(self, tmp_path, monkeypatch):
        pack = tmp_path / "some-pack"
        pack.mkdir()
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        return pack

    def test_a_nano_pack_is_detected_by_its_weights(self, pack_root) -> None:
        (pack_root / "t3_nano_v1.safetensors").write_bytes(b"\x00" * 4)
        assert gen._resolve_turbo_variant("some-pack") == "nano"

    def test_a_turbo_pack_is_detected_by_its_weights(self, pack_root) -> None:
        (pack_root / "t3_turbo_v1.safetensors").write_bytes(b"\x00" * 4)
        assert gen._resolve_turbo_variant("some-pack") == "turbo"

    def test_a_classic_pack_is_not_turbo(self, pack_root) -> None:
        (pack_root / "t3_cfg.safetensors").write_bytes(b"\x00" * 4)
        assert gen._resolve_turbo_variant("some-pack") is None

    def test_a_missing_pack_falls_back_to_the_registry(
        self, pack_root, monkeypatch
    ) -> None:
        # Nothing on disk: the official entry says what the download would be.
        monkeypatch.setattr(
            gen.model_info,
            "AVAILABLE_CHATTERBOX_MODELS",
            {"chatterbox-turbo-nano": {"architecture": "chatterbox_nano"}},
        )
        assert gen._resolve_turbo_variant("chatterbox-turbo-nano") == "nano"

    def test_a_missing_unknown_pack_is_not_turbo(
        self, pack_root, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            gen.model_info, "AVAILABLE_CHATTERBOX_MODELS", {}
        )
        assert gen._resolve_turbo_variant("whoever") is None

    def test_a_blank_name_is_not_turbo(self, monkeypatch) -> None:
        def _raise(_name):
            raise ValueError("no pack selected")

        monkeypatch.setattr(
            gen.ChatterboxLoader, "resolve_pack_dir", staticmethod(_raise)
        )
        assert gen._resolve_turbo_variant("") is None


# ---------------------------------------------------------------------------
# load_chatterbox_model routes; the entry point never stems the folder
# ---------------------------------------------------------------------------


class TestLoadChatterboxModelRouting:
    @pytest.fixture()
    def routed(self, tmp_path, monkeypatch):
        pack = tmp_path / "nano-pack"
        pack.mkdir()
        (pack / "t3_nano_v1.safetensors").write_bytes(b"\x00" * 4)
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        seen: dict = {}

        def _fake_entry_point(model_name, **kwargs):
            seen["model_name"] = model_name
            seen.update(kwargs)
            return object(), object()

        monkeypatch.setattr(gen, "load_chatterbox_entry_point", _fake_entry_point)
        return seen

    def test_a_turbo_pack_routes_to_the_turbo_entry_point(self, routed) -> None:
        patcher, model, vc = load_chatterbox_model("nano-pack")
        assert vc is None
        assert model is not None
        assert routed["entry_point"] == "turbo"
        assert routed["variant"] == "nano"
        assert routed["family"] == FAMILY_TURBO

    def test_the_classic_family_is_not_consulted(self, routed, monkeypatch) -> None:
        evictions: list[tuple] = []
        monkeypatch.setattr(
            gen, "evict_if_changed",
            lambda family, key, caches: evictions.append(family) or [],
        )
        load_chatterbox_model("nano-pack")
        assert FAMILY_TTS not in evictions


class TestEntryPointResolvesTheSelectedFolder:
    """The stem folds the variant into the *cache key* only.

    Deriving the pack directory from the stem downloaded a Turbo repo into
    'chatterbox-nano__turbo' next to the folder the user actually picked.
    """

    def test_the_handler_gets_the_selected_pack_dir(self, tmp_path, monkeypatch) -> None:
        pack = tmp_path / "chosen-pack"
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
        monkeypatch.setattr(gen, "evict_if_changed", lambda *a, **k: [])

        handler_paths: list[str] = []

        class RecordingHandler:
            size = 0

            def __init__(self, _name, **kwargs):
                handler_paths.append(kwargs["model_path"])

        class RecordingPatcher:
            def __init__(self, handler, **kwargs):
                self.model = type("M", (), {"tts_model": object()})()
                self.is_loaded = False

        monkeypatch.setattr(gen, "ChatterboxModelHandler", RecordingHandler)
        monkeypatch.setattr(
            gen, "select_patcher_class",
            lambda *a, **k: RecordingPatcher,
        )
        monkeypatch.setattr(gen, "load_to_device", lambda patcher: None)
        monkeypatch.setattr(gen, "CHATTERBOX_PATCHER_CACHE", {})

        patcher, model = gen.load_chatterbox_entry_point(
            "chosen-pack", entry_point="turbo", variant="nano", family=FAMILY_TURBO
        )
        assert model is not None
        assert handler_paths == [str(pack)]


# ---------------------------------------------------------------------------
# generate_audio dispatch
# ---------------------------------------------------------------------------


class TestGenerateAudioDispatch:
    def test_a_turbo_model_gets_the_top_k_sampler(self) -> None:
        model = ChatterboxTurboTTS()
        generate_audio(
            model,
            "Hello [laugh] there",
            cfg_weight=0.5,
            min_p=0.05,
            top_k=77,
            norm_loudness=False,
            seed=5,
            max_new_tokens=999,
        )
        call = model.calls[0]
        assert call["top_k"] == 77
        assert call["norm_loudness"] is False
        assert call["temperature"] == 0.8
        assert call["top_p"] == 1.0
        assert call["repetition_penalty"] == 1.2
        # Turbo has no CFG: the classic-only knobs must not reach it.
        assert "cfg_weight" not in call
        assert "min_p" not in call
        assert "max_new_tokens" not in call
        assert "flow_cfg_scale" not in call

    def test_turbo_tags_are_recognised_before_the_model_call(self) -> None:
        """The parser consumes the nine square-bracket tags; the model never
        sees one. Recognition is the point, not pass-through."""
        model = ChatterboxTurboTTS()
        generate_audio(model, "Hello [laugh] there")
        spoken = model.calls[0]["args"][0]
        assert "[laugh]" not in spoken
        assert "Hello" in spoken and "there" in spoken

    def test_a_classic_model_keeps_its_own_sampler(self) -> None:
        class PlainChatterboxTTS:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def generate(self, *args, **kwargs) -> torch.Tensor:
                self.calls.append({"args": list(args), **kwargs})
                return torch.zeros((1, 24000))

        model = PlainChatterboxTTS()
        generate_audio(model, "hello", top_k=77, max_new_tokens=123)
        call = model.calls[0]
        assert call["max_new_tokens"] == 123
        assert call["cfg_weight"] == 0.5
        assert "top_k" not in call
        assert "norm_loudness" not in call


# ---------------------------------------------------------------------------
# The main node's schema carries the Turbo controls
# ---------------------------------------------------------------------------


class TestTTSNodeCarriesTheTurboControls:
    def test_top_k_defaults_to_the_documented_range(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode

        top_k = ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"]["top_k"]
        assert top_k[0] == "INT"
        assert top_k[1]["default"] == 1000
        assert top_k[1]["min"] == 1
        assert top_k[1]["max"] == 5000

    def test_normalize_loudness_defaults_on(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode

        norm = ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "normalize_loudness"
        ]
        assert norm[0] == "BOOLEAN"
        assert norm[1]["default"] is True

    def test_the_widget_values_reach_generate_audio(self, monkeypatch) -> None:
        from ComfyUI_Chatterbox.nodes import tts_node

        seen: dict = {}
        monkeypatch.setattr(
            tts_node,
            "load_chatterbox_model",
            lambda *a, **k: (object(), object(), object()),
        )

        def _fake_generate_audio(model, text, **kwargs):
            seen.update(kwargs)
            return torch.zeros((1, 24000)), 24000

        monkeypatch.setattr(tts_node, "generate_audio", _fake_generate_audio)
        tts_node.ChatterboxTTSNode.execute(
            model_pack_name="chatterbox-turbo",
            text="Hello.",
            max_new_tokens=1000,
            flow_cfg_scale=0.7,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            seed=1,
            use_watermark=False,
            top_k=64,
            normalize_loudness=False,
        )
        assert seen["top_k"] == 64
        assert seen["norm_loudness"] is False
