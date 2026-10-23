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

import logging

import numpy as np
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
from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader
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
            {"chatterbox-nano": {"architecture": "chatterbox_nano"}},
        )
        assert gen._resolve_turbo_variant("chatterbox-nano") == "nano"

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

    def test_a_turbo_tag_reaches_the_model_verbatim(self) -> None:
        """The tag is the model's own vocabulary, so it is handed over as it
        was written.

        Upstream never parses the nine square-bracket tags: it injects them
        into the prompt and lets the tokenizer speak them
        (``src/chatterbox/tts_turbo.py:401-403``). A Turbo checkpoint only
        says ``[chuckle]`` if the text still carries it, so stripping it here
        made the feature silent while looking implemented.
        """
        model = ChatterboxTurboTTS()
        generate_audio(model, "Hello [chuckle] there")
        spoken = model.calls[0]["args"][0]
        assert "[chuckle]" in spoken
        assert "Hello" in spoken and "there" in spoken

    def test_a_turbo_model_receives_exaggeration_zero(self) -> None:
        """Turbo ignores exaggeration for sampling (``hp.emotion_adv`` is
        False on that family) and upstream warns about every non-zero
        value -- once per chunk."""
        model = ChatterboxTurboTTS()
        generate_audio(model, "Hello there", exaggeration=0.5)
        assert model.calls[0]["exaggeration"] == 0.0

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


class TestClassicPathTextSemantics:
    """The classic path goes through the text engine too.

    That is what makes chunking possible at all -- there is nowhere else to
    cut a long script -- and it is what makes ``[Alice]`` open a character
    segment instead of being read out loud as a name.
    """

    class _Plain:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, *args, **kwargs) -> torch.Tensor:
            self.calls.append({"args": list(args), **kwargs})
            return torch.zeros((1, 24000))

    def test_the_classic_path_consumes_a_character_tag(self) -> None:
        model = self._Plain()
        generate_audio(model, "[Alice] Hi.")
        assert model.calls[0]["args"][0] == "Hi."

    def test_the_classic_path_keeps_a_turbo_tag_name_as_text(self) -> None:
        """A Turbo tag on a pack with no Turbo vocabulary is *said*, not
        swallowed as a voice name."""
        model = self._Plain()
        generate_audio(model, "[laugh] Hi.")
        assert model.calls[0]["args"][0] == "[laugh] Hi."

    def test_the_classic_path_reports_a_notice(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=""):
            generate_audio(self._Plain(), "[laugh] Hi.")
        messages = [record.getMessage() for record in caplog.records]
        assert any("Turbo" in message for message in messages), (
            "a Turbo tag on a classic pack must be announced, not dropped: "
            f"{messages!r}"
        )

    def test_a_pause_tag_is_reported_rather_than_silently_dropped(
        self, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=""):
            generate_audio(self._Plain(), "Hi. [pause:1.5] There.")
        messages = [record.getMessage() for record in caplog.records]
        assert any("pause" in message.lower() for message in messages), (
            f"the parsed-but-unrendered pause was dropped on the floor: {messages!r}"
        )

    def test_a_clean_script_reports_nothing(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=""):
            generate_audio(self._Plain(), "Hello there.")
        messages = [record.getMessage() for record in caplog.records]
        assert messages == []


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

    def test_the_chunking_widgets_reach_generate_audio(self, monkeypatch) -> None:
        """The four long-script widgets are not decoration: the values the
        user picked have to arrive at the chunk loop."""
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
            model_pack_name="chatterbox",
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
            enable_chunking=False,
            max_chars_per_chunk=250,
            chunk_combination_method="crossfade",
            silence_between_chunks_ms=25,
        )
        assert seen["enable_chunking"] is False
        assert seen["max_chars_per_chunk"] == 250
        assert seen["chunk_combination_method"] == "crossfade"
        assert seen["silence_between_chunks_ms"] == 25

    def test_the_node_defaults_match_the_schema_defaults(self, monkeypatch) -> None:
        """A run that says nothing about chunking gets the schema's defaults."""
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
            model_pack_name="chatterbox",
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
        )
        assert seen["enable_chunking"] is True
        assert seen["max_chars_per_chunk"] == 400
        assert seen["chunk_combination_method"] == "auto"
        assert seen["silence_between_chunks_ms"] == 100


class TestSharedFamilyFolderRouting:
    """One folder, two members: the dropdown entry picks the weights."""

    @pytest.fixture()
    def family(self, tmp_path, monkeypatch):
        pack = tmp_path / "chatterbox-turbo"
        pack.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (pack / name).write_bytes(b"\x00" * 4)
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        return pack

    def test_a_nano_entry_beats_the_shared_contents(self, family, monkeypatch) -> None:
        monkeypatch.setattr(
            gen.model_info,
            "AVAILABLE_CHATTERBOX_MODELS",
            {"chatterbox-nano": {"architecture": "chatterbox_nano"}},
        )
        assert gen._resolve_turbo_variant("chatterbox-nano") == "nano"

    def test_a_turbo_entry_beats_the_shared_contents(self, family, monkeypatch) -> None:
        monkeypatch.setattr(
            gen.model_info,
            "AVAILABLE_CHATTERBOX_MODELS",
            {"chatterbox-turbo": {"architecture": "chatterbox_turbo"}},
        )
        assert gen._resolve_turbo_variant("chatterbox-turbo") == "turbo"

    def test_an_unregistered_shared_folder_still_answers_nano(self, family, monkeypatch) -> None:
        monkeypatch.setattr(gen.model_info, "AVAILABLE_CHATTERBOX_MODELS", {})
        assert gen._resolve_turbo_variant("chatterbox-turbo") == "nano"


# ---------------------------------------------------------------------------
# Flash routing (the vendored chatterbox-flash sampler)
# ---------------------------------------------------------------------------


class TestFlashDetection:
    @pytest.fixture()
    def flash_pack(self, tmp_path, monkeypatch):
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        for name in ("t3_flash.safetensors", "ve.safetensors", "tokenizer.json"):
            (pack / name).write_bytes(b"\x00" * 4)
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        return pack

    def test_a_flash_dir_is_detected_by_its_t3(self, flash_pack, monkeypatch) -> None:
        monkeypatch.setattr(gen.model_info, "AVAILABLE_CHATTERBOX_MODELS", {})
        assert gen._resolve_flash_pack("chatterbox-flash") is True

    def test_a_flash_registry_entry_answers_before_the_disk(
        self, flash_pack, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            gen.model_info,
            "AVAILABLE_CHATTERBOX_MODELS",
            {"chatterbox-flash": {"architecture": "chatterbox_flash"}},
        )
        assert gen._resolve_flash_pack("chatterbox-flash") is True

    def test_a_classic_pack_is_not_flash(self, tmp_path, monkeypatch) -> None:
        pack = tmp_path / "classic"
        pack.mkdir()
        (pack / "t3_cfg.safetensors").write_bytes(b"\x00" * 4)
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        monkeypatch.setattr(gen.model_info, "AVAILABLE_CHATTERBOX_MODELS", {})
        assert gen._resolve_flash_pack("classic") is False


class TestFlashRoutingAndDispatch:
    def test_load_chatterbox_model_routes_flash(self, tmp_path, monkeypatch) -> None:
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        (pack / "t3_flash.safetensors").write_bytes(b"\x00" * 4)
        monkeypatch.setattr(
            gen.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda name: str(pack)),
        )
        monkeypatch.setattr(gen.model_info, "AVAILABLE_CHATTERBOX_MODELS", {})
        seen: dict = {}

        def _fake_entry_point(model_name, **kwargs):
            seen["model_name"] = model_name
            seen.update(kwargs)
            return object(), object()

        monkeypatch.setattr(gen, "load_chatterbox_entry_point", _fake_entry_point)

        patcher, model, vc = load_chatterbox_model("chatterbox-flash")

        assert vc is None
        assert seen["entry_point"] == "flash"
        assert seen["variant"] == "flash"
        assert seen["family"] == gen.FAMILY_FLASH

    def test_a_flash_model_gets_the_block_diffusion_sampler(self) -> None:
        class ChatterboxFlashTTS:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.cond_calls: list[dict] = []

            def prepare_conditionals(self, path, exaggeration=0.5):
                self.cond_calls.append({"path": path, "exaggeration": exaggeration})

            def generate(self, *args, **kwargs) -> torch.Tensor:
                self.calls.append({"args": list(args), **kwargs})
                return torch.zeros((1, 24000))

        model = ChatterboxFlashTTS()
        waveform, rate = generate_audio(
            model,
            "hello flash",
            audio_prompt={"waveform": torch.zeros(1, 24000), "sample_rate": 24000},
            exaggeration=0.7,
            temperature=0.6,
            top_k=64,
            seed=3,
        )
        assert rate == 24000 and waveform.shape == (1, 24000)
        assert model.cond_calls[0]["exaggeration"] == 0.7
        # The reference clip reaches conditioning as an in-memory waveform
        # (P-039/P-044): the vendored loader takes a tensor, so there is no
        # temporary .wav any more -- just 24 000 mono float32 samples at
        # S3GEN_SAMPLE_RATE, the rate every conditioning path wants.
        prompt = model.cond_calls[0]["path"]
        assert isinstance(prompt, np.ndarray), (
            f"the reference clip was handed over as {type(prompt).__name__}, "
            "not an in-memory waveform; the temporary-file path is back"
        )
        assert prompt.dtype == np.float32
        assert prompt.shape == (24000,), (
            "the clip was resampled to the wrong rate -- one second at 24 kHz "
            f"is 24 000 samples, it came back {prompt.shape[0]}"
        )
        call = model.calls[0]
        assert call["exaggeration"] == 0.7
        assert call["temperature"] == 0.6
        assert "top_k" not in call and "cfg_weight" not in call
        assert "min_p" not in call and "max_new_tokens" not in call

    def test_flash_without_a_reference_clip_is_refused(self) -> None:
        class ChatterboxFlashTTS:
            pass

        with pytest.raises(ValueError, match="reference"):
            generate_audio(ChatterboxFlashTTS(), "hello flash")


class TestFlashCarriesItsSamplerControls:
    """num_steps / cfg_scale reach the flash generate; other families ignore them."""

    def test_flash_dispatch_forwards_num_steps_and_cfg_scale(self) -> None:
        class ChatterboxFlashTTS:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.cond_calls: list[dict] = []

            def prepare_conditionals(self, path, exaggeration=0.5) -> None:
                self.cond_calls.append({"path": path, "exaggeration": exaggeration})

            def generate(self, *args, **kwargs) -> torch.Tensor:
                self.calls.append({"args": list(args), **kwargs})
                return torch.zeros((1, 24000))

        model = ChatterboxFlashTTS()
        generate_audio(
            model,
            "hello flash",
            audio_prompt={"waveform": torch.zeros(1, 24000), "sample_rate": 24000},
            num_steps=4,
            cfg_scale=1.5,
        )
        assert model.calls[0]["num_steps"] == 4
        assert model.calls[0]["cfg_scale"] == 1.5


class TestSingleLanguagePackRouting:
    """The six Chatterbox-Multilingual finetunes load through the MTL path."""

    def _write_slp(self, tmp_path, t3_name="t3_zh_cmn.safetensors") -> str:
        pack = tmp_path / "chatterbox-multilingual-zh-cmn"
        pack.mkdir()
        for name in (
            t3_name,
            "s3gen_v3.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "ve.pt",
        ):
            (pack / name).write_bytes(b"\x00" * 4)
        return str(pack)

    def test_the_registered_t3_checkpoint_wins(self, tmp_path) -> None:
        pack = self._write_slp(tmp_path)
        resolved = ChatterboxLoader._resolve_mtl_t3_file(
            pack, "chatterbox-multilingual-zh-cmn", "t3_mtl23ls_v3.safetensors"
        )
        assert resolved == "t3_zh_cmn.safetensors"

    def test_a_community_slp_dir_resolves_its_single_language_t3(
        self, tmp_path, monkeypatch
    ) -> None:
        pack = self._write_slp(tmp_path, t3_name="t3_xx_yy.safetensors")
        monkeypatch.setitem(
            gen.model_info.MODEL_CONFIGS,
            "chatterbox-multilingual-zh-cmn",
            {"files": []},
        )
        resolved = ChatterboxLoader._resolve_mtl_t3_file(
            str(pack), "chatterbox-multilingual-zh-cmn", "t3_mtl23ls_v3.safetensors"
        )
        assert resolved == "t3_xx_yy.safetensors"

    def test_the_version_combo_wins_when_the_pack_carries_it(self, tmp_path) -> None:
        pack = tmp_path / "mtl-plain"
        pack.mkdir()
        for name in (
            "t3_mtl23ls_v3.safetensors",
            "ve.pt",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
        ):
            (pack / name).write_bytes(b"\x00" * 4)
        assert (
            ChatterboxLoader._resolve_mtl_t3_file(
                str(pack), "some-community-pack", "t3_mtl23ls_v3.safetensors"
            )
            == "t3_mtl23ls_v3.safetensors"
        )

    def test_the_s3gen_decoder_resolves_per_pack(self, tmp_path) -> None:
        pack = tmp_path / "slp"
        pack.mkdir()
        (pack / "s3gen_v3.pt").write_bytes(b"\x00" * 4)
        assert (
            ChatterboxLoader._resolve_mtl_s3gen_file(str(pack), "whatever")
            == "s3gen_v3.pt"
        )
        # No dir: the official SLP entries name the v3 decoder in their files.
        assert (
            ChatterboxLoader._resolve_mtl_s3gen_file(
                str(tmp_path / "missing"), "chatterbox-multilingual-zh-cmn"
            )
            == "s3gen_v3.safetensors"
        )
        assert (
            ChatterboxLoader._resolve_mtl_s3gen_file(
                str(tmp_path / "missing"), "unknown-local-pack"
            )
            == "s3gen_v3.safetensors"
        )

    def test_a_slp_dir_sniffs_as_multilingual(self, tmp_path) -> None:
        from ComfyUI_Chatterbox.modules.model_info import infer_pack_architecture

        assert (
            infer_pack_architecture(self._write_slp(tmp_path))
            == "chatterbox_multilingual"
        )


class TestMtlWithoutBuiltinVoice:
    def test_a_pack_without_conds_names_the_reference_clip(self) -> None:
        class MtlPack:
            conds = None

        with pytest.raises(ValueError, match="reference"):
            gen.generate_mtl_audio(MtlPack(), "hola", "es")
