"""S5.7 -- ``modules/generation.py``.

Three properties are load-bearing, and none of them is visible by reading the
code:

* **Ordering.** :func:`load_chatterbox_model` must resolve the attention mode,
  then compute the cache key, then evict the *previous* pack, and only then
  allocate. :class:`TestCallOrder` records the order with a spy rather than
  trusting inspection -- an eviction that happens after the first allocation is
  a CUDA OOM on a machine that can hold either pack but not both.
* **``client_id`` is not part of the cache key.** It names the websocket, not
  the model. :class:`TestClientId` proves a second browser tab reuses the
  loaded pack *and* gets its own id -- VibeVoice's ``generation.py:78`` stomped
  the first client's id onto every later caller.
* **Interrupts cancel.** ``InterruptProcessingException`` derives from
  ``BaseException``. :class:`TestInterruptHandling` proves it propagates rather
  than degrading into one second of silence, and -- with an interrupt that
  *also* derives from ``Exception`` -- that the ``except Exception`` clause
  cannot shadow it.

Nothing here loads weights or touches the network: the packs are stubs and
ComfyUI's memory manager is replaced per test.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import os

import pytest
import torch

from modules import attention_utils
from modules import generation as generation_module
from modules import model_registry
from modules import patcher as patcher_module
from modules.generation import (
    CHATTERBOX_PATCHER_CACHE,
    InterruptProgressBar,
    build_cache_key,
    convert_voice_audio,
    force_offload_model,
    generate_audio,
    load_chatterbox_model,
    resolve_attn_implementation,
    resolve_generation_family,
)
from modules.loader import ChatterboxModelHandler
from modules.patcher import ChatterboxPatcher


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class StubPack:
    """Stands in for a loaded ``ChatterboxTTS`` / ``ChatterboxVC``."""

    def __init__(self, samples: int = 24000) -> None:
        self.t3 = torch.nn.Linear(4, 4)
        self.s3gen = torch.nn.Linear(4, 4)
        self.ve = torch.nn.Linear(2, 2)
        self.sr = 24000
        self.samples = samples
        self.calls: list[dict] = []
        #: ``(path, existed)`` for every path handed to ``generate`` -- recorded
        #: *during* the call, because the temp file is gone by the time the
        #: caller gets control back.
        self.paths_at_call: list[tuple[str, bool]] = []

    def generate(self, *args, **kwargs) -> torch.Tensor:
        candidates = [a for a in args if isinstance(a, str) and a.endswith(".wav")]
        candidates += [
            v for v in kwargs.values() if isinstance(v, str) and v.endswith(".wav")
        ]
        self.paths_at_call = [(path, os.path.isfile(path)) for path in candidates]
        self.calls.append({"args": list(args), **kwargs})
        return torch.zeros((1, self.samples))


class InterruptingPack(StubPack):
    """Raises instead of generating, the way a cancelled run does."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def generate(self, *args, **kwargs) -> torch.Tensor:
        super().generate(*args, **kwargs)
        raise self.error


class FakeModelManagement:
    """A ``comfy.model_management`` stand-in that never touches a device."""

    class InterruptProcessingException(BaseException):
        pass

    def __init__(self, interrupt: bool = False) -> None:
        self._interrupt = interrupt
        self.unload_calls = 0
        self.empty_calls = 0

    def throw_exception_if_processing_interrupted(self) -> None:
        if self._interrupt:
            raise self.InterruptProcessingException("cancelled")

    def unload_all_models(self) -> None:
        self.unload_calls += 1

    def soft_empty_cache(self, force: bool = False) -> None:
        self.empty_calls += 1


class RecordingPatcher(ChatterboxPatcher):
    """A patcher class that reports its own construction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ORDER.append("build_patcher")


#: The shared call-order buffer every recorder appends to.
ORDER: list[str] = []


def make_pack_dir(tmp_path) -> str:
    directory = tmp_path / "resembleai_default_voice"
    directory.mkdir(exist_ok=True)
    (directory / "t3_cfg.safetensors").write_bytes(b"a" * 2048)
    (directory / "s3gen.safetensors").write_bytes(b"b" * 1024)
    (directory / "ve.safetensors").write_bytes(b"c" * 512)
    (directory / "tokenizer.json").write_text("{}")
    return str(directory)


def make_handler(name: str, pack_dir: str) -> ChatterboxModelHandler:
    """A real handler with a stub pack attached and no disk access needed."""
    handler = ChatterboxModelHandler(name, model_path=pack_dir)
    handler.tts_model = StubPack()
    handler.attach_weights()
    handler._refine_size()
    return handler


def make_patcher(name: str, pack_dir: str, device="cpu") -> ChatterboxPatcher:
    handler = make_handler(name, pack_dir)
    return ChatterboxPatcher(
        handler, torch.device(device), torch.device(device), handler.size
    )


def stub_load_to_device(order: list[str] | None = None):
    """A ``load_to_device`` that marks the pack available instead of loading it."""

    def _load(patcher, memory_required: int = 0):
        if order is not None:
            order.append("load_to_device")
        if patcher.model.tts_model is None:
            patcher.model.load_model(patcher.load_device)
        return patcher

    return _load


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace every collaborator with a recorder; return the record.

    ``order`` collects the call sequence, so the ordering contract is asserted
    against something that ran rather than something that reads well.
    """
    ORDER.clear()
    pack_dir = make_pack_dir(tmp_path)

    handlers: list[dict] = []
    evictions: list[tuple] = []

    def fake_resolve_attention_mode(requested_mode, quantize_4bit=False):
        ORDER.append("resolve_attention_mode")
        return requested_mode

    def fake_build_cache_key(*args, **kwargs):
        ORDER.append("build_cache_key")
        return build_cache_key(*args, **kwargs)

    def fake_evict_if_changed(family, new_key, patcher_caches):
        ORDER.append("evict_if_changed")
        evictions.append((family, new_key, patcher_caches))
        return model_registry.evict_if_changed(family, new_key, patcher_caches)

    class RecordingHandler:
        def __init__(self, model_pack_name, **kwargs):
            ORDER.append("build_handler")
            self.model_pack_name = model_pack_name
            self.attention_mode = kwargs.get("attention_mode")
            self.dtype_str = kwargs.get("dtype_str")
            self.model_path = kwargs.get("model_path")
            self.cache_key = ""
            self.size = 3584
            self.device = None
            self.tts_model = None
            self.vc_model = None
            handlers.append({"name": model_pack_name, **kwargs})

        def load_model(self, device, attention_mode="sdpa", client_id=None):
            self.tts_model = StubPack()
            self.vc_model = StubPack()

        def detach_weights(self):
            pass

    monkeypatch.setattr(generation_module, "resolve_attention_mode", fake_resolve_attention_mode)
    monkeypatch.setattr(generation_module, "build_cache_key", fake_build_cache_key)
    monkeypatch.setattr(generation_module, "evict_if_changed", fake_evict_if_changed)
    monkeypatch.setattr(generation_module, "get_torch_device", lambda device_type=None: torch.device("cpu"))
    monkeypatch.setattr(generation_module, "get_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(generation_module, "resolve_dtype", lambda dtype_str, device=None: torch.float32)
    monkeypatch.setattr(generation_module, "get_dtype_str", lambda dtype: "fp32")
    monkeypatch.setattr(
        generation_module.ChatterboxLoader,
        "resolve_pack_dir",
        staticmethod(lambda model_name: pack_dir),
    )
    monkeypatch.setattr(generation_module, "ChatterboxModelHandler", RecordingHandler)
    monkeypatch.setattr(generation_module, "ChatterboxPatcher", RecordingPatcher)
    monkeypatch.setattr(generation_module, "load_to_device", stub_load_to_device(ORDER))
    monkeypatch.setattr(
        generation_module, "_get_model_management", lambda: FakeModelManagement()
    )

    yield {
        "order": ORDER,
        "handlers": handlers,
        "evictions": evictions,
        "pack_dir": pack_dir,
    }

    CHATTERBOX_PATCHER_CACHE.clear()
    model_registry.clear_active_keys()


@pytest.fixture(autouse=True)
def _clean_state():
    CHATTERBOX_PATCHER_CACHE.clear()
    model_registry.clear_active_keys()
    model_registry.clear_bundle_registry()
    yield
    CHATTERBOX_PATCHER_CACHE.clear()
    model_registry.clear_active_keys()
    model_registry.clear_bundle_registry()


def audio(samples: int = 2400, sample_rate: int = 24000) -> dict:
    return {"waveform": torch.zeros((1, 1, samples)), "sample_rate": sample_rate}


def read_source(module) -> str:
    with open(module.__file__, "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Call order
# ---------------------------------------------------------------------------


class TestCallOrder:
    def test_the_recorded_sequence_is_exact(self, wired) -> None:
        load_chatterbox_model(
            "resembleai_default_voice", device="cpu", attention_mode="sdpa"
        )

        assert wired["order"] == [
            "resolve_attention_mode",
            "build_cache_key",
            "evict_if_changed",
            "build_handler",
            "build_patcher",
            "load_to_device",
        ], (
            "the load sequence changed; eviction must happen before any "
            "allocation, and the cache key must be built from the resolved "
            f"values. Recorded: {wired['order']}"
        )

    def test_eviction_precedes_the_first_allocation(self, wired) -> None:
        load_chatterbox_model("resembleai_default_voice", device="cpu")

        evict_index = wired["order"].index("evict_if_changed")
        for allocation in ("build_handler", "build_patcher", "load_to_device"):
            assert evict_index < wired["order"].index(allocation), (
                f"{allocation} ran before the eviction; the previous model was "
                "still resident when the new one started allocating"
            )

    def test_a_cached_second_call_skips_the_allocation(self, wired) -> None:
        load_chatterbox_model("resembleai_default_voice", device="cpu")
        wired["order"].clear()

        load_chatterbox_model("resembleai_default_voice", device="cpu")

        assert wired["order"] == [
            "resolve_attention_mode",
            "build_cache_key",
            "evict_if_changed",
            "load_to_device",
        ]
        assert len(wired["handlers"]) == 1, "a cached pack was rebuilt"

    def test_force_reload_rebuilds(self, wired) -> None:
        first, _, _ = load_chatterbox_model("resembleai_default_voice", device="cpu")
        second, _, _ = load_chatterbox_model(
            "resembleai_default_voice", device="cpu", force_reload=True
        )
        assert first is not second
        assert len(wired["handlers"]) == 2


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_two_calls_with_one_key_allocate_once(self, wired) -> None:
        first_patch, first_tts, _ = load_chatterbox_model("pack", device="cpu")
        second_patch, second_tts, _ = load_chatterbox_model("pack", device="cpu")

        assert first_patch is second_patch
        assert first_tts is second_tts
        assert len(CHATTERBOX_PATCHER_CACHE) == 1
        assert len(wired["handlers"]) == 1

    def test_a_changed_key_evicts_the_previous_pack(self, wired) -> None:
        first_patch, _, _ = load_chatterbox_model("pack_a", device="cpu")
        assert len(CHATTERBOX_PATCHER_CACHE) == 1

        second_patch, _, _ = load_chatterbox_model("pack_b", device="cpu")

        assert second_patch is not first_patch
        assert len(CHATTERBOX_PATCHER_CACHE) == 1
        assert list(CHATTERBOX_PATCHER_CACHE.values())[0] is second_patch
        assert first_patch.model.tts_model is None, (
            "the evicted patcher still holds its pack; the VRAM was never freed"
        )
        assert [entry[1] for entry in wired["evictions"]][1].startswith("pack_b")

    def test_every_call_consults_the_registry(self, wired) -> None:
        load_chatterbox_model("pack_a", device="cpu")
        load_chatterbox_model("pack_a", device="cpu")
        load_chatterbox_model("pack_b", device="cpu")

        assert len(wired["evictions"]) == 3
        assert model_registry.get_active(model_registry.FAMILY_TTS).startswith("pack_b")

    def test_a_changed_attention_mode_is_a_different_key(self) -> None:
        assert build_cache_key("pack", "sdpa", "fp32", torch.device("cpu")) != (
            build_cache_key("pack", "sage", "fp32", torch.device("cpu"))
        )

    def test_a_changed_dtype_is_a_different_key(self) -> None:
        assert build_cache_key("pack", "sdpa", "fp32", torch.device("cpu")) != (
            build_cache_key("pack", "sdpa", "bf16", torch.device("cpu"))
        )

    def test_sage_resolved_to_sdpa_shares_the_sdpa_entry(self, monkeypatch, tmp_path) -> None:
        """The B7 ordering constraint, made concrete.

        ``resolve_attention_mode("sage")`` returns ``"sdpa"`` on a machine with
        no usable sage kernel. If the key were built from the *requested* mode,
        that workflow would get its own cache entry and a second copy of the
        same 2.7 GB pack -- the exact thing eviction exists to prevent.
        """
        pack_dir = make_pack_dir(tmp_path)
        handlers: list[dict] = []

        class RecordingHandler:
            def __init__(self, model_pack_name, **kwargs):
                handlers.append(kwargs)
                self.model_pack_name = model_pack_name
                self.cache_key = ""
                self.size = 3584
                self.device = None
                self.tts_model = None
                self.vc_model = None

            def load_model(self, device, attention_mode="sdpa", client_id=None):
                self.tts_model = StubPack()

            def detach_weights(self):
                pass

        monkeypatch.setattr(
            attention_utils, "check_sage_attention_compatible", lambda: False
        )
        monkeypatch.setattr(
            generation_module.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda model_name: pack_dir),
        )
        monkeypatch.setattr(generation_module, "ChatterboxModelHandler", RecordingHandler)
        monkeypatch.setattr(generation_module, "load_to_device", stub_load_to_device())

        load_chatterbox_model("pack", device="cpu", attention_mode="sdpa")
        load_chatterbox_model("pack", device="cpu", attention_mode="sage")

        assert len(CHATTERBOX_PATCHER_CACHE) == 1, (
            "a sage request that resolved to sdpa created a second cache entry "
            f"for the same weights: {sorted(CHATTERBOX_PATCHER_CACHE)}"
        )
        assert len(handlers) == 1, "the pack was allocated twice"
        assert all(
            key.count("_attn_sdpa_") == 1 for key in CHATTERBOX_PATCHER_CACHE
        ), "the cache key still carries the unresolved 'sage' request"


# ---------------------------------------------------------------------------
# client_id
# ---------------------------------------------------------------------------


class TestClientId:
    def test_client_id_is_not_part_of_the_cache_key(self) -> None:
        key = build_cache_key("pack", "sdpa", "fp32", torch.device("cpu"))
        assert "client" not in key
        # The key is a pure function of four model-identity arguments; there is
        # no fifth input for a client id to hide in.
        assert set(inspect.signature(build_cache_key).parameters) == {
            "model_name",
            "attention_mode",
            "dtype_str",
            "device",
        }

    def test_two_clients_share_the_patch_and_keep_their_own_id(self, wired) -> None:
        first_patch, _, _ = load_chatterbox_model("pack", device="cpu", client_id="client-A")
        assert first_patch.client_id == "client-A"

        second_patch, _, _ = load_chatterbox_model("pack", device="cpu", client_id="client-B")

        assert second_patch is first_patch, "a second client forced a second load"
        assert second_patch.client_id == "client-B", (
            "the second client silently inherited the first client's id, so its "
            "progress events go to the wrong browser tab"
        )
        assert len(CHATTERBOX_PATCHER_CACHE) == 1
        assert len(wired["handlers"]) == 1

    def test_a_call_without_a_client_id_clears_the_previous_one(self, wired) -> None:
        load_chatterbox_model("pack", device="cpu", client_id="A")
        patcher, _, _ = load_chatterbox_model("pack", device="cpu")
        assert patcher.client_id is None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestGenerateAudio:
    def test_returns_a_tensor_and_the_model_sample_rate(self) -> None:
        waveform, rate = generate_audio(StubPack(samples=24000), "hello there", seed=7)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == (1, 24000)
        assert waveform.dtype is torch.float32
        assert rate == 24000

    def test_the_widget_values_reach_the_vendor(self) -> None:
        model = StubPack()
        generate_audio(
            model,
            "hello",
            audio_prompt=audio(),
            exaggeration=0.9,
            temperature=0.5,
            cfg_weight=0.7,
            repetition_penalty=1.4,
            min_p=0.02,
            top_p=0.8,
            max_new_tokens=512,
            flow_cfg_scale=0.6,
            use_watermark=True,
            seed=3,
        )
        call = model.calls[0]
        assert call["args"][0] == "hello"
        assert call["exaggeration"] == 0.9
        assert call["temperature"] == 0.5
        assert call["cfg_weight"] == 0.7
        assert call["repetition_penalty"] == 1.4
        assert call["min_p"] == 0.02
        assert call["top_p"] == 0.8
        assert call["max_new_tokens"] == 512
        assert call["flow_cfg_scale"] == 0.6
        assert call["watermark"] is True
        assert model.paths_at_call, "no path was handed to the model at all"
        assert all(existed for _path, existed in model.paths_at_call), (
            f"the voice prompt file did not exist while generate() ran: "
            f"{model.paths_at_call}"
        )

    def test_the_voice_prompt_file_is_gone_afterwards(self) -> None:
        model = StubPack()
        generate_audio(model, "hello", audio_prompt=audio())
        assert not os.path.exists(model.calls[0]["audio_prompt_path"])

    def test_the_voice_prompt_file_is_gone_after_a_failure(self) -> None:
        class Exploding(StubPack):
            def generate(self, *args, **kwargs):
                self.calls.append({"args": list(args), **kwargs})
                raise RuntimeError("boom")

        model = Exploding()
        with pytest.raises(RuntimeError):
            generate_audio(model, "hello", audio_prompt=audio())
        assert not os.path.exists(model.calls[0]["audio_prompt_path"])

    def test_no_voice_prompt_means_no_path(self) -> None:
        model = StubPack()
        generate_audio(model, "hello")
        assert model.calls[0]["audio_prompt_path"] is None

    def test_watermarking_defaults_off(self) -> None:
        model = StubPack()
        generate_audio(model, "hello")
        assert model.calls[0]["watermark"] is False

    def test_the_seed_is_applied_before_generating(self) -> None:
        first = StubPack()
        second = StubPack()
        generate_audio(first, "hello", seed=99)
        torch_first = torch.rand(3).tolist()
        generate_audio(second, "hello", seed=99)
        assert torch_first == torch.rand(3).tolist(), (
            "the seed was not set deterministically by generate_audio"
        )

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_text_is_an_error_not_silence(self, blank: str) -> None:
        with pytest.raises(ValueError, match="needs some text"):
            generate_audio(StubPack(), blank)

    def test_empty_output_is_reported(self) -> None:
        with pytest.raises(ValueError, match="produced no audio"):
            generate_audio(StubPack(samples=0), "hello")

    def test_a_generic_failure_is_logged_and_re_raised(self, caplog) -> None:
        class Exploding(StubPack):
            def generate(self, *args, **kwargs):
                raise RuntimeError("the s3 decoder fell over")

        with caplog.at_level(logging.ERROR, logger=""):
            with pytest.raises(RuntimeError, match="s3 decoder"):
                generate_audio(Exploding(), "hello")

        messages = [record.getMessage() for record in caplog.records]
        assert any("Generation failed" in message for message in messages)
        assert all(record.name == "root" for record in caplog.records)


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------


class TestInterruptHandling:
    def test_a_real_interrupt_propagates(self) -> None:
        """The headline behaviour: a cancelled run cancels.

        V1 turned every exception into ``torch.zeros((1, 24000))`` -- one
        second of silence, which the user hears as "the node worked and
        produced nothing".
        """
        import comfy.model_management as model_management

        error = model_management.InterruptProcessingException("stopped")
        model = InterruptingPack(error)

        with pytest.raises(model_management.InterruptProcessingException):
            generate_audio(model, "a very long sentence")

        assert model.calls, "the model was never asked to generate"

    def test_an_interrupt_that_is_also_an_exception_still_propagates(self) -> None:
        """The discriminating case for clause ordering.

        Core's interrupt derives from ``BaseException``, so a plain
        ``except Exception`` cannot catch it and the ordering would not matter.
        This synthetic interrupt derives from **both**, which is exactly the
        situation the ordering rule exists for: with ``except Exception``
        first, this run would return silence and this test would fail.
        """
        import comfy.model_management as model_management

        class Hybrid(model_management.InterruptProcessingException, RuntimeError):
            pass

        with pytest.raises(Hybrid):
            generate_audio(InterruptingPack(Hybrid("stopped")), "a very long sentence")

    def test_no_silence_is_returned_on_an_interrupt(self) -> None:
        """Not merely "it raises": it must not also hand back audio."""
        import comfy.model_management as model_management

        model = InterruptingPack(model_management.InterruptProcessingException())
        returned = None
        try:
            returned = generate_audio(model, "hello")
        except model_management.InterruptProcessingException:
            pass
        assert returned is None

    def test_the_interrupt_clause_precedes_the_generic_one(self) -> None:
        """The source-level contract, asserted structurally rather than by eye."""
        tree = ast.parse(read_source(generation_module), filename=generation_module.__file__)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "generate_audio"
        )

        guarded = [node for node in ast.walk(function) if isinstance(node, ast.Try) and len(node.handlers) >= 2]
        assert guarded, "generate_audio has no multi-clause try/except to inspect"

        for node in guarded:
            kinds = []
            for handler in node.handlers:
                names = [
                    part.id
                    for part in ast.walk(handler.type)
                    if isinstance(part, ast.Name)
                ] if handler.type is not None else []
                kinds.append(names)

            interrupt_at = next(
                (i for i, names in enumerate(kinds) if "interrupt_error" in names), None
            )
            generic_at = next(
                (i for i, names in enumerate(kinds) if "Exception" in names), None
            )
            assert interrupt_at is not None, "the interrupt clause is missing"
            assert generic_at is not None, "the generic clause is missing"
            assert interrupt_at < generic_at, (
                "the interrupt clause comes after the generic one; a cancel "
                "would be reported as a generation failure"
            )

    def test_no_bare_or_base_exception_handler(self) -> None:
        source = read_source(generation_module)
        assert "except BaseException" not in source
        assert "\n    except:" not in source

    def test_the_progress_bar_polls_for_cancellation(self) -> None:
        bar = InterruptProgressBar(10)
        bar._model_management = FakeModelManagement(interrupt=True)
        with pytest.raises(FakeModelManagement.InterruptProcessingException):
            bar.update(1)

    def test_the_progress_bar_never_polls_when_finishing(self) -> None:
        """A poll in ``finish`` would re-raise the interrupt from a cleanup path."""
        bar = InterruptProgressBar(10)
        bar._model_management = FakeModelManagement(interrupt=True)
        bar.finish()  # must not raise

    def test_an_interrupt_during_generation_still_closes_the_bar(self, monkeypatch) -> None:
        import comfy.model_management as model_management

        closed: list[bool] = []

        class Polling(InterruptProgressBar):
            def finish(self):
                closed.append(True)

        class PollingPack(StubPack):
            def generate(self, *args, **kwargs):
                kwargs["pbar"].update(1)
                raise model_management.InterruptProcessingException("stopped")

        monkeypatch.setattr(
            generation_module, "_build_progress_bar", lambda total: Polling(total)
        )

        with pytest.raises(model_management.InterruptProcessingException):
            generate_audio(PollingPack(), "hello")

        assert closed == [True], "the progress bar was left open by the cancel"


# ---------------------------------------------------------------------------
# Voice conversion and offload
# ---------------------------------------------------------------------------


class TestVoiceConversion:
    def test_returns_audio_and_the_sample_rate(self) -> None:
        model = StubPack(samples=12000)
        waveform, rate = convert_voice_audio(model, audio(), n_timesteps=4)

        assert waveform.shape == (1, 12000)
        assert rate == 24000
        call = model.calls[0]
        assert call["n_timesteps"] == 4
        assert model.paths_at_call, "no path was handed to the model at all"
        assert all(existed for _path, existed in model.paths_at_call), (
            f"the source clip was not on disk while generate() ran: "
            f"{model.paths_at_call}"
        )
        assert call["target_voice_path"] is None

    def test_the_target_voice_is_written_to_disk_for_the_call(self) -> None:
        model = StubPack()
        convert_voice_audio(model, audio(), target_voice_audio=audio(samples=1200))

        assert len(model.paths_at_call) == 2, "both clips should be written out"
        assert all(existed for _path, existed in model.paths_at_call)
        assert not os.path.exists(model.calls[0]["target_voice_path"]), (
            "the target voice clip survived the call"
        )

    @pytest.mark.parametrize(
        "empty", [None, {}, {"waveform": None}, {"waveform": torch.zeros(1, 1, 0)}]
    )
    def test_missing_source_audio_is_an_error(self, empty) -> None:
        with pytest.raises(ValueError, match="needs source audio"):
            convert_voice_audio(StubPack(), empty)

    def test_empty_output_is_reported(self) -> None:
        with pytest.raises(ValueError, match="produced no audio"):
            convert_voice_audio(StubPack(samples=0), audio())

    def test_an_interrupt_propagates(self) -> None:
        import comfy.model_management as model_management

        model = InterruptingPack(model_management.InterruptProcessingException())
        with pytest.raises(model_management.InterruptProcessingException):
            convert_voice_audio(model, audio())


class TestForceOffload:
    def test_a_destroying_offload_releases_the_pack(self, monkeypatch, tmp_path) -> None:
        fake_mm = FakeModelManagement()
        monkeypatch.setattr(generation_module, "_get_model_management", lambda: fake_mm)

        patcher = make_patcher("pack", make_pack_dir(tmp_path))
        patcher.load_device = torch.device("cpu")

        force_offload_model(patcher, "pack")

        assert patcher.model.tts_model is None
        assert fake_mm.unload_calls == 1
        assert fake_mm.empty_calls == 1

    def test_a_warm_offload_keeps_the_pack(self, monkeypatch, tmp_path) -> None:
        fake_mm = FakeModelManagement()
        monkeypatch.setattr(generation_module, "_get_model_management", lambda: fake_mm)

        patcher = make_patcher("pack", make_pack_dir(tmp_path))
        patcher.load_device = torch.device("cpu")

        force_offload_model(patcher, "pack", warm=True)

        assert patcher.model.tts_model is not None
        assert patcher._warm_offloaded is True


# ---------------------------------------------------------------------------
# Routing and module surface
# ---------------------------------------------------------------------------


class TestRouting:
    def test_a_named_pack_routes_to_tts(self) -> None:
        assert resolve_generation_family("resembleai_default_voice") == "tts"

    def test_a_blank_name_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="No Chatterbox model"):
            resolve_generation_family("")

    def test_an_external_family_wins(self) -> None:
        assert resolve_generation_family("x", {"family": "vc"}) == "vc"
        assert resolve_generation_family("x", {"type": "tts"}) == "tts"

    def test_an_unknown_external_family_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="implements 'tts' and 'vc'"):
            resolve_generation_family("x", {"family": "asr"})

    def test_sage_loads_as_sdpa(self) -> None:
        assert resolve_attn_implementation("sage") == "sdpa"
        assert resolve_attn_implementation("sdpa") == "sdpa"
        assert resolve_attn_implementation("eager") == "eager"


class TestModuleSurface:
    def test_there_is_exactly_one_cache(self) -> None:
        assert CHATTERBOX_PATCHER_CACHE is patcher_module.CHATTERBOX_PATCHER_CACHE, (
            "generation.py re-created the cache instead of re-exporting "
            "patcher.py's; an eviction would only sweep one of the two"
        )

    def test_no_comfy_api_anywhere_in_modules(self) -> None:
        """No module in ``modules/`` may reach for the node-API package."""
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modules")
        offenders = []
        for name in sorted(os.listdir(root)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name), "r", encoding="utf-8") as handle:
                source = handle.read()
            if "import comfy_api" in source:
                offenders.append(f"{name}: imports it")
                continue
            tree = ast.parse(source, filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "comfy_api":
                            offenders.append(f"{name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").split(".")[0] == "comfy_api":
                        offenders.append(f"{name}: from {node.module}")
        assert offenders == []

    def test_no_borrowed_vibevoice_modules_are_referenced(self) -> None:
        source = read_source(generation_module)
        for name in (
            "asr_loader",
            "memory_census",
            "gguf_quant",
            "progress_utils",
            "realtime_generation",
            "diagnostics",
        ):
            assert name not in source

    def test_generation_reaches_comfy_lazily(self) -> None:
        source = read_source(generation_module)
        assert "import comfy.model_management" not in source
        assert "import folder_paths" not in source
        assert "import server" not in source

    def test_no_logger_objects_or_handlers(self) -> None:
        for module in (generation_module, patcher_module):
            source = read_source(module)
            for forbidden in (
                "getLogger(",
                "logging.Formatter(",
                "LoggerAdapter(",
                ".addHandler(",
                ".setLevel(",
            ):
                assert forbidden not in source, f"{module.__name__} uses {forbidden}"

    def test_the_public_signatures_are_the_contract(self) -> None:
        assert list(inspect.signature(load_chatterbox_model).parameters) == [
            "model_name",
            "device",
            "dtype",
            "attention_mode",
            "force_reload",
            "client_id",
        ]
        assert list(inspect.signature(force_offload_model).parameters) == [
            "patcher",
            "model_name",
            "warm",
        ]
        generate_parameters = list(inspect.signature(generate_audio).parameters)
        assert generate_parameters[:2] == ["model", "text"]


    def test_the_set_seed_shim_is_gone(self) -> None:
        """``modules/utils.py`` was a one-name re-export; B11 deleted it.

        ``modules.audio_utils.set_seed`` is the single implementation and
        the node layer reaches it through this module. A second spelling of
        the same function is one more thing that can drift.
        """
        with pytest.raises(ImportError):
            importlib.import_module("modules.utils")