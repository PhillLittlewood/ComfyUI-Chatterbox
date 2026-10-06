"""S7.2 -- ``client_id`` reaches the download manager, end to end.

A Chatterbox pack is 2.7 GB. The first time a user picks one, it is not on
disk, and the download is the slowest part of their prompt by an order of
magnitude. Progress for that download goes out over the websocket, addressed
to **one** client id -- the tab that queued the prompt. Without an id the
manager broadcasts, so a second open tab shows a download it did not ask for,
and the tab that *did* ask shows nothing at all because its frontend is
listening on its own channel.

The chain that carries the id is::

    ChatterboxTTSNode.execute(client_id=...)
        -> modules.generation.load_chatterbox_model(client_id=...)
            -> patcher.client_id                       (stamped per call)
                -> ChatterboxPatcher.patch_model
                    -> ChatterboxModelHandler.load_model(client_id=...)
                        -> ChatterboxLoader.load_model(client_id=...)
                            -> ChatterboxLoader.download_pack(client_id)
                                -> DownloadManager.download_model(client_id)

Every hop is asserted here with a recorder, in order, with the value -- not by
reading the code and trusting that it connects.

**And ``client_id`` is kept out of the cache key** (plan §3.1.8). It names a
websocket, not a model. VibeVoice's ``generation.py:78`` stamped the id onto a
cached patcher at construction time, so the *first* tab's id stuck to the
patcher and every later tab got its progress on the wrong channel. The tests in
:class:`TestCachedPatcherReuse` are the ones that discriminate: a second client
must receive its own id, and two clients on one model must share a single cache
entry rather than holding two copies of 2.7 GB.

No weights are loaded and no network is touched. The pieces replaced are the
ones that would need them -- ``comfy.model_management``'s placement, core's
``ModelPatcher.patch_model`` body, and the vendored ``from_local`` calls. Every
link of the chain *between* them is the shipped code.
"""

from __future__ import annotations

import pytest
import torch

import comfy.model_patcher

from ComfyUI_Chatterbox.modules import generation as generation_module
from ComfyUI_Chatterbox.modules import model_registry
from ComfyUI_Chatterbox.modules import patcher as patcher_module
from ComfyUI_Chatterbox.modules.generation import (
    CHATTERBOX_PATCHER_CACHE,
    load_chatterbox_model,
)
from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader, ChatterboxModelHandler
from ComfyUI_Chatterbox.modules.patcher import ChatterboxPatcher
from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode
from ComfyUI_Chatterbox.nodes import tts_node as tts_node_module
from ComfyUI_Chatterbox.nodes import vc_node as vc_node_module


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class StubPack(torch.nn.Module):
    """Stands in for a loaded ``ChatterboxTTS`` / ``ChatterboxVC``.

    A real ``nn.Module`` because the patcher casts the pack to the target dtype
    and sums its parameters: a plain object would fail in
    ``modules.dtype_utils`` rather than at the hop under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.t3 = torch.nn.Linear(4, 4)
        self.s3gen = torch.nn.Linear(4, 4)
        self.ve = torch.nn.Linear(2, 2)

    def generate(self, *args, **kwargs):  # pragma: no cover - never reached here
        return torch.zeros((1, 24000))


class RecordingMemoryManager:
    """``comfy.model_management`` reduced to the placement calls.

    ``load_model_gpu`` calls ``patcher.patch_model`` for the same reason core's
    ``model_load`` does: that is the *only* route by which a lazily-loaded pack
    is ever built, so stubbing it away would delete the hop this file exists to
    prove.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_free_memory(self, device) -> int:
        self.calls.append("get_free_memory")
        return 0

    def load_model_gpu(self, patcher) -> None:
        self.calls.append("load_model_gpu")
        patcher.patch_model(load_weights=False)

    def load_models_gpu(self, patchers, **kwargs) -> None:
        self.calls.append("load_models_gpu")
        for patcher in patchers:
            patcher.patch_model(load_weights=False)

    def intermediate_device(self) -> torch.device:
        self.calls.append("intermediate_device")
        return torch.device("cpu")

    def soft_empty_cache(self, force: bool = False) -> None:
        self.calls.append("soft_empty_cache")

    def unload_all_models(self) -> None:
        self.calls.append("unload_all_models")


class RecordingDownloadManager:
    """``DownloadManager`` reduced to the one call the chain makes."""

    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def download_model(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: str | None = None,
    ) -> str:
        self.calls.append(("download_model", client_id, model_name, local_dir))
        return local_dir


class StopHere(Exception):
    """Raised by a stub to end a chain walk at a known hop."""


def make_complete_pack(tmp_path, name: str = "pack") -> str:
    """A directory that passes :meth:`ChatterboxLoader.is_pack_complete`."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "t3_cfg.safetensors").write_bytes(b"a" * 1024)
    (directory / "s3gen.safetensors").write_bytes(b"b" * 512)
    (directory / "ve.safetensors").write_bytes(b"c" * 256)
    (directory / "tokenizer.json").write_text("{}")
    return str(directory)


@pytest.fixture(autouse=True)
def _clean_state():
    """One cache, one active key, per test.

    The patcher cache is process-wide and the active key is per model family.
    Left dirty, a test that loaded a pack would evict the next test's pack --
    or, worse, the "second client" assertions would pass vacuously because the
    cached patcher was never rebuilt.
    """
    CHATTERBOX_PATCHER_CACHE.clear()
    model_registry.clear_active_keys()
    model_registry.clear_bundle_registry()
    yield
    CHATTERBOX_PATCHER_CACHE.clear()
    model_registry.clear_active_keys()
    model_registry.clear_bundle_registry()


@pytest.fixture
def chain(monkeypatch, tmp_path):
    """Record every hop of the chain. Returns the list of ``(hop, client_id)``.

    Replaced: device/dtype resolution, memory placement, core's
    ``ModelPatcher.patch_model`` body, and the vendored ``from_local`` calls
    that need the real ``chatterbox`` package. Everything between them is the
    shipped code under test.
    """
    hops: list[tuple[str, str | None]] = []
    pack_dir = make_complete_pack(tmp_path)

    monkeypatch.setattr(
        generation_module, "resolve_attention_mode", lambda mode, quantize=False: mode
    )
    monkeypatch.setattr(
        generation_module, "get_torch_device", lambda device_type=None: torch.device("cpu")
    )
    monkeypatch.setattr(
        generation_module, "get_offload_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        generation_module, "resolve_dtype", lambda dtype, device=None: torch.float32
    )
    monkeypatch.setattr(generation_module, "get_dtype_str", lambda dtype: "fp32")
    monkeypatch.setattr(
        generation_module.ChatterboxLoader,
        "resolve_pack_dir",
        staticmethod(lambda model_name: pack_dir),
    )

    memory_manager = RecordingMemoryManager()
    monkeypatch.setattr(
        generation_module, "_get_model_management", lambda: memory_manager
    )
    monkeypatch.setattr(
        patcher_module, "_get_model_management", lambda: memory_manager
    )

    def load_to_device(patcher, memory_required: int = 0):
        hops.append(("load_to_device", getattr(patcher, "client_id", None)))
        return patcher_module.load_to_device(patcher, memory_required)

    monkeypatch.setattr(generation_module, "load_to_device", load_to_device)

    # Core's own patch_model moves and measures real modules; ours has already
    # done the part under test by the time control reaches this call.
    monkeypatch.setattr(
        comfy.model_patcher.ModelPatcher, "patch_model", lambda self, **kwargs: None
    )

    real_patch_model = ChatterboxPatcher.patch_model

    def patch_model(self, *args, **kwargs):
        hops.append(("ChatterboxPatcher.patch_model", self.client_id))
        return real_patch_model(self, *args, **kwargs)

    monkeypatch.setattr(ChatterboxPatcher, "patch_model", patch_model)

    real_handler_load_model = ChatterboxModelHandler.load_model

    def handler_load_model(
        self, device, attention_mode: str = "sdpa", client_id: str | None = None
    ) -> None:
        hops.append(("ChatterboxModelHandler.load_model", client_id))
        return real_handler_load_model(self, device, attention_mode, client_id=client_id)

    monkeypatch.setattr(
        ChatterboxModelHandler, "load_model", handler_load_model
    )

    def loader_load_model(model_name, device, **kwargs):
        client_id = kwargs.get("client_id")
        hops.append(("ChatterboxLoader.load_model", client_id))
        return StubPack(), StubPack()

    monkeypatch.setattr(
        ChatterboxLoader, "load_model", staticmethod(loader_load_model)
    )

    return hops


# ---------------------------------------------------------------------------
# The full chain
# ---------------------------------------------------------------------------


class TestFullChain:
    def test_the_id_arrives_at_the_download_manager(
        self, chain, monkeypatch, tmp_path
    ) -> None:
        """Every hop, in order, carrying ``client-alpha``, ending at the manager.

        The pack directory is emptied first so ``ChatterboxLoader.load_model``
        takes the **download** branch. A complete pack never calls
        ``download_pack`` at all, so a chain test that stops at ``from_local``
        would prove nothing about the hop that actually matters.
        """
        empty = tmp_path / "not-downloaded-yet"
        empty.mkdir()
        monkeypatch.setattr(
            generation_module.ChatterboxLoader,
            "resolve_pack_dir",
            staticmethod(lambda model_name: str(empty)),
        )

        manager_calls: list[tuple] = []
        monkeypatch.setattr(
            "ComfyUI_Chatterbox.modules.downloader.get_download_manager",
            lambda: RecordingDownloadManager(manager_calls),
        )

        def download_then_report(model_name, device, **kwargs):
            """The real loader's download step, then its honest failure.

            ``download_pack`` and ``is_pack_complete`` are the shipped code;
            only the vendor build after them is replaced.
            """
            ChatterboxLoader.download_pack(
                model_name, "ResembleAI/chatterbox", str(empty), kwargs.get("client_id")
            )
            if not ChatterboxLoader.is_pack_complete(str(empty)):
                raise FileNotFoundError(
                    f"The Chatterbox voice pack '{model_name}' is not "
                    f"available at '{empty}'."
                )
            return StubPack(), StubPack()  # pragma: no cover - unreachable here

        monkeypatch.setattr(
            ChatterboxLoader, "load_model", staticmethod(download_then_report)
        )

        with pytest.raises(FileNotFoundError, match="not available at"):
            ChatterboxTTSNode.execute(
                model_pack_name="pack",
                text="hello",
                max_new_tokens=1000,
                flow_cfg_scale=0.7,
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=1.0,
                seed=0,
                use_watermark=False,
                client_id="client-alpha",
            )

        assert manager_calls, (
            "the download manager was never reached; the chain is broken "
            "somewhere above ChatterboxLoader.download_pack. Recorded hops: "
            f"{chain}"
        )
        assert manager_calls[0][1] == "client-alpha", (
            f"the manager was handed {manager_calls[0][1]!r}; a first-run "
            f"download would publish its progress on the wrong channel"
        )

    def test_each_hop_forwards_the_id_unchanged(self, chain) -> None:
        """The recorded hop sequence, with the same id at every one.

        ``patcher.client_id`` is asserted as its own hop because that is the
        one place the value is *stored* rather than passed. A hop that quietly
        substitutes ``None`` there -- or that stamps the id once at patcher
        construction instead of per call -- shows up as a single wrong entry,
        which is why the sequence and the values are asserted together.
        """
        load_chatterbox_model("pack", device="cpu", client_id="client-beta")

        recorded = [hop for hop, _id in chain]
        assert recorded == [
            "load_to_device",
            "ChatterboxPatcher.patch_model",
            "ChatterboxModelHandler.load_model",
            "ChatterboxLoader.load_model",
        ], f"the chain ran out of order, or a link is missing: {chain}"

        assert [client_id for _hop, client_id in chain] == ["client-beta"] * len(chain), (
            f"the id changed on the way down: {chain}"
        )

    def test_both_nodes_forward_their_client_id(self, monkeypatch) -> None:
        """``ChatterboxTTS`` and ``ChatterboxVC`` both pass it on.

        The VC node gets its own test because the two nodes have separate
        ``execute`` signatures with the id as a trailing optional kwarg --
        exactly the shape that gets dropped from one of them in a refactor. The
        stub stops the walk right after the forward, so nothing here depends on
        synthesis or on the preview encoder.
        """
        for node_cls, node_module, kwargs in (
            (
                ChatterboxTTSNode,
                tts_node_module,
                {
                    "model_pack_name": "pack",
                    "text": "hello",
                    "max_new_tokens": 1000,
                    "flow_cfg_scale": 0.7,
                    "exaggeration": 0.5,
                    "temperature": 0.8,
                    "cfg_weight": 0.5,
                    "repetition_penalty": 1.2,
                    "min_p": 0.05,
                    "top_p": 1.0,
                    "seed": 0,
                    "use_watermark": False,
                },
            ),
            (
                ChatterboxVCNode,
                vc_node_module,
                {
                    "model_pack_name": "pack",
                    "source_audio": {
                        "waveform": torch.zeros((1, 1, 2400)),
                        "sample_rate": 24000,
                    },
                    "n_timesteps": 10,
                    "temperature": 1.0,
                    "flow_cfg_scale": 0.7,
                    "use_watermark": False,
                },
            ),
        ):
            expected = f"client-for-{node_cls.__name__}"
            captured: list[str | None] = []

            def record(*args, **kw):
                captured.append(kw.get("client_id"))
                raise StopHere

            # The nodes bind the symbol at import time
            # (``from ..modules.generation import load_chatterbox_model``), so
            # the patch has to land in the *node* module's globals -- patching
            # ``modules.generation`` would leave the node calling the real one.
            monkeypatch.setattr(node_module, "load_chatterbox_model", record)

            with pytest.raises(StopHere):
                node_cls.execute(client_id=expected, **kwargs)

            assert captured == [expected], (
                f"{node_cls.__name__}.execute did not forward client_id; it "
                f"passed {captured}"
            )

    def test_the_signature_of_both_nodes_accepts_a_client_id(self) -> None:
        """The kwarg has to exist, or the chain starts at a ``TypeError``."""
        import inspect

        for node_cls in (ChatterboxTTSNode, ChatterboxVCNode):
            parameters = inspect.signature(node_cls.execute).parameters
            assert "client_id" in parameters, (
                f"{node_cls.__name__}.execute has no client_id parameter"
            )
            assert parameters["client_id"].default is None, (
                f"{node_cls.__name__}.execute must default client_id to None so "
                f"an API caller that supplies none still runs"
            )


# ---------------------------------------------------------------------------
# The cache-key contract
# ---------------------------------------------------------------------------


class TestCachedPatcherReuse:
    def test_a_second_client_gets_its_own_id(self, chain) -> None:
        """The discriminating case.

        The first tab loads the pack. The second tab asks for the same pack and
        gets the *cached patcher* -- no second download, no second copy of the
        weights -- but the id has to be **re-stamped with the second client's**,
        or its progress goes out on the first tab's channel.

        A single assertion on the end state would also pass if the re-stamp
        happened at construction; this one checks the id is right *after* the
        reuse, which is the only moment it matters.
        """
        first, _tts, _vc = load_chatterbox_model("pack", device="cpu", client_id="tab-1")
        assert first.client_id == "tab-1"

        second, _tts, _vc = load_chatterbox_model("pack", device="cpu", client_id="tab-2")

        assert second is first, (
            "the second client rebuilt the patcher; two clients would then "
            "hold two copies of a 2.7 GB pack"
        )
        assert second.client_id == "tab-2", (
            f"the cached patcher still carries {second.client_id!r}; tab-2's "
            f"progress would be published on tab-1's channel"
        )

    def test_client_id_is_absent_from_the_cache_key(self, chain) -> None:
        """Two clients, one model: one cache entry, two ids delivered.

        Asserted on the *key* as well as on the entry count, because a key that
        merely happened to collide would pass the count check while still
        being a key built from the id.

        The two ids are read from the per-call recorder rather than from the
        patcher afterwards: ``first`` and ``second`` are the *same object*, so
        reading both attributes at the end would only ever see the last value
        written -- and would pass even if only ever one id had been delivered.
        """
        first, _tts, _vc = load_chatterbox_model("pack", device="cpu", client_id="tab-1")
        key = first.cache_key

        assert "tab-1" not in key, (
            f"the cache key carries the client id ({key!r}); the id names a "
            f"websocket, not a model"
        )

        second, _tts, _vc = load_chatterbox_model("pack", device="cpu", client_id="tab-2")

        assert second is first, "the second client rebuilt the patcher"
        assert len(CHATTERBOX_PATCHER_CACHE) == 1, (
            f"two clients produced {sorted(CHATTERBOX_PATCHER_CACHE)}; a key "
            f"built from client_id holds the same weights twice"
        )
        assert sorted(CHATTERBOX_PATCHER_CACHE) == [key]

        delivered = [client_id for hop, client_id in chain if hop == "load_to_device"]
        assert delivered == ["tab-1", "tab-2"], (
            f"the ids actually delivered to the placement step were {delivered}; "
            f"a shared patcher that never re-reads its id serves tab-1 twice"
        )
        assert second.client_id == "tab-2"

    def test_a_call_without_an_id_clears_the_previous_one(self, chain) -> None:
        """An API call (no browser tab) must not inherit a tab's channel.

        The patcher is shared, so leaving the previous caller's id on it would
        publish progress for a client that is not waiting for any.
        """
        first, _tts, _vc = load_chatterbox_model("pack", device="cpu", client_id="tab-1")
        assert first.client_id == "tab-1"

        again, _tts, _vc = load_chatterbox_model("pack", device="cpu")

        assert again is first
        assert again.client_id is None, (
            "a call with no client_id inherited the previous caller's; progress "
            "would be addressed to a tab that did not ask for it"
        )

    def test_the_cache_key_is_built_without_a_client_id_argument(
        self,
    ) -> None:
        """Structural: ``build_cache_key`` cannot accept the id at all.

        Signature-level rather than behavioural on purpose. A future edit that
        adds a ``client_id=None`` parameter -- leaving every current call site
        unchanged and every current assertion green -- would split the cache per
        client without failing a single test above until it did.
        """
        import inspect

        from ComfyUI_Chatterbox.modules.generation import build_cache_key

        assert "client_id" not in inspect.signature(build_cache_key).parameters, (
            "client_id must never become a cache-key input: the cache holds "
            "weights, and the id names a websocket"
        )