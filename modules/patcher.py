"""ComfyUI ``ModelPatcher`` integration for Chatterbox.

Chatterbox ships two plain Python classes (``ChatterboxTTS`` and
``ChatterboxVC``) that are *not* ``torch.nn.Module`` subclasses, so ComfyUI's
memory manager has nothing to move around or account for on its own. This
module supplies the missing half:

* :class:`ChatterboxPatcher` -- a ``ModelPatcher`` subclass whose wrapped model
  is :class:`modules.loader.ChatterboxModelHandler`, an ``nn.Module`` that owns
  the pack, loads it lazily, and exposes its weights as real submodules.
* :data:`CHATTERBOX_PATCHER_CACHE` -- the single process-wide cache of live
  patchers. ``modules.generation`` re-exports this exact object rather than
  keeping a second dict, because two dicts means every eviction misses half the
  models.

Borrowed from VibeVoice's ``modules/patcher.py``; what changed on the way in is
listed per-symbol below.

The VibeVoice dynamic-VRAM (``aimdo`` / VBAR) subclass is **not** carried over:
the plan's §3.3 records it as dead weight there too -- ``select_patcher_class``
always returned the legacy class, so no path ever constructed it. Only the
probes that decide *not* to use it are kept.

``comfy.model_patcher`` is imported at module scope (it is core -- not the
node-API package -- and the class has to exist at import time to subclass);
``comfy.model_management`` is reached through
:func:`modules.device_utils._get_model_management`.
"""

from __future__ import annotations

import gc
import logging
from itertools import chain
from typing import Any, Optional

import torch

import comfy.model_patcher

from .device_utils import _get_model_management
from .dtype_utils import cast_model_to_dtype, representative_dtype

__all__ = [
    "CHATTERBOX_PATCHER_CACHE",
    "ACTIVATION_HEADROOM_BYTES",
    "WRAPPER_MODEL_FIELDS",
    "ChatterboxPatcher",
    "release_wrapper_references",
    "load_to_device",
    "select_patcher_class",
    "resolve_core_patcher_class",
    "dynamic_vram_available",
]


# The one cache. ``cache_key -> ChatterboxPatcher``.
#
# Keyed by *resolved* model identity (pack name, attention mode, dtype, device)
# and deliberately **not** by ``client_id``: the id identifies the websocket a
# progress toast goes to, not the model. A second browser tab asking for the
# same model must get the loaded weights, not a second copy of them.
CHATTERBOX_PATCHER_CACHE: dict[str, "ChatterboxPatcher"] = {}


# Headroom left free for activations when deciding to place a whole model in
# VRAM. 1.5 GB is what ComfyUI's own native full-load path budgets for the
# transient tensors of one step; a model that fits exactly leaves none.
ACTIVATION_HEADROOM_BYTES = int(1.5 * 1024**3)


# The attributes on the *wrapper* (i.e. ``patcher.model``) that hold the loaded
# pack. Offload has to clear these here, not on the patcher -- see
# :func:`release_wrapper_references`.
WRAPPER_MODEL_FIELDS = ("tts_model", "vc_model")


# Device types that count as "on the accelerator" for the resident-weight
# bookkeeping below. ``cuda`` is what a VRAM number means; the rest are there so
# an MPS/Intel/NPU run does not report a fully resident model as zero bytes.
_ACCELERATOR_DEVICE_TYPES = frozenset({"cuda", "mps", "xpu", "npu", "hip"})


def release_wrapper_references(patcher: Any) -> list:
    """Null ``tts_model`` / ``vc_model`` **on the wrapper**, never on the patcher.

    This is the fix for the VC-model leak: the V1 ``nodes.py:73`` wrote
    ``self.vc_model = None`` inside ``unpatch_model``, where ``self`` is the
    *patcher*. ``ModelPatcher.model`` is the wrapper those attributes actually
    live on, so the assignment created a brand-new attribute on the patcher,
    freed nothing, and the VC weights stayed resident for the rest of the
    session -- on every offload, not just the last one.

    Args:
        patcher: The patcher whose wrapper should be cleared.

    Returns:
        The list of attribute names actually nulled.
    """
    cleared: list[str] = []
    wrapper = getattr(patcher, "model", None)
    if wrapper is None:
        return cleared

    for attr in WRAPPER_MODEL_FIELDS:
        try:
            if getattr(wrapper, attr, None) is None:
                continue
            setattr(wrapper, attr, None)
            cleared.append(attr)
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] Could not null '%s' on the model wrapper: %s",
                attr,
                exc,
            )

    # The handler keeps its nn.Module children pointing at the same weight
    # objects the wrapper referenced. Clearing only the plain attributes would
    # leave those registrations holding every byte we just claimed to free.
    detach = getattr(wrapper, "detach_weights", None)
    if callable(detach):
        try:
            detach()
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] Could not detach the wrapper's weight modules: %s",
                exc,
            )
    return cleared


def _adopt_resident_weights(patcher: Any, model: Any) -> None:
    """Report to core the weight bytes this patcher already has in VRAM.

    The loader places every tensor on the load device *before* the patcher is
    built, so core's own counter still reads zero. ``partially_load`` then
    evaluates ``0 + extra_memory > model_size()`` as false, concludes the load
    is not full, and plans a partial load into whatever VRAM happens to be left.
    That is how a fully resident 2.7 GB pack gets reported as
    ``loaded partially; 1843.2 MB offloaded`` on an otherwise idle 16 GB card:
    the weights never move, only the bookkeeping is wrong -- and it is wrong in
    the direction that makes a *second* model look like there is room for it.

    Counting the bytes genuinely resident on an accelerator lets core's own
    guard do the right thing: ``partially_load`` returns 0 and leaves the
    resident model alone.
    """
    try:
        tensors = chain(model.parameters(), model.buffers())
    except Exception:
        return

    resident = 0
    on_accelerator = False
    try:
        for tensor in tensors:
            if tensor.is_meta:
                return  # nothing materialised yet; leave core's counter at zero
            if tensor.device.type in _ACCELERATOR_DEVICE_TYPES:
                resident += tensor.numel() * tensor.element_size()
                on_accelerator = True
    except Exception as exc:
        logging.debug(
            "[Chatterbox TTS] Resident-weight probe failed for patcher %s: %s",
            getattr(patcher, "cache_key", "<unknown>"),
            exc,
        )
        return

    if on_accelerator and resident > 0:
        patcher.model.model_loaded_weight_memory = resident


class ChatterboxPatcher(comfy.model_patcher.ModelPatcher):
    """A ``ModelPatcher`` for a Chatterbox voice pack.

    Loads the pack on first use, keeps ComfyUI's weight accounting honest, and
    implements the three-way offload contract:

    =======  ========  ==========================================================
    ``warm`` ``destroy`` Behaviour
    =======  ========  ==========================================================
    ``False`` ``False`` **Cold offload** (what ComfyUI asks for): the pack stays
                       loaded in RAM and is moved to ``device_to``. Next load is
                       a device move, not a disk read.
    ``True``  ``False`` **Warm offload**: the pack stays loaded *and* its
                       tensors are kept on the intermediate device, so re-loading
                       is the cheapest of the three.
    ``False`` ``True``  **Destroy**: every reference is dropped -- wrapper
                       attributes, submodule registrations, the cache entry --
                       and the CUDA cache is emptied. This is the eviction path.
    =======  ========  ==========================================================

    ``warm`` and ``destroy`` together are treated as a destroy: a caller that
    asked for the weights gone is not satisfied by tensors parked on the
    intermediate device.
    """

    def __init__(
        self,
        model: Any,
        *args: Any,
        attention_mode: str = "eager",
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        # ``attention_mode`` / ``dtype`` are keyword-only on purpose. Core's
        # ``ModelPatcher.clone()`` reconstructs the class **positionally** --
        # ``class_(model, self.load_device, self.offload_device, self.model_size(),
        # weight_inplace_update=..., fast_disk=...)`` at
        # ``comfy/model_patcher.py:446`` -- so a parameter declared before
        # ``*args`` would silently bind ``load_device`` to ``attention_mode``
        # and then die inside ``ModelPatcher.__init__`` with "missing 1
        # required positional argument: 'offload_device'". ``*args`` is the
        # only spelling that survives a positional reconstruction.
        super().__init__(model, *args, **kwargs)

        self.attention_mode = attention_mode
        self.target_dtype = dtype
        self.cache_key = getattr(model, "cache_key", "chatterbox_unknown")
        # Set by the warm offload path so ``is_loaded`` reports "not ready for
        # inference" while the tensors sit on the intermediate device.
        self._warm_offloaded = False
        # The websocket a progress toast should go to. Per-call, never part of
        # the cache key -- see :data:`CHATTERBOX_PATCHER_CACHE`.
        self.client_id: Optional[str] = None

        _adopt_resident_weights(self, model)

    # -- state ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the pack is loaded *and* on the load device.

        A pack that ComfyUI has offloaded to RAM is "loaded" in the sense that
        its weights exist, but not in the sense that it can be run: answering
        ``True`` for a CPU-resident model is how a caller ends up invoking
        ``generate`` on something that has not been moved to the GPU yet.
        """
        wrapper = getattr(self, "model", None)
        if wrapper is None:
            return False

        has_pack = any(
            getattr(wrapper, attr, None) is not None for attr in WRAPPER_MODEL_FIELDS
        )
        if not has_pack:
            return False
        if getattr(self, "_warm_offloaded", False):
            return False

        try:
            tensors = chain(wrapper.parameters(), wrapper.buffers())
            for tensor in tensors:
                device = tensor.device
                if not isinstance(device, torch.device):
                    # Undecidable (a test double, an exotic tensor subclass):
                    # the references exist, so treat it as loaded rather than
                    # claiming a load that is not there.
                    return True
                if device != self.load_device:
                    return False
        except Exception:
            return True
        return True

    # -- loading -------------------------------------------------------

    def patch_model(
        self,
        device_to: Optional[torch.device] = None,
        lowvram_model_memory: int = 0,
        load_weights: bool = True,
        force_patch_weights: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load the pack if needed, cast it, then defer to core.

        The lazy load has to happen *before* core's ``patch_model``: core moves
        and measures the modules it is handed, and a handler whose pack is
        still ``None`` has nothing to move.

        ``self.client_id`` is forwarded into that load: a first run is the run
        that downloads the pack, and the download progress has to reach the tab
        that asked for it. The id is read **here, per call**, because a cached
        patcher is shared -- ``modules.generation.load_chatterbox_model``
        re-stamps it with the current caller before every load, so a second tab
        that re-uses loaded weights still gets its own progress events.
        """
        self._warm_offloaded = False
        target_device = self.load_device if device_to is None else device_to
        wrapper = self.model

        if wrapper is None:
            raise RuntimeError(
                "ChatterboxPatcher has no model wrapper; it cannot be loaded."
            )

        if all(getattr(wrapper, attr, None) is None for attr in WRAPPER_MODEL_FIELDS):
            logging.info(
                "[Chatterbox TTS] Loading voice pack '%s' to %s (attention: %s)...",
                getattr(wrapper, "model_pack_name", "<unknown>"),
                target_device,
                self.attention_mode,
            )
            wrapper.load_model(
                target_device,
                attention_mode=self.attention_mode,
                client_id=self.client_id,
            )
            # Core's counter is refreshed from the refined size: the on-disk
            # estimate this patcher was built with is a guess about VRAM, the
            # measured parameter count is not.
            self.size = getattr(wrapper, "size", self.size) or self.size
        else:
            logging.debug(
                "[Chatterbox TTS] Voice pack '%s' is already in memory.",
                getattr(wrapper, "model_pack_name", "<unknown>"),
            )

        if self.target_dtype is not None:
            for attr in WRAPPER_MODEL_FIELDS:
                weights = getattr(wrapper, attr, None)
                if weights is None:
                    continue
                current = representative_dtype(weights)
                if current == self.target_dtype:
                    continue
                logging.debug(
                    "[Chatterbox TTS] Casting '%s' (%s) to %s (currently %s)",
                    attr,
                    getattr(wrapper, "model_pack_name", "<unknown>"),
                    self.target_dtype,
                    current,
                )
                cast_model_to_dtype(weights, self.target_dtype)

        return super().patch_model(
            device_to=target_device,
            lowvram_model_memory=lowvram_model_memory,
            load_weights=load_weights,
            force_patch_weights=force_patch_weights,
            *args,
            **kwargs,
        )

    # -- unloading -----------------------------------------------------

    def unpatch_model(
        self,
        device_to: Optional[torch.device] = None,
        unpatch_weights: bool = True,
        warm: bool = False,
        destroy: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Offload the pack: warm, cold or destroyed.

        See the class docstring for the contract table. The destroy path is the
        only one that drops references, and it drops them **on the wrapper** --
        see :func:`release_wrapper_references` for why that distinction is the
        whole point of D3.
        """
        if unpatch_weights and warm and not destroy:
            self._warm_offload(device_to)
            return super().unpatch_model(device_to, unpatch_weights=False, *args, **kwargs)

        if unpatch_weights and destroy:
            self._destroy(device_to)
            return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

        self._warm_offloaded = False
        if unpatch_weights:
            logging.debug(
                "[Chatterbox TTS] Offloading voice pack '%s' to %s (weights kept "
                "in RAM)...",
                getattr(self.model, "model_pack_name", "<unknown>"),
                device_to,
            )
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

    def _warm_offload(self, device_to: Optional[torch.device]) -> None:
        """Park the tensors on the intermediate device, keeping every reference."""
        offload_target = device_to
        try:
            offload_target = _get_model_management().intermediate_device()
        except Exception:
            offload_target = self.offload_device

        try:
            if self.model is not None:
                move = getattr(self.model, "to_device", None)
                if callable(move):
                    move(offload_target)
                else:
                    self.model.to(offload_target)
                    self.model.device = offload_target
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] Warm offload to %s failed (%s); falling back to a "
                "cold offload.",
                offload_target,
                exc,
            )
            self._warm_offloaded = False
            return

        self._warm_offloaded = True
        logging.debug(
            "[Chatterbox TTS] Warm offloaded voice pack '%s' to %s (tensors "
            "retained)...",
            getattr(self.model, "model_pack_name", "<unknown>"),
            offload_target,
        )

    def _destroy(self, device_to: Optional[torch.device]) -> None:
        """Drop every reference to the pack and empty the CUDA cache."""
        logging.debug(
            "[Chatterbox TTS] Destroying voice pack '%s' (weights freed)...",
            getattr(self.model, "model_pack_name", "<unknown>"),
        )

        try:
            from .model_registry import unregister_from_comfy

            unregister_from_comfy(self)
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] Could not unregister patcher '%s' from ComfyUI: %s",
                self.cache_key,
                exc,
            )

        release_wrapper_references(self)

        if self.cache_key in CHATTERBOX_PATCHER_CACHE:
            CHATTERBOX_PATCHER_CACHE.pop(self.cache_key, None)

        gc.collect()
        try:
            _get_model_management().soft_empty_cache()
        except Exception as exc:
            logging.debug("[Chatterbox TTS] soft_empty_cache failed: %s", exc)


# ---------------------------------------------------------------------------
# Patcher-class selection
# ---------------------------------------------------------------------------


def resolve_core_patcher_class() -> type:
    """The class ``comfy.model_patcher.CoreModelPatcher`` currently names."""
    return getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )


_PROBE_SENTINEL = object()


def dynamic_vram_available(patcher_cls: Any, load_device: Any) -> bool:
    """Whether ``patcher_cls`` is a real dynamic patcher aimed at CUDA.

    Used to decide that the *legacy* class is right, so the answer has to be a
    real ``False`` rather than an absence: a probe that raises would otherwise
    push the caller into a branch it cannot take.
    """
    try:
        device = torch.device(load_device)
    except (TypeError, ValueError, RuntimeError) as exc:
        logging.debug(
            "[Chatterbox TTS] Dynamic-VRAM probe: unusable load_device %r (%s)",
            load_device,
            exc,
        )
        return False

    if device.type != "cuda":
        return False

    is_dynamic = getattr(patcher_cls, "is_dynamic", None)
    if not callable(is_dynamic):
        return False

    try:
        return bool(is_dynamic(_PROBE_SENTINEL))
    except Exception as exc:
        logging.debug(
            "[Chatterbox TTS] Dynamic-VRAM probe: is_dynamic() failed on %s (%s)",
            patcher_cls,
            exc,
        )
        return False


def select_patcher_class(
    weight_family: Any,
    load_device: Any,
    legacy_cls: type = ChatterboxPatcher,
) -> type:
    """Pick the patcher class that should own ``weight_family``'s lifecycle.

    Always the legacy class. A Chatterbox pack is loaded in one pass onto the
    load device and stays there until something asks for it back; the
    demand-paged path would stream the same weights per step for no benefit, and
    upstream does not select it either. ``weight_family`` and ``load_device``
    are accepted (and probed through :func:`dynamic_vram_available`) so a
    future dynamic backend has one obvious place to branch from.
    """
    if dynamic_vram_available(legacy_cls, load_device):
        logging.debug(
            "[Chatterbox TTS] %s reports dynamic-VRAM support; Chatterbox still "
            "loads its packs in full (weight_family=%r).",
            getattr(legacy_cls, "__name__", legacy_cls),
            weight_family,
        )
    return legacy_cls


def _ensure_device_index(patcher: Any) -> None:
    """Give an accelerator load device the explicit index core expects."""
    device = getattr(patcher, "load_device", None)
    if device is None or getattr(device, "index", 0) is not None:
        return
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return
    patcher.load_device = torch.device("cuda", torch.cuda.current_device())


def load_to_device(patcher: Any, memory_required: int = 0) -> Any:
    """Put ``patcher`` on its load device under ComfyUI's memory orchestration.

    When the pack fits into free VRAM *with room left for activations*, it takes
    the native full-load shortcut: ``force_full_load=True``, one pass, no
    per-module paging. Otherwise it defers to ``mm.load_model_gpu``, which may
    keep part of the weights on the CPU.

    The 1.5 GB activation headroom is not decoration. A model that exactly fits
    passes ``lowvram_model_memory=0`` -- core's own "full load" branch -- and
    then has nowhere to put the transient tensors of a single sampling step,
    which surfaces as a CUDA OOM on a card that reported the model as fitting.

    Args:
        patcher: The patcher to place.
        memory_required: Extra bytes the *prompt* needs alongside the weights.

    Returns:
        The same patcher, for call-site chaining.
    """
    _ensure_device_index(patcher)
    model_management = _get_model_management()

    is_dynamic = getattr(patcher, "is_dynamic", None)
    if callable(is_dynamic) and is_dynamic() is True:
        model_management.load_models_gpu(
            [patcher], memory_required=memory_required
        )
        return patcher

    device = getattr(patcher, "load_device", None)
    if device is not None and getattr(device, "type", None) == "cuda":
        free_vram = model_management.get_free_memory(device)
        model_size = patcher.model_size()
        if model_size < (free_vram - ACTIVATION_HEADROOM_BYTES):
            patcher.patch_model(
                device_to=device, lowvram_model_memory=0, load_weights=True
            )
            model_management.load_models_gpu([patcher], force_full_load=True)
            return patcher

    model_management.load_model_gpu(patcher)
    return patcher