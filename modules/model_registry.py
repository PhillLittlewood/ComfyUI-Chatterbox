"""Active-model registry: unload-before-load eviction on model change.

Single source of truth for which Chatterbox model is active per family, and for
fully releasing a superseded model -- RAM, VRAM, and ComfyUI's own
``model_management.current_loaded_models`` registry -- **before any allocation
for a replacement begins**. Evicting after the fact is what produces
CUDA-OOM on a machine that could have held either model but never both.

Design notes:

* **Zero ComfyUI imports at module scope.** ``comfy.model_management`` is
  imported inside the two functions that need it, so this module can be
  imported (and tested) with ComfyUI absent.
* **Single-active-bundle semantics.** At most one live bundle per cache key.

Borrowed from VibeVoice's ``modules/model_registry.py``; see the plan's §3.2 for
what changed on the way in.
"""

import gc
import logging
import os

__all__ = [
    "FAMILY_TTS",
    "identity_for_external",
    "unregister_from_comfy",
    "evict_patcher",
    "evict_if_changed",
    "register_model_bundle",
    "release_model_bundles",
    "get_live_bundle",
    "clear_bundle_registry",
    "set_active",
    "get_active",
    "clear_active_keys",
]


# Model families tracked by set_active() / evict_if_changed().
FAMILY_TTS = "tts"

# family -> cache_key of the currently active model.
_ACTIVE_KEYS: dict = {}

# Fields of a model bundle that hold heavy references.
_BUNDLE_HEAVY_FIELDS = ("model", "processor")

# patcher cache_key -> the live model bundle registered for that key.
_BUNDLE_REGISTRY: dict = {}


# ---------------------------------------------------------------------------
# Bundle registry
# ---------------------------------------------------------------------------


def _neutralize_bundle(bundle) -> None:
    """Null a bundle's heavy fields, per-field guarded, never raising."""
    if bundle is None:
        return
    for field in _BUNDLE_HEAVY_FIELDS:
        try:
            bundle[field] = None
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] model_registry: could not null bundle field "
                "'%s': %s",
                field,
                exc,
            )


def register_model_bundle(cache_key: str, bundle) -> None:
    """Record the live model bundle for ``cache_key``."""
    try:
        previous = _BUNDLE_REGISTRY.get(cache_key)
        if previous is not None and previous is not bundle:
            _neutralize_bundle(previous)
        _BUNDLE_REGISTRY[cache_key] = bundle
    except Exception as exc:
        logging.warning(
            "[Chatterbox TTS] model_registry.register_model_bundle(%r) failed: %s",
            cache_key,
            exc,
        )


def release_model_bundles(cache_key: str) -> int:
    """Neutralize the bundle registered under ``cache_key`` and drop it.

    Returns:
        ``1`` when a bundle was released, ``0`` when there was none.
    """
    try:
        bundle = _BUNDLE_REGISTRY.pop(cache_key, None)
    except Exception as exc:
        logging.warning(
            "[Chatterbox TTS] model_registry.release_model_bundles(%r) failed: %s",
            cache_key,
            exc,
        )
        return 0
    if bundle is None:
        return 0
    _neutralize_bundle(bundle)
    return 1


def get_live_bundle(cache_key: str):
    """The bundle registered under ``cache_key``, or ``None`` when unusable.

    A bundle whose ``model`` has been neutralized counts as unusable: handing it
    back would produce a confusing "None has no attribute generate" instead of a
    clean reload.
    """
    try:
        bundle = _BUNDLE_REGISTRY.get(cache_key)
    except Exception as exc:
        logging.warning(
            "[Chatterbox TTS] model_registry.get_live_bundle(%r) failed: %s",
            cache_key,
            exc,
        )
        return None
    if bundle is None:
        return None
    try:
        if bundle.get("model") is None:
            return None
    except Exception:
        return None
    return bundle


def clear_bundle_registry() -> None:
    """Forget every bundle registration (test/teardown hook)."""
    _BUNDLE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Active-key bookkeeping
# ---------------------------------------------------------------------------


def set_active(family: str, cache_key: str) -> None:
    """Record ``cache_key`` as the active model for ``family``."""
    _ACTIVE_KEYS[family] = cache_key


def get_active(family: str) -> "str | None":
    """The active cache key for ``family``, or ``None``."""
    return _ACTIVE_KEYS.get(family)


def clear_active_keys() -> None:
    """Forget every active-key record (test/teardown hook)."""
    _ACTIVE_KEYS.clear()


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def identity_for_external(
    weight_path: str,
    config_name: str,
    attention_mode: str,
    use_llm_4bit: bool = False,
    dtype_str: str = "auto",
    prefix: str = "chatterbox",
) -> str:
    """Build a file-identity-aware cache key for an externally-loaded model.

    The ``basename@mtime_ns@size`` triple is what stops a re-downloaded model
    from being served out of a stale cache entry: same filename, new bytes, new
    key. A path alone is not enough (the file did not "change name"), and the
    name alone is worse (it never changes).

    A missing or unreadable file degrades to ``@0@0`` rather than raising, so a
    key can still be built for a path ComfyUI has not materialised yet.
    """
    basename = os.path.basename(weight_path) if weight_path else ""
    mtime_ns = "0"
    size = "0"
    try:
        if weight_path and os.path.isfile(weight_path):
            stat_result = os.stat(weight_path)
            mtime_ns = str(stat_result.st_mtime_ns)
            size = str(stat_result.st_size)
    except OSError as exc:
        logging.debug(
            "[Chatterbox TTS] model_registry: could not stat '%s' for the cache "
            "key (%s); falling back to a name-only identity",
            weight_path,
            exc,
        )

    return (
        f"{prefix}_{config_name}"
        f"@{basename}@{mtime_ns}@{size}"
        f"_attn_{attention_mode}"
        f"_q4_{int(bool(use_llm_4bit))}"
        f"_dtype_{dtype_str}"
    )


# ---------------------------------------------------------------------------
# ComfyUI registry + destructive eviction
# ---------------------------------------------------------------------------


def unregister_from_comfy(patcher) -> list:
    """Remove ``patcher``'s entries from ComfyUI's loaded-model registry.

    ComfyUI's ``current_loaded_models`` holds strong references *and* holds
    finalizers that free the weights. Leaving an entry behind after an explicit
    eviction means the replacement model allocates into VRAM that the "freed"
    model still owns.

    Returns:
        The list of removed registry entries (empty when there were none, or
        when ``comfy.model_management`` cannot be imported).
    """
    removed = []
    try:
        import comfy.model_management as model_management
    except Exception as exc:
        logging.warning(
            "[Chatterbox TTS] model_registry.unregister_from_comfy: cannot import "
            "comfy.model_management: %s",
            exc,
        )
        return removed

    loaded_list = getattr(model_management, "current_loaded_models", None)
    if loaded_list is None:
        return removed

    survivors = []
    for loaded in list(loaded_list):
        try:
            if loaded.model is not patcher:
                survivors.append(loaded)
                continue
        except Exception:
            # An entry we cannot even interrogate is somebody else's problem.
            survivors.append(loaded)
            continue

        for attr in ("model_finalizer", "_patcher_finalizer"):
            finalizer = getattr(loaded, attr, None)
            if finalizer is not None:
                try:
                    finalizer.detach()
                except Exception:
                    pass
                try:
                    setattr(loaded, attr, None)
                except Exception:
                    pass
        try:
            loaded.real_model = None
        except Exception:
            pass
        removed.append(loaded)

    if removed:
        loaded_list[:] = survivors
        logging.debug(
            "[Chatterbox TTS] Unregistered %d ComfyUI loaded-model entry/entries "
            "for patcher %s",
            len(removed),
            getattr(patcher, "cache_key", "<unknown>"),
        )
    return removed


def _release_wrapper_references(patcher) -> list:
    """Null the handler's model attributes **on the wrapper**.

    ``ModelPatcher.model`` is the wrapped handler; ``tts_model`` / ``vc_model``
    live on *that* object. Nulling the same-named attribute on the patcher
    instead is the defect this fixes: the patcher's attribute was never the one
    holding the weights, so the real model stayed reachable through the wrapper
    and the eviction freed nothing. (Batch B8's ``patcher.unpatch_model`` does
    the same on its own destroy path; doing it here as well means the registry
    cannot be defeated by a patcher that skipped it.)

    Returns:
        The list of attribute names actually nulled.
    """
    cleared = []
    wrapper = getattr(patcher, "model", None)
    if wrapper is None:
        return cleared

    for attr in ("tts_model", "vc_model"):
        try:
            if not hasattr(wrapper, attr):
                continue
            setattr(wrapper, attr, None)
            cleared.append(attr)
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] model_registry: could not null wrapper "
                "attribute '%s': %s",
                attr,
                exc,
            )
    return cleared


def evict_patcher(patcher, cache_dict: dict, key: str) -> list:
    """Atomically and destructively evict one patcher.

    Every step is attempted even after an earlier one fails, and every failure
    is collected and reported: a half-evicted model that raises out of the
    middle is worse than one that logs what it could not release.

    Returns:
        A list of human-readable failure descriptions (empty on full success).
    """
    errors = []

    try:
        unregister_from_comfy(patcher)
    except Exception as exc:
        errors.append(f"unregister_from_comfy failed: {exc}")

    try:
        _release_wrapper_references(patcher)
    except Exception as exc:
        errors.append(f"wrapper reference release failed: {exc}")

    try:
        patcher.unpatch_model(unpatch_weights=True, destroy=True)
    except Exception as exc:
        errors.append(f"unpatch_model(destroy=True) failed: {exc}")

    try:
        if cache_dict is not None:
            cache_dict.pop(key, None)
    except Exception as exc:
        errors.append(f"cache pop failed: {exc}")

    try:
        release_model_bundles(key)
    except Exception as exc:
        errors.append(f"release_model_bundles failed: {exc}")

    try:
        gc.collect()
    except Exception as exc:
        errors.append(f"gc.collect failed: {exc}")

    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception as exc:
        errors.append(f"soft_empty_cache failed: {exc}")

    for err in errors:
        logging.warning("[Chatterbox TTS] model_registry.evict_patcher(%s): %s", key, err)
    return errors


def evict_if_changed(family: str, new_key: str, patcher_caches) -> list:
    """Evict the previous active model of ``family`` when the key changed.

    Called *before* the replacement is allocated. A no-op when the key is
    unchanged, so a repeat run of the same workflow allocates nothing twice.

    Args:
        family: Model family, e.g. :data:`FAMILY_TTS`.
        new_key: The cache key about to be loaded.
        patcher_caches: Iterable of ``cache_key -> patcher`` dicts to sweep.

    Returns:
        The list of evicted cache keys.
    """
    active = get_active(family)
    if active == new_key:
        return []

    evicted = []
    for cache_dict in patcher_caches or ():
        if cache_dict is None:
            continue
        for key in [k for k in list(cache_dict.keys()) if k != new_key]:
            patcher = cache_dict.get(key)
            logging.debug(
                "[Chatterbox TTS] Model changed for family '%s' "
                "(active=%r -> %r); evicting '%s'...",
                family,
                active,
                new_key,
                key,
            )
            evict_patcher(patcher, cache_dict, key)
            evicted.append(key)

    set_active(family, new_key)
    return evicted