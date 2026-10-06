"""ComfyUI folder-registration helpers for Chatterbox TTS model assets.

``folder_paths`` arrives as an **injected parameter** (``folder_paths_module``)
rather than being imported here. That is what makes registration unit-testable
without booting ComfyUI, and it is why this module has no ``comfy`` /
``folder_paths`` import at all -- not even lazily.

Two rules shape everything below:

* **Register every TTS root, not the first one.** A machine resolves models
  across several roots at once -- the primary ``<models>/tts`` plus whatever
  ``extra_model_paths.yaml`` contributes. A ``chatterbox`` sub-directory has to
  be registered under *all* of them, otherwise a model installed under a
  secondary root is invisible even though the file is right there.
* **Never clobber another node's registration.** ``tts`` is a shared key: if
  another TTS node already registered it, its path list and its extension set
  are extended, never replaced.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

__all__ = [
    "TTS_FOLDER_KEY",
    "CHATTERBOX_FOLDER_KEY",
    "CHATTERBOX_SUBDIR",
    "CHATTERBOX_EXTENSIONS",
    "register_chatterbox_folders",
    "get_registered_tts_roots",
    "ensure_chatterbox_directories",
    "ensure_directory",
]


# Shared ComfyUI folder key. Not ours alone -- every TTS node may claim it.
TTS_FOLDER_KEY = "tts"

# Chatterbox-specific sub-key, registered *under every* TTS root.
CHATTERBOX_FOLDER_KEY = "chatterbox"
CHATTERBOX_SUBDIR = "chatterbox"

# Extensions ComfyUI's model browser uses to filter a folder listing.
CHATTERBOX_EXTENSIONS = frozenset(
    {".safetensors", ".pt", ".bin", ".ckpt", ".json", ".yaml", ".yml"}
)


def _normalize(path: str) -> str:
    """Absolute, normalised form of ``path`` -- the identity used for dedup."""
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _append_unique_path(paths: list[str], candidate: str) -> bool:
    """Append a normalised path once, preserving registration order.

    Returns ``True`` when the path was actually appended. ``paths`` is mutated
    in place because it is the list object ComfyUI's ``folder_names_and_paths``
    already holds; rebinding a new list would silently detach it.
    """
    if not candidate:
        return False
    normalized = _normalize(candidate)
    existing = {_normalize(path) for path in paths}
    if normalized in existing:
        return False
    paths.append(normalized)
    return True


def _ensure_entry(
    folder_names_and_paths: dict, key: str, extensions: Iterable[str]
) -> list[str]:
    """Return the path list for ``key``, creating the entry only if absent.

    An entry another node already created is returned untouched apart from the
    extensions being unioned in: replacing the tuple would drop that node's
    paths.
    """
    entry = folder_names_and_paths.get(key)
    if entry is None:
        paths: list[str] = []
        folder_names_and_paths[key] = (paths, set(extensions))
        return paths

    if not isinstance(entry, (tuple, list)) or len(entry) < 1:
        # Defensive: a foreign entry of an unexpected shape is replaced rather
        # than trusted, because a non-list first element cannot be appended to.
        paths = []
        folder_names_and_paths[key] = (paths, set(extensions))
        return paths

    paths = entry[0]
    if not isinstance(paths, list):
        paths = list(paths)
        folder_names_and_paths[key] = (paths, entry[1])

    if len(entry) > 1 and isinstance(entry[1], set):
        entry[1].update(extensions)
    return paths


def _supported_pt_extensions(folder_paths_module: Any) -> set:
    """ComfyUI's own weight extensions, or an empty set on an unfamiliar core."""
    return set(getattr(folder_paths_module, "supported_pt_extensions", ()) or ())


def register_chatterbox_folders(folder_paths_module: Any) -> list[str]:
    """Register the ``tts`` and ``chatterbox`` folder keys; return the TTS roots.

    Idempotent: ComfyUI re-imports a custom node on every startup, and a second
    call adds no duplicate path.

    Args:
        folder_paths_module: An object exposing ``models_dir``,
            ``folder_names_and_paths`` and (optionally)
            ``supported_pt_extensions``. The real ``folder_paths`` module in
            production; a fake in tests.

    Returns:
        Every registered TTS root, in registration order.
    """
    folder_names_and_paths = folder_paths_module.folder_names_and_paths
    models_dir = os.fspath(folder_paths_module.models_dir)

    tts_roots = _ensure_entry(
        folder_names_and_paths,
        TTS_FOLDER_KEY,
        _supported_pt_extensions(folder_paths_module) | CHATTERBOX_EXTENSIONS,
    )

    # Our own root goes last so the roots another node registered keep their
    # indices -- the multi-root case is the normal case, not the exception.
    _append_unique_path(tts_roots, os.path.join(models_dir, TTS_FOLDER_KEY))

    chatterbox_roots = _ensure_entry(
        folder_names_and_paths,
        CHATTERBOX_FOLDER_KEY,
        _supported_pt_extensions(folder_paths_module) | CHATTERBOX_EXTENSIONS,
    )
    for root in list(tts_roots):
        _append_unique_path(chatterbox_roots, os.path.join(root, CHATTERBOX_SUBDIR))

    logging.debug(
        "[Chatterbox TTS] folder_registration: %d TTS root(s) registered: %s",
        len(tts_roots),
        ", ".join(tts_roots),
    )
    return list(tts_roots)


def get_registered_tts_roots(folder_paths_module: Any) -> list[str]:
    """Return every TTS root ComfyUI currently knows about, normalised."""
    entry = folder_paths_module.folder_names_and_paths.get(TTS_FOLDER_KEY)
    if not entry:
        return []
    try:
        return [_normalize(path) for path in entry[0]]
    except (TypeError, IndexError):
        return []


def ensure_directory(path: str) -> bool:
    """``os.makedirs(path, exist_ok=True)``, reporting failure instead of raising.

    Directory creation is a convenience for the download path. A read-only or
    otherwise unwritable ``models`` tree must not take the whole node down at
    startup, so the failure is logged and ``False`` is returned.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logging.warning(
            "[Chatterbox TTS] folder_registration: could not create '%s': %s", path, exc
        )
        return False
    return True


def ensure_chatterbox_directories(tts_roots: Iterable[str]) -> list[str]:
    """Create ``<tts root>/chatterbox`` for each root; return what succeeded.

    Split out of :func:`register_chatterbox_folders` on purpose: registration
    only ever mutates ComfyUI's table, while this writes to disk and therefore
    belongs to the download/generation path (batch B8), not to node import.
    """
    created = []
    for root in tts_roots:
        if not root:
            continue
        target = os.path.join(os.fspath(root), CHATTERBOX_SUBDIR)
        if ensure_directory(target):
            created.append(_normalize(target))
    return created