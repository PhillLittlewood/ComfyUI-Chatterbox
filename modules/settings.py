"""Chatterbox user settings, read from ComfyUI's ``comfy.settings.json``.

The frontend writes settings via ``app.api.storeSetting("chatterbox.<key>",
value)``, which ComfyUI persists to ``user/default/comfy.settings.json``. This
module is the backend read side of that pair: it loads the file, keeps only the
``chatterbox.`` keys, and derives the effective model path from them.

Ported from ``ComfyUI-VoxCPM/modules/settings.py`` (plan S4.3) with two changes:
the prefix is ``chatterbox.`` instead of ``voxcpm.``, and ``folder_paths`` is
imported lazily inside ``try``/``except ImportError`` so the module imports
standalone -- without ComfyUI on ``sys.path`` at all.

This module is **read-only**. The frontend is the single write path; the backend
never edits ``comfy.settings.json``.

Settings keys (all prefixed with ``chatterbox.``):
- ``chatterbox.use_custom_path`` (bool): whether to use a custom model path
- ``chatterbox.custom_model_path`` (str): the custom model directory
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from .constants import CHATTERBOX_MODEL_SUBDIR, DEFAULT_MODEL_PACK_NAME


__all__ = [
    "SETTINGS_PREFIX",
    "SETTINGS_FILE",
    "ChatterboxSettings",
    "get_settings",
]

# Legacy settings file name, kept only so a future migration can clean it up.
SETTINGS_FILE = "chatterbox_settings.json"

# Settings key prefix in comfy.settings.json.
SETTINGS_PREFIX = "chatterbox."

# Subdirectory inside a custom path that holds the Chatterbox packs, mirroring
# the layout under ``models/tts/chatterbox``.
_CHATTERBOX_SUBDIR_NAME = os.path.basename(CHATTERBOX_MODEL_SUBDIR)


class ChatterboxSettings:
    """Reads ``chatterbox.*`` settings out of ComfyUI's settings file.

    Constructing an instance loads the file once; :meth:`reload` re-reads it.
    """

    def __init__(self):
        self._settings: Dict[str, Any] = {}
        self._load_settings()

    # ── Loading ────────────────────────────────────────────────────

    @staticmethod
    def _get_comfy_settings_path() -> Optional[str]:
        """Path of ``comfy.settings.json`` inside the ComfyUI user directory.

        Returns ``None`` when ``folder_paths`` is unavailable or the user
        directory cannot be determined -- both are legitimate states outside a
        running ComfyUI, and neither is an error.
        """
        try:
            import folder_paths

            user_dir = folder_paths.get_user_directory()
            if user_dir:
                return os.path.join(user_dir, "default", "comfy.settings.json")
        except ImportError:
            logging.debug(
                "[Chatterbox TTS] folder_paths is not importable; Chatterbox "
                "settings fall back to their defaults"
            )
        except Exception as exc:  # noqa: BLE001 -- a broken ComfyUI must not break us
            logging.debug("[Chatterbox TTS] Could not resolve the ComfyUI user directory: %s", exc)
        return None

    def _load_settings(self) -> None:
        """Load and prefix-filter the settings file, degrading to ``{}``."""
        try:
            settings_path = self._get_comfy_settings_path()
            if settings_path and os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as handle:
                    all_settings = json.load(handle)
                self._settings = {
                    key: value
                    for key, value in all_settings.items()
                    if key.startswith(SETTINGS_PREFIX)
                }
                logging.debug(
                    "[Chatterbox TTS] Loaded Chatterbox settings from comfy.settings.json: %s",
                    sorted(self._settings),
                )
            else:
                self._settings = {}
        except Exception as exc:  # noqa: BLE001 -- corrupt JSON must not be fatal
            logging.warning(
                "[Chatterbox TTS] Failed to load settings from comfy.settings.json: %s",
                exc,
            )
            self._settings = {}

    def reload(self) -> None:
        """Re-read the settings file.

        Call after the frontend writes a setting through ``storeSetting``, so
        the backend sees the new value without a restart.
        """
        self._load_settings()

    # ── Settings accessors ─────────────────────────────────────────

    @property
    def use_custom_path(self) -> bool:
        """Whether the user pointed Chatterbox at a directory of their own."""
        return bool(self._settings.get(f"{SETTINGS_PREFIX}use_custom_path", False))

    @property
    def custom_model_path(self) -> Optional[str]:
        """The custom model directory, or ``None`` when unset."""
        return self._settings.get(f"{SETTINGS_PREFIX}custom_model_path")

    # ── Path resolution ─────────────────────────────────────────────

    @staticmethod
    def _default_models_root() -> str:
        """``<models>/tts/chatterbox`` without needing ComfyUI at all.

        Two separate fallbacks, tried in order: the registered ``tts`` search
        path, then ``folder_paths.models_dir``, then a path relative to the
        working directory. They are separate ``try`` blocks on purpose -- a
        ComfyUI that has never seen a ``tts`` folder raises when the search
        path is asked for, and that must not also cost us ``models_dir``.
        """
        try:
            import folder_paths
        except ImportError:
            logging.debug(
                "[Chatterbox TTS] folder_paths is not importable; assuming a relative models root"
            )
            return os.path.join("models", CHATTERBOX_MODEL_SUBDIR)

        try:
            tts_paths = folder_paths.get_folder_paths("tts")
        except Exception as exc:  # noqa: BLE001 -- an unregistered folder is normal
            logging.debug("[Chatterbox TTS] No 'tts' folder registered with ComfyUI: %s", exc)
            tts_paths = None
        if tts_paths:
            return os.path.join(tts_paths[0], _CHATTERBOX_SUBDIR_NAME)

        try:
            return os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
        except Exception as exc:  # noqa: BLE001 -- a broken ComfyUI must not break us
            logging.debug("[Chatterbox TTS] Could not resolve models_dir: %s", exc)

        return os.path.join("models", CHATTERBOX_MODEL_SUBDIR)

    def get_effective_model_path(self) -> str:
        """The directory packs are actually read from."""
        if self.use_custom_path and self.custom_model_path:
            return self.custom_model_path
        return self._default_models_root()

    @staticmethod
    def _validate_model_path(path: str) -> bool:
        """True when ``path`` holds at least one usable Chatterbox pack.

        A pack is a directory carrying every file in
        ``CHATTERBOX_FILES_TO_DOWNLOAD`` -- or, for the wider Chatterbox family
        (Turbo, Nano, multilingual), the weights it needs. Kept deliberately
        permissive about *which* weights: the download stack fills the
        directory, and a partially downloaded pack is still worth reporting.
        """
        if not path or not os.path.isdir(path):
            return False

        # The path may point straight at a pack rather than at its parent.
        if ChatterboxSettings._looks_like_pack(path):
            return True

        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and ChatterboxSettings._looks_like_pack(item_path):
                    return True
        except OSError as exc:
            logging.debug("[Chatterbox TTS] Could not list '%s': %s", path, exc)
            return False
        return False

    @staticmethod
    def _looks_like_pack(path: str) -> bool:
        """True when ``path`` carries Chatterbox weights or a tokenizer."""
        try:
            entries = set(os.listdir(path))
        except OSError:
            return False
        if not entries:
            return False
        if "tokenizer.json" in entries:
            return True
        return any(
            entry.endswith(".safetensors") or entry.endswith(".bin") or entry.endswith(".pt")
            for entry in entries
        )

    def is_effective_path_valid(self) -> bool:
        """Whether the effective model path holds at least one usable pack."""
        return self._validate_model_path(self.get_effective_model_path())

    def to_dict(self) -> Dict[str, Any]:
        """The shape broadcast to the frontend as ``config["settings"]``."""
        return {
            "use_custom_path": self.use_custom_path,
            "custom_model_path": self.custom_model_path,
            "effective_path": self.get_effective_model_path(),
            "effective_path_valid": self.is_effective_path_valid(),
            "default_model_pack_name": DEFAULT_MODEL_PACK_NAME,
        }


# The process-wide settings instance, built on first use.
_settings_instance: Optional[ChatterboxSettings] = None


def get_settings() -> ChatterboxSettings:
    """The lazy settings singleton.

    Lazy so importing this module never touches the filesystem, and so a test
    can point ``folder_paths`` at a temporary directory before the first call.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = ChatterboxSettings()
    return _settings_instance