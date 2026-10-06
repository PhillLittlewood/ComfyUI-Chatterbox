"""Chatterbox model registry, filesystem discovery and feature flags.

Ported from ``ComfyUI-VoxCPM/modules/model_info.py`` (plan S4.6), keeping its
shape -- a static ``MODEL_CONFIGS`` table, an ``AVAILABLE_*`` registry with
``type``/``repo_id``/``architecture``/``sample_rate``/``size_gb`` metadata, and
``read_model_config`` / ``scan_*`` helpers -- and swapping the contents over to
the Chatterbox model family.

Two deliberate differences from the borrowed file:

* **Detection is file-based, not ``architecture``-based.** VoxCPM models carry
  a ``config.json`` with an ``architecture`` field; Chatterbox packs do not --
  a pack is a tokenizer plus safetensors blobs. So
  :func:`is_chatterbox_model_dir` looks for the tokenizer and the weights.
* **Local packs are registered, not baked in.** The registry starts with the
  official entries and gains local ones through :func:`refresh_local_models`,
  which delegates the filesystem walk to
  :mod:`modules.model_discovery` so there is exactly one cached scanner.

Nothing here imports the ComfyUI node-API package, ``comfy.*``, ``folder_paths`` or ``server``
at module scope, and no probe below ever imports the model stack: importing
``chatterbox.tts_turbo`` would drag in torch, librosa and transformers.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .constants import CHATTERBOX_FILES_TO_DOWNLOAD, CHATTERBOX_REPO_ID


__all__ = [
    "MODEL_CONFIGS",
    "AVAILABLE_CHATTERBOX_MODELS",
    "VALID_CHATTERBOX_ARCHITECTURES",
    "CHATTERBOX_WEIGHT_FILES",
    "TEXT_NORMALIZATION_AVAILABLE",
    "TURBO_AVAILABLE",
    "read_model_config",
    "describe_model_dir",
    "is_chatterbox_model_dir",
    "scan_chatterbox_models",
    "scan_custom_model_path",
    "register_local_model",
    "unregister_local_model",
    "refresh_local_models",
    "get_available_models",
    "refresh_feature_flags",
]

# ── Official model configurations ─────────────────────────────────────
#
# ``repo_id``/``architecture``/``sample_rate``/``size_gb`` are what the
# download UI renders and what the loader keys off. ``files`` is the pack's file
# list, used when downloading that specific model rather than the default pack.
#
# Sample rates are not a guess: every Chatterbox model synthesises through
# S3Gen at 24 kHz (``src/chatterbox/models/s3gen/const.py::S3GEN_SR``).

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # The pack a saved workflow names by default. Monolingual English, voice
    # cloning, the source of every other official pack.
    "resembleai_default_voice": {
        "repo_id": CHATTERBOX_REPO_ID,
        "architecture": "chatterbox_tts",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": list(CHATTERBOX_FILES_TO_DOWNLOAD),
        "description": "Original Chatterbox, English, voice cloning.",
    },
    # The multilingual family: same repo, same voice encoder, different T3
    # weights. v2 is upstream's default; v3 is the newer, more stable one.
    "T3_v1": {
        "repo_id": CHATTERBOX_REPO_ID,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_cfg.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "description": "Original single-language T3 weights, used by the multilingual loader as its v1.",
    },
    "T3_v2": {
        "repo_id": CHATTERBOX_REPO_ID,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_mtl23ls_v2.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "description": "23-language multilingual T3 (upstream default).",
    },
    "T3_v3": {
        "repo_id": CHATTERBOX_REPO_ID,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_mtl23ls_v3.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "description": "23-language multilingual T3, newer checkpoint: better similarity, fewer hallucinations.",
    },
    # Half the parameters, English only, paralinguistic tags.
    "tts_turbo": {
        "repo_id": "ResembleAI/chatterbox-turbo",
        "architecture": "chatterbox_turbo",
        "sample_rate": 24000,
        "size_gb": 1.4,
        "files": ["t3_turbo_v1.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "description": "350M English model with paralinguistic tags; lower VRAM.",
    },
    # A quarter of Turbo again; runs on CPU at realtime.
    "tts_nano": {
        "repo_id": "ResembleAI/chatterbox-nano",
        "architecture": "chatterbox_nano",
        "sample_rate": 24000,
        "size_gb": 0.5,
        "files": ["t3_nano_v1.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "description": "110M English model for CPU and tight latency budgets.",
    },
}

# Official entries, tagged so the API and the loader can tell them from local
# packs. Local entries are added by refresh_local_models().
AVAILABLE_CHATTERBOX_MODELS: Dict[str, Dict[str, Any]] = {
    name: {"type": "official", **config} for name, config in MODEL_CONFIGS.items()
}

# ── Recognised architectures ─────────────────────────────────────────

VALID_CHATTERBOX_ARCHITECTURES = (
    "chatterbox",
    "chatterbox_tts",
    "chatterbox_multilingual",
    "chatterbox_turbo",
    "chatterbox_nano",
)

# A file from this set makes a directory a model. ``conds.pt`` is deliberately
# absent: it is the smallest file in a pack and alone proves nothing.
CHATTERBOX_WEIGHT_FILES = (
    "t3_cfg.safetensors",
    "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v3.safetensors",
    "t3_turbo_v1.safetensors",
    "t3_nano_v1.safetensors",
    "ve.safetensors",
    "s3gen.safetensors",
)

# Every Chatterbox entry point needs a tokenizer to build its T3 conditions.
CHATTERBOX_TOKENIZER_FILE = "tokenizer.json"


# ── Feature flags ────────────────────────────────────────────────────

def _probe_text_normalization() -> bool:
    """True when at least one CJK segmenter is installed.

    Chatterbox does not ship a normaliser; it delegates word segmentation for
    Japanese and Chinese to optional packages. The nodes disable text
    normalisation when neither is present, and the frontend greys the widget
    out from ``config["normalization_available"]``.
    """
    for module_name in ("pykakasi", "spacy_pkuseg"):
        try:
            if importlib.util.find_spec(module_name) is not None:
                return True
        except (ImportError, ValueError, AttributeError):
            # ValueError: a parent package is mocked and has no __spec__.
            logging.debug("[Chatterbox TTS] Could not probe optional package '%s'", module_name)
    return False


def _probe_turbo() -> bool:
    """True when the vendored source ships the Turbo entry point.

    Checked on disk rather than by import: the module imports the full model
    stack, and a presence check must cost nothing at node-load time.
    """
    package_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "chatterbox",
    )
    return os.path.isfile(os.path.join(package_dir, "tts_turbo.py"))


def refresh_feature_flags() -> None:
    """Recompute the feature flags in this module's namespace."""
    global TEXT_NORMALIZATION_AVAILABLE, TURBO_AVAILABLE

    TEXT_NORMALIZATION_AVAILABLE = _probe_text_normalization()
    TURBO_AVAILABLE = _probe_turbo()
    logging.debug(
        "[Chatterbox TTS] Chatterbox feature flags: normalization=%s turbo=%s",
        TEXT_NORMALIZATION_AVAILABLE, TURBO_AVAILABLE,
    )


TEXT_NORMALIZATION_AVAILABLE: bool = False
TURBO_AVAILABLE: bool = False
refresh_feature_flags()


# ── Model scanning ───────────────────────────────────────────────────

def read_model_config(model_dir: str) -> Optional[dict]:
    """Parse ``<model_dir>/config.json``.

    Returns ``None`` when the file is absent or unreadable -- Chatterbox packs
    normally ship no ``config.json`` at all, so ``None`` is the common answer
    and not a failure. Use :func:`describe_model_dir` to characterise a pack
    that has none.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logging.debug("[Chatterbox TTS] Could not read config.json in %s: %s", model_dir, exc)
        return None


def _list_entries(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        logging.debug("[Chatterbox TTS] Could not list '%s': %s", path, exc)
        return []


def _infer_architecture(entries: List[str]) -> Optional[str]:
    """Guess a pack's architecture from the weights it carries."""
    if "t3_nano_v1.safetensors" in entries:
        return "chatterbox_nano"
    if "t3_turbo_v1.safetensors" in entries:
        return "chatterbox_turbo"
    if "t3_mtl23ls_v2.safetensors" in entries or "t3_mtl23ls_v3.safetensors" in entries:
        return "chatterbox_multilingual"
    if "t3_cfg.safetensors" in entries:
        return "chatterbox_tts"
    return None


def describe_model_dir(model_dir: str) -> Dict[str, Any]:
    """Summarise a pack directory.

    Prefers an explicit ``config.json`` (community checkpoints carry one), and
    otherwise infers the architecture from the weight filenames. Returns a dict
    with ``name``, ``architecture`` and ``path``; ``architecture`` is
    ``"unknown"`` when neither source says anything.
    """
    entries = _list_entries(model_dir)
    config = read_model_config(model_dir) or {}
    architecture = config.get("architecture") or _infer_architecture(entries) or "unknown"
    description: Dict[str, Any] = {
        "name": os.path.basename(os.path.normpath(model_dir)),
        "architecture": architecture,
        "path": model_dir,
    }
    if config:
        description["config"] = config
    return description


def is_chatterbox_model_dir(path: str) -> bool:
    """True when ``path`` is a usable Chatterbox pack directory.

    Requires both halves of a pack: the tokenizer that builds the T3
    conditions, and at least one weight blob. Either alone is not a model -- an
    empty directory, a partially cancelled download, or an unrelated
    HuggingFace checkpoint all fail one of the two checks.

    Any ``.safetensors`` counts as a weight, not only the names in
    :data:`CHATTERBOX_WEIGHT_FILES`: community checkpoints rename their blobs,
    and refusing to recognise them would hide a model the loader can actually
    load.
    """
    if not path or not os.path.isdir(path):
        return False

    config = read_model_config(path)
    if config is not None:
        architecture = config.get("architecture", "")
        if architecture and architecture not in VALID_CHATTERBOX_ARCHITECTURES:
            # An explicit config.json naming a foreign architecture settles it.
            return False

    entries = set(_list_entries(path))
    if CHATTERBOX_TOKENIZER_FILE not in entries:
        return False
    return any(weight in entries for weight in CHATTERBOX_WEIGHT_FILES) or any(
        entry.endswith(".safetensors") for entry in entries
    )


def scan_chatterbox_models(scan_dir: str) -> List[dict]:
    """Every valid Chatterbox pack directly under ``scan_dir``.

    Returns a list of ``{"name", "architecture", "path"}`` dicts, sorted by
    name. A missing directory yields ``[]`` rather than raising: the caller is
    usually a frontend poll that must not 500 because the user has no models
    yet.
    """
    models: List[dict] = []
    if not scan_dir or not os.path.isdir(scan_dir):
        return models

    for item in _list_entries(scan_dir):
        item_path = os.path.join(scan_dir, item)
        if not os.path.isdir(item_path):
            continue
        if not is_chatterbox_model_dir(item_path):
            continue
        description = describe_model_dir(item_path)
        models.append(description)
        logging.debug(
            "[Chatterbox TTS] Found Chatterbox pack %s (architecture=%s) at %s",
            item, description["architecture"], item_path,
        )

    return models


def scan_custom_model_path(path: str) -> List[dict]:
    """Scan a user-supplied model directory.

    Warns rather than raises when the path does not exist: this is the answer
    to an interactive "browse for your models" dialog, and a wrong path is the
    user correcting themselves, not a server fault.
    """
    if not path or not os.path.isdir(path):
        logging.warning("[Chatterbox TTS] Custom model path does not exist: %s", path)
        return []

    if is_chatterbox_model_dir(path):
        # The user pointed straight at a pack rather than at its parent.
        return [describe_model_dir(path)]

    return scan_chatterbox_models(path)


# ── Local model registry ─────────────────────────────────────────────

def register_local_model(name: str, path: str) -> None:
    """Add or replace a local pack in the registry."""
    description = describe_model_dir(path)
    description["name"] = name
    AVAILABLE_CHATTERBOX_MODELS[name] = {
        "type": "local",
        "repo_id": "",
        "architecture": description["architecture"],
        "sample_rate": 24000,
        "size_gb": 0.0,
        "path": path,
    }
    logging.debug("[Chatterbox TTS] Registered local Chatterbox pack '%s' at %s", name, path)


def unregister_local_model(name: str) -> bool:
    """Drop a local pack. Official entries are never removed."""
    entry = AVAILABLE_CHATTERBOX_MODELS.get(name)
    if entry is None or entry.get("type") != "local":
        return False
    del AVAILABLE_CHATTERBOX_MODELS[name]
    return True


def refresh_local_models() -> List[dict]:
    """Re-scan the models root and mirror its packs into the registry.

    The filesystem walk is delegated to :mod:`modules.model_discovery`, which
    caches on the models root's ``st_mtime_ns`` -- so calling this on every
    frontend refresh is cheap. Official entries are preserved; local entries
    for packs that have disappeared are dropped.
    """
    from . import model_discovery

    found = model_discovery.get_model_pack_entries()
    current = {
        name for name, entry in AVAILABLE_CHATTERBOX_MODELS.items()
        if entry.get("type") == "local"
    }

    for name in current - {name for name, _path in found}:
        unregister_local_model(name)

    for name, path in found:
        register_local_model(name, path)

    return [
        {"name": name, "path": path, "type": "local"}
        for name, path in found
    ]


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """The registry with local packs freshly scanned in."""
    refresh_local_models()
    return AVAILABLE_CHATTERBOX_MODELS