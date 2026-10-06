"""Device utilities for Chatterbox TTS nodes.

Unified device handling on top of ComfyUI's ``comfy.model_management``, so
device detection matches what the rest of ComfyUI is doing on this machine.

``comfy.model_management`` is imported **lazily**, inside
:func:`_get_model_management`, never at module scope: the node package has to be
importable (and unit-testable) before ComfyUI is guaranteed to be on
``sys.path``. ``dtype_utils`` imports ``_get_model_management`` from here
cross-module; the underscore is part of that contract, not a leftover.
"""

import logging
from typing import List, Optional

import torch

__all__ = [
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "DEVICE_MPS",
    "DEVICE_XPU",
    "DEVICE_NPU",
    "DEVICE_DEFAULT",
    "get_available_devices",
    "get_device_options",
    "get_device_option_labels",
    "get_device_display_name",
    "get_device_type",
    "get_torch_device",
    "get_offload_device",
    "get_device_memory_info",
    "is_gpu_device",
    "should_use_fp16",
    "should_use_bf16",
    "supports_fp8_compute",
    "get_autocast_device",
]


# Lazy import to avoid a circular dependency at package-import time.
_model_management = None


def _get_model_management():
    """Return ``comfy.model_management``, importing it on first use."""
    global _model_management
    if _model_management is None:
        import comfy.model_management as mm

        _model_management = mm
    return _model_management


# Device type constants. The first block are legacy aliases this project and
# VibeVoice accept as *input*; they are never produced by ``get_device_options``
# unless ComfyUI itself produced them.
DEVICE_AUTO = "auto"
DEVICE_DEFAULT = "default"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
DEVICE_XPU = "xpu"
DEVICE_NPU = "npu"
DEVICE_DIRECTML = "directml"
DEVICE_HIP = "hip"


def get_available_devices() -> List[str]:
    """Device option strings ComfyUI itself offers, in its own order.

    Delegates to ``mm.get_gpu_device_options()`` (``["default", "cpu"]``, plus
    ``"gpu:N"`` on a multi-GPU machine) so the widget offers exactly the labels
    ComfyUI's own value checks know about.

    Returns:
        List of device option strings, without the ``"auto"`` alias.
    """
    mm = _get_model_management()
    getter = getattr(mm, "get_gpu_device_options", None)
    if callable(getter):
        return [str(option) for option in getter()]

    # Older core without the helper: fall back to a capability probe.
    devices: List[str] = [DEVICE_DEFAULT]
    if mm.is_nvidia() or mm.is_amd() or mm.mps_mode():
        devices.append(DEVICE_CUDA)
    devices.append(DEVICE_CPU)
    return devices


def get_device_options() -> List[str]:
    """Device values for the node dropdown.

    ``"auto"`` is a real, supported value: :func:`get_torch_device` maps it to
    ComfyUI's default compute device. It has to appear in the combo options,
    otherwise ComfyUI's "Value not in list" check rejects a workflow saved with
    ``device="auto"`` before ``execute()`` ever runs.

    It is appended **last**, deliberately. ComfyUI binds a saved workflow's
    widget values **by index**, so inserting an option at the front silently
    re-binds every existing graph's device field to whatever now sits at that
    index. Appending is the only ordering that keeps old indices valid.

    Returns:
        List of device option strings for the UI dropdown.
    """
    return [*get_available_devices(), DEVICE_AUTO]


def get_device_option_labels() -> dict:
    """Mapping of device option values to human-readable labels."""
    return {device: get_device_display_name(device) for device in get_device_options()}


def get_device_display_name(device_type: str) -> str:
    """Human-readable label for a device option value.

    Args:
        device_type: A value from :func:`get_device_options`, or one of the
            legacy aliases this module still accepts as input.

    Returns:
        Human-readable device name.
    """
    if device_type == DEVICE_AUTO:
        return "Auto (ComfyUI default)"
    if device_type == DEVICE_DEFAULT:
        return "Default"
    if device_type.startswith("gpu:"):
        index = device_type.split(":", 1)[1]
        try:
            return f"GPU {index} ({torch.cuda.get_device_name(int(index))})"
        except Exception:
            return f"GPU {index}"

    mm = _get_model_management()

    if device_type == DEVICE_CUDA:
        if mm.is_nvidia():
            try:
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            except Exception:
                return "CUDA (NVIDIA GPU)"
        if mm.is_amd():
            return "CUDA (AMD ROCm/HIP)"
        return "CUDA"
    if device_type == DEVICE_MPS:
        return "MPS (Apple Silicon)"
    if device_type == DEVICE_XPU:
        return "XPU (Intel Arc)"
    if device_type == DEVICE_NPU:
        return "NPU (Huawei Ascend)"
    if device_type == DEVICE_DIRECTML:
        return "DirectML (Windows)"
    if device_type == DEVICE_CPU:
        return "CPU"

    return str(device_type).upper()


def get_torch_device(device_type: Optional[str] = None) -> torch.device:
    """Resolve a device option value to a ``torch.device``.

    Honours MPS / XPU / NPU requests when the backend is actually available on
    this machine; otherwise it logs a warning and falls back to ComfyUI's
    default compute device.

    Resolution order:

    1. ``None`` / ``"auto"`` / ``"default"`` / any ``"cuda*"`` -> ComfyUI's
       default compute device.
    2. ComfyUI's own option grammar (``"cpu"``, ``"gpu:N"``) via
       ``mm.resolve_gpu_device_option``.
    3. Explicit accelerator names (``"mps"`` / ``"xpu"`` / ``"npu"``), each
       guarded by an availability probe.
    4. Anything unrecognised falls back to the default device, loudly.

    Args:
        device_type: A device option value. ``None`` means "auto".

    Returns:
        A ``torch.device``.
    """
    mm = _get_model_management()

    if device_type is None or device_type in (DEVICE_AUTO, DEVICE_DEFAULT):
        return mm.get_torch_device()

    # CUDA and any "cuda:<index>" map to ComfyUI's default compute device.
    if isinstance(device_type, str) and device_type.startswith(DEVICE_CUDA):
        return mm.get_torch_device()

    resolver = getattr(mm, "resolve_gpu_device_option", None)
    if callable(resolver):
        resolved = resolver(device_type)
        if resolved is not None:
            return resolved
    elif device_type == DEVICE_CPU:
        return torch.device(DEVICE_CPU)

    if device_type == DEVICE_CPU:
        return torch.device(DEVICE_CPU)

    if device_type == DEVICE_MPS:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device(DEVICE_MPS)
        logging.warning(
            "[Chatterbox TTS] Device 'mps' requested but not available; using default device."
        )
        return mm.get_torch_device()

    if device_type == DEVICE_XPU:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device(DEVICE_XPU)
        logging.warning(
            "[Chatterbox TTS] Device 'xpu' requested but not available; using default device."
        )
        return mm.get_torch_device()

    if device_type == DEVICE_NPU:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device(DEVICE_NPU)
        logging.warning(
            "[Chatterbox TTS] Device 'npu' requested but not available; using default device."
        )
        return mm.get_torch_device()

    logging.warning(
        "[Chatterbox TTS] Device '%s' not available; using default device.", device_type
    )
    return mm.get_torch_device()


def get_device_type(device: torch.device) -> str:
    """Convert a ``torch.device`` to this project's device-type string."""
    if device.type in (DEVICE_CPU, DEVICE_MPS, DEVICE_XPU, DEVICE_NPU, DEVICE_CUDA):
        return device.type
    return device.type


def get_offload_device() -> torch.device:
    """The offload device (typically CPU), via ComfyUI's ``intermediate_device``."""
    return _get_model_management().intermediate_device()


def should_use_fp16(device: Optional[torch.device] = None) -> bool:
    """Whether FP16 is appropriate for ``device`` (ComfyUI's own answer)."""
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.should_use_fp16(device)


def should_use_bf16(device: Optional[torch.device] = None) -> bool:
    """Whether BF16 is appropriate for ``device`` (ComfyUI's own answer)."""
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.should_use_bf16(device)


def supports_fp8_compute(device: Optional[torch.device] = None) -> bool:
    """Whether FP8 compute is supported on ``device``."""
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.supports_fp8_compute(device)


def get_autocast_device(device: torch.device) -> str:
    """The device-type string ``torch.autocast`` expects."""
    return _get_model_management().get_autocast_device(device)


def is_gpu_device(device_type: str) -> bool:
    """Whether a device-type string names an accelerator rather than the CPU.

    Args:
        device_type: Device type string.

    Returns:
        ``True`` for a GPU-class device type.
    """
    gpu_types = {
        DEVICE_CUDA,
        DEVICE_MPS,
        DEVICE_XPU,
        DEVICE_NPU,
        DEVICE_DIRECTML,
        DEVICE_HIP,
    }
    if not isinstance(device_type, str):
        return False
    if device_type.startswith("gpu:") or device_type == DEVICE_DEFAULT:
        return True
    return device_type.lower() in gpu_types


def get_device_memory_info(device_type: Optional[str] = None) -> dict:
    """Total / used / free memory in bytes for ``device_type``.

    Returns an empty dict for anything that is not a CUDA device -- an honest
    "unknown" rather than a fabricated zero.
    """
    device = get_torch_device(device_type)

    if device.type == DEVICE_CUDA:
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free = total - reserved + (reserved - allocated)
            return {"total": total, "used": allocated, "free": free}
        except Exception:
            pass

    return {}