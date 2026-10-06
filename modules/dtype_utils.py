"""Dtype utilities for Chatterbox TTS nodes.

Dtype resolution, config bookkeeping and casting helpers built on ComfyUI's
``model_management`` so precision selection matches the rest of the UI.

Two things here are load-bearing rather than cosmetic:

* **transformers v5 renamed the config attribute.** ``config.torch_dtype`` is a
  deprecated property in transformers >= 5.0 that warns on every get *and* set;
  ``config.dtype`` is the replacement. Reading or writing the old name on the
  installed 5.3.0 floods the log. :func:`set_config_dtype` / :func:`get_config_dtype`
  probe for the deprecated property and use whichever name the installed
  transformers actually defines.
* **An explicit dtype is validated, not obeyed.** ``resolve_dtype("bf16")`` on
  hardware that cannot run bf16 raises instead of handing back a
  ``torch.bfloat16`` the caller will discover three minutes into a CUDA kernel.
  Only ``"auto"`` is a request that ComfyUI gets to answer.
"""

import logging
from typing import List, Optional

import torch

from .device_utils import _get_model_management, should_use_bf16, should_use_fp16

__all__ = [
    "DTYPE_AUTO",
    "DTYPE_BF16",
    "DTYPE_FP16",
    "DTYPE_FP32",
    "get_dtype_options",
    "set_config_dtype",
    "get_config_dtype",
    "resolve_dtype",
    "get_dtype_str",
    "cast_model_to_dtype",
    "cast_model_to_dtype_if_needed",
    "representative_dtype",
]


# Dtype string constants
DTYPE_AUTO = "auto"
DTYPE_BF16 = "bf16"
DTYPE_FP16 = "fp16"
DTYPE_FP32 = "fp32"

# Mapping from string to torch.dtype
_DTYPE_MAP = {
    DTYPE_BF16: torch.bfloat16,
    DTYPE_FP16: torch.float16,
    DTYPE_FP32: torch.float32,
}

# Reverse mapping
_DTYPE_STR_MAP = {
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
    torch.float32: DTYPE_FP32,
}

# Explicit dtypes validated against the target device. "fp32" is absent on
# purpose: it is the universally safe fallback and must always resolve.
_VALIDATED_DTYPES = frozenset({DTYPE_BF16})


def get_dtype_options() -> List[str]:
    """Dtype values for the node dropdown.

    ``"auto"`` leads because it is the default and the widget's index 0; the
    rest follow in the widget's own fixed order, which no existing workflow can
    be bound to yet (the V1 nodes had no dtype widget).

    Returns:
        List of dtype option strings.
    """
    return [DTYPE_AUTO, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32]


def _torch_dtype_is_deprecated_property(config) -> bool:
    """True when ``type(config).torch_dtype`` is the transformers >= 5.0
    deprecated property alias (which warns on every get and set)."""
    return isinstance(getattr(type(config), "torch_dtype", None), property)


def set_config_dtype(config, dtype) -> None:
    """Record the load dtype on a transformers config, without the v5 warning.

    Writes ``config.dtype`` when the installed transformers defines
    ``torch_dtype`` as the deprecated property, and ``config.torch_dtype``
    otherwise (transformers 4.x, and any duck-typed config that stores a plain
    attribute).
    """
    if _torch_dtype_is_deprecated_property(config):
        config.dtype = dtype
    else:
        config.torch_dtype = dtype


def get_config_dtype(config):
    """Return the torch dtype recorded on a transformers config, or ``None``."""
    if _torch_dtype_is_deprecated_property(config):
        dtype = getattr(config, "dtype", None)
    else:
        dtype = getattr(config, "torch_dtype", None)
        if dtype is None:
            dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    return dtype


def _validate_dtype_on_device(dtype_str: str, device: torch.device) -> None:
    """Raise when an explicit half dtype cannot run on ``device``.

    CPU is exempt: torch implements bf16 there, just slowly, and a user who
    asks for bf16 on a CPU-only box should get what they asked for.

    Raises:
        ValueError: when the dtype is unsupported on the target device.
    """
    if dtype_str not in _VALIDATED_DTYPES:
        return
    if device.type == "cpu":
        return

    supported = should_use_bf16(device) if dtype_str == DTYPE_BF16 else should_use_fp16(device)
    if supported:
        return

    raise ValueError(
        f"dtype '{dtype_str}' was requested explicitly but device '{device}' cannot "
        f"run it (ComfyUI's precision check says this hardware has no "
        f"{dtype_str} support). Use dtype 'auto' to let ComfyUI choose, or pick "
        f"'{DTYPE_FP32}' / '{DTYPE_FP16}'."
    )


def resolve_dtype(dtype_str: str, device: Optional[torch.device] = None) -> torch.dtype:
    """Resolve a dtype string to a ``torch.dtype``.

    Args:
        dtype_str: Dtype string (``"auto"``, ``"bf16"``, ``"fp16"``, ``"fp32"``).
        device: Device the dtype has to run on. ``None`` means ComfyUI's default
            compute device; it is only consulted for validation of an explicit
            choice.

    Returns:
        The resolved ``torch.dtype``.

    Raises:
        ValueError: If ``dtype_str`` is not recognised, or is an explicit dtype
            the target device cannot run.
    """
    if dtype_str == DTYPE_AUTO or dtype_str is None:
        mm = _get_model_management()
        if device is None:
            device = mm.get_torch_device()
        if should_use_bf16(device):
            return torch.bfloat16
        if should_use_fp16(device):
            return torch.float16
        return torch.float32

    if dtype_str in _DTYPE_MAP:
        if device is None:
            device = _get_model_management().get_torch_device()
        _validate_dtype_on_device(dtype_str, device)
        return _DTYPE_MAP[dtype_str]

    raise ValueError(
        f"Unknown dtype: {dtype_str}. Valid options: {get_dtype_options()}"
    )


def get_dtype_str(dtype: torch.dtype) -> str:
    """Convert a ``torch.dtype`` back to its widget string.

    Raises:
        ValueError: If the dtype is not one this module offers.
    """
    if dtype in _DTYPE_STR_MAP:
        return _DTYPE_STR_MAP[dtype]
    raise ValueError(
        f"Unknown torch.dtype: {dtype}. Supported: {sorted(_DTYPE_STR_MAP.values())}"
    )


def _quant_protected_names(model) -> set:
    """Parameter names excluded from bulk dtype casts.

    - Parameters of modules marked ``_quant_resident`` (GGUF / int8 / FP8
      linears), whose storage is raw packed bytes and is not castable at all.
    - Any parameter whose leaf name is ``weight_scale``, which is kept in the
      quantizer's own precision.
    """
    protected = set()
    for name, mod in model.named_modules():
        if getattr(mod, "_quant_resident", False):
            prefix = f"{name}." if name else ""
            protected.add(f"{prefix}weight")
            protected.add(f"{prefix}weight_scale")
    return protected


def _quant_resident_module_names(model) -> set:
    """Names of modules whose parameters hold packed quant data."""
    return {
        name
        for name, mod in model.named_modules()
        if getattr(mod, "_quant_resident", False)
    }


def representative_dtype(model):
    """The model's floating-point parameter dtype, ignoring quant storage.

    Returns ``None`` when the tree holds no floating parameter at all and the
    model carries no ``dtype`` attribute either.
    """
    try:
        resident_modules = _quant_resident_module_names(model)
        for name, param in model.named_parameters():
            if not param.dtype.is_floating_point:
                continue
            owner = name.rpartition(".")[0]
            if owner in resident_modules:
                continue
            return param.dtype
    except Exception:
        pass
    return getattr(model, "dtype", None)


def _cast_mismatched_params(model, dtype: torch.dtype) -> None:
    """Cast only floating, unprotected, mismatched parameters, then re-tie."""
    protected = _quant_protected_names(model)
    for name, param in model.named_parameters():
        if name in protected:
            continue
        if not param.dtype.is_floating_point:
            continue
        if param.dtype != dtype:
            param.data = param.data.to(dtype)

    config = getattr(model, "config", None)
    if config is not None and hasattr(model, "tie_weights"):
        decoder_config = getattr(config, "decoder_config", None)
        tied = bool(getattr(decoder_config, "tie_word_embeddings", False)) or bool(
            getattr(config, "tie_word_embeddings", False)
        )
        if tied:
            model.tie_weights()


def cast_model_to_dtype(model, dtype: torch.dtype) -> None:
    """Cast a model's floating parameters to ``dtype``.

    Quant-resident storage (raw packed GGUF blocks, int8 ConvRot weights, the
    quantizer's own fp32 ``weight_scale``) is **never** touched: those bytes are
    a payload, not numbers, and casting them corrupts the checkpoint.
    """
    if dtype is None:
        return
    _cast_mismatched_params(model, dtype)
    logging.debug(
        "[Chatterbox TTS] Model cast to dtype (filtered): %s", dtype
    )


def cast_model_to_dtype_if_needed(model, dtype: torch.dtype) -> None:
    """Cast to ``dtype`` only when a castable parameter actually mismatches."""
    if dtype is None:
        return

    protected = _quant_protected_names(model)

    mismatched = []
    for name, param in model.named_parameters():
        if name in protected or not param.dtype.is_floating_point:
            continue
        if param.dtype != dtype:
            mismatched.append((name, param.dtype))
    if not mismatched:
        return

    _cast_mismatched_params(model, dtype)
    sources = ", ".join(sorted({str(src) for _, src in mismatched}))
    logging.debug(
        "[Chatterbox TTS] Model cast %s -> %s (%d mismatched params)",
        sources,
        dtype,
        len(mismatched),
    )