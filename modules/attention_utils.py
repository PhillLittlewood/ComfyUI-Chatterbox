"""Attention-mode utilities for Chatterbox TTS nodes.

Capability probing and mode resolution for the attention widget: which backends
this machine can actually run, and what a saved workflow's mode string resolves
to once hardware is taken into account.

The central rule is that **a mode string carried by a saved workflow is not a
capability claim**. The dropdown is gated on
:func:`get_available_attention_modes` at schema-build time, but a graph saved on
a machine with SageAttention replays verbatim on one without it. Resolution
therefore re-checks availability and falls back rather than letting the loader
die deep inside the model constructor.
"""

import logging
from typing import List, Optional

import torch

__all__ = [
    "ATTENTION_MODES",
    "ATTENTION_AUTO",
    "DEFAULT_ATTENTION_MODE",
    "SAGE_SUPPORTED_ARCHS",
    "SAGE_UNSUPPORTED_DTYPES",
    "SAGE_ATTENTION_AVAILABLE",
    "sage_arch_code",
    "check_sage_attention_compatible",
    "check_flash_attention_available",
    "get_available_attention_modes",
    "resolve_attention_mode",
    "get_attn_implementation_for_load",
    "check_dtype_attention_compatible",
]


# sageattention is optional; its absence must not be an import error.
try:  # pragma: no cover - depends on the machine
    import sageattention  # noqa: F401

    SAGE_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the machine
    SAGE_ATTENTION_AVAILABLE = False


# Base attention modes always available.
ATTENTION_MODES: List[str] = ["eager", "sdpa", "flash_attention_2"]

#: The mode the attention widget starts on, and the one ``"auto"`` resolves to.
#: SDPA is the only backend that is both universally available and the one
#: vendored weights are actually built for (``llama_configs.py`` pins
#: ``attn_implementation="sdpa"``, see ``src/PATCHES.md`` U-005), so "let the
#: machine decide" and "what this project defaults to" are the same answer.
DEFAULT_ATTENTION_MODE = "sdpa"

#: The "let the machine decide" request. It is not offered in the dropdown -- it
#: is what an API caller or a hand-edited graph sends, and it is resolved here
#: rather than falling through to the unknown-mode branch, which would hand it
#: ``eager`` and silently cost the caller performance.
ATTENTION_AUTO = "auto"

# sage is a real option only when the library is importable at all. Hardware is
# checked separately (see check_sage_attention_compatible) because an importable
# kernel is not a supported GPU.
if SAGE_ATTENTION_AVAILABLE:
    ATTENTION_MODES.append("sage")


# CUDA architectures SageAttention is willing to run on, pinned as a literal set
# of ``sm<major><minor>`` strings -- the spelling sage's own dispatcher uses, so
# this membership test and the kernel dispatch cannot drift apart. Deriving it
# from the installed library instead would couple the check to whatever happens
# to be on disk.
#
# Two entries are deliberately absent from sage's own branch set:
#   * sm75 -- the check it replaces was "compute capability major >= 8", which
#     already excluded it, and sage routes it to a Triton kernel this project
#     never dispatches to.
#   * sm100/sm103 (Blackwell datacenter, CC 10.x) -- sage has no branch for
#     them and refuses with ``ValueError: Unsupported CUDA architecture``. A
#     plain "CC major >= 8" test lets them through, and an ``arch_code >= 90``
#     branch then hands the SM90 kernel to silicon it was not built for.
SAGE_SUPPORTED_ARCHS = frozenset({"sm80", "sm86", "sm89", "sm90", "sm120"})


def sage_arch_code() -> int:
    """Current CUDA device as ``major * 10 + minor`` (80, 86, 89, 90, 120).

    Spelled as an integer because that is the form the dispatcher and the
    ``smXY`` membership test both consume.
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def check_sage_attention_compatible() -> bool:
    """Whether SageAttention can run on this machine right now.

    Needs the library, CUDA, *and* membership of the device in
    :data:`SAGE_SUPPORTED_ARCHS`. Testing only "CC major >= 8" let sm100/sm103
    through, where sage's own ``sageattn()`` raises ``ValueError``.

    Returns:
        ``True`` when SageAttention can be used.
    """
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm{major * 10 + minor}"
    if arch not in SAGE_SUPPORTED_ARCHS:
        logging.warning(
            "[Chatterbox TTS] GPU (compute capability %d.%d, %s) is not one of the "
            "architectures SageAttention ships kernels for (%s). The sage option "
            "is unavailable and any request for it will fall back to 'sdpa'.",
            major,
            minor,
            arch,
            ", ".join(sorted(SAGE_SUPPORTED_ARCHS)),
        )
        return False
    return True


def check_flash_attention_available() -> bool:
    """Whether FlashAttention 2 is usable on this hardware.

    Requires the ``flash_attn`` package and a CUDA GPU. When either is missing
    the mode would fail at model-load time, so it is hidden from the options and
    downgraded when requested.
    """
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def get_available_attention_modes() -> List[str]:
    """Attention modes this machine can actually run.

    Always ``["eager", "sdpa"]``. ``flash_attention_2`` and ``sage`` are
    appended only when their real prerequisites are present, so the dropdown
    never offers a mode that cannot be loaded.

    Returns:
        List of attention mode strings, in widget order.
    """
    modes = ["eager", "sdpa"]
    if check_flash_attention_available():
        modes.append("flash_attention_2")
    if check_sage_attention_compatible():
        modes.append("sage")
    return modes


def resolve_attention_mode(requested_mode: str, quantize_4bit: bool = False) -> str:
    """Resolve a requested attention mode against this machine's capabilities.

    Fallbacks applied, in order:

    - 4-bit quantization + ``eager`` / ``flash_attention_2`` -> ``sdpa``
      (quantized linears are unstable under the non-SDPA kernels).
    - ``flash_attention_2`` / ``sage`` not usable here -> ``sdpa``. This is the
      case a *saved workflow* lands in: the graph carries the mode string
      verbatim while the dropdown that gated it was built on another machine.
      Without the check the string survives every guard and dies inside the
      loader with ``RuntimeError("Incompatible hardware/setup for
      SageAttention.")``.
- ``auto`` -> :data:`DEFAULT_ATTENTION_MODE`. Resolved before anything else,
      because "the machine decides" is not an unknown mode and answering it with
      ``eager`` would be an answer nobody asked for.
    - Unknown mode -> ``eager``, the one backend with no hardware requirement.

    Args:
        requested_mode: The mode the workflow or widget asked for.
        quantize_4bit: Whether 4-bit quantization is enabled.

    Returns:
        A mode that is guaranteed loadable on this machine.
    """
    mode = requested_mode

    if mode == ATTENTION_AUTO:
        return DEFAULT_ATTENTION_MODE

    if quantize_4bit and mode in ("eager", "flash_attention_2"):
        logging.warning(
            "[Chatterbox TTS] Attention mode '%s' is not recommended with 4-bit "
            "quantization. Falling back to 'sdpa' for stability and performance.",
            mode,
        )
        mode = "sdpa"

    if mode == "flash_attention_2" and not check_flash_attention_available():
        logging.warning(
            "[Chatterbox TTS] flash_attention_2 is not available on this hardware; "
            "falling back to 'sdpa'."
        )
        mode = "sdpa"

    if mode == "sage" and not check_sage_attention_compatible():
        logging.warning(
            "[Chatterbox TTS] sage is not usable on this machine (sageattention "
            "missing, no CUDA device, or an unsupported GPU architecture); "
            "falling back to 'sdpa'."
        )
        mode = "sdpa"

    if mode not in ATTENTION_MODES:
        logging.warning(
            "[Chatterbox TTS] Unknown attention mode '%s', falling back to eager.", mode
        )
        mode = "eager"

    return mode


def get_attn_implementation_for_load(attention_mode: str) -> str:
    """The ``attn_implementation`` string handed to ``from_pretrained``.

    SageAttention is applied *after* load, by patching, so the weights are
    built with ``"sdpa"``: transformers has no ``"sage"`` implementation to ask
    for, and passing one fails inside ``from_pretrained`` rather than at the
    patch site.
    """
    if attention_mode == "sage":
        return "sdpa"
    return attention_mode


# ---------------------------------------------------------------------------
# dtype / attention cross-checks
# ---------------------------------------------------------------------------
# The sage kernels hard-assert ``dtype in [torch.float16, torch.bfloat16]``, and
# the patched forward reads the *stored weight* dtype of a plain float linear.
# A user who picks "fp32" in the dtype widget and "sage" in the attention widget
# therefore crashes inside the kernel with an assert that names neither widget.
# The two widgets are independent inputs with no cross-check anywhere upstream, so
# it has to be caught before the model is built.
#
# "auto" is exempt: on any GPU sage supports (sm80+) ComfyUI's
# should_use_bf16/should_use_fp16 resolve it to a half dtype. The 4-bit case is
# handled by the loader (it forces bnb_compute_dtype=float32 while sage reads
# bf16 out of the quantized linears), so a node exposing 4-bit should pass the
# *effective* model dtype, not the widget.
SAGE_UNSUPPORTED_DTYPES = frozenset({"fp32"})


def check_dtype_attention_compatible(
    dtype_str: str,
    attention_mode: Optional[str],
    quantized_4bit: bool = False,
) -> "str | None":
    """Cross-check a dtype choice against an attention backend.

    Call from a node's ``validate_inputs`` so the user gets a queue-time message
    naming the two widgets, instead of an assert from inside a CUDA kernel
    several minutes into a model load.

    Args:
        dtype_str: The dtype widget value (``"auto"``, ``"bf16"``, ``"fp16"``,
            ``"fp32"``).
        attention_mode: The attention widget value as the user picked it
            (pre-resolution is fine -- ``"sage"`` is caught either way). ``None``
            means the widget was not part of the prompt, so there is nothing to
            cross-check.
        quantized_4bit: Whether 4-bit quantization is on. It exempts the check.

    Returns:
        An actionable error message, or ``None`` when the pair is fine.
    """
    if attention_mode != "sage":
        return None
    if dtype_str not in SAGE_UNSUPPORTED_DTYPES:
        return None
    if quantized_4bit:
        return None
    return (
        f"dtype '{dtype_str}' cannot run with attention_mode 'sage': the "
        f"SageAttention kernels require fp16 or bf16 inputs. Pick dtype "
        f"'bf16' (or 'fp16'), or switch attention_mode to 'sdpa' / "
        f"'flash_attention_2'."
    )