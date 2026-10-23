"""Shared pieces of the Chatterbox node schemas.

Every Chatterbox node exposes the same three appended widgets -- ``device``,
``dtype``, ``attention_mode`` -- and the same three queue-time checks. Those
are factored out here so the TTS, VC and (later) feature nodes cannot drift
apart: a divergence in how a node validates a voice pack is exactly the kind
of bug that only shows up on one machine, in one graph, weeks later.

Nothing in this module imports the vendored ``chatterbox`` package, and
nothing here builds a schema -- it returns plain values and ``io`` inputs that
the node modules arrange in their own frozen order.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from comfy_api.latest import io

from ..modules.attention_utils import (
    check_dtype_attention_compatible,
    get_available_attention_modes,
)
from ..modules.audio_utils import S3GEN_SAMPLE_RATE, to_comfy_audio
from ..modules.constants import DEFAULT_MODEL_PACK_NAME
from ..modules.device_utils import get_device_options
from ..modules.dtype_utils import DTYPE_AUTO, get_dtype_options
from ..modules.model_discovery import get_chatterbox_model_pack_names
from ..modules.model_info import canonical_pack_name

__all__ = [
    "default_pack_name",
    "silent_audio",
    "runtime_widgets",
    "validate_common_inputs",
]


def default_pack_name(pack_names: Optional[list[str]] = None) -> str:
    """The pack a combo carrying no stored value resolves to.

    ``get_chatterbox_model_pack_names`` hoists the default pack to index 0
    whenever it is on disk, and returns it alone when nothing is. The only case
    where index 0 is *not* the default pack is a machine holding exclusively
    third-party packs -- and naming one the default would put a value the combo
    does not offer into the schema.
    """
    names = list(pack_names) if pack_names is not None else list(get_chatterbox_model_pack_names())
    if DEFAULT_MODEL_PACK_NAME in names:
        return DEFAULT_MODEL_PACK_NAME
    return names[0] if names else DEFAULT_MODEL_PACK_NAME


def silent_audio() -> dict:
    """One second of silence, returned when a run is cancelled.

    A cancelled run still has to satisfy the node's output socket. Silence is
    the one answer that cannot be mistaken for a successful synthesis, and it
    is used for cancellation only -- every other failure raises.
    """
    waveform = torch.zeros((1, 1, S3GEN_SAMPLE_RATE), dtype=torch.float32)
    return to_comfy_audio(waveform, S3GEN_SAMPLE_RATE)


def runtime_widgets() -> list[io.Input]:
    """The appended ``device`` / ``dtype`` / ``attention_mode`` inputs.

    Callers spread these in **after** their frozen widget block. Appending is
    the only safe insertion: ComfyUI binds a saved workflow's widget values by
    index, so anything placed before the frozen tail would shift it.

    The Combo options are resolved here, at schema-build time, from this
    machine's capabilities:

    * ``get_device_options`` returns ComfyUI's own device list with ``"auto"``
      appended **last**, so a saved graph's device index never moves either.
    * ``get_available_attention_modes`` returns only the backends this machine
      can actually run, which drops ``flash_attention_2`` and ``sage`` when
      their prerequisites are missing -- an unavailable backend is therefore
      *removed from the schema here* rather than offered and then refused.
      A saved graph naming a backend that vanished is still fine:
      :func:`validate_common_inputs` does not refuse it, and
      ``modules.attention_utils.resolve_attention_mode`` downgrades it to
      something loadable at load time.
    """
    device_options = get_device_options()
    dtype_options = get_dtype_options()
    attention_modes = get_available_attention_modes()

    return [
        io.Combo.Input(
            "device",
            options=device_options,
            default=device_options[0] if device_options else "auto",
            tooltip=(
                "Device to run inference on. 'auto' follows ComfyUI's default "
                "compute device."
            ),
        ),
        io.Combo.Input(
            "dtype",
            options=dtype_options,
            default=DTYPE_AUTO,
            tooltip=(
                "Data type for model precision. 'auto' lets ComfyUI pick the "
                "best type for the target device."
            ),
        ),
        io.Combo.Input(
            "attention_mode",
            options=attention_modes,
            default="sdpa",
            tooltip=(
                "Attention implementation. Backends this machine cannot run are "
                "left out of the dropdown; a saved graph that names one anyway "
                "falls back to 'sdpa'."
            ),
        ),
    ]


def validate_common_inputs(
    model_pack_name: Optional[str] = None,
    dtype: Optional[str] = None,
    attention_mode: Optional[str] = None,
) -> Optional[str]:
    """The queue-time checks every Chatterbox node shares.

    Args:
        model_pack_name: The pack the combo carried.
        dtype: Dtype widget value.
        attention_mode: Attention widget value, as the user picked it.

    Returns:
        An actionable refusal message, or ``None`` when the prompt is fine.
        The nodes wrap this into the ``bool | str`` ComfyUI expects.
    """
    if model_pack_name is not None:
        pack_names = list(get_chatterbox_model_pack_names())
        # An alias is the pack it now names, so it passes the same gate the
        # current spelling does: a workflow saved before a pack was renamed
        # must still queue.
        canonical = canonical_pack_name(model_pack_name)
        if canonical not in pack_names:
            available = ", ".join(pack_names) if pack_names else "none"
            return (
                f"The Chatterbox voice pack '{canonical}' was not found "
                f"under models/tts/chatterbox. Available packs: {available}. "
                f"Install the pack, or pick one of the available names."
            )

    if dtype is not None and attention_mode is not None:
        message = check_dtype_attention_compatible(dtype, attention_mode)
        if message is not None:
            return message

    if (
        attention_mode is not None
        and attention_mode not in get_available_attention_modes()
    ):
        # Deliberately not a refusal. A mode string in a saved workflow is a
        # record of the machine that saved it, not a claim about this one;
        # resolve_attention_mode downgrades it to something loadable, which is
        # friendlier than refusing to open the graph.
        logging.info(
            "[Chatterbox TTS] attention_mode '%s' is not available on this "
            "machine; it will fall back to a loadable backend.",
            attention_mode,
        )

    return None