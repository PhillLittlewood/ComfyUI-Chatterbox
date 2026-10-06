"""hf-progress: plug-and-play progress tracking for HuggingFace Hub.

A library that provides real-time upload/download progress tracking
for HuggingFace Hub operations. Uses a **direct-first** strategy:

- **Xet uploads**: Direct ``hf_xet`` calls with detailed callbacks
  (dedup, transfer speed, per-file progress)
- **Xet downloads**: Direct ``hf_xet.download_files()`` with
  detailed ``(total_update, item_updates)`` callbacks for speed,
  dedup info, and per-item progress
- **HTTP downloads** (fallback): ``tqdm_class`` override for ``hf_hub_download()``
- **LFS uploads** (fallback): tqdm monkey-patching for ``HfApi.upload_file()``

Quick start::

    from ComfyUI_Chatterbox.src.hf_progress import HfProgressTracker, EventType

    tracker = HfProgressTracker(token="hf_...")

    # Upload with progress
    result = tracker.upload_file("model.bin", "username/repo")

    # Download with progress
    path = tracker.download_file("bert-base-uncased", "config.json")

    # Consume events
    for event in tracker.events(stop_on=EventType.COMPLETE):
        print(f"{event.filename}: {event.percentage:.1f}%")

Low-level callbacks::

    from ComfyUI_Chatterbox.src.hf_progress import (
        XetUploadProgressCallback,
        XetDownloadProgressCallback,
        DownloadProgressTqdm,
        tqdm_upload_patcher,
    )

SSE integration::

    from ComfyUI_Chatterbox.src.hf_progress.integrations.sse import create_progress_router

Every import inside this package is **relative** (plan S4.1 / §3.1.5). The
package must stay importable from whatever parent it is vendored into, so it
never names itself absolutely and never touches ``sys.path``.
"""

from __future__ import annotations

# Core types
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferProgressError,
    TransferResult,
    TokenError,
    generate_transfer_id,
)

# Callback classes
from .callbacks import (
    DownloadProgressTqdm,
    XetDownloadProgressCallback,
    XetProgressCallback,
    XetUploadProgressCallback,
    tqdm_upload_patcher,
)

# Token management
from .token import XetCredentials, XetTokenManager, is_xet_available

# Standard (non-Xet) HTTP transfer drivers. Re-exported at package level so a
# caller can swap ``tqdm_class`` in without needing to know the module layout.
from .standard_download import download_file, download_snapshot, patch_download_chunk_size
from .standard_upload import upload_bytes, upload_file, upload_folder

# Xet transfer drivers. These are what ``HfProgressTracker`` calls internally
# when Xet is enabled; they are exported so a caller can drive a single
# transfer with the same progress plumbing the tracker uses.
from .xet_download import download_file_with_xet, download_files_with_xet
from .xet_upload import upload_bytes_with_xet, upload_file_with_xet

# Subprocess isolation
from .subprocess_runner import XetSubprocessRunner
from .subprocess_messages import SubprocessMessage

# High-level tracker
from .tracker import HfProgressTracker

__version__ = "0.1.0"

__all__ = [
    # Core types
    "EventType",
    "ProgressEvent",
    "ProgressPhase",
    "TransferCancelledError",
    "TransferDirection",
    "TransferError",
    "TransferProgressError",
    "TransferResult",
    "TokenError",
    "generate_transfer_id",
    # Callback classes
    "DownloadProgressTqdm",
    "XetDownloadProgressCallback",
    "XetProgressCallback",
    "XetUploadProgressCallback",
    "tqdm_upload_patcher",
    # Token management
    "XetCredentials",
    "XetTokenManager",
    "is_xet_available",
    # Standard HTTP transfer drivers
    "download_file",
    "download_snapshot",
    "patch_download_chunk_size",
    "upload_file",
    "upload_bytes",
    "upload_folder",
    # Xet transfer drivers
    "download_file_with_xet",
    "download_files_with_xet",
    "upload_file_with_xet",
    "upload_bytes_with_xet",
    # Subprocess isolation
    "XetSubprocessRunner",
    "SubprocessMessage",
    # High-level tracker
    "HfProgressTracker",
]