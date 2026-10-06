"""Download manager for Chatterbox models: progress, retry and cancellation.

Ported from ``ComfyUI-VoxCPM/modules/downloader.py`` (plan S4.2) with the
corrections listed in plan §3.1 applied. They are listed here because every one
of them changes observable behaviour and each has a regression test:

1. ``_cleanup()`` no longer ``rmtree``s the download directory. The borrowed
   version deleted everything under ``local_dir``, destroying weights
   ``huggingface_hub`` had already cached. It now removes only ``.incomplete``
   artifacts -- partial blobs -- and never a directory.
2. Cancellation emits its own wire event type, ``"cancelled"``. The borrowed
   code emitted ``event_type="error"`` for both a genuine failure and a user
   cancellation, so the frontend could not tell "try again" from "you stopped
   this".
3. The retry arithmetic is right. See :data:`MAX_DOWNLOAD_RETRIES`.
4. There is no bare ``except: pass`` anywhere: notification failures are logged
   at debug level so a client that has gone away is visible in the log without
   being fatal.
5. The event channels are ``chatterbox.*`` instead of ``voxcpm.*``.
6. ``src.hf_progress`` is imported **relatively** (``..src.hf_progress``), so
   nothing here depends on ``sys.path`` or on the absolute name of this
   package.
7. Dead imports (``sys``, ``field``) are gone.

This module never imports the ComfyUI node-API package, ``comfy.*`` or ``folder_paths``.
``server`` is imported lazily inside the two emit helpers, so the whole module
imports and unit-tests without booting ComfyUI.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

__all__ = [
    "DISABLE_XET_DOWNLOADS",
    "DownloadCancelledError",
    "DownloadError",
    "DownloadManager",
    "DownloadState",
    "DownloadStatus",
    "MAX_DOWNLOAD_RETRIES",
    "MAX_DOWNLOAD_ATTEMPTS",
    "PROGRESS_CHANNEL",
    "STATUS_CHANNEL",
    "INCOMPLETE_SUFFIX",
    "get_download_manager",
]

# ── Event channels ──────────────────────────────────────────────────
# Renamed from the borrowed ``voxcpm.*`` namespace (plan §3.1.5).
PROGRESS_CHANNEL = "chatterbox.download_progress"
STATUS_CHANNEL = "chatterbox.status"

# ── Xet Configuration Flag ──────────────────────────────────────────
# Set to False to enable hf_xet downloading (if installed).
# Set to True to enforce standard HTTP/tqdm downloads (no-xet).
DISABLE_XET_DOWNLOADS = True

# ── Retry policy ────────────────────────────────────────────────────
# A transient network error on a multi-gigabyte safetensors pull is common.
# ``MAX_DOWNLOAD_RETRIES`` is the number of *retries*, so the total attempt
# count is one more than this. The borrowed code set ``MAX_DOWNLOAD_RETRIES =
# 2`` and looped ``range(1, MAX_DOWNLOAD_RETRIES + 1)``, which is two attempts
# total -- contradicting its own "1 initial + retries" comment -- while its
# toast printed ``attempt + 1 / MAX_DOWNLOAD_RETRIES``, reading ``2/2`` on the
# only retry it ever made. Both are fixed below (plan §3.1.3).
MAX_DOWNLOAD_RETRIES = 2
MAX_DOWNLOAD_ATTEMPTS = MAX_DOWNLOAD_RETRIES + 1

# ── Cleanup policy ──────────────────────────────────────────────────
# ``huggingface_hub`` writes partial blobs with this suffix and resumes into
# them. They are the only thing safe to delete between attempts; everything
# else in the directory is a completed file the user paid bandwidth for.
INCOMPLETE_SUFFIX = ".incomplete"

# How long the event loop waits on an empty queue before re-checking the
# cancellation flag. Small enough to feel instant, large enough not to spin.
_EVENT_POLL_TIMEOUT = 0.05

# How long to wait for the worker thread to notice a cancellation before
# giving up on a clean exit.
_CANCEL_JOIN_TIMEOUT = 5.0

# How long to wait for a worker thread to finish after the queue drains.
_FINISH_JOIN_TIMEOUT = 30.0


# ── Third-party log noise (deliberate, and deliberately narrow) ──────
#
# The borrowed VoxCPM code silenced four *global* loggers by hand --
# ``httpx``, ``urllib3.connectionpool``, ``filelock`` and the handlers hanging
# off ``huggingface_hub``'s own logger -- for the duration of a download, then
# reset them all to ``NOTSET`` afterwards.
#
# **The third-party ``setLevel`` calls are gone**, and that is the decision:
#
# * ComfyUI installs its own handler on the **root** logger and owns it. A custom
#   node that reaches into another package's logger, raises its level, and later
#   resets it to ``NOTSET``, is mutating shared state it does not own -- and
#   "restore to NOTSET" is not "restore": it clobbers whatever level another
#   extension had legitimately chosen. The S7.3 guard (``tests/
#   test_logging_idiom.py``) bans ``setLevel`` for exactly this reason, and this
#   file was the last exemption. The trade is accepted: ``httpx`` / ``urllib3``
#   may emit a few INFO lines during a first-run download, which is noise, not
#   breakage. ``tests/test_logging_idiom.py::TestDownloaderLoggingDecision`` pins
#   the decision so it cannot drift back by accident.
#
# * ``huggingface_hub`` is different, and it is why the function still exists:
#   that library *owns* its verbosity and publishes an API for it. Asking a
#   package to be quiet through its own documented entry point is not reaching
#   into anyone else's logging tree.


def _suppress_hf_http_logs() -> None:
    """Ask ``huggingface_hub`` to stop narrating a download, through its own API."""
    try:
        from huggingface_hub import logging as hf_logging
    except ImportError:
        logging.debug(
            "[Chatterbox TTS] huggingface_hub is not importable; there are no "
            "download logs to quiet."
        )
        return
    hf_logging.set_verbosity_error()
    hf_logging.disable_propagation()


def _restore_hf_http_logs() -> None:
    """Undo :func:`_suppress_hf_http_logs`."""
    try:
        from huggingface_hub import logging as hf_logging
    except ImportError:
        logging.debug(
            "[Chatterbox TTS] huggingface_hub is not importable; leaving its log "
            "levels alone."
        )
        return
    hf_logging.set_verbosity_warning()
    hf_logging.enable_propagation()


def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _format_speed(bytes_per_sec: float) -> str:
    return f"{_format_bytes(bytes_per_sec)}/s"


class DownloadCancelledError(Exception):
    """Raised when the user cancels an in-flight download."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Download cancelled for model '{model_name}'")


class DownloadError(Exception):
    """Raised when a download fails and no retries remain."""

    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        self.message = message
        super().__init__(f"Download failed for model '{model_name}': {message}")


class DownloadStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class DownloadState:
    """Mutable snapshot of one model's download, as the frontend sees it."""

    model_name: str
    repo_id: str
    local_dir: str
    status: DownloadStatus = DownloadStatus.IDLE
    progress_percentage: float = 0.0
    current_file: str = ""
    file_index: int = 0
    total_files: int = 0
    speed: float = 0.0
    bytes_completed: int = 0
    total_bytes: int = 0
    transfer_bytes_completed: int = 0
    transfer_bytes_total: int = 0
    transfer_speed: float = 0.0
    dedup_saved_bytes: int = 0
    error_message: Optional[str] = None
    cancel_requested_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_name": self.model_name,
            "repo_id": self.repo_id,
            "local_dir": self.local_dir,
            "status": self.status.value,
            "progress_percentage": round(self.progress_percentage, 2),
            "current_file": self.current_file,
            "file_index": self.file_index,
            "total_files": self.total_files,
            "speed": self.speed,
            "bytes_completed": self.bytes_completed,
            "total_bytes": self.total_bytes,
        }
        # Xet-only fields ride along only when they carry a value, so the
        # frontend's default shape is unchanged for plain HTTP downloads.
        if self.transfer_bytes_completed or self.transfer_bytes_total:
            d["transfer_bytes_completed"] = self.transfer_bytes_completed
            d["transfer_bytes_total"] = self.transfer_bytes_total
            d["transfer_speed"] = self.transfer_speed
        if self.dedup_saved_bytes:
            d["dedup_saved_bytes"] = self.dedup_saved_bytes
        if self.error_message is not None:
            d["error_message"] = self.error_message
        if self.cancel_requested_at is not None:
            d["cancel_requested_at"] = self.cancel_requested_at
        return d


def _default_tracker_factory(token: Optional[str]) -> Any:
    """Build the real progress tracker. Overridden in tests via DI."""
    from ..src.hf_progress import HfProgressTracker

    return HfProgressTracker(token=token)


class DownloadManager:
    """Owns in-flight downloads: progress events, retries and cancellation."""

    def __init__(self, tracker_factory: Optional[Callable[[Optional[str]], Any]] = None):
        self._active_downloads: Dict[str, DownloadState] = {}
        self._active_trackers: Dict[str, Any] = {}
        self._active_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._tracker_factory = tracker_factory or _default_tracker_factory

    # ── Public API ─────────────────────────────────────────────────

    def download_model(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str] = None,
    ) -> str:
        """Download ``repo_id`` into ``local_dir`` and return the local path.

        Args:
            model_name: Friendly pack name; also the transfer id and the key
                progress is reported under.
            repo_id: Hugging Face repository id.
            local_dir: Destination directory.
            client_id: Websocket SID to target. ``None`` broadcasts.

        Raises:
            DownloadCancelledError: the user cancelled.
            DownloadError: every attempt failed.
        """
        # Apply the Xet flag to both the environment and huggingface_hub's
        # in-process constants *before* the tracker asks whether Xet is
        # available -- it caches the answer on first use.
        if DISABLE_XET_DOWNLOADS:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            self._set_hf_xet_flag(True)
        else:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
            self._set_hf_xet_flag(False)

        from ..src.hf_progress import EventType, TransferCancelledError

        return self._download_model_inner(
            model_name, repo_id, local_dir, client_id,
            EventType, TransferCancelledError,
        )

    def cancel_download(self, model_name: str) -> bool:
        """Request cancellation. Returns False when nothing is in flight."""
        with self._lock:
            tracker = self._active_trackers.get(model_name)
            if tracker is None:
                return False

            tracker.cancel(transfer_id=model_name)

            state = self._active_downloads.get(model_name)
            if state is not None:
                state.cancel_requested_at = time.time()

            logging.info("[Chatterbox TTS] Cancel requested for model '%s'", model_name)
            return True

    def get_status(self, model_name: str) -> Optional[DownloadState]:
        with self._lock:
            return self._active_downloads.get(model_name)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                name: state.to_dict()
                for name, state in self._active_downloads.items()
            }

    # ── Download driver ────────────────────────────────────────────

    def _download_model_inner(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str],
        EventType: Any,
        TransferCancelledError: Any,
    ) -> str:
        from ..src.hf_progress import is_xet_available

        token = self._get_token()
        tracker = self._tracker_factory(token)

        xet_avail = not DISABLE_XET_DOWNLOADS and is_xet_available()
        xet_status = "ENABLED (subprocess)" if xet_avail else "DISABLED (standard HTTP)"
        logging.info(
            "[Chatterbox TTS] Starting download for '%s' from '%s' (Xet: %s)",
            model_name, repo_id, xet_status,
        )

        with self._lock:
            self._active_trackers[model_name] = tracker
            self._active_downloads[model_name] = DownloadState(
                model_name=model_name,
                repo_id=repo_id,
                local_dir=local_dir,
                status=DownloadStatus.DOWNLOADING,
            )

        self._emit_download_event(
            model_name=model_name,
            event_type="start",
            phase="downloading",
            client_id=client_id,
            extra_data={
                "repo_id": repo_id,
                "local_dir": local_dir,
            },
        )

        _suppress_hf_http_logs()

        try:
            for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
                result_path = None
                error_occurred = None
                was_cancelled = False

                def do_download() -> None:
                    nonlocal result_path, error_occurred, was_cancelled
                    try:
                        result_path = tracker.download_snapshot(
                            repo_id=repo_id,
                            local_dir=local_dir,
                            transfer_id=model_name,
                        )
                    except TransferCancelledError:
                        was_cancelled = True
                    except KeyboardInterrupt:
                        was_cancelled = True
                    except Exception as exc:  # noqa: BLE001 -- captured, then retried
                        error_occurred = exc

                # ALWAYS on a background thread: this keeps the main ComfyUI
                # thread free so the loop below drains the tracker's queue and
                # pushes UI updates in real time.
                download_thread = threading.Thread(target=do_download, daemon=True)
                with self._lock:
                    self._active_threads[model_name] = download_thread
                download_thread.start()

                try:
                    while download_thread.is_alive() or not tracker.event_queue.empty():
                        if tracker.is_cancelled(transfer_id=model_name) or was_cancelled:
                            self._handle_cancellation(model_name, tracker, client_id)
                            raise DownloadCancelledError(model_name)

                        try:
                            event = tracker.event_queue.get(timeout=_EVENT_POLL_TIMEOUT)
                        except queue.Empty:
                            continue

                        self._update_state_from_event(model_name, event, EventType)
                        self._emit_progress_event(model_name, event, client_id)

                        if event.event_type == EventType.PROGRESS:
                            self._print_console_progress(model_name)

                        if event.event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                            if event.event_type == EventType.ERROR:
                                error_occurred = event.error or "Unknown download error"
                            if event.event_type == EventType.CANCELLED:
                                was_cancelled = True
                            break

                    download_thread.join(timeout=_FINISH_JOIN_TIMEOUT)

                    # Drain whatever the tracker queued between the last poll
                    # and the thread exiting.
                    while not tracker.event_queue.empty():
                        try:
                            event = tracker.event_queue.get_nowait()
                        except queue.Empty:
                            break
                        self._update_state_from_event(model_name, event, EventType)
                        self._emit_progress_event(model_name, event, client_id)

                    # Final cancel check: the request may have arrived while the
                    # queue was draining.
                    if tracker.is_cancelled(transfer_id=model_name) or was_cancelled:
                        self._handle_cancellation(model_name, tracker, client_id)
                        raise DownloadCancelledError(model_name)

                    if error_occurred:
                        if attempt < MAX_DOWNLOAD_ATTEMPTS:
                            logging.warning(
                                "[Chatterbox TTS] Download attempt %d/%d failed for '%s': %s. Retrying...",
                                attempt, MAX_DOWNLOAD_ATTEMPTS, model_name, error_occurred,
                            )
                            self._emit_status_toast(
                                model_name=model_name,
                                client_id=client_id,
                                # `attempt` is the one that just failed, so the
                                # number to show is the one about to start.
                                detail=(
                                    f"Attempt {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS} "
                                    f"after network error: {error_occurred}"
                                ),
                            )
                            # Drop only the partial blobs; the completed files
                            # huggingface_hub already wrote stay on disk.
                            self._cleanup(model_name, local_dir)
                            continue

                        self._update_state(
                            model_name,
                            status=DownloadStatus.ERROR,
                            error_message=str(error_occurred),
                        )
                        self._emit_download_event(
                            model_name=model_name,
                            event_type="error",
                            phase="error",
                            client_id=client_id,
                            extra_data={"error": str(error_occurred)},
                        )
                        self._print_console_progress(model_name, error=True)
                        raise DownloadError(model_name, str(error_occurred))

                    self._update_state(model_name, status=DownloadStatus.COMPLETE)
                    self._emit_download_event(
                        model_name=model_name,
                        event_type="complete",
                        phase="complete",
                        client_id=client_id,
                        extra_data={"local_dir": local_dir},
                    )
                    self._print_console_progress(model_name, complete=True)

                    return result_path or local_dir

                except DownloadCancelledError:
                    self._cleanup(model_name, local_dir)
                    raise

            # Unreachable: the final attempt either returns or raises.
            raise DownloadError(model_name, "Download failed after all retries")
        finally:
            _restore_hf_http_logs()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _set_hf_xet_flag(disabled: bool) -> None:
        """Mirror the Xet policy into ``huggingface_hub.constants``."""
        try:
            import huggingface_hub.constants

            huggingface_hub.constants.HF_HUB_DISABLE_XET = disabled
        except ImportError:
            logging.debug(
                "[Chatterbox TTS] huggingface_hub is not importable; the Xet policy only reached "
                "the environment"
            )

    @staticmethod
    def _get_token() -> Optional[str]:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
            if token:
                return token
        except Exception as exc:  # noqa: BLE001 -- an unauthenticated download is fine
            logging.debug("[Chatterbox TTS] No stored Hugging Face token available: %s", exc)
        return None

    def _handle_cancellation(
        self,
        model_name: str,
        tracker: Any,
        client_id: Optional[str],
    ) -> None:
        """Cancel the tracker, tell the frontend, and drain the worker thread."""
        self._update_state(model_name, status=DownloadStatus.CANCELLED)
        # A distinct event type, not "error": the frontend shows a different
        # toast and does not offer a retry for a user-initiated stop.
        self._emit_download_event(
            model_name=model_name,
            event_type="cancelled",
            phase="cancelled",
            client_id=client_id,
            extra_data={"error": "Download cancelled by user"},
        )
        self._print_console_progress(model_name, cancelled=True)
        tracker.cancel(transfer_id=model_name)
        # Join the worker so a cancellation never leaves a thread writing into
        # a directory the caller is about to reuse.
        with self._lock:
            thread = self._active_threads.get(model_name)
        if thread is not None:
            thread.join(timeout=_CANCEL_JOIN_TIMEOUT)
            if thread.is_alive():
                logging.warning(
                    "[Chatterbox TTS] Download thread for '%s' did not exit within %.0fs after "
                    "cancellation. Resources may temporarily leak.",
                    model_name, _CANCEL_JOIN_TIMEOUT,
                )

    @staticmethod
    def _emit_status_toast(
        model_name: str,
        detail: str,
        client_id: Optional[str] = None,
    ) -> None:
        """Pop a transient frontend toast on the status channel."""
        try:
            from server import PromptServer

            if PromptServer.instance is None:
                return
            PromptServer.instance.send_sync(
                STATUS_CHANNEL,
                {
                    "severity": "warn",
                    "summary": f"Retrying download: {model_name}",
                    "detail": detail,
                    "life": 5000,
                },
                client_id,
            )
        except Exception as exc:  # noqa: BLE001 -- a dead client is not fatal
            logging.debug("[Chatterbox TTS] Could not send the retry status toast: %s", exc)

    @staticmethod
    def _emit_download_event(
        model_name: str,
        event_type: str,
        phase: str,
        client_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "model_name": model_name,
            "event_type": event_type,
            "phase": phase,
            "transfer_id": model_name,
        }
        if extra_data:
            payload.update(extra_data)
        _send_progress(payload, client_id)

    @staticmethod
    def _emit_progress_event(
        model_name: str,
        event: Any,
        client_id: Optional[str] = None,
    ) -> None:
        payload = event.to_dict()
        payload["model_name"] = model_name
        _send_progress(payload, client_id)

    def _update_state(
        self,
        model_name: str,
        status: Optional[DownloadStatus] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return
            if status is not None:
                state.status = status
            if error_message is not None:
                state.error_message = error_message

    def _update_state_from_event(
        self,
        model_name: str,
        event: Any,
        EventType: Any,
    ) -> None:
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return
            state.progress_percentage = event.percentage
            state.current_file = event.filename
            state.file_index = event.file_index
            state.total_files = event.total_files

            if event.event_type == EventType.COMPLETE:
                state.status = DownloadStatus.COMPLETE
                return
            if event.event_type == EventType.CANCELLED:
                state.status = DownloadStatus.CANCELLED
                return

            state.speed = event.speed
            state.bytes_completed = event.bytes_completed
            state.total_bytes = event.total_bytes

            state.transfer_bytes_completed = event.transfer_bytes_completed
            state.transfer_bytes_total = event.transfer_bytes_total
            state.transfer_speed = event.transfer_speed
            state.dedup_saved_bytes = event.dedup_saved_bytes

        if event.error:
            with self._lock:
                state = self._active_downloads.get(model_name)
                if state is not None:
                    state.error_message = str(event.error)
                    state.status = DownloadStatus.ERROR

    def _print_console_progress(
        self,
        model_name: str,
        complete: bool = False,
        error: bool = False,
        cancelled: bool = False,
    ) -> None:
        """Log download progress through the root logger, with the package prefix.

        Console progress goes to ``logging.debug``: the frontend already has a
        live progress bar over the websocket, so a stdout bar would be noise.
        """
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return

        if cancelled:
            logging.info("[Chatterbox TTS] Download cancelled: %s", model_name)
            return

        if error:
            logging.error(
                "[Chatterbox TTS] Download failed: %s - %s", model_name, state.error_message or "unknown error"
            )
            return

        if complete:
            total = _format_bytes(state.total_bytes) if state.total_bytes else "unknown size"
            logging.info("[Chatterbox TTS] Download complete: %s (%s)", model_name, total)
            return

        pct = state.progress_percentage
        speed_str = _format_speed(state.speed) if state.speed > 0 else "-"

        file_info = ""
        if state.total_files > 1:
            file_info = f" file {state.file_index + 1}/{state.total_files}"

        logging.debug("[Chatterbox TTS] Downloading %s: %.1f%% %s%s", model_name, pct, speed_str, file_info)

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self, model_name: str, local_dir: str) -> None:
        """Release the bookkeeping for ``model_name`` and drop partial blobs.

        CORRECTION (plan §3.1.1): the borrowed implementation deleted *every*
        file and subdirectory under ``local_dir``. That threw away the
        completed weights ``huggingface_hub`` had already written, so a retry
        re-downloaded gigabytes. Only ``.incomplete`` artifacts are removed
        here, and no directory is ever deleted.
        """
        self._remove_incomplete_artifacts(local_dir)
        with self._lock:
            self._active_downloads.pop(model_name, None)
            self._active_trackers.pop(model_name, None)
            self._active_threads.pop(model_name, None)

    @staticmethod
    def _remove_incomplete_artifacts(local_dir: str) -> None:
        """Delete every ``*.incomplete`` blob under ``local_dir``, recursively.

        Anything that is not a file carrying the suffix is left untouched --
        including completed weights and the ``.cache/huggingface`` tree that
        holds the download metadata.
        """
        if not local_dir or not os.path.isdir(local_dir):
            return
        for dirpath, _dirnames, filenames in os.walk(local_dir):
            for filename in filenames:
                if not filename.endswith(INCOMPLETE_SUFFIX):
                    continue
                artifact = os.path.join(dirpath, filename)
                try:
                    os.remove(artifact)
                    logging.debug("[Chatterbox TTS] Removed partial download artifact: %s", artifact)
                except OSError as exc:
                    logging.warning("[Chatterbox TTS] Could not remove partial download %s: %s", artifact, exc)


def _send_progress(payload: Dict[str, Any], client_id: Optional[str]) -> None:
    """Push ``payload`` on the progress channel, targeted or broadcast."""
    try:
        from server import PromptServer

        if PromptServer.instance is None:
            return
        if client_id:
            PromptServer.instance.send_sync(PROGRESS_CHANNEL, payload, client_id)
        else:
            PromptServer.instance.send_sync(PROGRESS_CHANNEL, payload)
    except Exception as exc:  # noqa: BLE001 -- a vanished client is not fatal
        logging.debug(
            "[Chatterbox TTS] Could not deliver a '%s' progress event to %s: %s",
            payload.get("event_type"), client_id or "all clients", exc,
        )


_download_manager: Optional[DownloadManager] = None
_manager_lock = threading.Lock()


def get_download_manager() -> DownloadManager:
    """The process-wide :class:`DownloadManager` singleton."""
    global _download_manager
    if _download_manager is None:
        with _manager_lock:
            if _download_manager is None:
                _download_manager = DownloadManager()
    return _download_manager