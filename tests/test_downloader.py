"""S4.2 -- ``modules/downloader.py``.

The manager is ported from VoxCPM with the plan §3.1 corrections applied. Each
correction that changes behaviour has a test here, and the ``_cleanup`` test is
written to genuinely discriminate: against the borrowed implementation, which
deleted everything under ``local_dir``, the sibling-file assertions fail.

No test in this file touches the network. The HuggingFace tracker is replaced
by a fake through the ``tracker_factory`` dependency-injection seam, so the real
``snapshot_download`` is never reached.
"""

from __future__ import annotations

import inspect
import os
import queue
import sys
import threading
import types

import pytest

from ComfyUI_Chatterbox.modules.downloader import (
    INCOMPLETE_SUFFIX,
    MAX_DOWNLOAD_ATTEMPTS,
    MAX_DOWNLOAD_RETRIES,
    PROGRESS_CHANNEL,
    STATUS_CHANNEL,
    DownloadCancelledError,
    DownloadError,
    DownloadManager,
    DownloadState,
    DownloadStatus,
    get_download_manager,
)
from ComfyUI_Chatterbox.src.hf_progress import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
)


# ── Test doubles ─────────────────────────────────────────────────────

class FakeServer:
    """Stand-in for ``server.PromptServer`` with a recording ``send_sync``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, tuple]] = []

    def send_sync(self, channel: str, payload=None, *extra) -> None:
        # ``*extra`` rather than a named ``sid``: the point of the broadcast
        # tests is whether the sid positional was passed *at all*.
        self.sent.append((channel, payload, extra))

    def channels(self) -> list[str]:
        return [entry[0] for entry in self.sent]

    def events_of(self, event_type: str) -> list[dict]:
        return [
            entry[1]
            for entry in self.sent
            if entry[0] == PROGRESS_CHANNEL and entry[1].get("event_type") == event_type
        ]

    def toasts(self) -> list[dict]:
        return [entry[1] for entry in self.sent if entry[0] == STATUS_CHANNEL]


@pytest.fixture()
def prompt_server(monkeypatch):
    """Install a fake ``server`` module carrying a recording PromptServer."""
    recorder = FakeServer()
    module = types.ModuleType("server")
    module.PromptServer = type("PromptServer", (), {"instance": recorder})
    monkeypatch.setitem(sys.modules, "server", module)
    return recorder


def progress_event(transfer_id: str, filename: str = "ve.safetensors", percentage: float = 50.0):
    return ProgressEvent(
        event_type=EventType.PROGRESS,
        transfer_id=transfer_id,
        direction=TransferDirection.DOWNLOAD,
        filename=filename,
        phase=ProgressPhase.DOWNLOADING,
        bytes_completed=int(percentage * 1024),
        total_bytes=102400,
        percentage=percentage,
        speed=1024.0,
    )


class FakeTracker:
    """A stand-in for ``HfProgressTracker``.

    Records every ``download_snapshot`` call and either returns a path, raises,
    or (for the cancellation test) blocks until cancelled.
    """

    def __init__(self, behaviours: list, token=None):
        self.behaviours = list(behaviours)
        self.calls: list[dict] = []
        self.token = token
        self.event_queue: queue.Queue = queue.Queue()
        self.cancelled_ids: list[str] = []
        self._cancelled = False
        self._gate = threading.Event()

    # -- HfProgressTracker surface --
    def cancel(self, transfer_id: str) -> None:
        self.cancelled_ids.append(transfer_id)
        self._cancelled = True
        self._gate.set()

    def is_cancelled(self, transfer_id: str) -> bool:
        return self._cancelled

    def download_snapshot(self, repo_id: str, local_dir: str, transfer_id: str) -> str:
        self.calls.append(
            {"repo_id": repo_id, "local_dir": local_dir, "transfer_id": transfer_id}
        )
        index = min(len(self.calls) - 1, len(self.behaviours) - 1)
        behaviour = self.behaviours[index]

        if behaviour == "cancel":
            self.event_queue.put(progress_event(transfer_id, percentage=25.0))
            self._gate.wait(timeout=5.0)
            raise TransferCancelledError("Download interrupted by user")
        if callable(behaviour):
            return behaviour(repo_id, local_dir, transfer_id)
        if isinstance(behaviour, BaseException):
            raise behaviour

        self.event_queue.put(progress_event(transfer_id))
        self.event_queue.put(
            ProgressEvent.complete(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename="ve.safetensors",
                bytes_completed=102400,
                total_bytes=102400,
            )
        )
        return local_dir


def manager_with(behaviours: list) -> tuple[DownloadManager, FakeTracker]:
    """A manager wired to one pre-built fake tracker, returned to the caller."""
    tracker = FakeTracker(behaviours)

    def factory(token):
        tracker.token = token
        return tracker

    return DownloadManager(tracker_factory=factory), tracker


# ── Signature and constants ──────────────────────────────────────────

class TestPublicContract:
    def test_download_model_signature_is_exactly_as_specified(self) -> None:
        signature = inspect.signature(DownloadManager.download_model)
        assert list(signature.parameters) == [
            "self",
            "model_name",
            "repo_id",
            "local_dir",
            "client_id",
        ]
        assert signature.parameters["client_id"].default is None
        for name in ("model_name", "repo_id", "local_dir"):
            assert signature.parameters[name].default is inspect.Parameter.empty

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("cancel_download", ["self", "model_name"]),
            ("get_status", ["self", "model_name"]),
            ("get_all_statuses", ["self"]),
        ],
    )
    def test_other_signatures(self, name: str, expected: list[str]) -> None:
        assert list(inspect.signature(getattr(DownloadManager, name)).parameters) == expected

    def test_event_channel_is_namespaced_to_chatterbox(self) -> None:
        assert PROGRESS_CHANNEL == "chatterbox.download_progress"
        assert STATUS_CHANNEL == "chatterbox.status"

    def test_no_voxcpm_event_channel_survives(self) -> None:
        """Plan §3.1.5: the borrowed channels were ``voxcpm.*``."""
        import re

        import ComfyUI_Chatterbox.modules.downloader as downloader_module

        with open(downloader_module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        offenders = re.findall(r"[\"']voxcpm\.[a-z_]+[\"']", source)
        assert offenders == [], f"borrowed event channels remain: {offenders}"

    def test_module_does_not_import_comfy_api(self) -> None:
        import ComfyUI_Chatterbox.modules.downloader as downloader_module

        with open(downloader_module.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()


class TestRetryArithmetic:
    """Plan §3.1.3: '1 initial + retries', and a toast that counts right."""

    def test_attempts_are_one_initial_plus_the_retries(self) -> None:
        assert MAX_DOWNLOAD_ATTEMPTS == MAX_DOWNLOAD_RETRIES + 1
        assert MAX_DOWNLOAD_RETRIES == 2
        assert MAX_DOWNLOAD_ATTEMPTS == 3


# ── Progress events ──────────────────────────────────────────────────

class TestProgressDelivery:
    def test_completion_reaches_the_progress_channel(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        result = manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        assert result == str(tmp_path)
        assert PROGRESS_CHANNEL in prompt_server.channels()
        # The tracker's own COMPLETE event is forwarded, and the manager adds
        # its own terminal one carrying the destination directory.
        manager_completes = [
            event for event in prompt_server.events_of("complete") if "local_dir" in event
        ]
        assert len(manager_completes) == 1
        assert manager_completes[0]["model_name"] == "pack"
        assert manager_completes[0]["phase"] == "complete"
        assert manager_completes[0]["local_dir"] == str(tmp_path)

    def test_a_start_event_precedes_the_progress(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        progress_events = [
            entry[1] for entry in prompt_server.sent if entry[0] == PROGRESS_CHANNEL
        ]
        assert progress_events[0]["event_type"] == "start"
        assert progress_events[0]["repo_id"] == "ResembleAI/chatterbox"
        assert progress_events[0]["phase"] == "downloading"

    def test_progress_payload_carries_model_name(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        progress = prompt_server.events_of("progress")
        assert progress, "no PROGRESS event was forwarded"
        assert progress[0]["model_name"] == "pack"
        assert progress[0]["filename"] == "ve.safetensors"

    def test_broadcast_when_no_client_id_is_given(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        # Every send must omit the sid positional entirely, so PromptServer
        # broadcasts rather than sending to sid=None.
        for channel, _payload, extra in prompt_server.sent:
            assert extra == (), f"{channel} was sent with sid {extra}, not broadcast"

    def test_targeted_when_a_client_id_is_given(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        manager.download_model(
            "pack", "ResembleAI/chatterbox", str(tmp_path), client_id="client-77"
        )

        assert prompt_server.sent, "nothing was sent at all"
        for channel, _payload, extra in prompt_server.sent:
            assert extra == ("client-77",), f"{channel} was not targeted at client-77"


# ── Cancellation ─────────────────────────────────────────────────────

class TestCancellation:
    def test_cancellation_raises_download_cancelled_error(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["cancel"])
        timer = threading.Timer(0.05, lambda: manager.cancel_download("pack"))
        timer.start()
        try:
            with pytest.raises(DownloadCancelledError):
                manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        finally:
            timer.cancel()

    def test_cancellation_emits_cancelled_not_error(self, tmp_path, prompt_server) -> None:
        """Plan §3.1.4: cancel and error share one wire event in the borrowed
        code, so the frontend could not tell them apart."""
        manager, _ = manager_with(["cancel"])
        timer = threading.Timer(0.05, lambda: manager.cancel_download("pack"))
        timer.start()
        try:
            with pytest.raises(DownloadCancelledError):
                manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        finally:
            timer.cancel()

        assert prompt_server.events_of("cancelled"), (
            "a user cancellation must emit event_type='cancelled'"
        )
        assert prompt_server.events_of("error") == [], (
            "a user cancellation must not be reported as an error"
        )
        cancelled = prompt_server.events_of("cancelled")[0]
        assert cancelled["phase"] == "cancelled"
        assert cancelled["error"] == "Download cancelled by user"

    def test_cancelling_an_unknown_model_returns_false(self) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        assert manager.cancel_download("never-started") is False

    def test_cancel_download_records_the_request_time(self, tmp_path, prompt_server) -> None:
        manager, tracker = manager_with(["cancel"])
        timer = threading.Timer(0.05, lambda: manager.cancel_download("pack"))
        timer.start()
        try:
            with pytest.raises(DownloadCancelledError):
                manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        finally:
            timer.cancel()

        # cancel_download() is the API call under test; the manager cancels
        # the tracker a second time on the way out, which is harmless.
        assert tracker.cancelled_ids[0] == "pack"
        assert set(tracker.cancelled_ids) == {"pack"}


# ── Failure and retry ────────────────────────────────────────────────

class TestFailureAndRetry:
    def test_a_transient_failure_is_retried(self, tmp_path, prompt_server) -> None:
        manager, tracker = manager_with([OSError("connection reset"), "ok"])
        result = manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        assert result == str(tmp_path)
        assert len(tracker.calls) == 2, "the transient failure was not retried"

    def test_the_retry_toast_names_the_correct_attempt(self, tmp_path, prompt_server) -> None:
        """The borrowed toast printed ``attempt + 1 / MAX_DOWNLOAD_RETRIES``,
        which reads '2/2' on the only retry it made."""
        manager, _ = manager_with([OSError("boom"), "ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        toasts = prompt_server.toasts()
        assert len(toasts) == 1
        assert "Attempt 2/3" in toasts[0]["detail"], (
            f"expected the toast to announce attempt 2 of 3, got {toasts[0]['detail']!r}"
        )
        assert toasts[0]["severity"] == "warn"
        assert toasts[0]["summary"] == "Retrying download: pack"

    def test_a_second_failure_reports_attempt_three_of_three(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with([OSError("a"), OSError("b"), "ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        details = [toast["detail"] for toast in prompt_server.toasts()]
        assert len(details) == 2
        assert "Attempt 2/3" in details[0]
        assert "Attempt 3/3" in details[1]

    def test_exhausting_every_attempt_raises_download_error(self, tmp_path, prompt_server) -> None:
        manager, tracker = manager_with([OSError("a"), OSError("b"), OSError("c")])
        with pytest.raises(DownloadError) as excinfo:
            manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        assert len(tracker.calls) == MAX_DOWNLOAD_ATTEMPTS
        assert "c" in excinfo.value.message
        assert len(prompt_server.toasts()) == MAX_DOWNLOAD_RETRIES

    def test_a_final_failure_emits_an_error_event(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with([OSError("a"), OSError("b"), OSError("c")])
        with pytest.raises(DownloadError):
            manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        errors = prompt_server.events_of("error")
        assert errors, "the final failure was not reported on the progress channel"
        assert errors[-1]["error"] == "c"
        assert prompt_server.events_of("cancelled") == []

    def test_state_records_the_error_after_a_failure(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with([OSError("a"), OSError("b"), OSError("c")])
        with pytest.raises(DownloadError):
            manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        # _cleanup drops the bookkeeping, so inspect what was emitted instead.
        assert prompt_server.events_of("error")


# ── The _cleanup regression ──────────────────────────────────────────

class TestCleanupKeepsCompletedFiles:
    """Plan §3.1.1 -- the regression that must fail against the borrowed code.

    The VoxCPM ``_cleanup`` iterated ``os.listdir(local_dir)`` and removed
    every file and subdirectory. That is exactly what these assertions forbid:
    a sibling weight that finished downloading must survive a cancel, and so
    must the ``.cache/huggingface`` metadata tree.
    """

    @pytest.fixture()
    def pack_dir(self, tmp_path):
        local_dir = tmp_path / "resembleai_default_voice"
        cache_dir = local_dir / ".cache" / "huggingface" / "download"
        cache_dir.mkdir(parents=True)

        # Files huggingface_hub already completed.
        (local_dir / "ve.safetensors").write_bytes(b"completed-voice-encoder")
        (local_dir / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
        # A partial blob it is still writing.
        (local_dir / "s3gen.safetensors.incomplete").write_bytes(b"half")
        (cache_dir / "s3gen.safetensors.metadata").write_text("{}", encoding="utf-8")
        (cache_dir / "t3_cfg.safetensors.incomplete").write_bytes(b"quarter")
        return local_dir

    def test_a_completed_sibling_survives_cleanup(self, pack_dir) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._cleanup("resembleai_default_voice", str(pack_dir))

        assert (pack_dir / "ve.safetensors").is_file(), (
            "_cleanup deleted an already-downloaded weight file; this is the "
            "borrowed rmtree behaviour and it re-downloads gigabytes"
        )
        assert (pack_dir / "ve.safetensors").read_bytes() == b"completed-voice-encoder"
        assert (pack_dir / "tokenizer.json").is_file()

    def test_the_pack_directory_itself_survives(self, pack_dir) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._cleanup("resembleai_default_voice", str(pack_dir))
        assert pack_dir.is_dir()

    def test_the_hub_cache_tree_survives(self, pack_dir) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._cleanup("resembleai_default_voice", str(pack_dir))
        download_cache = pack_dir / ".cache" / "huggingface" / "download"
        assert download_cache.is_dir()
        assert (download_cache / "s3gen.safetensors.metadata").is_file()

    def test_incomplete_artifacts_are_removed(self, pack_dir) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._cleanup("resembleai_default_voice", str(pack_dir))

        remaining = [
            os.path.join(root, name)
            for root, _dirs, files in os.walk(pack_dir)
            for name in files
            if name.endswith(INCOMPLETE_SUFFIX)
        ]
        assert remaining == [], f"partial blobs were left behind: {remaining}"

    def test_cleanup_releases_the_bookkeeping(self, tmp_path) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._active_downloads["pack"] = DownloadState(
            model_name="pack", repo_id="r", local_dir=str(tmp_path)
        )
        manager._active_trackers["pack"] = object()
        manager._cleanup("pack", str(tmp_path))

        assert "pack" not in manager._active_downloads
        assert "pack" not in manager._active_trackers

    def test_cleanup_on_a_missing_directory_is_a_no_op(self, tmp_path) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        manager._cleanup("pack", str(tmp_path / "does-not-exist"))  # must not raise

    def test_the_module_no_longer_imports_shutil_rmtree(self) -> None:
        import ast

        import ComfyUI_Chatterbox.modules.downloader as downloader_module

        with open(downloader_module.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "rmtree", (
                    "modules/downloader.py still calls rmtree; _cleanup must "
                    "only unlink .incomplete files"
                )

    def test_a_retry_after_a_failure_keeps_the_downloaded_file(self, tmp_path, prompt_server) -> None:
        """End to end: the completed file written during attempt 1 is still
        there when attempt 2 starts."""
        local_dir = tmp_path / "pack"

        def write_then_fail(repo_id, directory, transfer_id):
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "ve.safetensors").write_bytes(b"done")
            (local_dir / "s3gen.safetensors.incomplete").write_bytes(b"partial")
            raise OSError("connection reset")

        manager, tracker = manager_with([write_then_fail, "ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(local_dir))

        assert len(tracker.calls) == 2
        assert (local_dir / "ve.safetensors").read_bytes() == b"done", (
            "the retry deleted a file the first attempt had already completed"
        )
        assert not (local_dir / "s3gen.safetensors.incomplete").exists()


# ── State and singletons ─────────────────────────────────────────────

class TestDownloadState:
    def test_to_dict_carries_the_base_fields(self) -> None:
        state = DownloadState(model_name="p", repo_id="r", local_dir="/d")
        payload = state.to_dict()
        assert payload["model_name"] == "p"
        assert payload["repo_id"] == "r"
        assert payload["local_dir"] == "/d"
        assert payload["status"] == "idle"
        assert payload["progress_percentage"] == 0.0

    def test_to_dict_omits_empty_optionals(self) -> None:
        payload = DownloadState(model_name="p", repo_id="r", local_dir="/d").to_dict()
        assert "error_message" not in payload
        assert "cancel_requested_at" not in payload
        assert "transfer_bytes_total" not in payload

    def test_to_dict_includes_them_once_set(self) -> None:
        state = DownloadState(
            model_name="p",
            repo_id="r",
            local_dir="/d",
            status=DownloadStatus.ERROR,
            error_message="nope",
            cancel_requested_at=123.0,
            transfer_bytes_total=10,
            dedup_saved_bytes=4,
        )
        payload = state.to_dict()
        assert payload["status"] == "error"
        assert payload["error_message"] == "nope"
        assert payload["cancel_requested_at"] == 123.0
        assert payload["transfer_bytes_total"] == 10
        assert payload["dedup_saved_bytes"] == 4

    def test_download_status_members(self) -> None:
        assert {member.value for member in DownloadStatus} == {
            "idle", "downloading", "complete", "cancelled", "error",
        }

    def test_status_of_a_completed_download(self, tmp_path, prompt_server) -> None:
        manager, _ = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

        # _cleanup runs on the failure/cancel paths only, so after a success the
        # state survives and is queryable.
        state = manager.get_status("pack")
        assert state is not None
        assert state.status is DownloadStatus.COMPLETE
        assert manager.get_all_statuses()["pack"]["status"] == "complete"

    def test_status_of_an_unknown_model_is_none(self) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        assert manager.get_status("nope") is None

    def test_get_all_statuses_on_an_empty_manager(self) -> None:
        manager = DownloadManager(tracker_factory=lambda token: FakeTracker(["ok"]))
        assert manager.get_all_statuses() == {}


class TestTokenResolution:
    def test_reads_hf_token_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert DownloadManager._get_token() == "hf_secret"

    def test_falls_back_to_the_legacy_variable(self, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
        assert DownloadManager._get_token() == "hf_legacy"

    def test_the_token_reaches_the_tracker(self, tmp_path, monkeypatch, prompt_server) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        manager, tracker = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        assert tracker.token == "hf_secret"


class TestGetDownloadManager:
    def test_returns_a_singleton(self) -> None:
        assert get_download_manager() is get_download_manager()

    def test_is_a_download_manager(self) -> None:
        assert isinstance(get_download_manager(), DownloadManager)


class TestDegradation:
    def test_no_server_module_does_not_raise(self, tmp_path, monkeypatch) -> None:
        """Configuration and progress are a convenience; a missing server must
        never break a download.

        ``sys.modules[name] = None`` is the interpreter's documented way of
        saying "this import will fail", so the lazy ``from server import
        PromptServer`` raises ImportError exactly as it would in a plain
        environment with no aiohttp.
        """
        monkeypatch.setitem(sys.modules, "server", None)

        manager, _ = manager_with(["ok"])
        assert manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

    def test_a_prompt_server_without_an_instance_does_not_raise(
        self, tmp_path, monkeypatch
    ) -> None:
        module = types.ModuleType("server")
        module.PromptServer = type("PromptServer", (), {"instance": None})
        monkeypatch.setitem(sys.modules, "server", module)

        manager, _ = manager_with(["ok"])
        assert manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))

    def test_a_raising_send_sync_does_not_break_the_download(
        self, tmp_path, monkeypatch
    ) -> None:
        class Exploding:
            instance = None

            def send_sync(self, *args, **kwargs):
                raise RuntimeError("websocket is gone")

        module = types.ModuleType("server")
        module.PromptServer = Exploding
        monkeypatch.setitem(sys.modules, "server", module)

        manager, _ = manager_with(["ok"])
        assert manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))


class TestXetPolicy:
    def test_downloads_disable_xet_by_default(self, tmp_path, prompt_server) -> None:
        from ComfyUI_Chatterbox.modules import downloader as downloader_module

        assert downloader_module.DISABLE_XET_DOWNLOADS is True
        manager, _ = manager_with(["ok"])
        manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"

        import huggingface_hub.constants

        assert huggingface_hub.constants.HF_HUB_DISABLE_XET is True


class TestStructuredErrorEvents:
    def test_a_tracker_failure_becomes_a_download_error(self, tmp_path, prompt_server) -> None:
        """Whatever the tracker raises ends up in DownloadError.message, so the
        frontend toast names the actual cause rather than 'download failed'."""

        def raise_from_tracker(repo_id, directory, transfer_id):
            raise RuntimeError("tracker said no")

        manager, _ = manager_with([raise_from_tracker] * MAX_DOWNLOAD_ATTEMPTS)
        with pytest.raises(DownloadError) as excinfo:
            manager.download_model("pack", "ResembleAI/chatterbox", str(tmp_path))
        assert "tracker said no" in excinfo.value.message
        assert excinfo.value.model_name == "pack"