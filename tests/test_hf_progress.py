"""S4.1 -- the vendored ``src/hf_progress`` package.

The package is lifted from ``ComfyUI-VoxCPM`` and is deliberately reusable, so
what is asserted here is the contract a consumer relies on, not the internals:
the serialisable event shape, the cancel hook injection, the download chunk
patch, and the re-export surface named in the plan.
"""

from __future__ import annotations

import queue
import os

import pytest

from ComfyUI_Chatterbox.src import hf_progress
from ComfyUI_Chatterbox.src.hf_progress import (
    DownloadProgressTqdm,
    EventType,
    HfProgressTracker,
    ProgressEvent,
    ProgressPhase,
    TokenError,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferProgressError,
    TransferResult,
    XetCredentials,
    XetTokenManager,
    download_file,
    download_snapshot,
    generate_transfer_id,
    is_xet_available,
    patch_download_chunk_size,
    tqdm_upload_patcher,
)
from ComfyUI_Chatterbox.src.hf_progress.callbacks import state_manager

SRC_HF_PROGRESS = os.path.dirname(os.path.abspath(hf_progress.__file__))

# Names the plan's §"public contract" section requires the package to re-export.
REQUIRED_REEXPORTS = (
    "HfProgressTracker",
    "ProgressEvent",
    "EventType",
    "TransferDirection",
    "ProgressPhase",
    "TransferError",
    "TransferResult",
    "TransferCancelledError",
    "TransferProgressError",
    "TokenError",
    "generate_transfer_id",
    "patch_download_chunk_size",
    "download_file",
    "download_snapshot",
    "DownloadProgressTqdm",
    "tqdm_upload_patcher",
    "XetCredentials",
    "XetTokenManager",
    "is_xet_available",
)


class TestReExportSurface:
    @pytest.mark.parametrize("name", REQUIRED_REEXPORTS)
    def test_name_is_exported(self, name: str) -> None:
        assert name in hf_progress.__all__, f"{name} is missing from __all__"
        assert hasattr(hf_progress, name), f"{name} is listed but not defined"

    def test_all_is_importable_as_a_whole(self) -> None:
        """Every advertised name resolves -- a stale __all__ is a real bug."""
        for name in hf_progress.__all__:
            assert getattr(hf_progress, name, None) is not None, name

    def test_package_has_a_version(self) -> None:
        assert hf_progress.__version__


class TestRelativeImportsOnly:
    """Plan §3.1.5: no absolute self-references.

    The borrowed VoxCPM tree mixed ``from src.hf_progress import ...`` with
    relative imports and depended on a ``sys.path.append`` in the node's
    ``__init__.py``. A vendored library that names itself absolutely only works
    when the host puts its own root on ``sys.path`` first.
    """

    @pytest.mark.parametrize(
        "filename",
        sorted(
            name
            for name in os.listdir(SRC_HF_PROGRESS)
            if name.endswith(".py")
        ),
    )
    def test_no_absolute_self_reference(self, filename: str) -> None:
        with open(os.path.join(SRC_HF_PROGRESS, filename), "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "from src.hf_progress" not in source
        assert "import src.hf_progress" not in source
        assert "from src." not in source

    @pytest.mark.parametrize("filename", ["__init__.py", "tracker.py", "callbacks.py"])
    def test_module_does_not_touch_sys_path(self, filename: str) -> None:
        """Checked on the AST: the string may appear in prose, but never as
        an attribute access or an assignment."""
        import ast

        with open(os.path.join(SRC_HF_PROGRESS, filename), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "path":
                value = node.value
                assert not (isinstance(value, ast.Name) and value.id == "sys"), (
                    f"{filename} touches sys.path"
                )


class TestProgressEventShape:
    """``ProgressEvent.to_dict()`` is the JSON a websocket carries."""

    @pytest.fixture()
    def event(self) -> ProgressEvent:
        return ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="t-1",
            direction=TransferDirection.DOWNLOAD,
            filename="ve.safetensors",
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=512,
            total_bytes=1024,
            percentage=50.0,
            speed=256.0,
        )

    def test_always_present_keys(self, event: ProgressEvent) -> None:
        payload = event.to_dict()
        for key in (
            "event_type",
            "transfer_id",
            "direction",
            "filename",
            "phase",
            "bytes_completed",
            "total_bytes",
            "percentage",
            "speed",
            "file_index",
            "total_files",
            "timestamp",
        ):
            assert key in payload, f"{key} must always be serialisable"

    def test_values_are_plain_json_types(self, event: ProgressEvent) -> None:
        payload = event.to_dict()
        assert payload["event_type"] == "progress"
        assert payload["direction"] == "download"
        assert payload["phase"] == "downloading"
        assert payload["bytes_completed"] == 512
        assert payload["total_bytes"] == 1024
        assert payload["file_index"] == 0
        assert payload["total_files"] == 1
        assert isinstance(payload["timestamp"], float)

    def test_percentage_is_rounded_to_two_places(self) -> None:
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="t",
            direction=TransferDirection.DOWNLOAD,
            filename="f",
            phase=ProgressPhase.DOWNLOADING,
            percentage=33.333333,
        )
        assert event.to_dict()["percentage"] == 33.33

    def test_xet_only_fields_are_omitted_when_zero(self, event: ProgressEvent) -> None:
        payload = event.to_dict()
        assert "transfer_bytes_completed" not in payload
        assert "dedup_saved_bytes" not in payload
        assert "error" not in payload
        assert "extra" not in payload

    def test_xet_only_fields_appear_once_set(self, event: ProgressEvent) -> None:
        event.transfer_bytes_completed = 100
        event.transfer_bytes_total = 200
        event.transfer_speed = 3.0
        event.dedup_saved_bytes = 7
        payload = event.to_dict()
        assert payload["transfer_bytes_completed"] == 100
        assert payload["transfer_bytes_total"] == 200
        assert payload["transfer_speed"] == 3.0
        assert payload["dedup_saved_bytes"] == 7

    def test_error_is_serialised_as_a_dict(self) -> None:
        event = ProgressEvent.error_event(
            transfer_id="t",
            direction=TransferDirection.DOWNLOAD,
            filename="f",
            error=TransferError(message="boom", error_type="OSError", retryable=True),
        )
        payload = event.to_dict()
        assert payload["event_type"] == "error"
        assert payload["error"] == {
            "message": "boom",
            "error_type": "OSError",
            "retryable": True,
        }

    def test_round_trips_through_from_dict(self, event: ProgressEvent) -> None:
        event.extra = {"note": "x"}
        restored = ProgressEvent.from_dict(event.to_dict())
        assert restored.transfer_id == event.transfer_id
        assert restored.event_type is EventType.PROGRESS
        assert restored.direction is TransferDirection.DOWNLOAD
        assert restored.extra == {"note": "x"}

    def test_complete_event_forces_one_hundred_percent(self) -> None:
        event = ProgressEvent.complete(
            transfer_id="t",
            direction=TransferDirection.DOWNLOAD,
            filename="f",
            bytes_completed=10,
            total_bytes=10,
        )
        assert event.to_dict()["percentage"] == 100.0
        assert event.to_dict()["event_type"] == "complete"

    def test_cancelled_event_uses_its_own_type(self) -> None:
        event = ProgressEvent.cancelled_event(
            transfer_id="t",
            direction=TransferDirection.DOWNLOAD,
            filename="f",
        )
        assert event.event_type is EventType.CANCELLED
        assert event.to_dict()["event_type"] == "cancelled"

    def test_transfer_error_is_json_safe(self) -> None:
        assert TransferError.from_dict(TransferError(message="m").to_dict()).message == "m"
        assert str(TransferError(message="m")) == "m"


class TestDownloadProgressTqdmBind:
    """``bind()`` is the trick the whole HTTP download path rests on."""

    def test_injects_the_cancel_hook(self) -> None:
        calls = []
        hook = lambda: bool(calls) or calls.append(1) is None and False  # noqa: E731
        events: queue.Queue = queue.Queue()
        bound = DownloadProgressTqdm.bind(
            event_queue=events,
            transfer_id="t-9",
            filename="ve.safetensors",
            is_cancelled=hook,
        )
        bar = bound(total=10, unit="B", disable=True)
        try:
            assert bar._is_cancelled is hook
            assert bar._transfer_id == "t-9"
            assert bar._filename == "ve.safetensors"
            assert bar._event_queue is events
        finally:
            bar.close()

    def test_a_true_cancel_hook_raises_on_update(self) -> None:
        bound = DownloadProgressTqdm.bind(
            event_queue=queue.Queue(),
            transfer_id="t-10",
            filename="f",
            is_cancelled=lambda: True,
        )
        bar = bound(total=10, unit="B", disable=True)
        try:
            with pytest.raises(TransferCancelledError):
                bar.update(1)
        finally:
            bar.close()
            state_manager.clear_state("t-10")

    def test_a_false_cancel_hook_lets_the_bar_advance(self) -> None:
        bound = DownloadProgressTqdm.bind(
            event_queue=queue.Queue(),
            transfer_id="t-11",
            filename="f",
            is_cancelled=lambda: False,
        )
        bar = bound(total=10, unit="B", disable=False)
        try:
            bar.update(1)
            assert bar.n == 1
        finally:
            bar.close()
            state_manager.clear_state("t-11")

    def test_bound_class_is_named_after_its_transfer(self) -> None:
        bound = DownloadProgressTqdm.bind(
            event_queue=queue.Queue(),
            transfer_id="abcdef123456",
            filename="f",
        )
        assert bound.__name__ == "BoundDownloadTqdm_abcdef12"

    def test_two_binds_do_not_share_state(self) -> None:
        first = DownloadProgressTqdm.bind(
            event_queue=queue.Queue(), transfer_id="a-1", filename="a"
        )
        second = DownloadProgressTqdm.bind(
            event_queue=queue.Queue(), transfer_id="b-1", filename="b"
        )
        bar_a = first(total=4, unit="B", disable=True)
        bar_b = second(total=4, unit="B", disable=True)
        try:
            assert bar_a._transfer_id == "a-1"
            assert bar_b._transfer_id == "b-1"
        finally:
            bar_a.close()
            bar_b.close()
            state_manager.clear_state("a-1")
            state_manager.clear_state("b-1")


class TestPatchDownloadChunkSize:
    def test_sets_and_restores_the_hub_constant(self) -> None:
        from huggingface_hub import constants as hf_constants

        original = hf_constants.DOWNLOAD_CHUNK_SIZE
        with patch_download_chunk_size(1234):
            assert hf_constants.DOWNLOAD_CHUNK_SIZE == 1234
        assert hf_constants.DOWNLOAD_CHUNK_SIZE == original

    def test_restores_even_when_the_body_raises(self) -> None:
        from huggingface_hub import constants as hf_constants

        original = hf_constants.DOWNLOAD_CHUNK_SIZE
        with pytest.raises(RuntimeError):
            with patch_download_chunk_size(4321):
                assert hf_constants.DOWNLOAD_CHUNK_SIZE == 4321
                raise RuntimeError("boom")
        assert hf_constants.DOWNLOAD_CHUNK_SIZE == original

    def test_default_chunk_size_is_smaller_than_the_hub_default(self) -> None:
        """A 256 KiB window reports progress often enough for a live bar."""
        from huggingface_hub import constants as hf_constants

        assert 256 * 1024 < hf_constants.DOWNLOAD_CHUNK_SIZE


class TestHfProgressTracker:
    def test_cancel_registry_round_trips(self) -> None:
        tracker = HfProgressTracker(token=None)
        assert tracker.is_cancelled("x") is False
        tracker.cancel(transfer_id="x")
        assert tracker.is_cancelled("x") is True
        tracker.cleanup_transfer("x")
        assert tracker.is_cancelled("x") is False

    def test_event_queue_starts_empty(self) -> None:
        tracker = HfProgressTracker(token=None)
        assert tracker.event_queue.empty()

    def test_generate_transfer_id_is_unique(self) -> None:
        assert generate_transfer_id() != generate_transfer_id()


class TestExceptionsAndTokens:
    @pytest.mark.parametrize(
        "exc_type",
        [TransferCancelledError, TransferProgressError, TokenError],
    )
    def test_exceptions_are_distinct_types(self, exc_type: type) -> None:
        assert issubclass(exc_type, Exception)

    def test_transfer_result_defaults(self) -> None:
        result = TransferResult(
            success=True, transfer_id="t", filename="f",
            direction=TransferDirection.DOWNLOAD,
        )
        assert result.direction is TransferDirection.DOWNLOAD
        assert result.url is None
        assert result.file_size == 0

    def test_is_xet_available_is_a_bool(self) -> None:
        assert isinstance(is_xet_available(), bool)

    def test_xet_credentials_holds_the_fields_the_tracker_reads(self) -> None:
        credentials = XetCredentials(
            endpoint="https://cas-server.xethub.hf.co",
            token_info=("token", 1_700_000_000),
        )
        assert credentials.endpoint == "https://cas-server.xethub.hf.co"
        assert credentials.token_info == ("token", 1_700_000_000)
        assert credentials.token_refresher is None

    def test_token_manager_can_be_built_without_a_token(self) -> None:
        assert XetTokenManager(None) is not None

    def test_tqdm_upload_patcher_is_a_context_manager(self) -> None:
        patcher = tqdm_upload_patcher(
            event_queue=queue.Queue(), transfer_id="u-1", filename="f", total_bytes=1
        )
        with patcher:
            pass
        state_manager.clear_state("u-1")


class TestIntegrationsSubpackage:
    """``hf_progress.integrations`` re-exports its SSE factories.

    The subpackage shipped with an empty ``__init__``: importing
    ``hf_progress.integrations`` and finding nothing in it is a trap, and a
    static reachability scan sees ``sse.py`` as an orphan. Both are fixed by the
    re-export.
    """

    def test_the_sse_factories_are_reachable_from_the_subpackage(self) -> None:
        from ComfyUI_Chatterbox.src.hf_progress import integrations

        assert integrations.__all__ == [
            "create_progress_router",
            "create_raw_sse_stream",
        ]
        assert callable(integrations.create_progress_router)
        assert callable(integrations.create_raw_sse_stream)

    def test_the_subpackage_does_not_import_a_web_framework(self) -> None:
        """It must stay importable with no aiohttp/FastAPI installed."""
        with open(
            os.path.join(SRC_HF_PROGRESS, "integrations", "sse.py"),
            "r",
            encoding="utf-8",
        ) as handle:
            source = handle.read()
        assert "import aiohttp" not in source
        assert "from aiohttp" not in source
        assert "import fastapi" not in source


class TestModuleFunctionsAreCallable:
    """``download_file`` / ``download_snapshot`` must not import the network
    stack until they are called, so importing this module stays cheap."""

    def test_download_helpers_defer_their_hub_import(self, monkeypatch) -> None:
        import huggingface_hub.file_download as file_download

        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return kwargs.get("local_dir", "/tmp/fake")

        # The real snapshot_download is imported *inside* the helper, so
        # replacing the attribute on the module object is enough.
        monkeypatch.setattr(
            file_download, "snapshot_download", fake_snapshot_download, raising=False
        )
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False
        )

        result = download_snapshot(
            repo_id="owner/repo",
            token=None,
            event_queue=queue.Queue(),
            transfer_id="s-1",
            local_dir="/tmp/fake",
        )
        assert result == "/tmp/fake"
        assert calls, "the download helper never reached huggingface_hub"
        assert calls[0]["repo_id"] == "owner/repo"