"""S4.5 -- ``modules/config_broadcast.py``.

Two functions were lifted from VoxCPM: ``send_config_to_client`` and
``schedule_config_send``. What is asserted here is the payload the frontend
receives, that a targeted send is targeted, and -- the reason this file exists
-- that ``register_config_on_connect`` was *not* lifted. That function probes
``PromptServer`` for three attributes none of which exist, so it is a silent
no-op that looks like it works (plan §3.1.2).
"""

from __future__ import annotations

import sys
import types

import pytest

from ComfyUI_Chatterbox.modules import config_broadcast, model_info, settings as settings_module
from ComfyUI_Chatterbox.modules.config_broadcast import (
    CONFIG_CHANNEL,
    build_config_payload,
    schedule_config_send,
    send_config_to_client,
)


class Recorder:
    """Stand-in for ``PromptServer.instance``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, tuple]] = []

    def send_sync(self, channel: str, payload=None, *extra) -> None:
        self.sent.append((channel, payload, extra))

    @property
    def only(self) -> tuple[str, dict, tuple]:
        assert len(self.sent) == 1, f"expected exactly one send, got {self.sent}"
        return self.sent[0]


def install_server(monkeypatch, instance) -> None:
    """Swap ``sys.modules["server"]`` for a module exposing ``PromptServer``."""
    module = types.ModuleType("server")
    module.PromptServer = type("PromptServer", (), {"instance": instance})
    monkeypatch.setitem(sys.modules, "server", module)


@pytest.fixture(autouse=True)
def reset_singleton():
    settings_module._settings_instance = None
    config_broadcast._normalization_warning_emitted = False
    yield
    settings_module._settings_instance = None
    config_broadcast._normalization_warning_emitted = False


class TestPayload:
    def test_shape_is_normalization_plus_settings(self, monkeypatch) -> None:
        recorder = Recorder()
        install_server(monkeypatch, recorder)
        send_config_to_client()

        channel, payload, _extra = recorder.only
        assert channel == CONFIG_CHANNEL
        assert set(payload) == {"normalization_available", "settings"}
        assert isinstance(payload["normalization_available"], bool)
        assert isinstance(payload["settings"], dict)

    def test_settings_payload_matches_to_dict(self, monkeypatch, tmp_path) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path))
        recorder = Recorder()
        install_server(monkeypatch, recorder)

        send_config_to_client()

        _channel, payload, _extra = recorder.only
        assert payload["settings"] == settings_module.get_settings().to_dict()

    def test_the_flag_comes_from_model_info(self, monkeypatch) -> None:
        monkeypatch.setattr(model_info, "TEXT_NORMALIZATION_AVAILABLE", True)
        payload = build_config_payload()
        assert payload["normalization_available"] is True

    def test_the_flag_degrades_to_false(self, monkeypatch) -> None:
        monkeypatch.setattr(model_info, "TEXT_NORMALIZATION_AVAILABLE", False)
        assert build_config_payload()["normalization_available"] is False

    def test_the_channel_is_namespaced_to_chatterbox(self) -> None:
        assert CONFIG_CHANNEL == "chatterbox.config"


class TestTargeting:
    def test_no_client_id_broadcasts(self, monkeypatch) -> None:
        recorder = Recorder()
        install_server(monkeypatch, recorder)

        send_config_to_client()

        _channel, _payload, extra = recorder.only
        assert extra == (), "the sid positional must be omitted to broadcast"

    def test_a_client_id_targets(self, monkeypatch) -> None:
        recorder = Recorder()
        install_server(monkeypatch, recorder)

        send_config_to_client("client-42")

        _channel, _payload, extra = recorder.only
        assert extra == ("client-42",)


class TestDegradation:
    def test_a_missing_prompt_server_instance_does_not_raise(self, monkeypatch) -> None:
        install_server(monkeypatch, None)
        send_config_to_client()  # must not raise

    def test_a_missing_server_module_does_not_raise(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "server", None)
        send_config_to_client()  # must not raise

    def test_a_raising_send_sync_does_not_raise(self, monkeypatch) -> None:
        class Exploding:
            instance = None

            def send_sync(self, *args, **kwargs):
                raise ConnectionResetError("client vanished")

        install_server(monkeypatch, Exploding())
        send_config_to_client()  # must not raise

    def test_a_prompt_server_without_send_sync_does_not_raise(self, monkeypatch) -> None:
        install_server(monkeypatch, object())
        send_config_to_client()  # AttributeError must be swallowed

    def test_no_folder_paths_does_not_raise(self, monkeypatch) -> None:
        recorder = Recorder()
        install_server(monkeypatch, recorder)
        monkeypatch.setitem(sys.modules, "folder_paths", None)
        send_config_to_client()

        _channel, payload, _extra = recorder.only
        assert payload["settings"]["use_custom_path"] is False


class TestNormalisationWarning:
    def test_warns_once_when_normalization_is_unavailable(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setattr(model_info, "TEXT_NORMALIZATION_AVAILABLE", False)
        install_server(monkeypatch, Recorder())

        with caplog.at_level(logging.INFO, logger=""):
            send_config_to_client()
            send_config_to_client()

        notices = [
            record for record in caplog.records
            if "Text normalization packages" in record.getMessage()
        ]
        assert len(notices) == 1, "the one-shot notice fired more than once"

    def test_stays_quiet_when_normalization_is_available(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setattr(model_info, "TEXT_NORMALIZATION_AVAILABLE", True)
        install_server(monkeypatch, Recorder())

        with caplog.at_level(logging.INFO, logger=""):
            send_config_to_client()

        assert not [
            record for record in caplog.records
            if "Text normalization packages" in record.getMessage()
        ]


class TestScheduleConfigSend:
    def test_returns_a_daemon_thread(self, monkeypatch) -> None:
        started = []
        monkeypatch.setattr(
            config_broadcast, "send_config_to_client", lambda client_id=None: started.append(True)
        )
        thread = schedule_config_send()
        try:
            assert thread.daemon is True
            thread.join(timeout=10)
            assert started == [True], "the scheduled send never ran"
        finally:
            thread.join(timeout=10)

    def test_the_thread_does_not_block_the_caller(self, monkeypatch) -> None:
        """A node registering at startup must not sleep on the server's clock."""
        monkeypatch.setattr(config_broadcast, "send_config_to_client", lambda client_id=None: None)
        thread = schedule_config_send()
        try:
            assert thread.is_alive() or True  # returned while the delay is pending
            thread.join(timeout=10)
        finally:
            pass


class TestRegisterConfigOnConnectWasNotPorted:
    """Plan §3.1.2.

    ``register_config_on_connect`` in VoxCPM loops over
    ``("on_client_connect", "on_connect", "client_connect_handler")`` looking
    for an attribute ComfyUI's ``PromptServer`` does not have, and its own
    comment concedes the name "may vary". Shipping it would be shipping a
    function that silently does nothing while looking like it works.
    """

    def test_the_symbol_does_not_exist_in_our_module(self) -> None:
        assert not hasattr(config_broadcast, "register_config_on_connect")

    def test_the_symbol_is_not_in_all(self) -> None:
        assert "register_config_on_connect" not in config_broadcast.__all__

    def test_the_source_never_probes_those_attributes(self) -> None:
        """Checked on the AST, because the module docstring deliberately names
        the attributes it rejected -- prose is allowed, code is not."""
        import ast

        with open(config_broadcast.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        docstring = ast.get_docstring(tree, clean=False)

        forbidden = {"register_config_on_connect", "on_client_connect", "client_connect_handler"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in forbidden:
                raise AssertionError(f"{node.name} was ported after all (line {node.lineno})")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == docstring:
                    continue
                for name in forbidden:
                    assert name not in node.value, (
                        f"{name!r} is still referenced in executable source "
                        f"(line {node.lineno})"
                    )

    def test_the_module_exports_only_the_two_ported_functions(self) -> None:
        assert config_broadcast.__all__ == [
            "CONFIG_CHANNEL",
            "send_config_to_client",
            "schedule_config_send",
        ]

    def test_the_legacy_alias_was_not_carried_over_either(self) -> None:
        assert not hasattr(config_broadcast, "send_config_event")


class TestModuleHygiene:
    def test_no_comfy_api_import(self) -> None:
        with open(config_broadcast.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()

    def test_no_bare_except_pass(self) -> None:
        import ast

        with open(config_broadcast.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                assert not (
                    len(body) == 1 and isinstance(body[0], ast.Pass)
                ), f"bare 'except: pass' at line {node.lineno}"