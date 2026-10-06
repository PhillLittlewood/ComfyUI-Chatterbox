"""Broadcast the Chatterbox configuration to frontend clients.

Ported from ``ComfyUI-VoxCPM/modules/config_broadcast.py`` (plan S4.5) with
three changes:

* the event channel is ``chatterbox.config`` instead of ``voxcpm.config``;
* **only** :func:`send_config_to_client` and :func:`schedule_config_send` were
  taken. ``register_config_on_connect`` was *not*: it probes ``PromptServer``
  for ``on_client_connect`` / ``on_connect`` / ``client_connect_handler``,
  none of which ComfyUI's ``PromptServer`` has, so it is a silent no-op whose
  own comment concedes the attribute name "may vary" (plan §3.1.2). A client
  that connects after the scheduled send gets the configuration from the
  ``model_info`` HTTP response instead, which embeds the same payload; and
* no bare ``except: pass`` -- every failure path logs (plan §3.1.4).

``server`` is imported lazily, so this module imports without ComfyUI running.
"""

from __future__ import annotations

import logging
import threading


__all__ = ["CONFIG_CHANNEL", "send_config_to_client", "schedule_config_send"]

# Renamed from the borrowed ``voxcpm.config`` namespace (plan §3.1.5).
CONFIG_CHANNEL = "chatterbox.config"

# Set once the "text normalization unavailable" notice has been logged, so a
# restart-free reconnect does not repeat it.
_normalization_warning_emitted = False


def build_config_payload() -> dict:
    """Assemble the config payload: feature flag plus user settings."""
    from .model_info import TEXT_NORMALIZATION_AVAILABLE
    from .settings import get_settings

    return {
        "normalization_available": TEXT_NORMALIZATION_AVAILABLE,
        "settings": get_settings().to_dict(),
    }


def send_config_to_client(client_id=None):
    """Send the configuration to one client, or broadcast to all.

    Args:
        client_id: Websocket SID to target. ``None`` broadcasts.

    Degrades quietly in every environment the node runs in: no server module,
    no ``PromptServer.instance`` yet, or a client that has already gone away.
    None of those may raise -- configuration is a convenience, not a
    precondition for generating audio.
    """
    global _normalization_warning_emitted

    try:
        from server import PromptServer
    except ImportError as exc:
        logging.debug("[Chatterbox TTS] No 'server' module available; config not sent: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001 -- a broken server module is not fatal
        logging.warning(
            "[Chatterbox TTS] Failed to import 'server' for the config broadcast: %s",
            exc,
        )
        return

    try:
        if PromptServer.instance is None:
            logging.debug("[Chatterbox TTS] PromptServer.instance is None; config not sent yet")
            return

        config_data = build_config_payload()
        logging.debug(
            "[Chatterbox TTS] Sending config to %s: %s",
            client_id or "all clients",
            config_data,
        )
        if client_id:
            PromptServer.instance.send_sync(CONFIG_CHANNEL, config_data, client_id)
        else:
            PromptServer.instance.send_sync(CONFIG_CHANNEL, config_data)
        logging.debug("[Chatterbox TTS] Config sent successfully")

        if not _normalization_warning_emitted and not config_data["normalization_available"]:
            _normalization_warning_emitted = True
            logging.info(
                "[Chatterbox TTS] Text normalization packages (inflect, wetext) not found. "
                "Normalization is disabled. Install them with: pip install inflect wetext"
            )
    except AttributeError as exc:
        logging.warning(
            "[Chatterbox TTS] Could not access PromptServer for the config broadcast: %s",
            exc,
        )
    except (ConnectionError, OSError) as exc:
        logging.warning("[Chatterbox TTS] Network error sending config: %s", exc)
    except Exception as exc:  # noqa: BLE001 -- never let a toast break a generation
        logging.error("[Chatterbox TTS] Unexpected error sending config: %s", exc, exc_info=True)


def schedule_config_send():
    """Send the configuration once, shortly after the server starts.

    A daemon thread with a short delay: ComfyUI's frontend is not connected in
    the first moments of startup, and a blocking ``time.sleep`` on the main
    thread would delay node registration.
    """

    def send_after_delay() -> None:
        import time

        time.sleep(0.5)
        logging.debug("[Chatterbox TTS] Sending the initial config event")
        send_config_to_client()
        logging.debug("[Chatterbox TTS] Initial config event sent")

    thread = threading.Thread(target=send_after_delay, daemon=True)
    thread.start()
    return thread