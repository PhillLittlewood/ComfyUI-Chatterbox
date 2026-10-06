"""Integration helpers for external frameworks.

Provides optional integrations with web frameworks and real-time
streaming protocols. Each integration is an opt-in dependency
controlled by extras in pyproject.toml:

- **sse**: Server-Sent Events endpoint for progress streaming.

Usage::

    from .hf_progress.integrations import create_progress_router

The names are re-exported here rather than left to be reached through
``integrations.sse`` directly, so the subpackage has a real public surface --
importing ``integrations`` and finding nothing in it is a trap.
"""

from __future__ import annotations

from .sse import create_progress_router, create_raw_sse_stream

__all__ = ["create_progress_router", "create_raw_sse_stream"]