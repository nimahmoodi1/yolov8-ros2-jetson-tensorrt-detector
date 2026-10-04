"""Thread-safe latest-only frame handoff for real-time inference."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    """One camera frame and its local receipt metadata."""

    sequence: int
    image: Any
    received_monotonic: float


class LatestFrameSlot:
    """A capacity-one buffer that replaces stale pending frames.

    The inference worker can never build a FIFO backlog: while it is busy, the
    newest camera callback atomically replaces the previous pending frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._latest: FramePacket | None = None
        self._closed = False

    def put(self, packet: FramePacket) -> bool:
        """Store ``packet`` and return True when an older pending frame was dropped."""
        with self._lock:
            if self._closed:
                return False
            replaced = self._latest is not None
            self._latest = packet
            self._ready.set()
            return replaced

    def take(self, timeout_s: float = 0.1) -> FramePacket | None:
        """Take the newest pending frame, or None on timeout/closed-empty."""
        if not self._ready.wait(timeout=max(0.0, float(timeout_s))):
            return None
        with self._lock:
            packet = self._latest
            self._latest = None
            self._ready.clear()
            return packet

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._latest = None
            self._ready.set()
