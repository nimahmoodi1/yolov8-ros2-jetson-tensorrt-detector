"""Fast ``sensor_msgs/Image`` -> BGR ndarray conversion.

``cv_bridge.imgmsg_to_cv2`` allocates and copies the whole frame through the
C++ layer on every callback. At 1280x720 bgr8 / 15 Hz that is ~41 MB/s of pure
memcpy on the CPU before any detection work starts.

For the encodings a camera actually publishes we can build a *view* over the
message buffer instead, which costs nothing. ``cv_bridge`` stays as the
fallback for anything unusual (Bayer, 16-bit depth, ...), so behaviour never
regresses on an unexpected stream.

The returned array may be read-only because it aliases the DDS buffer. Callers
that need to draw on it must copy first; :func:`ensure_writable` is provided
for that.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# encoding -> (channels, cvtColor code or None)
_DIRECT = {
    'bgr8': (3, None),
    '8UC3': (3, None),
    'rgb8': (3, cv2.COLOR_RGB2BGR),
    'bgra8': (4, cv2.COLOR_BGRA2BGR),
    'rgba8': (4, cv2.COLOR_RGBA2BGR),
    'mono8': (1, cv2.COLOR_GRAY2BGR),
    '8UC1': (1, cv2.COLOR_GRAY2BGR),
}


def _buffer_to_array(data) -> Optional[np.ndarray]:
    """Wrap the message payload as a uint8 ndarray without copying."""
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except (TypeError, BufferError):
        pass
    try:
        arr = np.asarray(data)
        return arr.view(np.uint8) if arr.dtype != np.uint8 else arr
    except Exception:
        return None


def to_bgr(msg, bridge=None, logger=None) -> Optional[np.ndarray]:
    """Return an 8-bit BGR image for ``msg``, or ``None`` if unusable.

    ``bridge`` is an optional ``cv_bridge.CvBridge`` used only for encodings
    this module does not handle natively.
    """
    encoding = str(getattr(msg, 'encoding', '') or '')
    spec = _DIRECT.get(encoding)
    if spec is not None:
        channels, convert = spec
        flat = _buffer_to_array(msg.data)
        if flat is not None:
            height = int(msg.height)
            width = int(msg.width)
            step = int(msg.step) or width * channels
            expected = height * step
            if flat.size >= expected > 0:
                rows = flat[:expected].reshape(height, step)
                frame = rows[:, : width * channels].reshape(height, width, channels)
                if convert is None:
                    return frame
                return cv2.cvtColor(frame, convert)

    if bridge is None:
        if logger is not None:
            logger.warn(f"Unsupported image encoding '{encoding}' and no cv_bridge fallback")
        return None

    try:
        return bridge.imgmsg_to_cv2(msg, 'bgr8')
    except Exception:
        pass
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
    except Exception as exc:
        if logger is not None:
            logger.warn(f'cv_bridge conversion failed (encoding={encoding}): {exc}')
        return None
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and encoding.startswith('rgb'):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def ensure_writable(frame: np.ndarray) -> np.ndarray:
    """Return a writable array, copying only when the input aliases DDS memory.

    ``np.ascontiguousarray`` is not enough here: a read-only array that is
    already C-contiguous is returned unchanged, still read-only, and the first
    ``cv2.rectangle`` on it raises.
    """
    if frame.flags.writeable and frame.flags.c_contiguous:
        return frame
    return frame.copy()
