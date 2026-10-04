"""Overlay drawing and a self-contained threaded OpenCV display.

``SHOW_DISPLAY=true`` is a first-class mode here, not an afterthought, so the
display path is written to be cheap:

* **Downscale first, draw second.** The window is 960x540; annotating a full
  1280x720 frame and letting X scale it down wastes ~45 % of the pixels twice
  over. The frame is resized once and the box coordinates are scaled with it.
* **Rate-capped.** A debug overlay does not need 30 Hz. ``display_max_fps``
  (default 10) drops surplus frames *before* the resize and the drawing, so the
  saving applies to the producer thread as well as the display thread.
* **Repaint only when something changed.** The old loop pushed a full frame to
  the X server on every iteration, which is expensive over the xrdp/X11 socket
  this Jetson uses. Now ``imshow`` is called on a new frame or on a slow
  heartbeat, while ``waitKey`` still runs every iteration so Qt stays
  responsive.

Qt requires the window to be created and serviced from one thread, so the
display keeps its own thread fed by a capacity-one queue. Inference never
blocks on rendering.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from vision_perception.geometry import Box

_HEAD_COLOR = (255, 0, 255)      # magenta (BGR)
_BODY_COLOR = (0, 255, 0)        # green
_RAW_HEAD_COLOR = (0, 255, 255)  # yellow
_RAW_BODY_COLOR = (255, 255, 0)  # cyan
_GUIDE_COLOR = (0, 0, 255)       # red
_HUD_COLOR = (255, 255, 255)
_HUD_SHADOW = (0, 0, 0)

_HEARTBEAT_S = 0.5
# Resize only if it removes at least 15 % of the width.
_RESIZE_WORTH_IT = 0.85


def _scaled_corners(box: Box, scale: float) -> Tuple[int, int, int, int]:
    return (
        int(box['x1'] * scale), int(box['y1'] * scale),
        int(box['x2'] * scale), int(box['y2'] * scale),
    )


def draw_overlay(
    frame: np.ndarray,
    head_box: Optional[Box],
    body_box: Optional[Box],
    *,
    raw_head: Optional[Box] = None,
    raw_body: Optional[Box] = None,
    scale: float = 1.0,
    hud: str = '',
) -> np.ndarray:
    """Draw guides, raw candidates and mission-consumed stable boxes.

    ``scale`` maps source-image coordinates onto ``frame``, so the caller can
    hand in an already-downscaled frame and full-resolution boxes.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    guide_thickness = 2 if w <= 1000 else 4
    cv2.line(frame, (cx, 0), (cx, h), _GUIDE_COLOR, guide_thickness)
    cv2.line(frame, (0, cy), (w, cy), _GUIDE_COLOR, guide_thickness)

    for raw, label, color in (
        (raw_head, 'raw head', _RAW_HEAD_COLOR),
        (raw_body, 'raw body', _RAW_BODY_COLOR),
    ):
        if raw is None:
            continue
        x1, y1, x2, y2 = _scaled_corners(raw, scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            f'{label} {float(raw.get("conf", 0.0)):.2f}',
            (max(0, x1), max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    for box, label, color in (
        (head_box, 'head', _HEAD_COLOR),
        (body_box, 'body', _BODY_COLOR),
    ):
        if box is None:
            continue
        x1, y1, x2, y2 = _scaled_corners(box, scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (max(0, x1), max(10, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if hud:
        cv2.putText(frame, hud, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _HUD_SHADOW, 3)
        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _HUD_COLOR, 1)
    return frame


def render_frame(
    source: np.ndarray,
    head_box: Optional[Box],
    body_box: Optional[Box],
    *,
    raw_head: Optional[Box] = None,
    raw_body: Optional[Box] = None,
    target_width: int = 960,
    hud: str = '',
) -> np.ndarray:
    """Downscale ``source`` to ``target_width`` and annotate it.

    ``cv2.resize`` allocates a new array, so this also removes the separate
    ``img.copy()`` the old render path needed to avoid drawing on the frame
    still referenced by the duplicate-frame guard.
    """
    src_h, src_w = source.shape[:2]
    target_width = int(target_width)
    # Only resize when it actually buys something. On a 1024-wide camera a
    # display_width of 960 is a 0.94x "downscale": it costs a full INTER_AREA
    # pass and saves 6 % of the pixels. Below the threshold a plain copy (which
    # we need anyway, to get a writable buffer) is strictly cheaper.
    if target_width > 0 and target_width <= src_w * _RESIZE_WORTH_IT:
        scale = target_width / float(src_w)
        # INTER_AREA is the better downscaler but costs more; it only matters
        # below ~0.5x, where bilinear starts to alias visibly.
        interpolation = cv2.INTER_AREA if scale < 0.5 else cv2.INTER_LINEAR
        canvas = cv2.resize(
            source, (target_width, max(1, int(round(src_h * scale)))),
            interpolation=interpolation)
    else:
        scale = 1.0
        canvas = source.copy()
    return draw_overlay(
        canvas, head_box, body_box,
        raw_head=raw_head, raw_body=raw_body, scale=scale, hud=hud,
    )


class DisplayWindow:
    """A background thread that owns and updates a single OpenCV window."""

    def __init__(
        self,
        window_name: str = 'Vision Feedback',
        logger=None,
        max_fps: float = 10.0,
        width: int = 960,
    ):
        self.window_name = window_name
        self.width = int(width)
        self._logger = logger
        self._min_period = (1.0 / float(max_fps)) if float(max_fps) > 0 else 0.0
        self._next_due = 0.0
        # One pending scene only: the newest frame always wins, so retaining
        # older ones would only add latency.
        self._queue: 'queue.Queue' = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.frames_shown = 0
        self.frames_skipped = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='vision-display', daemon=True)
        self._thread.start()

    def due(self) -> bool:
        """True when the next frame should be rendered.

        Asked by the producer *before* it resizes and annotates, so a skipped
        frame costs nothing at all.
        """
        if not self._running:
            return False
        if self._min_period <= 0.0:
            return True
        now = time.monotonic()
        if now < self._next_due:
            self.frames_skipped += 1
            return False
        self._next_due = now + self._min_period
        return True

    def _enqueue(self, item) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def submit(self, frame: np.ndarray) -> None:
        """Hand an already-annotated frame to the display thread."""
        if not self._running or frame is None or frame.size == 0:
            return
        self._enqueue(('frame', frame))

    def submit_scene(
        self,
        frame: np.ndarray,
        head_box: Optional[Box],
        body_box: Optional[Box],
        *,
        raw_head: Optional[Box] = None,
        raw_body: Optional[Box] = None,
        hud: str = '',
    ) -> None:
        """Hand a raw frame plus boxes over; the display thread renders them.

        Queueing the scene instead of a finished image keeps the resize, the
        rectangle drawing and the text rendering off the inference thread,
        which is otherwise the busiest thread in the process.
        """
        if not self._running or frame is None or frame.size == 0:
            return
        self._enqueue(('scene', (frame, head_box, body_box, raw_head, raw_body, hud)))

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    @staticmethod
    def _to_displayable(frame: np.ndarray) -> Optional[np.ndarray]:
        """Coerce a frame to 8-bit BGR, or None if unusable.

        Float frames are scaled rather than truncated so a [0, 1] image does
        not floor to black through ``astype(uint8)``.
        """
        if frame is None or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            fmax = float(frame.max()) if frame.size else 0.0
            if fmax <= 1.0 + 1e-6:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _is_black(frame: np.ndarray) -> bool:
        """Cheap check: sampled pixels are (near) all zero."""
        return bool(frame[::16, ::16].max() < 4)

    def _loop(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 960, 540)
            cv2.moveWindow(self.window_name, 100, 100)
        except Exception as exc:  # pragma: no cover - depends on the display
            if self._logger is not None:
                self._logger.error(
                    f'Failed to create display window: {exc}. With SHOW_DISPLAY=true the '
                    'container needs a reachable X server: check DISPLAY, that '
                    '/tmp/.X11-unix is mounted, and that `xhost +local:docker` has run.'
                )
            self._running = False
            return

        last_good: Optional[np.ndarray] = None
        last_paint = 0.0
        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            frame = None
            if item is not None:
                kind, payload = item
                if kind == 'frame':
                    frame = payload
                else:
                    try:
                        source, head, body, raw_head, raw_body, hud = payload
                        frame = render_frame(
                            source, head, body,
                            raw_head=raw_head, raw_body=raw_body,
                            target_width=self.width, hud=hud,
                        )
                    except Exception as exc:  # pragma: no cover
                        if self._logger is not None:
                            self._logger.warn(f'Overlay render failed: {exc}')
                        frame = None

            dirty = False
            if frame is not None:
                shown = self._to_displayable(frame)
                # Never let a transient empty/black frame replace a good one.
                if shown is not None and not self._is_black(shown):
                    last_good = shown
                    dirty = True

            now = time.monotonic()
            try:
                if last_good is not None and (dirty or now - last_paint > _HEARTBEAT_S):
                    cv2.imshow(self.window_name, last_good)
                    last_paint = now
                    self.frames_shown += 1
                cv2.waitKey(1)
            except Exception as exc:  # pragma: no cover
                if self._logger is not None:
                    self._logger.warn(f'Display error: {exc}')
                time.sleep(0.1)
