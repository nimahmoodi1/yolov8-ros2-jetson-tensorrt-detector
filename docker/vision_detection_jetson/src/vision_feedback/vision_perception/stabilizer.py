"""Bounding-box stabilizer.

Smooths published boxes and holds the last good box through short YOLO
dropouts, while rejecting geometrically implausible measurements (e.g. a head
that suddenly balloons into the whole upper tower). This is a *pure* transform:
it consumes a ``BoundingBoxes`` message and returns a new, stabilized one.
"""

from __future__ import annotations

import copy
import time
from typing import List, Optional

from ros2_interface.msg import BoundingBoxes

from vision_perception.geometry import estimate_body_box


class _PartState:
    """Per-part (head/body) smoothing state."""

    def __init__(self, has_field: str, fields: List[str]):
        self.has_field = has_field
        self.fields = fields
        self.last_box: Optional[List[float]] = None
        self.last_seen: Optional[float] = None


class BoundingBoxStabilizer:
    """Exponential smoothing + dropout hold + sanity gating for head/body boxes."""

    def __init__(
        self,
        hold_time_s: float = 2.0,
        alpha: float = 0.55,
        img_w: int = 640,
        img_h: int = 480,
    ):
        self.hold_time_s = float(hold_time_s)
        self.alpha = float(alpha)
        self.img_w = img_w
        self.img_h = img_h
        self.rejected_measurements = 0
        self._parts = {
            "head": _PartState("has_head", ["head_x1", "head_y1", "head_x2", "head_y2"]),
            "body": _PartState("has_body", ["body_x1", "body_y1", "body_x2", "body_y2"]),
        }

    def set_frame_size(self, img_w: int, img_h: int) -> None:
        """Update geometry to the frame that produced the current boxes.

        Camera drivers do not always publish their advertised mode (the real
        Sony stream is 1024x680).  Bounding-box validity must therefore follow
        the received image, not a build-time resolution.  A resolution change
        also invalidates pixel-space smoothing history from the old frame.
        """
        img_w = int(img_w)
        img_h = int(img_h)
        if img_w <= 0 or img_h <= 0:
            raise ValueError("frame dimensions must be positive")
        if (img_w, img_h) == (self.img_w, self.img_h):
            return
        self.img_w = img_w
        self.img_h = img_h
        for state in self._parts.values():
            state.last_box = None
            state.last_seen = None

    # --- geometry helpers -------------------------------------------------
    @staticmethod
    def _dims(box: List[float]):
        x1, y1, x2, y2 = (float(v) for v in box)
        return x2 - x1, y2 - y1

    def _basic_valid(self, box: List[float]) -> bool:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 5 or h <= 5:
            return False
        if x1 < -5 or y1 < -5:
            return False
        if x2 > self.img_w + 5 or y2 > self.img_h + 5:
            return False
        return True

    def _valid_measurement(self, part_name: str, box: List[float], msg: BoundingBoxes) -> bool:
        if not self._basic_valid(box):
            return False
        w, h = self._dims(box)
        x1, y1 = float(box[0]), float(box[1])

        if part_name == "head":
            # Preserve the original 640x480 tuning as normalized geometry so
            # it behaves identically at any camera resolution.
            if h > self.img_h * (310.0 / 480.0) or w > self.img_w * (270.0 / 640.0):
                return False
            if y1 <= self.img_h * (3.0 / 480.0) and h > self.img_h * (300.0 / 480.0):
                return False
            if bool(getattr(msg, "has_body", False)):
                body = [msg.body_x1, msg.body_y1, msg.body_x2, msg.body_y2]
                _, body_h = self._dims(body)
                if (body_h > self.img_h * (20.0 / 480.0)
                        and (h / body_h) > 1.10
                        and h > self.img_h * (280.0 / 480.0)):
                    return False
            old = self._parts["head"].last_box
            if old is not None:
                _, old_h = self._dims(old)
                if (old_h > self.img_h * (20.0 / 480.0)
                        and h > old_h * 2.20
                        and h > self.img_h * (300.0 / 480.0)):
                    return False

        if part_name == "body":
            if h < self.img_h * (50.0 / 480.0) or w > self.img_w * (330.0 / 640.0):
                return False

        return True

    def _smooth(self, old: Optional[List[float]], new: List[float]) -> List[float]:
        if old is None:
            return [float(v) for v in new]
        return [self.alpha * float(n) + (1.0 - self.alpha) * float(o) for o, n in zip(old, new)]

    # --- main transform ---------------------------------------------------
    def _update_part(self, msg_in: BoundingBoxes, msg_out: BoundingBoxes, part_name: str) -> None:
        now = time.monotonic()
        st = self._parts[part_name]
        detected = bool(getattr(msg_in, st.has_field, False))

        if detected:
            measured = [getattr(msg_in, f) for f in st.fields]
            if self._valid_measurement(part_name, measured, msg_in):
                st.last_box = self._smooth(st.last_box, measured)
                st.last_seen = now
                setattr(msg_out, st.has_field, True)
                for f, v in zip(st.fields, st.last_box):
                    setattr(msg_out, f, int(round(v)))
                return
            self.rejected_measurements += 1
            detected = False  # bad measurement -> treat as dropout

        if st.last_box is not None and st.last_seen is not None and (now - st.last_seen) <= self.hold_time_s:
            setattr(msg_out, st.has_field, True)
            for f, v in zip(st.fields, st.last_box):
                setattr(msg_out, f, int(round(v)))
            return

        setattr(msg_out, st.has_field, False)
        for f in st.fields:
            setattr(msg_out, f, 0)

    def _derive_body_from_head(self, msg_out: BoundingBoxes) -> None:
        """Keep body locked under the stabilized head (no independent body smoothing)."""
        head = {
            "x1": int(msg_out.head_x1),
            "y1": int(msg_out.head_y1),
            "x2": int(msg_out.head_x2),
            "y2": int(msg_out.head_y2),
        }
        body = estimate_body_box(head, self.img_w, self.img_h)
        st = self._parts["body"]
        st.last_box = [float(body["x1"]), float(body["y1"]), float(body["x2"]), float(body["y2"])]
        st.last_seen = time.monotonic()
        msg_out.has_body = True
        msg_out.body_x1 = int(body["x1"])
        msg_out.body_y1 = int(body["y1"])
        msg_out.body_x2 = int(body["x2"])
        msg_out.body_y2 = int(body["y2"])

    def stabilize(self, msg_in: BoundingBoxes) -> BoundingBoxes:
        """Return a new stabilized BoundingBoxes derived from ``msg_in``."""
        msg_out = copy.deepcopy(msg_in)
        self._update_part(msg_in, msg_out, "head")
        if msg_out.has_head:
            self._derive_body_from_head(msg_out)
        else:
            self._update_part(msg_in, msg_out, "body")
        return msg_out
