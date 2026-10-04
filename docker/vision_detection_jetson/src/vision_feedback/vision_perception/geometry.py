"""Pixel/meter geometry helpers and body-box estimation.

A *box* throughout this package is a plain dict with integer pixel corners and
optional metadata::

    {"x1": int, "y1": int, "x2": int, "y2": int, "conf": float, "label": str}
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

Box = Dict[str, Any]


def to_float(value: Any) -> float:
    """Best-effort conversion of scalars/tensors to float (0.0 on failure)."""
    try:
        return float(value)
    except Exception:
        try:
            return float(np.array(value).reshape(-1)[0])
        except Exception:
            return 0.0


def to_int(value: Any) -> int:
    """Best-effort conversion of scalars/tensors to int (0 on failure)."""
    try:
        return int(value)
    except Exception:
        try:
            return int(np.array(value).reshape(-1)[0])
        except Exception:
            return 0


def pixels_to_meters_vertical(
    delta_px: float,
    img_height_px: int,
    distance_to_target_m: float,
    vertical_fov_deg: float,
) -> float:
    """Convert a vertical pixel displacement to an approximate metric distance.

    Uses the pinhole relation: the image height spans
    ``2 * d * tan(vfov / 2)`` meters at distance ``d``.
    """
    vfov_rad = math.radians(float(vertical_fov_deg))
    span_m = 2.0 * float(distance_to_target_m) * math.tan(vfov_rad / 2.0)
    return abs(span_m * (float(delta_px) / float(max(1, img_height_px))))


def estimate_body_box(head_box: Box, frame_w: int, frame_h: int) -> Box:
    """Derive a body box from a head box.

    Heuristic (unchanged from the validated pipeline): widen ~20 % beyond the
    head and extend ~2.5x the head height downward.
    """
    hx1, hy1, hx2, hy2 = head_box["x1"], head_box["y1"], head_box["x2"], head_box["y2"]
    head_h = max(1, hy2 - hy1)
    head_w = max(1, hx2 - hx1)

    width_expansion = int(head_w * 0.2)
    body_height = int(head_h * 2.5)

    bx1 = max(0, hx1 - width_expansion // 2)
    bx2 = min(frame_w - 1, hx2 + width_expansion // 2)
    by1 = hy2
    by2 = min(frame_h - 1, by1 + body_height)
    return {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2}
