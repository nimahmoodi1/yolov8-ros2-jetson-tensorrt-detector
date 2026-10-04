"""Head/body box selection from a backend detection array.

The selection rules are unchanged from the validated pipeline:

* only boxes whose class name contains ``head`` / ``body`` are considered,
  and only above the per-class confidence threshold;
* aspect-ratio and area-fraction gates reject poles, people and background
  regions that YOLO occasionally labels as tower parts;
* the winner is the highest confidence with a mild centre preference;
* head fragments that sit on the same tier are unioned into one box.

What changed is *how* they are evaluated. The previous implementation looped
over Ultralytics ``Boxes`` objects in Python and pulled each box off the GPU
individually. Here the whole frame's detections arrive as one ``(N, 6)`` array
(see :mod:`vision_perception.backends.base`) and every gate is a vectorised
NumPy comparison. With ``max_det=20`` this turns ~40 GPU synchronisations and a
few hundred Python-level attribute lookups per frame into a handful of array
operations on at most 20 rows.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vision_perception.backends.base import EMPTY_DETECTIONS, is_cuda_oom  # noqa: F401
from vision_perception.geometry import Box

# Candidate array columns after :func:`collect_candidates`.
_C_X1, _C_Y1, _C_X2, _C_Y2, _C_CONF, _C_CLS, _C_CX, _C_CY = range(8)

_EMPTY_CANDIDATES = np.zeros((0, 8), dtype=np.float64)


def meters_to_pixels_vertical(
    delta_m: float,
    img_height_px: int,
    distance_to_target_m: float,
    vertical_fov_deg: float,
) -> float:
    """Inverse of :func:`geometry.pixels_to_meters_vertical`.

    Converting the metric gate into pixels once per frame is cheaper than (and
    numerically identical to) converting every candidate's pixel delta into
    metres.
    """
    vfov_rad = math.radians(float(vertical_fov_deg))
    span_m = 2.0 * float(distance_to_target_m) * math.tan(vfov_rad / 2.0)
    if span_m <= 0.0:
        return float('inf')
    metres_per_px = span_m / float(max(1, int(img_height_px)))
    return float(delta_m) / max(metres_per_px, 1e-12)


def collect_candidates(
    detections: np.ndarray,
    img_shape: Sequence[int],
    conf_thresh: float,
    class_ids: Iterable[int],
    *,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_area_fraction: float,
    max_area_fraction: float,
) -> np.ndarray:
    """Return the gated candidates as an ``(M, 8)`` array.

    Columns: ``x1, y1, x2, y2, conf, cls, cx, cy`` with integral pixel corners
    clipped to the frame, matching the original dict-based behaviour.
    """
    if detections is None or len(detections) == 0:
        return _EMPTY_CANDIDATES
    wanted = np.asarray(list(class_ids), dtype=np.int64)
    if wanted.size == 0:
        return _EMPTY_CANDIDATES

    dets = np.asarray(detections, dtype=np.float64)
    if dets.ndim != 2 or dets.shape[1] < 6:
        return _EMPTY_CANDIDATES

    height, width = int(img_shape[0]), int(img_shape[1])

    # Truncate toward zero, as the previous int() conversion did.
    corners = dets[:, :4].astype(np.int64)
    x1, y1, x2, y2 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    conf = dets[:, 4]
    cls = dets[:, 5].astype(np.int64)

    box_w = (x2 - x1).astype(np.float64)
    box_h = (y2 - y1).astype(np.float64)

    keep = (x2 > x1) & (y2 > y1)
    keep &= conf >= float(conf_thresh)
    keep &= np.isin(cls, wanted)
    aspect = box_w / np.maximum(1.0, box_h)
    area_fraction = (box_w * box_h) / max(1.0, float(width * height))
    keep &= (aspect >= float(min_aspect_ratio)) & (aspect <= float(max_aspect_ratio))
    keep &= (area_fraction >= float(min_area_fraction)) & (
        area_fraction <= float(max_area_fraction))

    if not keep.any():
        return _EMPTY_CANDIDATES

    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    out = np.empty((x1.size, 8), dtype=np.float64)
    out[:, _C_X1] = np.maximum(0, x1)
    out[:, _C_Y1] = np.maximum(0, y1)
    out[:, _C_X2] = np.minimum(width - 1, x2)
    out[:, _C_Y2] = np.minimum(height - 1, y2)
    out[:, _C_CONF] = conf[keep]
    out[:, _C_CLS] = cls[keep]
    # Centres use the pre-clip corners, as before.
    out[:, _C_CX] = (x1 + x2) // 2
    out[:, _C_CY] = (y1 + y2) // 2
    return out


def _best_index(candidates: np.ndarray, width: int, height: int) -> int:
    """Highest confidence, with a mild preference for the image centre.

    The centre term only breaks near-ties; it is small enough that an
    unrelated central object never outranks a confident off-centre target.
    """
    cx_img, cy_img = width // 2, height // 2
    dx = candidates[:, _C_CX] - cx_img
    dy = candidates[:, _C_CY] - cy_img
    centre_distance = np.hypot(dx, dy) / float(max(width, height))
    return int(np.argmax(candidates[:, _C_CONF] - 0.10 * centre_distance))


def _as_box(row: np.ndarray, label: str) -> Box:
    return {
        'x1': int(row[_C_X1]),
        'y1': int(row[_C_Y1]),
        'x2': int(row[_C_X2]),
        'y2': int(row[_C_Y2]),
        'conf': float(row[_C_CONF]),
        'label': label,
        'cx': int(row[_C_CX]),
        'cy': int(row[_C_CY]),
    }


def select_and_merge_head_box(
    detections: np.ndarray,
    img_shape: Sequence[int],
    conf_thresh: float,
    merge_center_delta_m: float,
    distance_to_target_m: float,
    vertical_fov_deg: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_area_fraction: float,
    max_area_fraction: float,
    class_ids: Iterable[int] = (),
) -> Optional[Box]:
    """Pick the best head box and union near-collinear head fragments."""
    height, width = int(img_shape[0]), int(img_shape[1])
    candidates = collect_candidates(
        detections, img_shape, conf_thresh, class_ids,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    )
    if candidates.shape[0] == 0:
        return None

    best = candidates[_best_index(candidates, width, height)]
    if candidates.shape[0] == 1:
        return _as_box(best, 'head')

    merge_px = meters_to_pixels_vertical(
        merge_center_delta_m, height, distance_to_target_m, vertical_fov_deg)

    vertical_delta = np.abs(candidates[:, _C_CY] - best[_C_CY])
    horizontal_gap = np.maximum(
        0.0,
        np.maximum(candidates[:, _C_X1], best[_C_X1])
        - np.minimum(candidates[:, _C_X2], best[_C_X2]),
    )
    merged = candidates[(vertical_delta <= merge_px) & (horizontal_gap <= merge_px)]
    if merged.shape[0] == 0:            # defensive: `best` always qualifies
        merged = best[None, :]

    return {
        'x1': int(max(0, merged[:, _C_X1].min())),
        'y1': int(max(0, merged[:, _C_Y1].min())),
        'x2': int(min(width - 1, merged[:, _C_X2].max())),
        'y2': int(min(height - 1, merged[:, _C_Y2].max())),
        'conf': float(merged[:, _C_CONF].max()),
        'label': 'head',
    }


def select_body_box(
    detections: np.ndarray,
    img_shape: Sequence[int],
    conf_thresh: float,
    *,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_area_fraction: float,
    max_area_fraction: float,
    class_ids: Iterable[int] = (),
) -> Optional[Box]:
    """Pick the body box closest to image centre, or ``None``."""
    height, width = int(img_shape[0]), int(img_shape[1])
    candidates = collect_candidates(
        detections, img_shape, conf_thresh, class_ids,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    )
    if candidates.shape[0] == 0:
        return None
    return _as_box(candidates[_best_index(candidates, width, height)], 'body')


def detections_from_results(results: Iterable[Any]) -> np.ndarray:
    """Bridge Ultralytics ``Results`` objects into the ``(N, 6)`` contract.

    Only needed by callers that still hold raw Ultralytics output (tests,
    ad-hoc scripts). The production backends already return the array form.
    """
    rows: List[Tuple[float, float, float, float, float, float]] = []
    for result in results or ():
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue
        data = getattr(boxes, 'data', None)
        if data is not None and hasattr(data, 'shape'):
            array = data.detach().cpu().numpy() if hasattr(data, 'detach') else np.asarray(data)
            rows.extend(map(tuple, np.asarray(array, dtype=np.float64)[:, :6]))
            continue
        for box in boxes:
            try:
                corners = np.asarray(box.xyxy[0], dtype=np.float64).reshape(-1)[:4]
            except Exception:
                continue
            rows.append((
                float(corners[0]), float(corners[1]),
                float(corners[2]), float(corners[3]),
                float(np.asarray(getattr(box, 'conf', 0.0)).reshape(-1)[0]),
                float(np.asarray(getattr(box, 'cls', 0)).reshape(-1)[0]),
            ))
    if not rows:
        return EMPTY_DETECTIONS
    return np.asarray(rows, dtype=np.float32)
