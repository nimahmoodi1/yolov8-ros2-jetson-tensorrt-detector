"""Pixel<->meter gates aligned with the real camera geometry."""

import math

import numpy as np
import pytest

from vision_perception.detection import (
    meters_to_pixels_vertical,
    select_and_merge_head_box,
    select_body_box,
)
from vision_perception.geometry import pixels_to_meters_vertical
from vision_perception.temporal_filter import TemporalFilter

# The deployed checkpoints are 2-class: {0: 'body', 1: 'head-YJtu'}.
HEAD_IDS = (1,)
BODY_IDS = (0,)

HEAD_GATES = dict(
    min_aspect_ratio=0.60,
    max_aspect_ratio=1.50,
    min_area_fraction=0.0025,
    max_area_fraction=0.12,
)


def _dets(*rows):
    """Build the (N, 6) backend contract: x1 y1 x2 y2 conf cls."""
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _head(coords, confidence=0.9):
    return _dets((*coords, confidence, 1))


def test_sim_7m_standoff_vertical_scale():
    dist = 7.0
    vfov = 46.8
    height_px = 480
    span_m = 2.0 * dist * math.tan(math.radians(vfov) / 2.0)
    assert 5.8 < span_m < 6.3
    one_px_m = pixels_to_meters_vertical(1.0, height_px, dist, vfov)
    assert abs(one_px_m - span_m / height_px) < 1e-6
    merge_px = 1.2 / one_px_m
    assert 90 < merge_px < 110


def test_meters_to_pixels_is_the_exact_inverse():
    for height in (480, 680, 720):
        for delta_m in (0.4, 1.2, 3.0):
            px = meters_to_pixels_vertical(delta_m, height, 12.5, 45.47)
            back = pixels_to_meters_vertical(px, height, 12.5, 45.47)
            assert abs(back - delta_m) < 1e-9


def test_temporal_filter_rejects_large_horizontal_target_jump():
    filt = TemporalFilter(
        max_jump_m=1.0,
        reset_after_n_missed=6,
        distance_to_target_m=12.5,
        vertical_fov_deg=51.4,
        acquisition_frames=1,
    )
    first = {"x1": 400, "y1": 200, "x2": 560, "y2": 360}
    switched_target = {"x1": 700, "y1": 200, "x2": 860, "y2": 360}

    assert filt.step(first, img_height_px=680) == first
    assert filt.step(switched_target, img_height_px=680) is None


def test_new_track_requires_three_consistent_detections_with_one_dropout():
    filt = TemporalFilter(
        max_jump_m=1.0,
        reset_after_n_missed=6,
        distance_to_target_m=12.5,
        vertical_fov_deg=45.47,
        acquisition_frames=3,
        acquisition_max_missed_frames=1,
    )
    boxes = [
        {"x1": 400, "y1": 200, "x2": 550, "y2": 350},
        {"x1": 404, "y1": 202, "x2": 554, "y2": 352},
        {"x1": 407, "y1": 204, "x2": 557, "y2": 354},
    ]

    assert filt.step(boxes[0], 680) is None
    assert filt.step(None, 680) is None
    assert filt.step(boxes[1], 680) is None
    assert filt.step(boxes[2], 680) == boxes[2]


def test_head_geometry_rejects_thin_false_positive():
    result = select_and_merge_head_box(
        _head((480, 80, 520, 500)), (680, 1024, 3),
        0.78, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    )

    assert result is None


def test_head_geometry_accepts_measured_real_box_shape():
    result = select_and_merge_head_box(
        _head((665, 339, 851, 515)), (680, 1024, 3),
        0.78, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    )

    assert result is not None
    assert (result["x1"], result["y1"], result["x2"], result["y2"]) == (
        665, 339, 851, 515)


def test_body_class_is_never_selected_as_a_head():
    body_shaped_as_a_head = _dets((665, 339, 851, 515, 0.99, 0))

    assert select_and_merge_head_box(
        body_shaped_as_a_head, (680, 1024, 3),
        0.78, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    ) is None


def test_confidence_threshold_is_applied_per_class():
    weak_head = _head((665, 339, 851, 515), confidence=0.70)

    assert select_and_merge_head_box(
        weak_head, (680, 1024, 3),
        0.78, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    ) is None


def test_nearby_head_fragments_are_unioned():
    # Two antenna fragments on the same tier, ~30 px apart vertically.
    fragments = _dets(
        (665, 339, 851, 515, 0.91, 1),
        (700, 360, 880, 530, 0.88, 1),
    )
    result = select_and_merge_head_box(
        fragments, (680, 1024, 3),
        0.78, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    )

    assert result is not None
    assert (result["x1"], result["y1"]) == (665, 339)
    assert (result["x2"], result["y2"]) == (880, 530)
    assert result["conf"] == pytest.approx(0.91, abs=1e-5)


def test_far_apart_heads_are_not_unioned():
    # Same tower, but tiers separated by far more than merge_center_delta_m.
    far = _dets(
        (665, 60, 851, 236, 0.91, 1),
        (665, 500, 851, 676, 0.88, 1),
    )
    result = select_and_merge_head_box(
        far, (680, 1024, 3),
        0.05, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    )

    assert result is not None
    assert result["y2"] - result["y1"] < 300     # one tier only, not the union


def test_empty_detections_select_nothing():
    empty = np.zeros((0, 6), dtype=np.float32)

    assert select_and_merge_head_box(
        empty, (680, 1024, 3), 0.78, 1.2, 12.5, 45.47,
        0.60, 1.50, 0.0025, 0.12, class_ids=HEAD_IDS) is None
    assert select_body_box(
        empty, (680, 1024, 3), 0.82,
        min_aspect_ratio=0.22, max_aspect_ratio=0.90,
        min_area_fraction=0.004, max_area_fraction=0.30,
        class_ids=BODY_IDS) is None


def test_detector_conf_floor_never_discards_a_usable_box():
    """The NMS floor is min(head, body) conf, so nothing acceptable is lost."""
    head_conf, body_conf = 0.78, 0.82
    floor = min(head_conf, body_conf)

    just_above_floor = _dets((665, 339, 851, 515, floor + 0.001, 1))
    result = select_and_merge_head_box(
        just_above_floor, (680, 1024, 3),
        head_conf, 1.2, 12.5, 45.47,
        HEAD_GATES["min_aspect_ratio"], HEAD_GATES["max_aspect_ratio"],
        HEAD_GATES["min_area_fraction"], HEAD_GATES["max_area_fraction"],
        class_ids=HEAD_IDS,
    )

    assert result is not None
