"""Static-shape letterbox geometry, the contract between preprocessing and TRT."""

import numpy as np
import pytest

from vision_perception.letterbox import (
    anchor_count,
    auto_input_shape,
    compute_letterbox,
    parse_input_shape,
)


def test_auto_shape_matches_the_720p_camera_without_square_padding():
    # The key optimisation: 1280x720 needs 384x640, not 640x640.
    assert auto_input_shape(720, 1280, 640) == (384, 640)


def test_auto_shape_handles_the_1024x680_sony_stream():
    assert auto_input_shape(680, 1024, 640) == (448, 640)


def test_auto_shape_is_cheaper_than_square_for_the_same_detail():
    auto_h, auto_w = auto_input_shape(720, 1280, 640)
    assert auto_h * auto_w < 640 * 640
    # And far cheaper than the previous 960 square setting.
    assert (auto_h * auto_w) * 3.7 < 960 * 960


def test_parse_input_shape_modes():
    assert parse_input_shape('auto', 720, 1280, 640) == (384, 640)
    assert parse_input_shape('square', 720, 1280, 640) == (640, 640)
    assert parse_input_shape('576x960', 720, 1280, 640) == (576, 960)
    # Non-stride-aligned values are rounded up, never down.
    assert parse_input_shape('370x630', 720, 1280, 640) == (384, 640)


def test_parse_input_shape_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_input_shape('big', 720, 1280, 640)


def test_letterbox_round_trip_recovers_source_pixels():
    tf = compute_letterbox(720, 1280, 384, 640)
    assert tf.gain == pytest.approx(0.5)
    assert (tf.resized_h, tf.resized_w) == (360, 640)
    assert tf.pad_left == 0 and tf.pad_top == 12

    source = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float64)
    net = source.copy()
    net[:, [0, 2]] = net[:, [0, 2]] * tf.gain + tf.pad_left
    net[:, [1, 3]] = net[:, [1, 3]] * tf.gain + tf.pad_top

    recovered = tf.to_source(net)
    assert recovered == pytest.approx(source, abs=1e-6)


def test_letterbox_clips_to_the_source_frame():
    tf = compute_letterbox(720, 1280, 384, 640)
    out_of_frame = np.array([[-50.0, -50.0, 5000.0, 5000.0]], dtype=np.float64)

    clipped = tf.to_source(out_of_frame)

    assert clipped[0, 0] >= 0 and clipped[0, 1] >= 0
    assert clipped[0, 2] <= 1279 and clipped[0, 3] <= 719


def test_letterbox_pads_the_axis_that_is_not_gain_limited():
    # 1280x720 into 384x640 is width-limited, so the padding is top/bottom.
    wide = compute_letterbox(720, 1280, 384, 640)
    assert wide.pad_left == 0 and wide.pad_top == 12

    # 640x480 into the same network is height-limited, so it pads left/right.
    squarish = compute_letterbox(480, 640, 384, 640)
    assert squarish.pad_top == 0 and squarish.pad_left == 64


def test_anchor_count_matches_yolov8_heads():
    assert anchor_count(640, 640) == 8400
    assert anchor_count(384, 640) == 5040
