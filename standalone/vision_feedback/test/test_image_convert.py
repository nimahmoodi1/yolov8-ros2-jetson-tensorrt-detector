"""Zero-copy ROS Image conversion, the replacement for cv_bridge on the hot path."""

import array

import numpy as np

from vision_perception import image_convert


class _Msg:
    def __init__(self, height, width, encoding, data, step=None, channels=3):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step if step is not None else width * channels
        self.data = data


def _payload(arr):
    return array.array('B', arr.tobytes())


def test_bgr8_is_converted_without_copying():
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    msg = _Msg(48, 64, 'bgr8', _payload(frame))

    out = image_convert.to_bgr(msg)

    assert out.shape == (48, 64, 3)
    assert np.array_equal(out, frame)
    # A view over the message buffer, not a fresh allocation.
    assert out.base is not None


def test_rgb8_channel_order_is_corrected():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 255          # red in an rgb8 payload
    msg = _Msg(4, 4, 'rgb8', _payload(frame))

    out = image_convert.to_bgr(msg)

    assert out[0, 0, 2] == 255   # must land in the B-G-R blue-last slot
    assert out[0, 0, 0] == 0


def test_mono8_is_expanded_to_three_channels():
    frame = np.full((8, 8), 120, dtype=np.uint8)
    msg = _Msg(8, 8, 'mono8', _payload(frame), step=8, channels=1)

    out = image_convert.to_bgr(msg)

    assert out.shape == (8, 8, 3)
    assert np.all(out == 120)


def test_row_padding_in_step_is_respected():
    width, height, pad = 10, 4, 6
    padded = np.random.randint(0, 255, (height, width * 3 + pad), dtype=np.uint8)
    msg = _Msg(height, width, 'bgr8', _payload(padded), step=width * 3 + pad)

    out = image_convert.to_bgr(msg)

    assert out.shape == (height, width, 3)
    assert np.array_equal(out, padded[:, : width * 3].reshape(height, width, 3))


def test_unknown_encoding_without_bridge_returns_none():
    msg = _Msg(4, 4, 'bayer_rggb8', _payload(np.zeros((4, 4), dtype=np.uint8)))

    assert image_convert.to_bgr(msg, bridge=None) is None


def test_ensure_writable_only_copies_when_needed():
    writable = np.zeros((4, 4, 3), dtype=np.uint8)
    assert image_convert.ensure_writable(writable) is writable

    readonly = np.frombuffer(bytes(4 * 4 * 3), dtype=np.uint8).reshape(4, 4, 3)
    out = image_convert.ensure_writable(readonly)
    assert out.flags.writeable
