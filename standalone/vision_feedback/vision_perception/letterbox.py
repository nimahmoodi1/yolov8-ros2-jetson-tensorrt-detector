"""Letterbox geometry shared by every detector backend.

A TensorRT engine has a *static* input shape, so the letterbox maths has to be
explicit and identical on both the preprocessing and the box-rescaling side.
This module holds that maths and nothing else, which makes it cheap to unit
test without CUDA, torch or ROS.

The convention matches Ultralytics ``LetterBox(auto=False, center=True)`` so a
TensorRT engine and the PyTorch fallback produce the same boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_STRIDE = 32


@dataclass(frozen=True)
class LetterboxTransform:
    """Mapping between source-image pixels and network-input pixels."""

    src_h: int
    src_w: int
    net_h: int
    net_w: int
    gain: float
    pad_left: int
    pad_top: int
    resized_h: int
    resized_w: int

    def to_source(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map boxes from network pixels back to source pixels (in place).

        Column slices are used deliberately: ``boxes[:, [0, 2]]`` is fancy
        indexing and returns a *copy*, so an ``out=`` write there would be
        silently discarded and the boxes would never get clipped.
        """
        if boxes_xyxy.size == 0:
            return boxes_xyxy
        inv_gain = 1.0 / self.gain
        for index, pad, limit in (
            (0, self.pad_left, self.src_w - 1),
            (2, self.pad_left, self.src_w - 1),
            (1, self.pad_top, self.src_h - 1),
            (3, self.pad_top, self.src_h - 1),
        ):
            column = boxes_xyxy[:, index]          # basic slice -> a view
            column -= pad
            column *= inv_gain
            np.clip(column, 0, limit, out=column)
        return boxes_xyxy


def compute_letterbox(src_h: int, src_w: int, net_h: int, net_w: int) -> LetterboxTransform:
    """Fit ``src`` inside ``net`` preserving aspect ratio, padding symmetrically."""
    src_h = int(src_h)
    src_w = int(src_w)
    net_h = int(net_h)
    net_w = int(net_w)
    if min(src_h, src_w, net_h, net_w) <= 0:
        raise ValueError('letterbox dimensions must be positive')

    gain = min(net_w / src_w, net_h / src_h)
    resized_w = int(round(src_w * gain))
    resized_h = int(round(src_h * gain))
    dw = (net_w - resized_w) / 2.0
    dh = (net_h - resized_h) / 2.0
    return LetterboxTransform(
        src_h=src_h,
        src_w=src_w,
        net_h=net_h,
        net_w=net_w,
        gain=gain,
        pad_left=int(round(dw - 0.1)),
        pad_top=int(round(dh - 0.1)),
        resized_h=resized_h,
        resized_w=resized_w,
    )


def auto_input_shape(src_h: int, src_w: int, long_side: int, stride: int = _STRIDE) -> Tuple[int, int]:
    """Return the (h, w) engine shape that matches the camera aspect ratio.

    YOLO only needs the *long* side to reach ``long_side``; padding the short
    side out to a square wastes compute on grey pixels that carry no signal.
    A 1280x720 frame at ``long_side=640`` becomes 384x640 instead of 640x640,
    which is 40 % fewer pixels through the network for identical detail.
    """
    src_h = int(src_h)
    src_w = int(src_w)
    long_side = int(long_side)
    if min(src_h, src_w) <= 0 or long_side <= 0:
        raise ValueError('auto_input_shape needs positive dimensions')

    gain = long_side / float(max(src_h, src_w))
    net_h = int(math.ceil(src_h * gain / stride) * stride)
    net_w = int(math.ceil(src_w * gain / stride) * stride)
    return max(stride, net_h), max(stride, net_w)


def parse_input_shape(spec: str, src_h: int, src_w: int, long_side: int) -> Tuple[int, int]:
    """Resolve the ``inference_input_shape`` parameter.

    ``auto``      -> :func:`auto_input_shape` from the expected camera geometry
    ``square``    -> ``long_side`` x ``long_side``
    ``HxW``       -> explicit, rounded up to the stride
    """
    text = str(spec or 'auto').strip().lower()
    if text in ('', 'auto'):
        return auto_input_shape(src_h, src_w, long_side)
    if text == 'square':
        side = int(math.ceil(long_side / _STRIDE) * _STRIDE)
        return side, side
    if 'x' not in text:
        raise ValueError(f"inference_input_shape '{spec}' must be 'auto', 'square' or 'HxW'")
    raw_h, raw_w = text.split('x', 1)
    net_h = int(math.ceil(int(raw_h) / _STRIDE) * _STRIDE)
    net_w = int(math.ceil(int(raw_w) / _STRIDE) * _STRIDE)
    if net_h <= 0 or net_w <= 0:
        raise ValueError(f"inference_input_shape '{spec}' resolved to a non-positive size")
    return net_h, net_w


def anchor_count(net_h: int, net_w: int) -> int:
    """Number of YOLOv8 anchor points for a given input shape (P3/P4/P5)."""
    return sum((net_h // s) * (net_w // s) for s in (8, 16, 32))
