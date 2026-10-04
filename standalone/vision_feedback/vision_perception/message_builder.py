"""Build ``vision_feedback/BoundingBoxes`` messages from box dicts."""

from __future__ import annotations

from typing import Optional

from vision_feedback.msg import BoundingBoxes

from vision_perception.geometry import Box


def build_bounding_boxes_msg(head_box: Optional[Box], body_box: Optional[Box]) -> BoundingBoxes:
    """Create a BoundingBoxes message.

    Missing detections are encoded explicitly (``has_* = False`` and zeroed
    coordinates) so downstream consumers always receive a well-formed message.
    """
    msg = BoundingBoxes()

    msg.has_head = head_box is not None
    msg.head_x1 = int(head_box["x1"]) if head_box else 0
    msg.head_y1 = int(head_box["y1"]) if head_box else 0
    msg.head_x2 = int(head_box["x2"]) if head_box else 0
    msg.head_y2 = int(head_box["y2"]) if head_box else 0

    msg.has_body = body_box is not None
    msg.body_x1 = int(body_box["x1"]) if body_box else 0
    msg.body_y1 = int(body_box["y1"]) if body_box else 0
    msg.body_x2 = int(body_box["x2"]) if body_box else 0
    msg.body_y2 = int(body_box["y2"]) if body_box else 0

    return msg
