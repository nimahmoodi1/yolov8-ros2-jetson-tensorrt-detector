"""Stabilizer keeps body geometrically coupled to the smoothed head."""

from ros2_interface.msg import BoundingBoxes
from vision_perception.stabilizer import BoundingBoxStabilizer


def _msg(head, body=None):
    msg = BoundingBoxes()
    if head is not None:
        msg.has_head = True
        msg.head_x1, msg.head_y1, msg.head_x2, msg.head_y2 = head
    if body is not None:
        msg.has_body = True
        msg.body_x1, msg.body_y1, msg.body_x2, msg.body_y2 = body
    return msg


def test_body_derived_from_stabilized_head_not_smoothed_independently():
    stab = BoundingBoxStabilizer(hold_time_s=2.0, alpha=0.5, img_w=640, img_h=480)
    first = _msg((200, 40, 440, 180), (180, 200, 460, 400))
    out1 = stab.stabilize(first)
    assert out1.has_head and out1.has_body
    assert out1.body_y1 == out1.head_y2

    second = _msg((210, 50, 450, 190), (160, 50, 480, 350))
    out2 = stab.stabilize(second)
    assert out2.body_y1 == out2.head_y2
    assert out2.body_y1 >= out2.head_y2


def test_body_only_path_when_no_head():
    stab = BoundingBoxStabilizer(hold_time_s=2.0, alpha=1.0, img_w=640, img_h=480)
    msg = _msg(None, (100, 120, 300, 420))
    out = stab.stabilize(msg)
    assert not out.has_head
    assert out.has_body
    assert out.body_y1 == 120


def test_real_1024x680_box_is_not_rejected_by_old_640x480_bounds():
    stab = BoundingBoxStabilizer(hold_time_s=2.0, alpha=1.0)
    stab.set_frame_size(1024, 680)

    out = stab.stabilize(_msg((665, 339, 851, 515)))

    assert out.has_head
    assert out.has_body
    assert 0 <= out.body_x1 < out.body_x2 <= 1023
    assert 0 <= out.body_y1 < out.body_y2 <= 679


def test_resolution_change_resets_pixel_space_history():
    stab = BoundingBoxStabilizer(hold_time_s=2.0, alpha=0.5, img_w=640, img_h=480)
    stab.stabilize(_msg((200, 40, 440, 180)))

    stab.set_frame_size(1024, 680)
    out = stab.stabilize(_msg((665, 339, 851, 515)))

    assert (out.head_x1, out.head_y1, out.head_x2, out.head_y2) == (665, 339, 851, 515)
