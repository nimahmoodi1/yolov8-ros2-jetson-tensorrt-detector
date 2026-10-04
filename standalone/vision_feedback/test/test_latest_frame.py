import numpy as np

from vision_perception.latest_frame import FramePacket, LatestFrameSlot


def _packet(sequence):
    return FramePacket(sequence, np.full((2, 2, 3), sequence, dtype=np.uint8), 1.0)


def test_latest_frame_replaces_stale_pending_frame():
    slot = LatestFrameSlot()

    assert slot.put(_packet(1)) is False
    assert slot.put(_packet(2)) is True

    packet = slot.take(timeout_s=0.01)
    assert packet.sequence == 2
    assert int(packet.image[0, 0, 0]) == 2


def test_closed_slot_wakes_and_returns_no_frame():
    slot = LatestFrameSlot()
    slot.close()

    assert slot.take(timeout_s=0.01) is None
    assert slot.put(_packet(1)) is False
