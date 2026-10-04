"""Display path: downscale before drawing, and cap the render rate."""

import numpy as np

from vision_perception.visualization import DisplayWindow, render_frame


def _frame():
    return np.full((720, 1280, 3), 90, dtype=np.uint8)


def test_render_downscales_to_the_window_width():
    out = render_frame(_frame(), None, None, target_width=960)

    assert out.shape[1] == 960
    assert out.shape[0] == 540          # aspect ratio preserved


def test_render_scales_box_coordinates_with_the_frame():
    head = {'x1': 640, 'y1': 360, 'x2': 700, 'y2': 420}

    out = render_frame(_frame(), head, None, target_width=640)

    # The magenta head rectangle must land at half the source coordinates.
    # (ys.min() would catch the "head" label drawn above the box, so check the
    # bottom edge, which nothing else can touch.)
    magenta = np.all(out == np.array([255, 0, 255], dtype=np.uint8), axis=2)
    ys, xs = np.nonzero(magenta)
    assert xs.size > 0
    assert 315 <= xs.min() <= 325       # 640 * 0.5
    assert 205 <= ys.max() <= 215       # 420 * 0.5


def test_render_never_upscales_a_small_frame():
    small = np.zeros((240, 320, 3), dtype=np.uint8)

    out = render_frame(small, None, None, target_width=960)

    assert out.shape[:2] == (240, 320)


def test_render_does_not_mutate_the_source_frame():
    source = _frame()
    original = source.copy()

    render_frame(source, {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100}, None)

    assert np.array_equal(source, original)


def test_render_accepts_a_read_only_frame():
    readonly = np.frombuffer(bytes(720 * 1280 * 3), dtype=np.uint8).reshape(720, 1280, 3)

    out = render_frame(readonly, None, None, target_width=960)

    assert out.shape[1] == 960


def test_display_rate_cap_drops_surplus_frames():
    window = DisplayWindow(max_fps=10.0)
    window._running = True              # no X server needed for the gate itself

    assert window.due() is True         # first call always passes
    assert window.due() is False        # immediately after -> throttled
    assert window.frames_skipped == 1


def test_display_rate_cap_can_be_disabled():
    window = DisplayWindow(max_fps=0.0)
    window._running = True

    assert all(window.due() for _ in range(5))
    assert window.frames_skipped == 0


def test_display_gate_is_closed_before_start():
    assert DisplayWindow(max_fps=10.0).due() is False


def test_marginal_downscale_is_skipped():
    """960 on a 1024-wide camera is a 0.94x resize: all cost, no saving."""
    source = np.full((680, 1024, 3), 90, dtype=np.uint8)

    out = render_frame(source, None, None, target_width=960)

    assert out.shape[:2] == (680, 1024)


def test_worthwhile_downscale_still_happens():
    source = np.full((680, 1024, 3), 90, dtype=np.uint8)

    out = render_frame(source, None, None, target_width=640)

    assert out.shape[1] == 640
    assert out.shape[0] == 425


def test_scene_submission_defers_drawing_to_the_display_thread():
    window = DisplayWindow(max_fps=0.0, width=640)
    window._running = True
    source = np.full((680, 1024, 3), 90, dtype=np.uint8)

    window.submit_scene(source, {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100}, None)

    kind, payload = window._queue.get_nowait()
    assert kind == 'scene'
    # The raw frame is queued untouched; nothing was rendered on this thread.
    assert payload[0] is source


def test_scene_submission_is_ignored_before_start():
    window = DisplayWindow(max_fps=0.0)
    window.submit_scene(np.zeros((10, 10, 3), dtype=np.uint8), None, None)

    assert window._queue.empty()


def test_newest_scene_replaces_a_pending_one():
    window = DisplayWindow(max_fps=0.0, width=640)
    window._running = True
    first = np.full((680, 1024, 3), 10, dtype=np.uint8)
    second = np.full((680, 1024, 3), 20, dtype=np.uint8)

    window.submit_scene(first, None, None)
    window.submit_scene(second, None, None)

    _, payload = window._queue.get_nowait()
    assert payload[0] is second
    assert window._queue.empty()
