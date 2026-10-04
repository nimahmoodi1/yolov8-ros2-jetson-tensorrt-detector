"""The inference rate cap must not alias the camera down to a lower rate."""

import pytest

from vision_perception.rate_limit import PhaseTolerantRateLimiter


def _run(limiter, source_fps, seconds):
    """Feed evenly spaced arrivals and return how many were accepted."""
    interval = 1.0 / source_fps
    count = int(seconds * source_fps)
    accepted = 0
    for i in range(count):
        if limiter.step(i * interval):
            accepted += 1
    return accepted / seconds


def test_cap_disabled_accepts_everything():
    limiter = PhaseTolerantRateLimiter(0.0)

    assert not limiter.enabled
    assert _run(limiter, 30.0, 4.0) == pytest.approx(30.0, rel=0.02)
    assert limiter.rejected == 0


def test_slower_camera_is_never_throttled():
    limiter = PhaseTolerantRateLimiter(20.0)

    assert _run(limiter, 15.0, 6.0) == pytest.approx(15.0, rel=0.03)
    assert limiter.rejected == 0


def test_30hz_camera_against_a_20hz_cap_lands_near_20hz():
    """The regression this class exists for: the naive cap gave 15 Hz here."""
    limiter = PhaseTolerantRateLimiter(20.0)

    rate = _run(limiter, 29.8, 10.0)

    assert 18.5 <= rate <= 21.0


def test_cap_is_not_exceeded_over_the_long_run():
    for source in (24.0, 29.8, 30.0, 60.0):
        limiter = PhaseTolerantRateLimiter(20.0)
        assert _run(limiter, source, 10.0) <= 21.0


def test_measured_source_rate_is_reported():
    limiter = PhaseTolerantRateLimiter(20.0)
    _run(limiter, 29.8, 5.0)

    assert limiter.measured_fps == pytest.approx(29.8, rel=0.05)


def test_measured_rate_is_zero_before_two_frames():
    limiter = PhaseTolerantRateLimiter(20.0)
    limiter.step(0.0)

    assert limiter.measured_fps == 0.0


def test_a_stalled_feed_does_not_poison_the_measured_interval():
    limiter = PhaseTolerantRateLimiter(20.0)
    for i in range(50):
        limiter.step(i / 30.0)
    steady = limiter.measured_fps

    limiter.step(50 / 30.0 + 8.0)     # 8 second gap: camera froze

    assert limiter.measured_fps == pytest.approx(steady, rel=0.01)


def test_rejected_frames_are_counted():
    limiter = PhaseTolerantRateLimiter(10.0)
    _run(limiter, 30.0, 3.0)

    assert limiter.rejected > 0
