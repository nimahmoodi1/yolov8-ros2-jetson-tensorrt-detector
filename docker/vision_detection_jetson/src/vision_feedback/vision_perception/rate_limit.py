"""Frame-rate limiting for the inference path.

A naive rate cap ("reject until ``now >= deadline``") aliases badly against a
camera whose frame interval does not divide the cap period.

Measured on the vehicle: a 29.8 Hz camera (33.6 ms interval) against a 20 Hz
cap (50 ms period) settled at **14.5 Hz**, not 20 Hz. Frame *n* lands at 33.6 ms
against a 50 ms deadline and is rejected; frame *n+1* lands at 67.2 ms and is
accepted, pushing the deadline to 117.2 ms. The pattern locks to "every second
frame" and the node loses a third of the throughput it was allowed to have.

:class:`PhaseTolerantRateLimiter` fixes both halves of that:

* a frame arriving within ``tolerance`` of its deadline still counts, so the
  accept pattern can alternate 1-2-1-2 and average out at the requested rate;
* the next deadline advances from the *previous deadline* rather than from the
  arrival time, so late frames do not drag the average down.

The tolerance is derived from the measured inter-arrival interval, so it
adapts to whatever the camera is actually doing.
"""

from __future__ import annotations

# Half an inter-arrival interval: large enough to break the aliasing, small
# enough that the cap is never exceeded by more than one frame in a row.
_TOLERANCE_FRACTION = 0.5

# Intervals outside this range are treated as startup noise or a stalled feed
# rather than a real camera period.
_MIN_INTERVAL_S = 1e-4
_MAX_INTERVAL_S = 1.0

_SMOOTHING = 0.1


class PhaseTolerantRateLimiter:
    """Decide whether a frame arriving at ``now`` should be processed."""

    def __init__(self, max_fps: float):
        max_fps = float(max_fps)
        self.period = (1.0 / max_fps) if max_fps > 0 else 0.0
        self.interval_ema = 0.0
        self.rejected = 0
        self._next_due = 0.0
        self._last_arrival = 0.0

    @property
    def enabled(self) -> bool:
        return self.period > 0.0

    @property
    def measured_fps(self) -> float:
        """Observed source frame rate, or 0.0 before two frames have arrived."""
        return (1.0 / self.interval_ema) if self.interval_ema > 0.0 else 0.0

    def observe(self, now: float) -> None:
        """Record an arrival, updating the measured interval."""
        if self._last_arrival > 0.0:
            interval = now - self._last_arrival
            if _MIN_INTERVAL_S < interval < _MAX_INTERVAL_S:
                self.interval_ema = (
                    interval if self.interval_ema <= 0.0
                    else (1.0 - _SMOOTHING) * self.interval_ema + _SMOOTHING * interval
                )
        self._last_arrival = now

    def allow(self, now: float) -> bool:
        """Return True when this frame should be processed.

        Call :meth:`observe` first; this method only decides.
        """
        if not self.enabled:
            return True
        tolerance = _TOLERANCE_FRACTION * self.interval_ema
        if now + tolerance < self._next_due:
            self.rejected += 1
            return False
        self._next_due = max(now, self._next_due) + self.period
        return True

    def step(self, now: float) -> bool:
        """:meth:`observe` then :meth:`allow`, which is what callers usually want."""
        self.observe(now)
        return self.allow(now)
