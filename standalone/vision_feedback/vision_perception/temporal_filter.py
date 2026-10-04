"""Temporal consistency filter for head boxes.

Rejects detections that jump implausibly far (in metric terms) from the last
accepted head box, which suppresses YOLO flicker. The body is never filtered
here; it is always derived from the most recently accepted head.
"""

from __future__ import annotations

from typing import Optional

from vision_perception.geometry import Box, pixels_to_meters_vertical


class TemporalFilter:
    """Accept/reject head boxes based on frame-to-frame metric motion.

    - A new head needs several spatially consistent observations before it is
      published (single-frame false positives never establish a track).
    - Subsequent heads must move <= ``max_jump_m`` in bbox height and in both
      center axes, otherwise they are rejected as noise.
    - After ``reset_after_n_missed`` consecutive rejections/misses the track is
      forgotten and the next detection becomes a fresh baseline.
    """

    def __init__(
        self,
        max_jump_m: float,
        reset_after_n_missed: int,
        distance_to_target_m: float,
        vertical_fov_deg: float,
        acquisition_frames: int = 3,
        acquisition_max_missed_frames: int = 1,
    ):
        self.max_jump_m = float(max_jump_m)
        self.reset_after_n_missed = int(reset_after_n_missed)
        self.distance_to_target_m = float(distance_to_target_m)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.acquisition_frames = max(1, int(acquisition_frames))
        self.acquisition_max_missed_frames = max(
            0, int(acquisition_max_missed_frames))

        self.last_valid_box: Optional[Box] = None
        self.missed_counter = 0
        self._candidate_box: Optional[Box] = None
        self._candidate_hits = 0
        self._candidate_misses = 0

    def set_distance_to_target_m(self, distance_m: float) -> None:
        """Update standoff depth used for pixel->meter gates."""
        self.distance_to_target_m = float(distance_m)

    def _px_to_m(self, delta_px: float, img_height_px: int) -> float:
        return pixels_to_meters_vertical(
            delta_px, img_height_px, self.distance_to_target_m, self.vertical_fov_deg
        )

    def _register_miss(self) -> None:
        self.missed_counter += 1
        if self.missed_counter >= self.reset_after_n_missed:
            self.last_valid_box = None
            self.missed_counter = 0
            self._reset_acquisition()

    def _reset_acquisition(self) -> None:
        self._candidate_box = None
        self._candidate_hits = 0
        self._candidate_misses = 0

    def _compatible(self, previous: Box, candidate: Box, img_height_px: int) -> bool:
        prev_h = max(1, previous["y2"] - previous["y1"])
        curr_h = max(1, candidate["y2"] - candidate["y1"])
        delta_h_m = self._px_to_m(abs(curr_h - prev_h), img_height_px)

        prev_cy = 0.5 * (previous["y1"] + previous["y2"])
        curr_cy = 0.5 * (candidate["y1"] + candidate["y2"])
        delta_cy_m = self._px_to_m(abs(curr_cy - prev_cy), img_height_px)

        prev_cx = 0.5 * (previous["x1"] + previous["x2"])
        curr_cx = 0.5 * (candidate["x1"] + candidate["x2"])
        delta_cx_m = self._px_to_m(abs(curr_cx - prev_cx), img_height_px)
        return (
            delta_h_m <= self.max_jump_m
            and delta_cy_m <= self.max_jump_m
            and delta_cx_m <= self.max_jump_m
        )

    def _acquire(self, candidate: Optional[Box], img_height_px: int) -> Optional[Box]:
        if candidate is None:
            if self._candidate_box is not None:
                self._candidate_misses += 1
                if self._candidate_misses > self.acquisition_max_missed_frames:
                    self._reset_acquisition()
            return None
        if self._candidate_box is None or not self._compatible(
                self._candidate_box, candidate, img_height_px):
            self._candidate_box = candidate
            self._candidate_hits = 1
            self._candidate_misses = 0
            return None
        self._candidate_box = candidate
        self._candidate_hits += 1
        self._candidate_misses = 0
        if self._candidate_hits < self.acquisition_frames:
            return None
        self.last_valid_box = candidate
        self.missed_counter = 0
        self._reset_acquisition()
        return candidate

    def step(self, candidate: Optional[Box], img_height_px: int) -> Optional[Box]:
        """Return the head box to publish this frame, or None to suppress."""
        # No active track yet.
        if self.last_valid_box is None:
            if self.acquisition_frames <= 1 and candidate is not None:
                self.last_valid_box = candidate
                self.missed_counter = 0
                return candidate
            return self._acquire(candidate, img_height_px)

        # Active track but nothing detected this frame.
        if candidate is None:
            self._register_miss()
            return None

        if self._compatible(self.last_valid_box, candidate, img_height_px):
            self.last_valid_box = candidate
            self.missed_counter = 0
            return candidate

        self._register_miss()
        return None
