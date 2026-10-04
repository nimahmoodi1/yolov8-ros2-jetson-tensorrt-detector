"""Minimal REP-107 style diagnostics for the vision node.

Publishes a ``diagnostic_msgs/DiagnosticArray`` on ``/diagnostics`` summarising
camera input, model state, and output publishing health. Self-contained (no
external diagnostic_updater dependency).

The model status now reports which backend actually loaded. A silent downgrade
from TensorRT to PyTorch (or worse, to CPU) roughly quadruples latency, so it
is surfaced as a WARN rather than being buried in the startup log.
"""

from __future__ import annotations

import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class VisionDiagnostics:
    """Collects counters from the node and renders a DiagnosticArray."""

    def __init__(self, node, hardware_id: str = "vision_feedback"):
        self._node = node
        self._hardware_id = hardware_id
        self._pub = node.create_publisher(DiagnosticArray, "/diagnostics", 1)

        self.last_frame_time: float | None = None
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped_before_inference = 0
        self.frames_throttled = 0
        self.last_processed_sequence = 0
        self.processing_latency_ms = 0.0
        self.inference_ms = 0.0
        self.detections = 0
        self.stable_detections = 0
        self.stabilizer_rejections = 0
        self.published = 0
        self.frame_width = 0
        self.frame_height = 0
        self.model_loaded = False
        self.device = "unknown"
        self.backend = "unknown"
        self.precision = "unknown"
        self.net_shape = "unknown"
        self.backend_degraded = False
        self.backend_note = ""
        self.camera_timeout_s = 3.0

    def describe_backend(self, info, degraded: bool = False, note: str = "") -> None:
        """Record the loaded backend so it shows up in /diagnostics."""
        self.model_loaded = True
        self.backend = info.kind
        self.device = info.device
        self.precision = info.precision
        self.net_shape = f"{info.net_w}x{info.net_h}"
        self.backend_degraded = bool(degraded)
        self.backend_note = str(note)

    def _status(self, name: str, level, message: str, values: dict) -> DiagnosticStatus:
        st = DiagnosticStatus()
        st.name = name
        st.hardware_id = self._hardware_id
        st.level = level
        st.message = message
        st.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
        return st

    def publish(self) -> None:
        now = time.monotonic()
        arr = DiagnosticArray()
        arr.header.stamp = self._node.get_clock().now().to_msg()

        # Camera input.
        if self.last_frame_time is None:
            cam_level, cam_msg = DiagnosticStatus.WARN, "No frames received yet"
            age = -1.0
        else:
            age = now - self.last_frame_time
            if age <= self.camera_timeout_s:
                cam_level, cam_msg = DiagnosticStatus.OK, "Receiving frames"
            else:
                cam_level, cam_msg = DiagnosticStatus.ERROR, "Camera feed stale"
        arr.status.append(self._status(
            "vision: camera input", cam_level, cam_msg,
            {"frames_received": self.frames_received,
             "seconds_since_last_frame": f"{age:.2f}"},
        ))

        # Model / backend.
        if not self.model_loaded:
            model_level, model_msg = DiagnosticStatus.ERROR, "Not loaded"
        elif self.backend_degraded:
            model_level = DiagnosticStatus.WARN
            model_msg = f"Loaded on the fallback backend: {self.backend_note}"
        else:
            model_level, model_msg = DiagnosticStatus.OK, f"Loaded ({self.backend})"
        arr.status.append(self._status(
            "vision: model", model_level, model_msg,
            {
                "backend": self.backend,
                "device": self.device,
                "precision": self.precision,
                "network_input": self.net_shape,
                "inference_ms": f"{self.inference_ms:.1f}",
            },
        ))

        # Output.
        arr.status.append(self._status(
            "vision: output", DiagnosticStatus.OK, "Publishing bounding boxes",
            {
                "frames_processed": self.frames_processed,
                "frames_dropped_before_inference": self.frames_dropped_before_inference,
                "frames_throttled": self.frames_throttled,
                "last_processed_sequence": self.last_processed_sequence,
                "processing_latency_ms": f"{self.processing_latency_ms:.1f}",
                "raw_head_detections": self.detections,
                "stable_detections": self.stable_detections,
                "stabilizer_rejections": self.stabilizer_rejections,
                "published": self.published,
                "frame_size": f"{self.frame_width}x{self.frame_height}",
            },
        ))

        self._pub.publish(arr)
