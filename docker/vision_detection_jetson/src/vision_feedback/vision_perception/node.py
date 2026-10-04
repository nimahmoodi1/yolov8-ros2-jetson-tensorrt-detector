"""ROS 2 node: camera image -> YOLO -> filtered/stabilized BoundingBoxes.

Pipeline per frame
------------------
1. Convert the image (zero-copy view over the DDS buffer where possible).
2. Optional inference-rate cap, applied before any per-frame work.
3. Duplicate/frozen-frame guard (re-publish last good box, then suppress).
4. Detector backend -> ``(N, 6)`` array -> select+merge head, select body.
5. Temporal filter on the head (rejects implausible jumps).
6. Publish the post-filter message on the *raw* topic (debug).
7. Stabilize (smooth + dropout-hold + sanity gate) and publish on the *stable*
   topic that the mission consumes.
8. Optional downscaled overlay + display.

Threading
---------
Camera callbacks only hand off the newest frame through a capacity-one slot; a
single worker thread owns inference, filtering, publication and rendering. That
guarantees chronological output with no inference backlog.

Because every callback is short and non-blocking, the node spins on a
``SingleThreadedExecutor``. The previous ``MultiThreadedExecutor(num_threads=4)``
added three permanently-resident threads that had no work to do but still
contended for the same eight Orin cores as the flight stack.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from ros2_interface.msg import BoundingBoxes

from vision_perception import config, detection, geometry, image_convert
from vision_perception.backends import ClassRouter, build_detector, resolve_device
from vision_perception.diagnostics import VisionDiagnostics
from vision_perception.latest_frame import FramePacket, LatestFrameSlot
from vision_perception.letterbox import auto_input_shape
from vision_perception.message_builder import build_bounding_boxes_msg
from vision_perception.rate_limit import PhaseTolerantRateLimiter
from vision_perception.runtime import limit_cpu_threads
from vision_perception.stabilizer import BoundingBoxStabilizer
from vision_perception.temporal_filter import TemporalFilter
from vision_perception.visualization import DisplayWindow

# Stride used by the frozen-frame guard. Comparing every 8th pixel in both axes
# looks at ~1/64 of the frame (43 kB at 1280x720 instead of 690 kB) which is
# still far more entropy than any real camera can repeat by chance.
_DUP_STRIDE = 8


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.params = config.load_params(self)
        self.log = self.get_logger()

        limit_cpu_threads(self.params.cpu_threads, self.log)

        # cv_bridge is only a fallback for encodings image_convert cannot map.
        try:
            from cv_bridge import CvBridge

            self.bridge = CvBridge()
        except Exception:  # pragma: no cover - cv_bridge is a hard ROS dep
            self.bridge = None

        # --- model / device / backend ---
        import torch  # imported here so the module import stays light

        device = resolve_device(self.params.device, torch.cuda.is_available(), self.log)
        net_h, net_w = self.params.net_shape()
        conf_floor = self.params.detector_conf_floor
        self.log.info(
            f'Camera geometry: expecting {self.params.expected_frame_width}x'
            f'{self.params.expected_frame_height} -> network input {net_w}x{net_h} '
            f"(inference_input_shape='{self.params.inference_input_shape}', "
            f'imgsz={self.params.inference_imgsz})'
        )
        self.log.info(
            f'Detector request: backend={self.params.backend} device={device} '
            f'input={net_w}x{net_h} precision={self.params.inference_precision} '
            f'fp16_io={self.params.trt_fp16_io} '
            f'conf_floor={conf_floor:.2f} iou={self.params.inference_iou_threshold:.2f}'
        )
        self.detector, backend_warnings = build_detector(
            model_path=self.params.model_path,
            backend=self.params.backend,
            device=device,
            net_shape=(net_h, net_w),
            precision=self.params.inference_precision,
            conf_threshold=conf_floor,
            iou_threshold=self.params.inference_iou_threshold,
            max_detections=self.params.inference_max_detections,
            engine_cache_dir=self.params.engine_cache_dir,
            auto_build_engine=self.params.auto_build_engine,
            trt_workspace_mib=self.params.trt_workspace_mib,
            onnx_opset=self.params.onnx_opset,
            trt_fp16_io=self.params.trt_fp16_io,
            warmup_frames=self.params.warmup_frames,
            # Warm up at the geometry the camera will actually send, so the
            # first live frame reuses these buffers instead of reallocating.
            warmup_shape=(self.params.expected_frame_height,
                          self.params.expected_frame_width),
            logger=self.log,
        )
        self._backend_warnings = backend_warnings
        self.classes = ClassRouter(self.detector.names)
        if not self.classes.is_valid():
            self.log.error(
                f'Model classes {self.detector.names} contain neither "head" nor "body"; '
                'no detection can ever be published. Check model_path / mission.yaml.'
            )
        else:
            self.log.info(f'Class routing: {self.classes.describe()}')

        # --- processing components ---
        filter_kwargs = dict(
            max_jump_m=self.params.max_jump_m,
            reset_after_n_missed=self.params.reset_after_n_missed,
            distance_to_target_m=self.params.distance_to_target_m,
            vertical_fov_deg=self.params.vertical_fov_deg,
            acquisition_frames=self.params.acquisition_frames,
            acquisition_max_missed_frames=self.params.acquisition_max_missed_frames,
        )
        self.filter = TemporalFilter(**filter_kwargs)
        self.body_filter = TemporalFilter(**filter_kwargs)
        self.stabilizer = BoundingBoxStabilizer(
            hold_time_s=self.params.stabilizer_hold_time_s,
            alpha=self.params.stabilizer_alpha,
            img_w=self.params.expected_frame_width,
            img_h=self.params.expected_frame_height,
        )
        self._observed_frame_size: Optional[Tuple[int, int]] = None
        self._log_geometry_profile()
        self._live_distance: Optional[float] = None
        self._live_distance_time: float = 0.0

        # --- publishers ---
        self.stable_pub = self.create_publisher(BoundingBoxes, self.params.bbox_topic, 10)
        self.raw_pub = self.create_publisher(BoundingBoxes, self.params.raw_bbox_topic, 10)

        # --- display sink ---
        self.display: Optional[DisplayWindow] = None
        if self.params.show_display:
            self._check_display_env()
            self.display = DisplayWindow(
                "Vision Feedback",
                self.log,
                max_fps=self.params.display_max_fps,
                width=self.params.display_width,
            )
            self.display.start()

        # --- duplicate-frame state ---
        self._prev_signature: Optional[np.ndarray] = None
        self._pending_signature: Optional[np.ndarray] = None
        self._dup_first_seen: Optional[float] = None

        # --- inference rate cap ---
        # See rate_limit.py: a naive deadline cap aliased a 29.8 Hz camera down
        # to 14.5 Hz instead of the requested 20 Hz.
        self._rate_limit = PhaseTolerantRateLimiter(self.params.max_inference_fps)

        # --- timing stats ---
        self._stats_lock = threading.Lock()
        self._infer_ms_ema = 0.0
        self._total_ms_ema = 0.0
        self._processed_since_log = 0
        self._last_stats_log = time.monotonic()

        self._frame_slot = LatestFrameSlot()
        self._frame_sequence = 0
        self._worker_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name='vision-inference', daemon=True)
        self._inference_thread.start()

        # --- diagnostics ---
        self.diagnostics: Optional[VisionDiagnostics] = None
        if self.params.publish_diagnostics:
            self.diagnostics = VisionDiagnostics(self)
            self.diagnostics.describe_backend(
                self.detector.info,
                degraded=self.detector.info.kind != 'tensorrt',
                note='; '.join(backend_warnings) or 'TensorRT not in use',
            )
            self.create_timer(
                1.0, self._publish_diagnostics,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        # Depth one plus the latest-only handoff means overload drops stale
        # camera frames instead of replaying them after inference catches up.
        camera_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._image_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image, self.params.image_topic, self._on_image,
            camera_qos, callback_group=self._image_callback_group,
        )
        if self.params.use_live_distance:
            self.create_subscription(
                Float32, self.params.live_distance_topic, self._on_live_distance,
                qos_profile_sensor_data, callback_group=ReentrantCallbackGroup(),
            )
            self.log.info(
                f"Live standoff from '{self.params.live_distance_topic}' "
                f"(fallback radius={self.params.distance_to_target_m:.1f} m)"
            )
        self.log.info(
            f"Subscribed to '{self.params.image_topic}'; publishing stable boxes on "
            f"'{self.params.bbox_topic}' and raw boxes on '{self.params.raw_bbox_topic}'"
        )
        self._no_frame_warned = False
        self.create_timer(
            5.0, self._check_camera_feed,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        if self.params.stats_log_interval_s > 0:
            self.create_timer(
                float(self.params.stats_log_interval_s), self._log_stats,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

    # ------------------------------------------------------------------ #
    # startup helpers
    # ------------------------------------------------------------------ #
    def _check_display_env(self) -> None:
        display = os.environ.get('DISPLAY', '')
        if not display:
            self.log.warning(
                'show_display is true but DISPLAY is unset. Set DISPLAY in .env '
                '(this Jetson runs its desktop on :10 via xrdp) and mount /tmp/.X11-unix.'
            )
        else:
            self.log.info(
                f'Display enabled on DISPLAY={display} '
                f'(max {self.params.display_max_fps:.0f} fps, {self.params.display_width}px wide)'
            )

    def _log_geometry_profile(self) -> None:
        dist = self.params.distance_to_target_m
        vfov = self.params.vertical_fov_deg
        reference_height = self.params.expected_frame_height
        mm_per_px = geometry.pixels_to_meters_vertical(
            1.0, reference_height, dist, vfov) * 1000.0
        merge_px_equiv = detection.meters_to_pixels_vertical(
            self.params.merge_center_delta_m, reference_height, dist, vfov)
        net_h, net_w = self.params.net_shape()
        self.log.info(
            'Vision tuning: vfov=%.1f deg standoff=%.1f m (live lidar when fresh) '
            'merge=%.2f m (~%.0f px) max_jump=%.2f m '
            'conf(head/body)=%.2f/%.2f net=%dx%d '
            'stabilizer alpha=%.2f hold=%.1fs | ~%.1f mm/px vertical @%dpx height'
            % (
                vfov, dist,
                self.params.merge_center_delta_m, merge_px_equiv,
                self.params.max_jump_m,
                self.params.head_confidence_threshold,
                self.params.body_confidence_threshold,
                net_w, net_h,
                self.params.stabilizer_alpha,
                self.params.stabilizer_hold_time_s,
                mm_per_px, reference_height,
            )
        )

    # ------------------------------------------------------------------ #
    # live standoff (LIDAR distance topic from mission.yaml)
    # ------------------------------------------------------------------ #
    def _on_live_distance(self, msg: Float32) -> None:
        value = float(msg.data)
        if 0.5 < value < 30.0:
            self._live_distance = value
            self._live_distance_time = time.monotonic()

    def _effective_distance(self) -> float:
        if self.params.use_live_distance and self._live_distance is not None:
            age = time.monotonic() - self._live_distance_time
            if age <= float(self.params.live_distance_stale_s):
                return self._live_distance
        return float(self.params.distance_to_target_m)

    # ------------------------------------------------------------------ #
    # image handling
    # ------------------------------------------------------------------ #
    def _check_camera_feed(self) -> None:
        if self.diagnostics is None or self.diagnostics.frames_received > 0:
            return
        if self._no_frame_warned:
            return
        self._no_frame_warned = True
        self.log.warning(
            f"No camera frames on '{self.params.image_topic}' yet. "
            f"Check the publisher with: ros2 topic hz {self.params.image_topic}"
        )

    def _is_duplicate(self, img: np.ndarray) -> bool:
        """Frozen-feed guard on a strided subsample of the frame."""
        signature = np.ascontiguousarray(img[::_DUP_STRIDE, ::_DUP_STRIDE])
        prev = self._prev_signature
        if prev is None or prev.shape != signature.shape:
            self._pending_signature = signature
            return False
        self._pending_signature = signature
        return bool(np.array_equal(prev, signature))

    def _on_image(self, msg: Image) -> None:
        """Convert and enqueue only; inference is owned by one worker thread."""
        now = time.monotonic()
        if self.diagnostics is not None:
            self.diagnostics.frames_received += 1
            self.diagnostics.last_frame_time = now
        self._no_frame_warned = False

        # Rate cap before any per-frame work, so a 30 Hz camera cannot force
        # 30 Hz of GPU work when the mission consumes boxes at a lower rate.
        if not self._rate_limit.step(now):
            if self.diagnostics is not None:
                self.diagnostics.frames_throttled += 1
            return

        img = image_convert.to_bgr(msg, self.bridge, self.log)
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return

        self._frame_sequence += 1
        packet = FramePacket(
            sequence=self._frame_sequence,
            image=img,
            received_monotonic=now,
        )
        replaced = self._frame_slot.put(packet)
        if replaced and self.diagnostics is not None:
            self.diagnostics.frames_dropped_before_inference += 1

    def _inference_loop(self) -> None:
        """Process frames serially and publish results in acquisition order."""
        while self._worker_running:
            packet = self._frame_slot.take(timeout_s=0.1)
            if packet is None:
                continue
            try:
                self._process_image(packet)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self.log.error(
                    f'Vision frame {packet.sequence} processing failed: {exc}')

    def _process_image(self, packet: FramePacket) -> None:
        img = packet.image
        h, w = img.shape[:2]

        if self._is_duplicate(img):
            self._handle_duplicate(img, w, h)
            self._record_processed_frame(packet)
            return
        self._prev_signature = self._pending_signature
        self._dup_first_seen = None

        standoff_m = self._effective_distance()
        self.filter.set_distance_to_target_m(standoff_m)
        self.body_filter.set_distance_to_target_m(standoff_m)

        detections = self.detector.infer(img)

        head = detection.select_and_merge_head_box(
            detections, img.shape, self.params.head_confidence_threshold,
            self.params.merge_center_delta_m, standoff_m,
            self.params.vertical_fov_deg,
            self.params.head_min_aspect_ratio,
            self.params.head_max_aspect_ratio,
            self.params.head_min_area_fraction,
            self.params.head_max_area_fraction,
            class_ids=self.classes.head_ids,
        )
        body = detection.select_body_box(
            detections,
            img.shape,
            self.params.body_confidence_threshold,
            min_aspect_ratio=self.params.body_min_aspect_ratio,
            max_aspect_ratio=self.params.body_max_aspect_ratio,
            min_area_fraction=self.params.body_min_area_fraction,
            max_area_fraction=self.params.body_max_area_fraction,
            class_ids=self.classes.body_ids,
        )
        self._publish_raw(build_bounding_boxes_msg(head, body))

        if self.diagnostics is not None and head is not None:
            self.diagnostics.detections += 1

        accepted_head = self.filter.step(head, img_height_px=h)
        if accepted_head is not None:
            head_box = accepted_head
            body_box = geometry.estimate_body_box(accepted_head, w, h)
        else:
            head_box = None
            body_box = self.body_filter.step(body, img_height_px=h)

        self._publish(head_box, body_box, img, raw_head=head, raw_body=body)
        self._record_processed_frame(packet)

    def _publish_raw(self, msg: BoundingBoxes) -> None:
        """Publish debug boxes only when something is listening.

        Serialising a message nobody reads is small but constant per-frame CPU;
        skipping it costs nothing when a debugger *is* attached.
        """
        if self.raw_pub.get_subscription_count() > 0:
            self.raw_pub.publish(msg)

    def _record_processed_frame(self, packet: FramePacket) -> None:
        total_ms = (time.monotonic() - packet.received_monotonic) * 1000.0
        infer_ms = self.detector.last_infer_ms
        weight = 0.1
        with self._stats_lock:
            self._infer_ms_ema = (
                infer_ms if self._infer_ms_ema <= 0.0
                else (1.0 - weight) * self._infer_ms_ema + weight * infer_ms)
            self._total_ms_ema = (
                total_ms if self._total_ms_ema <= 0.0
                else (1.0 - weight) * self._total_ms_ema + weight * total_ms)
            self._processed_since_log += 1

        if self.diagnostics is None:
            return
        self.diagnostics.frames_processed += 1
        self.diagnostics.last_processed_sequence = packet.sequence
        self.diagnostics.processing_latency_ms = total_ms
        self.diagnostics.inference_ms = infer_ms

    def _handle_duplicate(self, img: np.ndarray, w: int, h: int) -> None:
        """Frozen feed: re-publish last good head until timeout, then suppress."""
        now = time.monotonic()
        if self._dup_first_seen is None:
            self._dup_first_seen = now
        within_timeout = (now - self._dup_first_seen) < self.params.duplicate_timeout_s

        if within_timeout and self.filter.last_valid_box is not None:
            head_box = self.filter.last_valid_box
            body_box = geometry.estimate_body_box(head_box, w, h)
        else:
            head_box, body_box = None, None
        self._publish_raw(build_bounding_boxes_msg(None, None))
        self._publish(head_box, body_box, img)

    def _publish(
        self,
        head_box,
        body_box,
        img: np.ndarray,
        *,
        raw_head=None,
        raw_body=None,
    ) -> None:
        h, w = img.shape[:2]
        frame_size = (w, h)
        if frame_size != self._observed_frame_size:
            previous = self._observed_frame_size
            self.stabilizer.set_frame_size(w, h)
            self._observed_frame_size = frame_size
            if previous is None:
                expected = (
                    self.params.expected_frame_width,
                    self.params.expected_frame_height,
                )
                if frame_size == expected:
                    self.log.info(f'Vision frame geometry verified: {w}x{h}')
                else:
                    ideal_h, ideal_w = auto_input_shape(
                        h, w, self.params.inference_imgsz)
                    info = self.detector.info
                    self.log.warning(
                        f'Vision received {w}x{h}, but expected_frame_* says '
                        f'{expected[0]}x{expected[1]}. Detection still works - the '
                        'letterbox adapts to any frame size - but the engine shape '
                        'is no longer ideal for this camera.\n'
                        f'  current engine input : {info.net_w}x{info.net_h}\n'
                        f'  ideal for {w}x{h}    : {ideal_w}x{ideal_h}\n'
                        f'  fix: set expected_frame_width: {w} / expected_frame_height: {h} '
                        'in vision.yaml (or let mission.yaml supply them), then rebuild:\n'
                        '       ros2 run vision_feedback build_engine.py\n'
                        'Stabilizer history reset for the new geometry.'
                    )
            else:
                self.log.warning(
                    f'Vision frame resolution changed {previous[0]}x{previous[1]} '
                    f'-> {w}x{h}; stabilizer history reset'
                )
        filtered_msg = build_bounding_boxes_msg(head_box, body_box)
        stable_msg = self.stabilizer.stabilize(filtered_msg)
        self.stable_pub.publish(stable_msg)

        if self.diagnostics is not None:
            self.diagnostics.frame_width = w
            self.diagnostics.frame_height = h
            self.diagnostics.stabilizer_rejections = self.stabilizer.rejected_measurements
            if stable_msg.has_head or stable_msg.has_body:
                self.diagnostics.stable_detections += 1
            self.diagnostics.published += 1

        if self.display is not None and self.display.due():
            self._render(stable_msg, img, raw_head=raw_head, raw_body=raw_body)

    def _render(
        self,
        msg: BoundingBoxes,
        img: np.ndarray,
        *,
        raw_head=None,
        raw_body=None,
    ) -> None:
        """Hand the frame and boxes to the display thread.

        The resize and the drawing happen *there*, not here. Doing them on the
        inference thread put ~10-15 ms of OpenCV work between one frame's
        publish and the next frame's inference, which is most of the gap
        between the measured `infer` and `end-to-end` times.
        """
        head = {
            'x1': msg.head_x1, 'y1': msg.head_y1,
            'x2': msg.head_x2, 'y2': msg.head_y2,
        } if msg.has_head else None
        body = {
            'x1': msg.body_x1, 'y1': msg.body_y1,
            'x2': msg.body_x2, 'y2': msg.body_y2,
        } if msg.has_body else None
        self.display.submit_scene(
            img, head, body,
            raw_head=raw_head, raw_body=raw_body,
            hud=self._hud_text(),
        )

    def _hud_text(self) -> str:
        info = self.detector.info
        with self._stats_lock:
            infer_ms = self._infer_ms_ema
            total_ms = self._total_ms_ema
        fps = 1000.0 / total_ms if total_ms > 0.05 else 0.0
        return (
            f'{info.kind}/{info.precision} {info.net_w}x{info.net_h} | '
            f'{fps:5.1f} fps | infer {infer_ms:5.1f} ms'
        )

    def _log_stats(self) -> None:
        now = time.monotonic()
        with self._stats_lock:
            processed = self._processed_since_log
            infer_ms = self._infer_ms_ema
            total_ms = self._total_ms_ema
            self._processed_since_log = 0
        elapsed = max(1e-6, now - self._last_stats_log)
        self._last_stats_log = now
        if processed == 0:
            return
        throttled = self.diagnostics.frames_throttled if self.diagnostics else 0
        dropped = self.diagnostics.frames_dropped_before_inference if self.diagnostics else 0
        info = self.detector.info
        camera_fps = self._rate_limit.measured_fps
        self.log.info(
            f'{info.kind}/{info.precision} {info.net_w}x{info.net_h}: '
            f'{processed / elapsed:.1f} fps processed of {camera_fps:.1f} fps camera | '
            f'infer {infer_ms:.1f} ms | end-to-end {total_ms:.1f} ms | '
            f'dropped {dropped} | throttled {throttled}'
        )

    def _publish_diagnostics(self) -> None:
        if self.diagnostics is not None:
            self.diagnostics.publish()

    def destroy_node(self) -> bool:
        self._worker_running = False
        self._frame_slot.close()
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
        if self.display is not None:
            self.display.stop()
        try:
            self.detector.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
