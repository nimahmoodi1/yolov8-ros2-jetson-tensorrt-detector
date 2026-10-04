"""vision_perception: GPU-accelerated YOLO tower-detection pipeline for ROS 2.

Public layout
-------------
- :mod:`vision_perception.config`          runtime parameters (ROS params + dataclass)
- :mod:`vision_perception.backends`        detector backend selection + class routing
- :mod:`vision_perception.engine_builder`  ``.pt`` -> ONNX -> cached TensorRT engine
- :mod:`vision_perception.letterbox`       static-shape letterbox geometry
- :mod:`vision_perception.image_convert`   zero-copy ``sensor_msgs/Image`` -> BGR
- :mod:`vision_perception.geometry`        pixel<->meter math and body-box estimation
- :mod:`vision_perception.detection`       vectorised head/body box selection
- :mod:`vision_perception.temporal_filter` rejects implausible head jumps over time
- :mod:`vision_perception.stabilizer`      smooths and gates the published boxes
- :mod:`vision_perception.message_builder` builds ``vision_feedback/BoundingBoxes``
- :mod:`vision_perception.visualization`   overlay drawing + threaded OpenCV window
- :mod:`vision_perception.diagnostics`     REP-107 style ``/diagnostics`` publisher
- :mod:`vision_perception.runtime`         process-wide CPU thread limits
- :mod:`vision_perception.node`            the rclpy node wiring it all together
"""

__all__ = [
    "backends",
    "config",
    "detection",
    "diagnostics",
    "engine_builder",
    "geometry",
    "image_convert",
    "latest_frame",
    "letterbox",
    "message_builder",
    "node",
    "runtime",
    "stabilizer",
    "temporal_filter",
    "visualization",
]
