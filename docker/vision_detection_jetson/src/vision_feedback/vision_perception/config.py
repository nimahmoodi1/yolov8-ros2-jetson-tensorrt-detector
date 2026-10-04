"""Configuration handler for the vision node.

``config/vision.yaml`` holds the vision node tunables. This module loads the
YAML, declares the values as ROS 2 parameters (so launch/CLI can still override
them), and returns an immutable :class:`VisionParams` snapshot. No defaults are
duplicated in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Tuple

import yaml

from vision_perception.backends import resolve_device  # noqa: F401  (re-export)
from vision_perception.letterbox import parse_input_shape
from vision_perception.mission_config import load_mission_shared

_MODEL_NAME = "best_fake_tower_behshahr.pt"
# IO params whose empty-string launch override means "keep the YAML value".
_STRING_OVERRIDE_KEYS = (
    "image_topic",
    "bbox_topic",
    "raw_bbox_topic",
    "device",
    "model_path",
    "backend",
    "inference_precision",
    "inference_input_shape",
    "engine_cache_dir",
)
# Map a vision string param to the matching mission.yaml field.
_MISSION_TOPIC_KEYS = {"image_topic": "camera_topic", "bbox_topic": "bbox_topic"}

_VALID_BACKENDS = ("auto", "tensorrt", "trt", "torch", "pytorch")
_VALID_PRECISIONS = ("fp16", "fp32")


def _share_dir() -> str:
    from ament_index_python.packages import get_package_share_directory
    return get_package_share_directory("vision_feedback")


def config_file_path() -> str:
    return os.path.join(_share_dir(), "config", "vision.yaml")


def default_model_path() -> str:
    """Path to the packaged YOLO weights (share dir, with a source-tree fallback)."""
    candidate = os.path.join(_share_dir(), "models", _MODEL_NAME)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(os.path.dirname(__file__), "models", _MODEL_NAME)


def packaged_model_path(model_name: str) -> str:
    """Resolve a profile-selected model filename from the package share directory."""
    return os.path.join(_share_dir(), "models", os.path.basename(model_name))


def _load_yaml_defaults() -> dict:
    with open(config_file_path()) as handle:
        data = yaml.safe_load(handle) or {}
    return data["vision_node"]["ros__parameters"]


def yaml_defaults() -> dict:
    """Raw ``vision.yaml`` values, for tools that run without a ROS node."""
    return dict(_load_yaml_defaults())


@dataclass(frozen=True)
class VisionParams:
    """Immutable snapshot of node parameters (schema mirrors vision.yaml)."""

    use_mission_config: bool
    use_live_distance: bool
    live_distance_topic: str
    live_distance_stale_s: float
    image_topic: str
    bbox_topic: str
    raw_bbox_topic: str
    model_path: str
    device: str
    # --- inference backend ---
    backend: str
    inference_precision: str
    inference_input_shape: str
    engine_cache_dir: str
    auto_build_engine: bool
    trt_workspace_mib: int
    onnx_opset: int
    trt_fp16_io: bool
    warmup_frames: int
    # --- detection ---
    inference_imgsz: int
    inference_iou_threshold: float
    inference_max_detections: int
    max_inference_fps: float
    head_confidence_threshold: float
    body_confidence_threshold: float
    head_min_aspect_ratio: float
    head_max_aspect_ratio: float
    head_min_area_fraction: float
    head_max_area_fraction: float
    body_min_aspect_ratio: float
    body_max_aspect_ratio: float
    body_min_area_fraction: float
    body_max_area_fraction: float
    # --- geometry ---
    expected_frame_width: int
    expected_frame_height: int
    vertical_fov_deg: float
    distance_to_target_m: float
    merge_center_delta_m: float
    max_jump_m: float
    acquisition_frames: int
    acquisition_max_missed_frames: int
    reset_after_n_missed: int
    stabilizer_hold_time_s: float
    stabilizer_alpha: float
    duplicate_timeout_s: float
    # --- runtime / IO ---
    cpu_threads: int
    show_display: bool
    display_max_fps: float
    display_width: int
    publish_diagnostics: bool
    stats_log_interval_s: float

    @property
    def detector_conf_floor(self) -> float:
        """Lowest confidence that can ever reach an output.

        Selection applies ``head_confidence_threshold`` / ``body_confidence_threshold``
        afterwards, so decoding and running NMS on anything below the smaller of
        the two is guaranteed-wasted GPU work.
        """
        return float(min(self.head_confidence_threshold, self.body_confidence_threshold))

    def net_shape(self) -> Tuple[int, int]:
        """Resolved (height, width) of the network input."""
        return parse_input_shape(
            self.inference_input_shape,
            self.expected_frame_height,
            self.expected_frame_width,
            self.inference_imgsz,
        )


def load_params(node) -> VisionParams:
    """Declare every YAML parameter on ``node`` and return a VisionParams.

    Resolution order for each value:
      1. explicit launch/CLI override (non-empty)         -- highest
      2. mission.yaml (when use_mission_config)            -- shared quantities
      3. vision.yaml literal                               -- fallback
    """
    defaults = _load_yaml_defaults()
    logger = node.get_logger()

    values = {}
    for name, default in defaults.items():
        node.declare_parameter(name, default)
        values[name] = node.get_parameter(name).value

    # Pull shared quantities from mission.yaml (no-op if disabled/missing).
    use_mission = bool(values.get("use_mission_config", False))
    mission = load_mission_shared(logger=logger) if use_mission else None

    # IO overrides: explicit launch arg wins; else mission.yaml topic; else YAML.
    for key in _STRING_OVERRIDE_KEYS:
        raw = values.get(key)
        if isinstance(raw, str) and not raw.strip():
            mission_topic = None
            if mission is not None and key in _MISSION_TOPIC_KEYS:
                mission_topic = getattr(mission, _MISSION_TOPIC_KEYS[key])
            values[key] = mission_topic or defaults.get(key, "")

    # Shared scalars have no launch arg: mission.yaml overrides the YAML fallback.
    if mission is not None:
        if mission.vertical_fov_deg is not None:
            values["vertical_fov_deg"] = mission.vertical_fov_deg
        if mission.camera_width is not None:
            values["expected_frame_width"] = mission.camera_width
        if mission.camera_height is not None:
            values["expected_frame_height"] = mission.camera_height
        if mission.distance_to_target_m is not None:
            values["distance_to_target_m"] = mission.distance_to_target_m
        if mission.lidar_distance_topic:
            values["live_distance_topic"] = mission.lidar_distance_topic
        if mission.vision_model:
            values["model_path"] = packaged_model_path(mission.vision_model)

    # Resolve the packaged model when none is specified.
    if not str(values.get("model_path", "")).strip():
        values["model_path"] = default_model_path()

    # Environment escape hatches for container operation (compose/.env).
    env_cache = os.environ.get("ENGINE_CACHE_DIR", "").strip()
    if env_cache:
        values["engine_cache_dir"] = env_cache

    params = VisionParams(**{f.name: values[f.name] for f in fields(VisionParams)})
    _validate_params(params)
    return params


def _validate_params(params: VisionParams) -> None:
    if int(params.inference_imgsz) < 320 or int(params.inference_imgsz) % 32 != 0:
        raise ValueError('inference_imgsz must be >= 320 and divisible by 32')
    if not (0.0 < float(params.inference_iou_threshold) <= 1.0):
        raise ValueError('inference_iou_threshold must be in (0, 1]')
    if int(params.inference_max_detections) < 1:
        raise ValueError('inference_max_detections must be >= 1')
    if str(params.backend).strip().lower() not in _VALID_BACKENDS:
        raise ValueError(f'backend must be one of {_VALID_BACKENDS}')
    if str(params.inference_precision).strip().lower() not in _VALID_PRECISIONS:
        raise ValueError(f'inference_precision must be one of {_VALID_PRECISIONS}')
    if int(params.trt_workspace_mib) < 64:
        raise ValueError('trt_workspace_mib must be >= 64')
    if int(params.onnx_opset) < 11:
        raise ValueError('onnx_opset must be >= 11')
    if bool(params.trt_fp16_io) and str(params.inference_precision).lower() != 'fp16':
        raise ValueError('trt_fp16_io requires inference_precision: fp16')
    if float(params.max_inference_fps) < 0.0:
        raise ValueError('max_inference_fps must be >= 0 (0 disables the cap)')
    if float(params.display_max_fps) < 0.0:
        raise ValueError('display_max_fps must be >= 0 (0 disables the cap)')
    if int(params.display_width) < 0:
        raise ValueError('display_width must be >= 0 (0 keeps the source width)')
    for name in ('head_confidence_threshold', 'body_confidence_threshold'):
        if not (0.0 <= float(getattr(params, name)) <= 1.0):
            raise ValueError(f'{name} must be in [0, 1]')
    for prefix in ('head', 'body'):
        min_aspect = float(getattr(params, f'{prefix}_min_aspect_ratio'))
        max_aspect = float(getattr(params, f'{prefix}_max_aspect_ratio'))
        min_area = float(getattr(params, f'{prefix}_min_area_fraction'))
        max_area = float(getattr(params, f'{prefix}_max_area_fraction'))
        if not (0.0 < min_aspect < max_aspect):
            raise ValueError(f'{prefix} aspect-ratio limits are invalid')
        if not (0.0 < min_area < max_area <= 1.0):
            raise ValueError(f'{prefix} area-fraction limits are invalid')
    if int(params.acquisition_frames) < 1:
        raise ValueError('acquisition_frames must be >= 1')
    if int(params.acquisition_max_missed_frames) < 0:
        raise ValueError('acquisition_max_missed_frames must be >= 0')
    # Raises on a malformed inference_input_shape.
    params.net_shape()
