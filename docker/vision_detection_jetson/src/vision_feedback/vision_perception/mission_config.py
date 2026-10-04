"""Read shared values from tower_inspection mission.yaml.

Several packages need the same camera FOV, inspection standoff distance, and
topic names. This helper reads those values from mission.yaml instead of
duplicating them in ``vision.yaml``.

It stays dependency-free: it does not import ``tower_inspection``. If mission.yaml
is not available, getters return ``None`` and the caller uses the local
``vision.yaml`` value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML ships with ROS images
    yaml = None

# nav_sim ros_gz_bridge camera topics (mirror tower_inspection bringup defaults).
_FRONT_CAMERA_TOPIC = "/camera/fixed/image"
_GIMBAL_CAMERA_TOPIC = "/camera/gimbal/image"


@dataclass(frozen=True)
class MissionShared:
    """Subset of mission.yaml that the vision node needs to stay consistent.

    Any field may be ``None`` when mission.yaml does not define it; the caller then
    keeps its own vision.yaml fallback for that field.
    """

    vertical_fov_deg: Optional[float] = None
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    distance_to_target_m: Optional[float] = None   # := profiles.<mode>.inspection.radius
    lidar_distance_topic: Optional[str] = None
    camera_topic: Optional[str] = None
    bbox_topic: Optional[str] = None
    show_display: Optional[bool] = None
    vision_model: Optional[str] = None


def locate_mission_config() -> Optional[str]:
    """Resolve the path to mission.yaml, or None if it cannot be found.

    Order (matches tower_inspection/mission/schema.py):
      1. $MISSION_CONFIG_PATH
      2. $TOWER_INSPECTION_CONFIG
      3. installed tower_inspection share dir
    """
    for var in ("MISSION_CONFIG_PATH", "TOWER_INSPECTION_CONFIG"):
        path = os.environ.get(var)
        if path and os.path.exists(path):
            return path
    try:
        from ament_index_python.packages import get_package_share_directory
        candidate = os.path.join(
            get_package_share_directory("tower_inspection"), "config", "mission.yaml"
        )
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return None


def _coerce_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _resolved_camera_topic(
    bringup: dict,
    mission_camera: dict,
    interfaces: dict,
) -> Optional[str]:
    explicit = str(bringup.get("camera_topic") or "").strip()
    if explicit:
        return explicit
    use_gimbal = _coerce_bool(mission_camera.get("use_gimbal"))
    if use_gimbal:
        return str(
            interfaces.get("gimbal_camera_topic") or _GIMBAL_CAMERA_TOPIC
        ).strip()
    return str(
        interfaces.get("fixed_camera_topic") or _FRONT_CAMERA_TOPIC
    ).strip()


def load_mission_shared(path: Optional[str] = None, logger=None) -> MissionShared:
    """Read the shared subset from mission.yaml for the ACTIVE runtime profile.

    Returns an all-None :class:`MissionShared` (never raises) when the file is
    missing or unparseable, so the vision node degrades gracefully to its own
    config.
    """
    resolved = path or locate_mission_config()
    if not resolved or yaml is None:
        if logger is not None:
            logger.warn(
                "mission.yaml not found; vision will use its local vision.yaml values."
            )
        return MissionShared()

    try:
        with open(resolved) as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive
        if logger is not None:
            logger.warn(f"Failed to read mission.yaml ({resolved}): {exc}")
        return MissionShared()

    mode = str(((data.get("runtime") or {}).get("mode")) or "real")
    profile = (data.get("profiles") or {}).get(mode) or {}
    camera = profile.get("camera") or {}
    inspection = profile.get("inspection") or {}
    interfaces = profile.get("interfaces") or {}
    bringup = data.get("bringup") or {}
    mission_camera = data.get("mission_camera") or {}

    vfov = camera.get("vertical_fov_deg")
    radius = inspection.get("radius")
    bbox = bringup.get("bbox_topic")

    shared = MissionShared(
        vertical_fov_deg=float(vfov) if vfov is not None else None,
        camera_width=int(camera["width"]) if camera.get("width") is not None else None,
        camera_height=int(camera["height"]) if camera.get("height") is not None else None,
        distance_to_target_m=float(radius) if radius is not None else None,
        lidar_distance_topic=str(bringup.get("lidar_distance_topic") or "/dist_to_struct"),
        camera_topic=_resolved_camera_topic(bringup, mission_camera, interfaces),
        bbox_topic=str(bbox) if bbox else None,
        show_display=_coerce_bool(bringup.get("show_vision_display")),
        vision_model=str(interfaces.get("vision_model") or "").strip() or None,
    )
    if logger is not None:
        logger.info(
            f"Loaded shared config from mission.yaml (mode='{mode}'): "
            f"vertical_fov_deg={shared.vertical_fov_deg}, "
            f"camera={shared.camera_width}x{shared.camera_height}, "
            f"distance_to_target_m(radius)={shared.distance_to_target_m}, "
            f"bbox_topic={shared.bbox_topic}"
        )
    return shared
