# Changelog

All notable public-release changes to this repository are documented here.

## 1.1.0 - 2026-10-04

First public release of the YOLOv8 ROS 2 Jetson TensorRT Detector.

### Added

- ROS 2 Humble tower head/body detection from live camera streams.
- NVIDIA Jetson Orin NX deployment support.
- Native TensorRT FP16 inference backend.
- PyTorch/Ultralytics fallback backend.
- CUDA preprocessing and GPU NMS.
- Device-specific TensorRT engine generation and caching.
- Latest-frame processing to prevent stale-frame backlogs.
- Confidence, geometry, temporal, and stabilization filtering.
- Stable and raw/debug bounding-box outputs.
- ROS diagnostics and optional visualization.
- Standalone ROS 2 deployment.
- Docker deployment with NVIDIA runtime.
- Fast DDS host networking and shared-memory configuration.
- Multi-resolution camera and automatic inference-shape support.
- Detailed deployment and optimization documentation.

### Public repository preparation

- Added GitHub repository documentation and metadata.
- Centralized deployment documentation under `docs/`.
- Removed development-only ZIP/archive instructions.
- Removed local operator scratch notes.
- Added release, contribution, security, and citation metadata.
