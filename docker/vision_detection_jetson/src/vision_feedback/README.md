# vision_feedback

ROS 2 Humble package that detects a power-line tower's **head** and **body** in
a camera stream and publishes stabilized bounding boxes for the
`tower_inspection` mission.

Inference runs on a **cached TensorRT FP16 engine** with GPU preprocessing and
GPU NMS, and falls back to PyTorch/Ultralytics when TensorRT is unavailable.

## Architecture

The package defines the `vision_feedback/BoundingBoxes` message and ships a
modular Python implementation in `vision_perception`:

| Module | Responsibility |
| --- | --- |
| `config.py` | Declares ROS parameters, resolves device/backend/model, returns an immutable `VisionParams`. |
| `backends/base.py` | The backend contract: one `(N, 6)` float32 array per frame, `x1 y1 x2 y2 conf cls`, in source pixels. |
| `backends/tensorrt_backend.py` | Native TensorRT: pinned H2D, GPU letterbox, engine, GPU NMS, one D2H copy. |
| `backends/torch_backend.py` | Ultralytics PyTorch fallback, half precision on CUDA. |
| `backends/__init__.py` | Backend selection, fallback chain, and `ClassRouter`. |
| `engine_builder.py` | `.pt` → ONNX → cached `.engine`, content-addressed. |
| `letterbox.py` | Static-shape letterbox geometry shared by both backends. |
| `rate_limit.py` | Phase-tolerant inference rate cap (see below). |
| `image_convert.py` | Zero-copy `sensor_msgs/Image` → BGR, `cv_bridge` fallback. |
| `runtime.py` | Process-wide CPU thread limits. |
| `geometry.py` | Pixel↔meter math and body-box estimation. |
| `detection.py` | Vectorised head/body selection, gating and merging. |
| `temporal_filter.py` | Rejects implausible frame-to-frame head jumps. |
| `stabilizer.py` | Exponential smoothing, dropout hold, sanity gating. |
| `message_builder.py` | Builds the `BoundingBoxes` message. |
| `visualization.py` | Downscale-then-draw overlay + threaded OpenCV window. |
| `diagnostics.py` | REP-107 style `/diagnostics` publisher. |
| `node.py` | The rclpy node + `main()` wiring it together. |

### Pipeline (per frame)

1. Convert the image (zero-copy view over the DDS buffer where possible).
2. Optional inference-rate cap, applied before any per-frame work.
3. Duplicate/frozen-frame guard (stride-8 subsample comparison).
4. Detector backend → `(N, 6)` array → select+merge head, select body.
5. Temporal filter on the head.
6. Publish the post-filter message on the **raw** topic (only when subscribed).
7. Stabilize and publish on the **stable** topic the mission consumes.
8. Optional downscaled overlay + display.

Camera callbacks only hand off the newest frame through a capacity-one slot; a
single worker thread owns inference, filtering, publication and rendering, so
overload drops stale frames instead of backlogging. The node spins on a
`SingleThreadedExecutor` because every callback is short and non-blocking.

## Topics

| Direction | Default topic | Type |
| --- | --- | --- |
| subscribe | `/camera/out/live_view` | `sensor_msgs/Image` |
| subscribe | `/dist_to_struct` | `std_msgs/Float32` (live standoff) |
| publish (stable) | `/art/bounding_boxes` | `vision_feedback/BoundingBoxes` |
| publish (raw/debug) | `/art/bounding_boxes_raw` | `vision_feedback/BoundingBoxes` |
| publish | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` |

## Requirements

```text
ROS 2 Humble       ros-humble-ros-base, cv-bridge, diagnostic-msgs
torch 2.3.0        NVIDIA aarch64 JetPack 6 wheel (not the PyPI x86 build)
torchvision 0.18   NVIDIA aarch64 wheel  (torchvision.ops.nms runs on the GPU)
tensorrt 8.6.2     from JetPack: python3-libnvinfer
numpy 1.26.x       not numpy 2.x
opencv-python 4.11
ultralytics 8.4.72 export only, not used at runtime with TensorRT
onnx, onnxslim     export only
```

## Build

```bash
cd ~/ardu_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vision_feedback --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Run

```bash
# TensorRT FP16, auto shape (1280x720 -> 384x640), display on
ros2 launch vision_feedback vision.launch.py

# force a backend
ros2 launch vision_feedback vision.launch.py backend:=torch
ros2 launch vision_feedback vision.launch.py backend:=tensorrt device:=cuda:0

# headless
ros2 launch vision_feedback vision.launch.py show_display:=false

# extra reach at long standoff (rebuild the engine for this shape first)
ros2 launch vision_feedback vision.launch.py inference_input_shape:=576x960
```

Any YAML value can also be overridden directly:

```bash
ros2 run vision_feedback vision_node --ros-args \
  -p display_max_fps:=5.0 -p max_inference_fps:=10.0 -p cpu_threads:=2
```

## The TensorRT engine cache

A TensorRT engine is tied to the exact GPU, TensorRT version and JetPack
release, so it is built once on the target device and cached:

```text
<engine_cache_dir>/best_gazebo_new__384x640__fp16__trt862__orin__<hash>.engine
<engine_cache_dir>/best_gazebo_new__384x640__fp16__trt862__orin__<hash>.json
```

The node builds it automatically on first launch (`auto_build_engine: true`).
The default `engine_cache_dir` is `/engine_cache`; the `ENGINE_CACHE_DIR`
environment variable overrides it, which is what the container uses.

Rebuild after changing the model, the input shape, the precision, or after a
JetPack/TensorRT upgrade. Deleting the cache is always safe — it is rebuilt.

## Input shape

`inference_input_shape` controls the static engine input:

| Value | 1280x720 camera | Notes |
| --- | --- | --- |
| `auto` (default) | `384x640` | Long side = `inference_imgsz`, short side padded to the next multiple of 32. Derived from `expected_frame_*`. |
| `square` | `640x640` | Classic YOLO letterbox. 41 % of the tensor is grey padding on a 16:9 camera. |
| `576x960` | `576x960` | Explicit. More reach for very small distant objects. |

`auto` carries exactly the same scene detail as `square` at the same
`inference_imgsz` — the removed pixels are padding, not image content.

### Camera resolution

**Detection is correct at any incoming resolution, whatever the engine shape** —
`compute_letterbox()` fits any source into any engine input and
`LetterboxTransform.to_source()` maps the boxes back. The shape only affects
efficiency.

| Camera | `expected_frame_width` / `_height` | auto shape |
| --- | --- | --- |
| 1280x720 | 1280 / 720 | 384x640 |
| 1024x680 | 1024 / 680 | 448x640 |
| 1920x1080 | 1920 / 1080 | 384x640 |
| 640x480 | 640 / 480 | 480x640 |

```bash
ros2 run vision_feedback build_engine.py --width 1024 --height 680 --verify
ros2 run vision_feedback build_engine.py --width 1280 --height 720 --verify
```

Both engines coexist in the cache; switching camera is a config change and a
restart. If the received frames do not match `expected_frame_*` the node logs
the ideal shape and the exact command to build it, and keeps running.

On a 1024-wide camera set `display_width: 640` — the renderer skips a resize
that removes less than 15 % of the width, so the default 960 does nothing.

### Inference rate cap

`max_inference_fps` is phase tolerant. A naive deadline cap aliases: a 29.8 Hz
camera against a 20 Hz cap settles at 14.9 Hz because every second frame lands
just short of the deadline. `rate_limit.PhaseTolerantRateLimiter` accepts a
frame arriving within half an inter-arrival interval of its deadline and
advances the deadline from the deadline rather than from `now`, giving 19.9 Hz
from the same input. Set `0` to process every frame.

### Precision reporting

`/diagnostics` reports the precision the engine was **built** with (recorded in
the engine's sidecar JSON), not the dtype of its I/O tensors. TensorRT keeps
the network boundary in FP32 by default even for an FP16 engine, so deriving
the label from the I/O dtype made working FP16 engines report `fp32`. The I/O
dtype is shown separately in the backend detail string. `trt_fp16_io: true`
forces the boundary to FP16 too.

## Configuration

All tunables live in [`config/vision.yaml`](config/vision.yaml). Shared
quantities (camera FOV, size, standoff radius, topics, selected model) are read
from `mission.yaml` when `use_mission_config: true`; launch arguments override
both.

## Tests

```bash
cd src/vision_feedback && python3 -m pytest test -q
```

68 tests, none of which need CUDA, TensorRT or a camera — they run on a laptop.
