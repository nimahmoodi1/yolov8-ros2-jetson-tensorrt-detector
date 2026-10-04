# Tower Vision Detection — ROS 2 Humble + YOLOv8 + TensorRT on NVIDIA Jetson

Jetson-optimized ROS 2 vision detector for **power-line tower head and body detection** from a live camera stream. The repository contains both deployment forms of the project:

1. **Standalone / non-Docker ROS 2 package** for an existing ROS 2 Humble workspace.
2. **Dockerized Jetson deployment** with NVIDIA runtime, TensorRT engine caching, Fast DDS tuning, host networking, and host IPC/shared memory support for communication with other ROS 2 containers and host nodes.

The detector is built around YOLOv8 and is optimized for NVIDIA Jetson Orin NX 16 GB. The preferred inference path is a cached **TensorRT FP16** engine; a PyTorch/Ultralytics backend is retained as a fallback.

> The two deployments share nearly all perception code, but they intentionally differ in the ROS 2 message package used for bounding-box output. Read **ROS 2 interface compatibility** before connecting another node.

---

## Contents

1. [Repository layout](#repository-layout)
2. [What the detector does](#what-the-detector-does)
3. [Runtime architecture](#runtime-architecture)
4. [ROS 2 topics](#ros-2-topics)
5. [ROS 2 interface compatibility](#ros-2-interface-compatibility)
6. [Standalone vs Docker differences](#standalone-vs-docker-differences)
7. [Supported target](#supported-target)
8. [Clone the repository](#clone-the-repository)
9. [Standalone installation](#standalone-installation)
10. [Docker installation](#docker-installation)
11. [Communication with host and other ROS 2 containers](#communication-with-host-and-other-ros-2-containers)
12. [TensorRT engine cache](#tensorrt-engine-cache)
13. [Camera resolution and inference shape](#camera-resolution-and-inference-shape)
14. [Configuration](#configuration)
15. [Verification](#verification)
16. [Tests](#tests)
17. [Troubleshooting](#troubleshooting)
18. [Publishing checklist](#publishing-checklist)

---

## Repository layout

```text
tower-vision-yolov8-ros2-jetson/
├── README.md
├── .gitignore
├── standalone/
│   └── vision_feedback/
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── config/vision.yaml
│       ├── launch/vision.launch.py
│       ├── models/
│       ├── msg/
│       ├── scripts/
│       ├── test/
│       └── vision_perception/
└── docker/
    └── vision_detection_jetson/
        ├── Dockerfile
        ├── compose.yaml
        ├── Makefile
        ├── entrypoint.sh
        ├── config/fastdds_profile.xml
        ├── engine_cache/.gitkeep
        ├── scripts/
        └── src/
            ├── ros2_interface/
            └── vision_feedback/
```

The two folders are kept as deployment snapshots rather than silently merging them because their ROS message API and default camera configuration differ.

---

## What the detector does

For each accepted camera frame, the node:

1. receives `sensor_msgs/msg/Image`;
2. converts the image to BGR, using a zero-copy view where possible;
3. applies an inference-rate limiter before expensive processing;
4. rejects/handles frozen duplicate frames;
5. performs YOLO inference through TensorRT or PyTorch;
6. routes model classes whose names contain `head` and `body`;
7. applies confidence, area, aspect-ratio, geometry and temporal gates;
8. merges compatible tower-head fragments;
9. temporally filters the detected head/body;
10. stabilizes the accepted result and estimates a body box from the stabilized head when applicable;
11. publishes stable and optional raw/debug bounding boxes;
12. publishes diagnostics and optionally renders a debug window.

To avoid latency buildup, the image callback only stores the newest frame in a capacity-one handoff slot. A single inference worker processes frames serially. If the camera produces frames faster than inference can consume them, stale frames are replaced rather than queued.

---

## Runtime architecture

```text
Camera Image
    |
    v
/camera/out/live_view  (sensor_msgs/Image)
    |
    v
Latest-frame slot (capacity 1)
    |
    v
Rate limit -> duplicate/frozen-frame guard
    |
    v
TensorRT FP16 backend
    |  fallback
    +-----------> PyTorch / Ultralytics
    |
    v
Class routing: head / body
    |
    v
Confidence + geometry gates
    |
    v
Temporal filter + stabilization
    |
    +----> raw/debug bounding boxes
    |
    +----> stable bounding boxes
    |
    +----> /diagnostics
```

The TensorRT path uses a static engine shape, pinned transfer buffers, GPU preprocessing/NMS, engine caching, and device-specific engine metadata.

---

## ROS 2 topics

Default topics are:

| Direction | Topic | Type | Purpose |
|---|---|---|---|
| Subscribe | `/camera/out/live_view` | `sensor_msgs/msg/Image` | live camera image |
| Subscribe | `/dist_to_struct` | `std_msgs/msg/Float32` | optional live standoff distance |
| Publish | `/art/bounding_boxes` | custom `BoundingBoxes` | stabilized output used by mission/control code |
| Publish | `/art/bounding_boxes_raw` | custom `BoundingBoxes` | raw/debug output; only serialized when subscribed |
| Publish | `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | backend, rate, latency and health diagnostics |

The camera subscription is `KEEP_LAST`, depth `1`, `BEST_EFFORT`, `VOLATILE` to avoid replaying stale image data.

### BoundingBoxes fields

```text
bool has_head
int32 head_x1
int32 head_y1
int32 head_x2
int32 head_y2
bool has_body
int32 body_x1
int32 body_y1
int32 body_x2
int32 body_y2
```

Coordinates are source-image pixel coordinates.

---

## ROS 2 interface compatibility

This is the most important difference between the two deployment variants.

### Standalone package

The standalone node imports and publishes:

```text
vision_feedback/msg/BoundingBoxes
```

### Docker package

The Docker node imports and publishes:

```text
ros2_interface/msg/BoundingBoxes
```

The field definitions are the same, but the ROS 2 **type names are not the same**. ROS/DDS type matching requires the exact message type, not merely identical fields.

Therefore, if a Dockerized detector publishes `/art/bounding_boxes`, every host or container subscriber to that topic must be compiled against:

```text
ros2_interface/msg/BoundingBoxes
```

A subscriber compiled against `vision_feedback/msg/BoundingBoxes` will not become compatible just because both `.msg` files contain the same fields.

Verify the Docker output type with:

```bash
ros2 topic type /art/bounding_boxes
ros2 interface show ros2_interface/msg/BoundingBoxes
ros2 topic info /art/bounding_boxes -v
```

Expected type for the Docker deployment:

```text
ros2_interface/msg/BoundingBoxes
```

---

## Standalone vs Docker differences

| Area | Standalone | Docker |
|---|---|---|
| Main package | `vision_feedback` | `vision_feedback` + `ros2_interface` |
| Stable output type | `vision_feedback/msg/BoundingBoxes` | `ros2_interface/msg/BoundingBoxes` |
| Default backend in YAML | `tensorrt` | `auto` |
| Default device in YAML | `cuda:0` | `auto` |
| Compose launch default | n/a | overrides to `tensorrt` + `cuda:0` |
| Expected frame default | `1024x680` | `1280x720` |
| TensorRT cache | set a writable host directory | bind-mounted `./engine_cache:/engine_cache` |
| DDS/networking | normal host ROS setup | `network_mode: host`, `ipc: host`, Fast DDS profile |
| GPU access | native Jetson host | NVIDIA container runtime |
| Display | native X/OpenCV | X11 socket + Xauthority, optional |

Apart from these API/configuration changes, the perception modules are effectively the same implementation.

---

## Supported target

Primary target reflected by the current configuration:

- NVIDIA Jetson Orin NX 16 GB
- Ubuntu/JetPack 6.0 family
- L4T R36.3
- ROS 2 Humble
- CUDA from JetPack
- TensorRT from JetPack/container image
- Python 3.10 environment used by ROS 2 Humble/JetPack image
- YOLOv8 model packaged as `best_fake_tower_behshahr.pt`

The Dockerfile defaults to:

```text
nvcr.io/nvidia/l4t-jetpack:r36.3.0
```

and uses NVIDIA AArch64 PyTorch/torchvision wheels matching that environment. If you upgrade JetPack, re-check the matching NVIDIA wheels, TensorRT API compatibility, and rebuild the TensorRT engine on the target Jetson.

---

## Clone the repository

```bash
git clone https://github.com/nimahmoodi1/tower-vision-yolov8-ros2-jetson.git
cd tower-vision-yolov8-ros2-jetson
```

---

# Standalone installation

Use this path when ROS 2 Humble and the rest of your robotics stack already run directly on the Jetson host.

## 1. Install ROS-side dependencies

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-pytest \
  python3-yaml \
  python3-libnvinfer \
  ros-humble-ament-cmake-python \
  ros-humble-cv-bridge \
  ros-humble-diagnostic-msgs \
  ros-humble-rosidl-default-generators \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs
```

Jetson PyTorch/torchvision must be installed from NVIDIA-compatible AArch64 builds for your JetPack release. Do not assume generic PyPI `torch` wheels provide the correct Jetson CUDA build.

The current project was developed around:

```text
torch          2.3.x NVIDIA Jetson build
torchvision    0.18.x NVIDIA Jetson build
numpy          1.26.x
opencv         4.11.x
ultralytics    8.4.72
onnx           1.16.2
onnxslim       0.1.34
```

Ultralytics/ONNX tooling is primarily needed for `.pt -> ONNX -> TensorRT engine` generation. Native TensorRT is the preferred runtime backend.

## 2. Copy the package into a ROS workspace

Example workspace:

```bash
mkdir -p ~/ardu_ws/src
cp -a standalone/vision_feedback ~/ardu_ws/src/
cd ~/ardu_ws
```

## 3. Choose a writable TensorRT cache directory

The YAML default is `/engine_cache`, which may not be writable on a normal host account. Recommended host configuration:

```bash
export ENGINE_CACHE_DIR="$HOME/.cache/vision_feedback_engines"
mkdir -p "$ENGINE_CACHE_DIR"
```

Add that export to your deployment environment if you want it to persist.

## 4. Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ardu_ws
colcon build \
  --packages-select vision_feedback \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 5. Runtime CPU/GPU environment

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export YOLO_OFFLINE=1
export CUDA_MODULE_LOADING=LAZY
```

## 6. Build/verify the TensorRT engine

For the configured camera shape:

```bash
ros2 run vision_feedback build_engine.py --verify
```

Examples for common source resolutions:

```bash
ros2 run vision_feedback build_engine.py --width 1024 --height 680 --verify
ros2 run vision_feedback build_engine.py --width 1280 --height 720 --verify
```

## 7. Launch

```bash
ros2 launch vision_feedback vision.launch.py \
  backend:=tensorrt \
  device:=cuda:0 \
  inference_precision:=fp16 \
  inference_input_shape:=auto \
  image_topic:=/camera/out/live_view \
  show_display:=false
```

For desktop debugging:

```bash
ros2 launch vision_feedback vision.launch.py show_display:=true
```

---

# Docker installation

Use this deployment when the detector should be isolated in its own Jetson container while still participating in the host/robot ROS 2 graph.

```bash
cd docker/vision_detection_jetson
```

## 1. Verify Jetson container support

```bash
sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-jetpack:r36.3.0 \
  bash -lc 'uname -m; ls /usr/local/cuda >/dev/null && echo CUDA-mounted'
```

Expected architecture:

```text
aarch64
```

## 2. Raise host socket buffer limits

Large raw image topics are fragmented into many packets. The included Fast DDS profile requests 16 MiB socket buffers, so raise host limits first:

```bash
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
```

For a production system, make these persistent with an appropriate `/etc/sysctl.d/*.conf` file.

## 3. Check Jetson power mode

```bash
sudo nvpmodel -q
```

If your deployment allows MAXN and has adequate thermal/power headroom:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 4. Generate `.env`

Do not commit the generated `.env`. The repository keeps `.env.example` only.

```bash
make env
```

Review it:

```bash
cat .env
```

Important defaults:

```text
ROS_DOMAIN_ID=0
ROS_LOCALHOST_ONLY=0
VISION_BACKEND=tensorrt
VISION_DEVICE=cuda:0
VISION_PRECISION=fp16
VISION_INPUT_SHAPE=auto
IMAGE_TOPIC=/camera/out/live_view
BBOX_TOPIC=/art/bounding_boxes
RAW_BBOX_TOPIC=/art/bounding_boxes_raw
AUTOSTART=1
PREBUILD_ENGINE=1
```

## 5. Headless vs display

For flight/production operation, use:

```text
SHOW_DISPLAY=false
```

For X11 display, ensure the `.env` `DISPLAY` matches the active Jetson session and prepare X authentication:

```bash
make xauth
```

## 6. Build the image

```bash
make build
```

Equivalent command:

```bash
docker compose build
```

## 7. Start

```bash
make up
```

Follow logs:

```bash
make logs
```

The entrypoint checks architecture, PyTorch CUDA visibility, TensorRT bindings, display reachability, and pre-builds the device-specific TensorRT engine when enabled.

## 8. Verify GPU and ROS

```bash
make gpu-check
make check
make topics
make camera-hz
make bbox
make diag
```

## 9. Useful Docker workflow

```bash
make up-idle      # start without detector autostart
make shell        # shell with ROS workspace sourced
make run          # launch detector manually
make engine       # build active TensorRT engine
make engine-720p  # 1280x720 source -> auto 384x640 engine
make engine-1024  # 1024x680 source -> auto 448x640 engine
make bench        # compare backends on target device
make restart
make down
```

---

# Communication with host and other ROS 2 containers

The Docker deployment uses all of the following:

```yaml
network_mode: host
ipc: host
runtime: nvidia
```

and:

```text
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE=/ws_vision_detection/config/fastdds_profile.xml
ROS_LOCALHOST_ONLY=0
```

The Fast DDS profile defines:

- a `64 MiB` shared-memory transport segment;
- UDPv4 transport with `16 MiB` send/receive buffers;
- asynchronous DataWriter publish mode;
- both SHM and UDP user transports.

### What `network_mode: host` does

It removes Docker NAT from DDS discovery/data traffic and lets the container participate directly on the host network interfaces.

### What `ipc: host` does

It puts the container in the host IPC namespace so Fast DDS shared-memory transport can use the same host shared-memory namespace. Other ROS containers that are expected to use host SHM with this detector should use a compatible IPC arrangement as well.

### Requirements for another ROS 2 container

At minimum, align:

```text
ROS_DOMAIN_ID
ROS_LOCALHOST_ONLY=0
```

For the closest behavior/performance, use Fast DDS and host networking. If shared-memory transport is expected across containers, use compatible host IPC configuration and Fast DDS versions/settings.

Most importantly, the subscriber must have the same message package:

```text
ros2_interface/msg/BoundingBoxes
```

### Requirements for a ROS 2 node on the Jetson host

The host workspace must build/source the `ros2_interface` package used by Docker consumers/producers.

One simple method is to copy it into the host workspace:

```bash
cp -a docker/vision_detection_jetson/src/ros2_interface ~/ardu_ws/src/
cd ~/ardu_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_interface
source install/setup.bash
```

Then verify:

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ros2 interface show ros2_interface/msg/BoundingBoxes
ros2 topic type /art/bounding_boxes
ros2 topic echo /art/bounding_boxes
```

### Important note about `QT_X11_NO_MITSHM`

The Docker environment sets:

```text
QT_X11_NO_MITSHM=1
```

This disables the X11/Qt MIT-SHM display mechanism and is unrelated to Fast DDS ROS shared-memory transport. It does **not** disable DDS SHM.

---

# TensorRT engine cache

TensorRT engines are not portable binaries. They are tied to the target environment, including GPU architecture, TensorRT/JetPack version, network shape, model content and precision settings.

The project therefore builds engines on the target Jetson and caches them.

Cache identity includes important inputs such as:

```text
model hash
network HxW
requested precision
FP16 I/O choice
TensorRT version
GPU identity
```

The sidecar JSON records build provenance and the precision that TensorRT actually built.

### Standalone cache

Recommended:

```bash
export ENGINE_CACHE_DIR="$HOME/.cache/vision_feedback_engines"
```

### Docker cache

The container uses:

```text
/engine_cache
```

backed by the host directory:

```text
docker/vision_detection_jetson/engine_cache/
```

Generated `.engine` and sidecar cache files are intentionally ignored by Git.

Rebuild engines after changing:

- JetPack/TensorRT version;
- model weights;
- static input shape;
- precision;
- FP16 I/O behavior;
- target GPU.

---

# Camera resolution and inference shape

`inference_input_shape: auto` keeps the long side at `inference_imgsz` and pads the shorter side to a multiple of 32.

Examples:

| Source camera | Auto network shape |
|---|---|
| `1280x720` | `384x640` |
| `1024x680` | `448x640` |
| `1920x1080` | `384x640` |
| `640x480` | `480x640` |

The letterbox mapping converts detections back into source-image coordinates. Detection can still operate if the incoming resolution differs, but keeping `expected_frame_width` / `expected_frame_height` aligned with the actual camera avoids using a suboptimal engine shape.

Current defaults differ by deployment:

```text
standalone: 1024x680
docker:     1280x720
```

Set these to match your actual camera and rebuild the matching engine.

---

# Configuration

Primary configuration file:

```text
config/vision.yaml
```

Important groups include:

### Mission integration

```yaml
use_mission_config: true
use_live_distance: true
live_distance_topic: /dist_to_struct
live_distance_stale_s: 0.6
```

When mission integration is enabled, the node attempts to load shared values from `mission.yaml`, using:

```text
MISSION_CONFIG_PATH
TOWER_INSPECTION_CONFIG
installed tower_inspection package share directory
```

If no mission config is found, the node falls back to local `vision.yaml` values.

### Backend

```yaml
backend: tensorrt        # standalone default
# backend: auto          # docker YAML default; compose overrides to tensorrt
inference_precision: fp16
inference_input_shape: auto
trt_fp16_io: false
```

### Detection

```yaml
inference_imgsz: 640
inference_iou_threshold: 0.50
inference_max_detections: 20
max_inference_fps: 20.0
head_confidence_threshold: 0.78
body_confidence_threshold: 0.82
```

### Temporal behavior

```yaml
max_jump_m: 1.0
acquisition_frames: 3
acquisition_max_missed_frames: 1
reset_after_n_missed: 6
stabilizer_hold_time_s: 0.4
stabilizer_alpha: 0.60
duplicate_timeout_s: 0.75
```

### Runtime/display

```yaml
cpu_threads: 1
show_display: true
display_max_fps: 10.0
display_width: 960
publish_diagnostics: true
stats_log_interval_s: 10.0
```

Launch-time overrides include:

```text
image_topic
bbox_topic
raw_bbox_topic
device
model_path
backend
inference_precision
inference_input_shape
engine_cache_dir
show_display
auto_build_engine
trt_fp16_io
```

Example:

```bash
ros2 launch vision_feedback vision.launch.py \
  backend:=tensorrt \
  device:=cuda:0 \
  image_topic:=/camera/out/live_view \
  show_display:=false
```

---

# Verification

## Check camera input

```bash
ros2 topic info /camera/out/live_view -v
ros2 topic hz /camera/out/live_view
```

## Check detector output

Standalone:

```bash
ros2 topic type /art/bounding_boxes
# expected: vision_feedback/msg/BoundingBoxes
ros2 topic echo /art/bounding_boxes
```

Docker:

```bash
ros2 topic type /art/bounding_boxes
# expected: ros2_interface/msg/BoundingBoxes
ros2 topic echo /art/bounding_boxes
```

## Check diagnostics

```bash
ros2 topic echo --once /diagnostics
```

Look for backend type, TensorRT precision, source/inference rate, inference latency, end-to-end latency, dropped/throttled frames and degraded fallback warnings.

---

# Tests

The package includes unit tests for:

- TensorRT/PyTorch backend selection and fallback;
- CUDA OOM fallback handling;
- class routing;
- engine cache identity and sidecars;
- camera geometry and meter/pixel gates;
- head/body filtering;
- image conversion and row padding;
- latest-frame replacement behavior;
- letterbox round-trip mapping;
- phase-tolerant inference rate limiting;
- stabilizer behavior across resolution changes;
- visualization/downscaling/display rate limiting.

Run standalone tests from the package source:

```bash
cd standalone/vision_feedback
python3 -m pytest test -q
```

Inside the Docker deployment:

```bash
cd docker/vision_detection_jetson
make test
```

---

# Troubleshooting

## Topic is visible but subscriber receives no bounding boxes

Check the exact type:

```bash
ros2 topic type /art/bounding_boxes
```

For Docker it must be consumed as:

```text
ros2_interface/msg/BoundingBoxes
```

Also verify `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, RMW choice, discovery/networking and that the host/other container has sourced the interface package.

## Camera topic is only a few Hz in Docker

Check host socket buffer limits:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
```

Expected at least:

```text
16777216
```

Then inspect:

```bash
ros2 topic hz /camera/out/live_view
```

## TensorRT falls back to PyTorch

Check:

```bash
python3 -c 'import tensorrt; print(tensorrt.__version__)'
python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

Docker:

```bash
make gpu-check
make logs
```

## First launch is slow

The first launch may need to build the TensorRT engine. Pre-build it:

```bash
ros2 run vision_feedback build_engine.py --verify
```

or in Docker:

```bash
make engine
```

## `/engine_cache` permission error on standalone

Use a user-writable cache:

```bash
export ENGINE_CACHE_DIR="$HOME/.cache/vision_feedback_engines"
mkdir -p "$ENGINE_CACHE_DIR"
```

## Display fails in Docker

Prefer headless mode for production:

```text
SHOW_DISPLAY=false
```

For display mode, verify `DISPLAY`, Xauthority, X11 socket mount, and run:

```bash
make xauth
```

## Model publishes nothing

The class router only accepts model class names containing `head` and/or `body`. If a newly trained model uses different class names, update the model naming or class-routing logic/configuration.

---

# Publishing checklist

Before making this repository public:

- [ ] Replace/verify maintainer metadata in both ROS packages.
- [ ] Decide the repository license and add the actual license text at repository root. The current package metadata declares BSD-3-Clause for `vision_feedback` and Apache-2.0 for `ros2_interface`; make the public licensing strategy explicit.
- [ ] Confirm you have permission to redistribute `models/best_fake_tower_behshahr.pt` and document the dataset/model license or provenance.
- [ ] Keep `.env` out of Git; publish `.env.example` only.
- [ ] Do not commit TensorRT `.engine`, generated ONNX, build/install/log folders, or local cache data.
- [ ] Replace old ZIP/archive references in release documentation if filenames change again.
- [ ] Verify the actual camera resolution and update `expected_frame_width` / `expected_frame_height` for the intended deployment.
- [ ] Verify Docker and standalone consumers use the intended message type.
- [ ] Build on the target Jetson and run the unit tests before tagging a release.

---

## Additional detailed documentation

Deployment-specific documentation is retained inside each snapshot:

```text
standalone/vision_feedback/README.md
standalone/vision_feedback/RUNBOOK.md
standalone/vision_feedback/OPTIMIZATION_REPORT.md

docker/vision_detection_jetson/README.md
docker/vision_detection_jetson/RUNBOOK.md
docker/vision_detection_jetson/OPTIMIZATION_REPORT.md
```

The root README is the authoritative guide for repository layout and the differences between the two deployment variants.
