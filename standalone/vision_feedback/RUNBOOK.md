# Runbook — step by step with expected output

* **Part A — the full container** (`vision_detection_jetson`).
* **Part B — the standalone ROS 2 package** (`vision_feedback`) dropped into an
  existing workspace on the Jetson host.
* **Part C — changing camera resolution** (1280x720, 1024x680, anything else).
  Read this if your camera is not 1280x720.

Everything below runs **on the Jetson**, not on a laptop.

---

# Part A — the full container

## A0. One-time host preparation

```bash
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo nvpmodel -q
echo $DISPLAY
```

Expected:

```text
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
NV Power Mode: MAXN
0
:10
```

**The power mode matters.** An Orin in a 10 W or 15 W profile runs the GPU at a
fraction of its MAXN clock, which presents as inference times 2-3x higher with
no other symptom. If `nvpmodel -q` shows anything other than MAXN and you have
the thermal headroom:

```bash
sudo nvpmodel -m 0        # MAXN
sudo jetson_clocks        # pin clocks to maximum
```

`:10` is your xrdp session. If you get something else (`:0`, `:1`), use that
value in step A1.

Without the two `sysctl` lines Fast DDS silently clamps its socket buffers and
`ros2 topic hz` on the camera shows a few Hz instead of 15.

---

## A1. Unpack and configure

```bash
unzip vision_detection_container_jetson_v3_fixed_ros2_interface.zip
cd vision_detection_jetson
make env
```

Expected (the UID/GID are yours, `DISPLAY` follows your session):

```text
wrote .env:
HOST_UID=1000
HOST_GID=1000
ROS_DOMAIN_ID=0
ROS_LOCALHOST_ONLY=0
VISION_BACKEND=tensorrt
VISION_DEVICE=cuda:0
VISION_PRECISION=fp16
VISION_INPUT_SHAPE=auto
IMAGE_TOPIC=/camera/out/live_view
BBOX_TOPIC=/art/bounding_boxes
RAW_BBOX_TOPIC=/art/bounding_boxes_raw
SHOW_DISPLAY=true
AUTOSTART=1
PREBUILD_ENGINE=1
DISPLAY=:10
XAUTHORITY=/tmp/.docker.xauth
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

If `DISPLAY` came out wrong, edit `.env` now.

---

## A2. Pull the JetPack base image

```bash
make pull
```

Expected (~14 GB, several minutes):

```text
docker pull nvcr.io/nvidia/l4t-jetpack:r36.3.0
r36.3.0: Pulling from nvidia/l4t-jetpack
...
Status: Downloaded newer image for nvcr.io/nvidia/l4t-jetpack:r36.3.0
```

---

## A3. Build the image

```bash
make build
```

Expected, ending with:

```text
 => [ 6/11] RUN python3 -c "import tensorrt; print('TensorRT bindings OK:', tensorrt.__version__)"
#0 1.234 TensorRT bindings OK: 8.6.2
 => [10/11] RUN source /opt/ros/humble/setup.bash     && colcon build ...
#0 45.6 Starting >>> vision_feedback
#0 78.9 Finished <<< vision_feedback [33.1s]
#0 79.0 Summary: 1 package finished [33.4s]
 => exporting to image
 => => naming to docker.io/library/vision_detection_jetson:latest
```

First build is 15-30 minutes (the torch wheels are large).

> **If it stops at the TensorRT line** with `ModuleNotFoundError: No module
> named 'tensorrt'`, you are not building on the JetPack base image. Check
> `BASE_IMAGE` in `.env`/`compose.yaml`.

---

## A4. Start the detector

Start `lr1_camera` the way you normally do first, then:

```bash
make up
```

Expected:

```text
X cookie ready at /tmp/.docker.xauth for DISPLAY=:10
[+] Running 1/1
 ✔ Container vision_detection_jetson  Started
NAME                       IMAGE                            STATUS         PORTS
vision_detection_jetson    vision_detection_jetson:latest   Up 2 seconds
```

---

## A5. Watch the first launch (this is where the engine is built)

```bash
make logs
```

Expected on the **first** run — note the one-off engine build:

```text
[entrypoint] arch=aarch64
[entrypoint] torch=2.3.0 torch_cuda=12.2
[entrypoint] CUDA available: True
[entrypoint] GPU: Orin
[entrypoint] TensorRT: 8.6.2
[entrypoint] SHOW_DISPLAY=true, DISPLAY=:10
[entrypoint] X server reachable on :10
[entrypoint] Ensuring the TensorRT engine cache is populated...
[build_engine] TensorRT 8.6.2
[build_engine] GPU: Orin
[build_engine] Network input 640x384 (fp16)
[build_engine] Building TensorRT engine (fp16); this takes a few minutes and only happens once per model/shape/device.
[build_engine] Engine written to /engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine in 118 s
[build_engine] OK best_gazebo_new.pt -> /engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine (9.4 MB, 121 s)
[build_engine] All engines ready
```

then the node itself:

```text
[vision_node]: CPU thread pools limited to 1 thread(s)
[vision_node]: Loaded shared config from mission.yaml (mode='real'): ... camera=1280x720 ...
[vision_node]: Camera geometry: expecting 1280x720 -> network input 640x384 (inference_input_shape='auto', imgsz=640)
[vision_node]: Detector request: backend=tensorrt device=cuda:0 input=640x384 precision=fp16 fp16_io=False conf_floor=0.78 iou=0.50
[vision_node]: Using cached TensorRT engine /engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine
[vision_node]: TensorRT letterbox 1280x720 -> 640x360 padded into 640x384 (gain=0.500)
[vision_node]: Detector ready: tensorrt/fp16 640x384 on cuda:0 (TensorRT 8.6.2, nc=2, io fp32, private stream)
[vision_node]: Class routing: head classes [1:head-YJtu] | body classes [0:body]
[vision_node]: Vision tuning: vfov=46.8 deg standoff=7.0 m (live lidar when fresh) merge=1.20 m (~142 px) max_jump=1.00 m conf(head/body)=0.78/0.82 net=640x384 stabilizer alpha=0.60 hold=0.4s | ~8.5 mm/px vertical @720px height
[vision_node]: Display enabled on DISPLAY=:10 (max 10 fps, 960px wide)
[vision_node]: Live standoff from '/dist_to_struct' (fallback radius=7.0 m)
[vision_node]: Subscribed to '/camera/out/live_view'; publishing stable boxes on '/art/bounding_boxes' and raw boxes on '/art/bounding_boxes_raw'
```

Once frames arrive:

```text
[vision_node]: Vision frame geometry verified: 1280x720
```

and then every 10 seconds:

```text
[vision_node]: tensorrt/fp16 640x384: 19.8 fps processed of 29.8 fps camera | infer 4.3 ms | end-to-end 9.8 ms | dropped 0 | throttled 98
```

Four things to check on these lines:

| Field | Expect | If not |
|---|---|---|
| `tensorrt/fp16` | this exactly | `torch/...` = fallback; `.../fp32` = the FP16 flag was not applied. See Troubleshooting. |
| `io fp32, private stream` | this exactly | Missing `private stream` means an old build; the TensorRT default-stream warning will also appear. |
| `X fps processed of Y fps camera` | X ≈ `max_inference_fps`, or ≈ Y if the camera is slower | If X is about half of Y, you are on a pre-v3 build. |
| `TensorRT letterbox` | appears **once** | Twice means warmup ran at the wrong geometry (pre-v3). |

There must be **no** line like this:

```text
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues ...
```

If you see it, the build is older than v3.

An OpenCV window titled **Vision Feedback** appears with a HUD in the corner:

```text
tensorrt/fp16 640x384 |  19.8 fps | infer   4.3 ms
```

On every **later** start the engine build line is replaced by:

```text
[build_engine] Using cached TensorRT engine /engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine
```

and startup takes a couple of seconds.

---

## A6. Verify the GPU stack

```bash
make gpu-check
```

Expected:

```text
=== Jetson vision GPU check ===
architecture: aarch64
python: 3.10.12
numpy: 1.26.4
opencv: 4.11.0
torch: 2.3.0
torch CUDA build: 12.2
CUDA available: True
cuDNN: 8904
GPU 0: Orin
CUDA tensor test: (1024, 1024) cuda:0 12.345678
CUDA FP16 tensor test: (1024, 1024) torch.float16 12.34
torchvision: 0.18.0a0+6043bc2
GPU NMS test: kept 1 of 2 overlapping boxes
tensorrt: 8.6.2
TensorRT fast FP16: True
TensorRT fast INT8: True
ultralytics: 8.4.72

=== Jetson power mode ===
nvpmodel default: < PM_CONFIG DEFAULT=0 >
GPU current clock: 918 MHz
GPU max clock:     918 MHz
PASS: Jetson GPU stack (PyTorch CUDA + FP16 + TensorRT) is operational
```

A `GPU max clock` well below the module's rating means a low `nvpmodel`
profile; see step A0.

---

## A7. Verify ROS wiring

```bash
make topics
```

Expected (abridged):

```text
/art/bounding_boxes
/art/bounding_boxes_raw
/camera/out/live_view
/diagnostics
/dist_to_struct
/parameter_events
/rosout
```

```bash
make camera-hz
```

Expected — press `Ctrl+C` to stop:

```text
average rate: 15.012
        min: 0.062s max: 0.071s std dev: 0.00203s window: 16
```

```bash
make bbox
```

Expected:

```text
has_head: true
head_x1: 612
head_y1: 288
head_x2: 741
head_y2: 407
has_body: true
body_x1: 599
body_y1: 407
body_x2: 754
body_y2: 704
---
```

```bash
make diag
```

Expected (abridged) — check `backend: tensorrt`:

```text
status:
- level: "\0"
  name: 'vision: camera input'
  message: Receiving frames
  values:
  - {key: frames_received, value: '451'}
  - {key: seconds_since_last_frame, value: '0.07'}
- level: "\0"
  name: 'vision: model'
  message: Loaded (tensorrt)
  values:
  - {key: backend, value: tensorrt}
  - {key: device, value: 'cuda:0'}
  - {key: precision, value: fp16}        # <- must be fp16, not fp32
  - {key: network_input, value: 640x384}
  - {key: inference_ms, value: '4.3'}
- level: "\0"
  name: 'vision: output'
  message: Publishing bounding boxes
  values:
  - {key: frames_processed, value: '451'}
  - {key: frames_dropped_before_inference, value: '0'}
  - {key: frames_throttled, value: '0'}
  - {key: processing_latency_ms, value: '9.8'}
  ...
```

---

## A8. Confirm the CPU actually dropped

```bash
htop -p $(docker inspect -f '{{.State.Pid}}' vision_detection_jetson)
```

Or straight from the host:

```bash
top -b -n 2 -d 2 | grep vision_node | tail -3
```

Before this change the `vision_node` process sat near **~175 %** of one core's
worth across its threads. Expect roughly **25-45 %** now, with the load average
noticeably lower because the OpenMP/TBB pools are gone.

---

## A9. Benchmark TensorRT vs PyTorch on your board

```bash
make bench
```

Expected shape of the result (your exact numbers will differ):

```text
==============================================================================
model  : best_gazebo_new.pt
frames : 120 x 1280x720 BGR
network: 640x384 (fp16)
==============================================================================
...
backend                                                 mean     p95     fps    cpu%
------------------------------------------------------------------------------
tensorrt/fp16 640x384 on cuda:0 (TensorRT 8.6.2, nc=2)   4.4m    5.6m   225.1      38
torch/fp16 640x384 on cuda:0 (ultralytics predictor)    18.9m   23.4m    52.8      96

tensorrt is 4.3x faster and uses 2.5x less CPU than torch

cpu% is single-process CPU time / wall time: 100 % = one saturated core.
```

---

## A10. Run the unit tests

```bash
make test
```

Expected:

```text
68 passed in 0.9s
```

---

## A11. Everyday operations

```bash
make logs        # follow
make restart     # restart the container
make down        # stop and remove
make recreate    # apply .env / compose.yaml changes
make shell       # interactive shell with the workspace sourced
make up-idle     # start the container without launching the node
make run         # then launch it manually in the foreground
```

---

# Part B — the standalone `vision_feedback` package

Use this when you want the node inside an existing workspace on the Jetson host
(for example `~/ardu_ws`, alongside `tower_inspection`) rather than in its own
container.

## B0. Host requirements

The host (or whatever container you build in) needs:

```text
ROS 2 Humble           ros-humble-ros-base, cv-bridge, diagnostic-msgs
Python 3.10
torch 2.3.0            NVIDIA aarch64 JetPack 6 wheel  (NOT the PyPI x86 one)
torchvision 0.18.0     NVIDIA aarch64 wheel
tensorrt 8.6.2         from JetPack: python3-libnvinfer
numpy 1.26.x           NOT numpy 2.x
opencv-python 4.11
ultralytics 8.4.72     export only
onnx, onnxslim         export only
```

Check what you have:

```bash
python3 -c "import torch, torchvision, tensorrt, numpy, cv2; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
print('torchvision', torchvision.__version__); \
print('tensorrt', tensorrt.__version__); \
print('numpy', numpy.__version__, '| cv2', cv2.__version__)"
```

Expected:

```text
torch 2.3.0 | cuda True
torchvision 0.18.0a0+6043bc2
tensorrt 8.6.2
numpy 1.26.4 | cv2 4.11.0
```

If `tensorrt` is missing:

```bash
sudo apt-get install -y python3-libnvinfer
```

If `ultralytics`/`onnx` are missing (needed only for the one-off export):

```bash
pip3 install "ultralytics==8.4.72" "onnx==1.16.2" "onnxslim==0.1.34" \
             "numpy==1.26.4" "opencv-python==4.11.0.86"
```

---

## B1. Drop the package into your workspace

```bash
unzip vision_feedback_ros2_package_v3.zip
cp -r vision_feedback ~/ardu_ws/src/
cd ~/ardu_ws
```

---

## B2. Choose where engines are cached

The default `engine_cache_dir` is `/engine_cache`, which exists inside the
container but probably not on your host. Either create it:

```bash
sudo mkdir -p /engine_cache && sudo chown "$USER:$USER" /engine_cache
```

or point the node somewhere else — the environment variable wins over the YAML:

```bash
export ENGINE_CACHE_DIR="$HOME/.cache/vision_feedback_engines"
mkdir -p "$ENGINE_CACHE_DIR"
```

---

## B3. Build

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select vision_feedback --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Expected:

```text
Starting >>> vision_feedback
Finished <<< vision_feedback [33.4s]

Summary: 1 package finished [33.7s]
```

---

## B4. Set the CPU thread limits before launching

OpenMP reads these at import time, so exporting them afterwards has no effect:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export YOLO_OFFLINE=1
export CUDA_MODULE_LOADING=LAZY
```

---

## B5. Build the TensorRT engine

```bash
ros2 run vision_feedback build_engine.py
```

Expected (1-4 minutes):

```text
[build_engine] TensorRT 8.6.2
[build_engine] GPU: Orin
[build_engine] Network input 640x384 (fp16)
[build_engine] Building TensorRT engine (fp16); this takes a few minutes and only happens once per model/shape/device.
[build_engine] Engine written to /home/flyby/.cache/vision_feedback_engines/best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine in 118 s
[build_engine] OK best_gazebo_new.pt -> .../best_gazebo_new__384x640__fp16__trt862__orin__3f9c1a2b7d.engine (9.4 MB, 121 s)
[build_engine] All engines ready
```

Useful variants:

```bash
ros2 run vision_feedback build_engine.py --all                    # every packaged model
ros2 run vision_feedback build_engine.py --width 1024 --height 680  # 1024x680 camera
ros2 run vision_feedback build_engine.py --width 1280 --height 720  # 1280x720 camera
ros2 run vision_feedback build_engine.py --shape 576x960          # long-standoff profile
ros2 run vision_feedback build_engine.py --fp16-io                # FP16 network I/O too
ros2 run vision_feedback build_engine.py --verify                 # time it after building
ros2 run vision_feedback build_engine.py --force                  # rebuild from scratch
```

With `--verify` the build ends with a timing line measured on this device:

```text
[build_engine]    built precision=fp16 (fp16 flag True) io=fp32 tensorrt=10.3.0 gpu=orin
[build_engine]    verify: tensorrt/fp16 640x448 on cuda:0 (TensorRT 10.3.0, nc=2, io fp32, private stream) | mean 5.9 ms | p50 5.7 ms | p95 7.1 ms
```

You can skip this step — the node builds it on first launch — but doing it now
keeps the first launch fast.

---

## B6. Launch

```bash
ros2 launch vision_feedback vision.launch.py \
    backend:=tensorrt \
    device:=cuda:0 \
    inference_precision:=fp16 \
    inference_input_shape:=auto \
    image_topic:=/camera/out/live_view \
    show_display:=true
```

Expected, for a **1024x680** camera (the shape is derived automatically):

```text
[vision_node-1] [INFO] [vision_node]: Loaded shared config from mission.yaml (mode='real'): vertical_fov_deg=45.47, camera=1024x680, distance_to_target_m(radius)=5.0, bbox_topic=/art/bounding_boxes
[vision_node-1] [INFO] [vision_node]: CPU thread pools limited to 1 thread(s)
[vision_node-1] [INFO] [vision_node]: Camera geometry: expecting 1024x680 -> network input 640x448 (inference_input_shape='auto', imgsz=640)
[vision_node-1] [INFO] [vision_node]: Detector request: backend=tensorrt device=cuda:0 input=640x448 precision=fp16 fp16_io=False conf_floor=0.78 iou=0.50
[vision_node-1] [INFO] [vision_node]: Using cached TensorRT engine /home/intplatform/.cache/vision_feedback_engines/best_fake_tower_behshahr__448x640__fp16__trt1030__orin__679908eca5.engine
[vision_node-1] [INFO] [vision_node]: TensorRT letterbox 1024x680 -> 640x425 padded into 640x448 (gain=0.625)
[vision_node-1] [INFO] [vision_node]: Detector ready: tensorrt/fp16 640x448 on cuda:0 (TensorRT 10.3.0, nc=2, io fp32, private stream)
[vision_node-1] [INFO] [vision_node]: Class routing: head classes [1:head-YJtu] | body classes [0:body]
[vision_node-1] [INFO] [vision_node]: Display enabled on DISPLAY=:0 (max 10 fps, 960px wide)
[vision_node-1] [INFO] [vision_node]: Subscribed to '/camera/out/live_view'; ...
[vision_node-1] [INFO] [vision_node]: Vision frame geometry verified: 1024x680
[vision_node-1] [INFO] [vision_node]: tensorrt/fp16 640x448: 19.8 fps processed of 29.8 fps camera | infer 6.2 ms | end-to-end 14.1 ms | dropped 0 | throttled 99
```

Compare against the pre-v3 log for the same hardware:

```text
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues ...
Detector ready: tensorrt/fp32 640x448 on cuda:0 (TensorRT 10.3.0, nc=2)
tensorrt/fp32 640x448: 14.8 fps processed | infer 15.4 ms | end-to-end 32.3 ms | dropped 0 | throttled 413
```

The warning is gone, the precision reads `fp16`, and the processed rate is no
longer half the camera rate.

Other useful launches:

```bash
# force the PyTorch fallback, e.g. to compare
ros2 launch vision_feedback vision.launch.py backend:=torch

# headless - recommended for flight
ros2 launch vision_feedback vision.launch.py show_display:=false

# explicit shape for a 1024x680 camera
ros2 launch vision_feedback vision.launch.py inference_input_shape:=448x640

# extra reach at long standoff (rebuild the engine for this shape first)
ros2 launch vision_feedback vision.launch.py inference_input_shape:=576x960

# FP16 network I/O as well (rebuild the engine with --fp16-io first)
ros2 launch vision_feedback vision.launch.py trt_fp16_io:=true

# any YAML value can still be overridden directly
ros2 run vision_feedback vision_node --ros-args \
  -p display_max_fps:=5.0 -p max_inference_fps:=25.0 -p display_width:=640
```

---

## B7. Verify

```bash
ros2 topic hz /camera/out/live_view      # ~15 Hz
ros2 topic hz /art/bounding_boxes        # ~15 Hz
ros2 topic echo --once /art/bounding_boxes
ros2 topic echo --once /diagnostics      # backend must read 'tensorrt'
```

```bash
ros2 run vision_feedback benchmark.py
```

Expected (numbers will differ on your board):

```text
backend                                                 mean     p95     fps    cpu%
------------------------------------------------------------------------------
tensorrt/fp16 640x384 on cuda:0 (TensorRT 8.6.2, nc=2)   4.4m    5.6m   225.1      38
torch/fp16 640x384 on cuda:0 (ultralytics predictor)    18.9m   23.4m    52.8      96

tensorrt is 4.3x faster and uses 2.5x less CPU than torch
```

```bash
cd ~/ardu_ws/src/vision_feedback && python3 -m pytest test -q
```

Expected:

```text
68 passed in 0.9s
```

---

# Part C — changing camera resolution

**Short version: it already works at any resolution.** The letterbox adapts to
whatever frame arrives, so an engine built for one camera produces *correct*
detections on another. Matching the engine shape to the camera only makes it
*faster*.

## C1. Which shape does my camera want?

`inference_input_shape: auto` derives the shape from `expected_frame_width` /
`expected_frame_height`: the long side becomes `inference_imgsz` (640), the
short side is padded to the next multiple of 32.

| Camera | `expected_frame_width` | `expected_frame_height` | auto shape |
|---|---:|---:|---|
| 1280x720 (720p) | 1280 | 720 | **384x640** |
| 1024x680 (Sony RX0 II) | 1024 | 680 | **448x640** |
| 1920x1080 (1080p) | 1920 | 1080 | 384x640 |
| 1280x960 | 1280 | 960 | 480x640 |
| 640x480 (VGA) | 640 | 480 | 480x640 |

Check it without running anything:

```bash
python3 -c "
from vision_perception.letterbox import auto_input_shape
h, w = auto_input_shape(680, 1024, 640)   # (height, width, imgsz)
print(f'network input {w}x{h}')"
```

Expected:

```text
network input 640x448
```

## C2. Two ways to set it

### Via `mission.yaml` (preferred — nothing to edit here)

If `use_mission_config: true`, `mission.yaml` supplies the camera size and the
node picks it up automatically:

```yaml
profiles:
  real:
    camera:
      width: 1024
      height: 680
      vertical_fov_deg: 45.47
```

Confirm it landed:

```text
[vision_node]: Loaded shared config from mission.yaml (mode='real'): ... camera=1024x680 ...
[vision_node]: Camera geometry: expecting 1024x680 -> network input 640x448 (inference_input_shape='auto', imgsz=640)
```

### Via `vision.yaml` (standalone, or to override mission.yaml)

Edit exactly these two lines in `config/vision.yaml`:

```yaml
    expected_frame_width: 1024      # <- your camera width
    expected_frame_height: 680      # <- your camera height
```

Leave everything else alone:

```yaml
    inference_input_shape: auto     # derives the shape from the two lines above
    inference_imgsz: 640            # matches how the models were trained
    inference_precision: fp16
```

Or skip the YAML entirely and pass the shape at launch:

```bash
ros2 launch vision_feedback vision.launch.py inference_input_shape:=448x640
```

## C3. Build the engine for that camera

The engine is static, so it has to be built for the new shape. The cache holds
as many as you like — the filename encodes the shape.

```bash
ros2 run vision_feedback build_engine.py --width 1024 --height 680 --verify
```

Expected:

```text
[build_engine] TensorRT 10.3.0
[build_engine] GPU: Orin
[build_engine] Camera 1024x680 -> network input 640x448 (fp16, shape='auto', io fp32)
[build_engine] Building TensorRT engine (requested fp16, fp16 flag applied, io fp32); this takes a few minutes and only happens once per model/shape/device.
[build_engine] Engine written to /home/intplatform/.cache/vision_feedback_engines/best_fake_tower_behshahr__448x640__fp16__trt1030__orin__679908eca5.engine in 96 s
[build_engine] OK best_fake_tower_behshahr.pt -> .../best_fake_tower_behshahr__448x640__fp16__trt1030__orin__679908eca5.engine (8.9 MB, 99 s)
[build_engine]    built precision=fp16 (fp16 flag True) io=fp32 tensorrt=10.3.0 gpu=orin
[build_engine]    verify: tensorrt/fp16 640x448 on cuda:0 (TensorRT 10.3.0, nc=2, io fp32, private stream) | mean 5.9 ms | p50 5.7 ms | p95 7.1 ms
```

The two lines that matter:

* `built precision=fp16 (fp16 flag True)` — FP16 was genuinely applied. If this
  says `fp32 (fp16 flag False)` the platform reported no fast FP16; see
  Troubleshooting.
* `verify: ... mean 5.9 ms` — measured on *your* board, not an estimate.

In the container the same thing is:

```bash
make engine-1024      # 1024x680 -> 448x640
make engine-720p      # 1280x720 -> 384x640
```

## C4. Keeping both cameras ready

Build both engines once; then switching camera is a YAML edit and a restart,
with no rebuild:

```bash
ros2 run vision_feedback build_engine.py --width 1280 --height 720
ros2 run vision_feedback build_engine.py --width 1024 --height 680
ls -1 "$ENGINE_CACHE_DIR"
```

Expected:

```text
best_gazebo_new__384x640__fp16__trt1030__orin__3f9c1a2b7d.engine
best_gazebo_new__384x640__fp16__trt1030__orin__3f9c1a2b7d.json
best_gazebo_new__448x640__fp16__trt1030__orin__3f9c1a2b7d.engine
best_gazebo_new__448x640__fp16__trt1030__orin__3f9c1a2b7d.json
```

Build for every packaged model at both shapes:

```bash
ros2 run vision_feedback build_engine.py --all --width 1280 --height 720
ros2 run vision_feedback build_engine.py --all --width 1024 --height 680
```

## C5. What happens if the shape does not match

Nothing breaks. The node tells you, and keeps going:

```text
[vision_node]: Vision received 1280x720, but expected_frame_* says 1024x680. Detection
still works - the letterbox adapts to any frame size - but the engine shape is no
longer ideal for this camera.
  current engine input : 640x448
  ideal for 1280x720   : 640x384
  fix: set expected_frame_width: 1280 / expected_frame_height: 720 in vision.yaml
       (or let mission.yaml supply them), then rebuild:
       ros2 run vision_feedback build_engine.py
Stabilizer history reset for the new geometry.
```

Cost of the mismatch, measured as padded pixels:

| Engine | Camera | Wasted compute |
|---|---|---:|
| 448x640 | 1280x720 | ~17 % |
| 384x640 | 1024x680 | ~10 % |

So you can fly a mismatched engine safely; just fix it when convenient.

## C6. Also worth adjusting per camera

`display_width` is the only other setting that is resolution-sensitive. The
resize is skipped when it would remove less than 15 % of the width, so on a
1024-wide camera the default of 960 does nothing:

| Camera width | `display_width` | Result |
|---:|---:|---|
| 1280 | 960 | resized to 960 (44 % fewer pixels) |
| 1024 | 960 | **skipped** — 0.94x is not worth a resize pass |
| 1024 | 640 | resized to 640 (61 % fewer pixels) |

For a 1024x680 camera with the display on, set:

```yaml
    display_width: 640
```

The geometry gates (`vertical_fov_deg`, `distance_to_target_m`,
`merge_center_delta_m`, `max_jump_m`, the area fractions) are all expressed as
*fractions of the frame* or in metres, so they carry across resolutions
unchanged. Only `vertical_fov_deg` needs to match the lens, and `mission.yaml`
already supplies it.

---

# Troubleshooting

### The log says `tensorrt/fp32` when I asked for fp16

Check the engine's own record:

```bash
cat "$ENGINE_CACHE_DIR"/*.json | python3 -m json.tool | grep -E "precision|fp16_flag|io_dtype"
```

| Output | Meaning |
|---|---|
| `"precision_built": "fp16", "fp16_flag_applied": true` | FP16 is working. If the node still reports `fp32` you are on a build older than v3, which derived the label from the engine's I/O dtype — and TensorRT keeps that FP32 by default even for an FP16 engine. Update and restart; no rebuild needed. |
| `"precision_built": "fp32", "fp16_flag_applied": false` | TensorRT reported no fast FP16 on this platform and genuinely built FP32. Confirm with `make gpu-check` (`TensorRT fast FP16: True` expected on any Orin). If it says `False`, the container is not seeing the real GPU. |
| No `precision_built` key at all | The engine predates v3. `make engine-clean` then rebuild. |

Note that `io_dtype: fp32` is normal and correct — it describes the network
boundary tensors, not the compute precision. To remove the boundary casts too,
rebuild with `--fp16-io` and launch with `trt_fp16_io:=true`.

### Inference time is 2-3x higher than expected

Almost always the Jetson power mode:

```bash
sudo nvpmodel -q
```

Anything other than MAXN runs the GPU well below its rated clock. Fix with:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Then re-measure:

```bash
ros2 run vision_feedback build_engine.py --verify
```

### I see `[TRT] [W] Using default stream in enqueueV3()`

You are running a build older than v3. That warning is not cosmetic — TensorRT
inserts extra `cudaStreamSynchronize()` calls around every inference because
the default stream has implicit cross-stream semantics. v3 gives the backend
its own CUDA stream. Update the package; no engine rebuild needed.

### Processed rate is about half the camera rate

Also a pre-v3 build. The old rate cap aliased: a 29.8 Hz camera against a 20 Hz
cap settled at 14.9 Hz because every second frame landed just short of the
deadline. v3's phase-tolerant cap gives 19.9 Hz from the same input. Confirm
from the stats line, which now shows both rates:

```text
tensorrt/fp16 640x448: 19.8 fps processed of 29.8 fps camera | ...
```

If you want every frame, set `max_inference_fps: 0`.

### The log says `torch/fp16` instead of `tensorrt/fp16`

The node fell back. The reason is on the line just above it, and also in
`/diagnostics` under `vision: model` as a `WARN`. Common causes:

| Log line | Fix |
|---|---|
| `TensorRT Python bindings not importable` | `sudo apt-get install python3-libnvinfer`, or you are not on the JetPack base image |
| `No cached engine at ... and auto_build_engine is disabled` | `make engine` |
| `Failed to deserialize ...` | The engine was built by a different TensorRT/GPU. `make engine-clean && make up` |
| `TensorRT returned no engine` | Not enough free GPU memory during the build. Stop other GPU work, or lower `trt_workspace_mib` |

### The display window never appears

```bash
docker exec -it vision_detection_jetson bash -lc 'echo $DISPLAY; xdpyinfo | head -3'
```

If `xdpyinfo` fails, run `xhost +local:docker` **in the Jetson desktop
session**, confirm `DISPLAY` in `.env` matches `echo $DISPLAY` there, and
`make recreate`.

`could not load the Qt platform plugin "xcb"` means the container is missing X
libraries — rebuild the image, the Dockerfile installs them.

### `No camera frames on '/camera/out/live_view' yet`

```bash
ros2 topic hz /camera/out/live_view
```

If that is empty, the problem is upstream (`lr1_camera`). If it shows only a
few Hz for a 15 Hz camera, you skipped the `sysctl` step in A0.

### `frames_dropped_before_inference` keeps climbing

Inference is slower than the camera. Check the `infer` figure in the 10-second
stats line. If it is high, confirm you are on `tensorrt`, then reduce
`max_inference_fps` or `inference_imgsz`.

### Detections got worse after the change

Compare against the old behaviour directly:

```bash
ros2 launch vision_feedback vision.launch.py \
    inference_input_shape:=square inference_imgsz:=960 inference_precision:=fp32
```

(rebuild the engine for that shape first with
`make engine` after editing `vision.yaml`, or pass `backend:=torch` to skip
TensorRT entirely). If that genuinely recovers detections you were losing, you
are detecting objects smaller than `head_min_area_fraction: 0.0025` — lower
that gate and keep the larger input shape. Section 6 of
`OPTIMIZATION_REPORT.md` covers this case.

### After a JetPack upgrade

```bash
make engine-clean
make rebuild
make up
```

Engines never survive a TensorRT version change.
