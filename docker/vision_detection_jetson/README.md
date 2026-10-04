# vision_detection_jetson

Jetson-native ROS 2 Humble tower head/body detector for the Orin NX 16 GB.
Runs **YOLOv8n through a cached TensorRT FP16 engine** with GPU preprocessing
and GPU NMS, and falls back to PyTorch only if TensorRT is unavailable.

See [`OPTIMIZATION_REPORT.md`](../../docs/OPTIMIZATION_REPORT.md) for what changed versus
the previous PyTorch-only version and why.

---

## Target platform

| Component | Version |
|---|---|
| Board | CTI Hadron + NVIDIA Jetson Orin NX 16 GB, `aarch64` |
| OS | Ubuntu 22.04 / Jetson Linux (L4T) **R36.3.0** |
| CUDA | 12.2 |
| cuDNN | 8.9.4 |
| TensorRT | **8.6.2** |
| PyTorch | 2.3.0 (NVIDIA aarch64 wheel) |
| Ultralytics | 8.4.72 (export only) |

Base image: `nvcr.io/nvidia/l4t-jetpack:r36.3.0`.
**Do not** replace the NVIDIA aarch64 torch wheels with the normal x86 PyPI
`torch`, and **do not** `pip install tensorrt` — TensorRT comes from the
JetPack base image.

---

## Quick start on the Jetson

```bash
cd vision_detection_jetson
make env          # writes .env with your UID/GID and DISPLAY
make pull         # ~14 GB JetPack base image, first time only
make build        # ~15-30 min first time
make up           # starts the detector; builds the engine on first run
make logs
```

Step-by-step commands with the exact output to expect are in
[`RUNBOOK.md`](../../docs/RUNBOOK.md).

---

## What `make up` runs

```text
backend:=tensorrt
device:=cuda:0
inference_precision:=fp16
inference_input_shape:=auto        # 1280x720 -> 384x640
image_topic:=/camera/out/live_view
show_display:=true
```

Output topics are unchanged:

```text
/art/bounding_boxes         stable boxes, consumed by the mission
/art/bounding_boxes_raw     post-filter debug boxes (published only when subscribed)
/diagnostics                camera / model / output health
```

---

## The TensorRT engine cache

A TensorRT engine is tied to the exact GPU, TensorRT version and JetPack
release, so it cannot ship inside the image. It is built **once on this
Jetson** and cached on the host:

```text
./engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__<hash>.engine
./engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__<hash>.json
```

The first `make up` builds it (1-4 minutes for YOLOv8n) and every later start
loads it instantly. The filename encodes the weights hash, shape, precision,
TensorRT version and GPU, so a stale engine can never be reused by accident.

```bash
make engine        # build/refresh for the active model
make engine-all    # build for every packaged model
make engine-clean  # delete the cache and force a rebuild
```

Rebuild after: changing the model, changing `inference_imgsz` or
`inference_input_shape`, changing precision, or a JetPack/TensorRT upgrade.

---

## Display costs CPU — turn it off for flight

Measured on an Orin with a 1024x680 camera and the same TensorRT engine:

| | `vision_node` CPU |
|---|---:|
| `show_display:=true` | 64.6 % of one core |
| `show_display:=false` | 24.9 % of one core |

Detection is identical either way. Use `SHOW_DISPLAY=false` in `.env` for
flight and turn it on only when an operator is actually watching.

---

## Proving it is really on the GPU

```bash
make gpu-check
```

must end with:

```text
PASS: Jetson GPU stack (PyTorch CUDA + FP16 + TensorRT) is operational
```

It deliberately **fails** if the TensorRT bindings are missing, because the
node would otherwise silently fall back to PyTorch and use several times more
CPU. The same downgrade also shows up as a `WARN` on `/diagnostics` and in the
on-screen HUD. It also reports the Jetson power mode and GPU clocks — a low
`nvpmodel` profile is the most common cause of inference times 2-3x higher than
expected, with no other symptom:

```bash
make power                 # show the profile and clocks
sudo nvpmodel -m 0         # MAXN
sudo jetson_clocks         # pin clocks to maximum
```

Measure the difference on your own board:

```bash
make bench
```

---

## Display (`SHOW_DISPLAY=true`)

This is a supported, tuned mode — not an afterthought.

- The overlay is drawn on a **downscaled** frame (`display_width: 960`), not on
  the full 1280x720 one.
- Rendering is capped at `display_max_fps: 10` and surplus frames are dropped
  *before* the resize and the drawing.
- `imshow` runs on a new frame or a 0.5 s heartbeat, not on every loop.

Your Jetson serves its desktop on `:10` via xrdp, so `.env` defaults to
`DISPLAY=:10`. `make up` runs `make xauth` first, which prepares
`/tmp/.docker.xauth` and runs `xhost +local:docker`.

If the window does not appear, run **on the Jetson desktop session**:

```bash
echo $DISPLAY              # confirm which display your session is on
xhost +local:docker
```

then set that value in `.env` and `make recreate`.

To run headless, set `SHOW_DISPLAY=false` in `.env` and `make recreate`.

---

## Camera resolution (1280x720, 1024x680, anything else)

**Detection works at any incoming resolution regardless of configuration** —
the letterbox adapts per frame. The engine shape only affects efficiency.

`inference_input_shape: auto` derives the shape from `expected_frame_width` /
`expected_frame_height` (which `mission.yaml` supplies when
`use_mission_config: true`):

| Camera | auto shape | Build with |
|---|---|---|
| 1280x720 | 384x640 | `make engine-720p` |
| 1024x680 | 448x640 | `make engine-1024` |
| 1920x1080 | 384x640 | `make engine` |
| 640x480 | 480x640 | `make engine` |

To switch camera, set the two `expected_frame_*` lines in `vision.yaml` (or let
`mission.yaml` do it) and rebuild the engine. Both engines can live in the
cache at once; the filename encodes the shape. Full walkthrough: **Part C** of
[`RUNBOOK.md`](../../docs/RUNBOOK.md).

On a 1024-wide camera also set `display_width: 640` — the default 960 is a
0.94x "resize" that the renderer skips, so you gain nothing from it.

---

## Tuning the precision / compute tradeoff

Everything lives in `src/vision_feedback/config/vision.yaml`.

| Setting | Default | Effect |
|---|---|---|
| `inference_imgsz` | `640` | Long side. Matches how every packaged model was trained. |
| `inference_input_shape` | `auto` | Derived from `expected_frame_*`. `square` gives `640x640`. `HxW` is explicit. |
| `inference_precision` | `fp16` | `fp32` for validating a new model. |
| `trt_fp16_io` | `false` | Also make the engine's I/O tensors FP16. Removes two cast nodes; costs ~0.5 px box quantisation. Needs an engine rebuild. |
| `max_inference_fps` | `20.0` | Hard cap on detector invocations, phase tolerant so a 30 Hz camera lands near 20 Hz rather than 15. `0` disables. |
| `display_max_fps` | `10.0` | Overlay render rate. |
| `display_width` | `960` | Overlay/annotation width. Set `640` for a 1024-wide camera. |
| `cpu_threads` | `1` | Clamps torch/OpenCV/BLAS pools. `0` restores library defaults. |
| `backend` | `auto` | `tensorrt`, `torch`, or `auto`. |

More reach at long standoff (rebuild the engine afterwards):

```yaml
inference_imgsz: 960
inference_input_shape: 576x960
```

```bash
make engine && make recreate
```

---

## Isolation from `lr1_camera`

The two containers stay separate:

| | `lr1_camera` | this detector |
|---|---|---|
| workspace | `/ws` | `/ws_vision_detection` |
| home | | `/home/vision_detection` |
| container | | `vision_detection_jetson` |
| image | | `vision_detection_jetson:latest` |

They share only host networking and host IPC so ROS 2 DDS can move the camera
images between them efficiently. Their filesystems do not overlap.

---

## Compatibility notes

1. **Build on the ARM64 Jetson.** `platform: linux/arm64` is intentional.
2. **Keep R36.3 with the R36.3 base image.** If the Jetson moves to another
   JetPack/L4T release, update the base image *and* the PyTorch wheels
   together, then `make engine-clean` — the cached engine will no longer match
   the new TensorRT.
3. `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all` and
   `NVIDIA_DRIVER_CAPABILITIES=all` are intentional and match `lr1_camera`.
4. Both containers use `network_mode: host`, `ipc: host`, ROS domain 0 and
   Fast DDS. On the host, raise the socket buffers or DDS will drop image
   fragments:
   ```bash
   sudo sysctl -w net.core.rmem_max=16777216
   sudo sysctl -w net.core.wmem_max=16777216
   ```
5. The `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` variables in `compose.yaml`
   are load-bearing: OpenMP reads them at import time, so setting them only
   inside the node would be too late.
6. TensorRT **8.6** (JetPack 6.0) and **10.3** (JetPack 6.2) are both supported
   — the backend handles the old binding-index API and the newer tensor API.
   Engines never transfer between them; `make engine-clean` after any upgrade.
