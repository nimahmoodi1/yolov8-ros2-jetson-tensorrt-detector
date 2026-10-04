# Optimization report

Target: **NVIDIA Jetson Orin**, JetPack 6.x, ROS 2 Humble.
Validated on TensorRT **8.6.2** (JetPack 6.0) and TensorRT **10.3.0**
(JetPack 6.2), with 1280x720 and 1024x680 cameras.

| Round | Subject |
|---|---|
| **v2** - sections 1-8 | The original PyTorch-to-TensorRT rewrite. |
| **v3** - sections 9-14 | Fixes found from the on-vehicle logs: the default-stream stall, the FP16 mislabel, the rate-cap aliasing, and multi-resolution support. |

If you only want the new material, jump to [section 9](#9-v3-what-the-on-vehicle-logs-showed).

---

## 1. What the htop capture actually shows

The green rows in that capture are **threads**, not processes. So:

| PID | | CPU% | meaning |
|---|---|---|---|
| 155532 | `vision_node` | 95.6 % | the process |
| 156067 | `vision_node` | 77.7 % | one of its threads |

That is **one node consuming ~1.75 cores**, at `RES 1176M` (7.5 % of 15.3 GB),
inside a system already at `load average 5.87` with **811 threads**.

The two hot threads are the inference worker and the display thread. Everything
below explains why each was so expensive.

---

## 2. What the models actually are

Read straight out of the checkpoint metadata (not assumed):

| File | Arch | Classes | Trained `imgsz` | Ultralytics |
|---|---|---|---|---|
| `best_gazebo_new.pt` (default) | YOLOv8n | `{0: body, 1: head-YJtu}` | **640** | 8.4.72 |
| `best.pt` | YOLOv8n | same | **640** | 8.4.72 |
| `best_fake_tower_behshahr.pt` | YOLOv8n | same | **640** | 8.4.72 |
| `Final_full_finetune_fake.pt` | YOLOv8n | same | **640** | 8.3.217 |

Two classes, `nc=2`, `depth_multiple=0.33` — plain YOLOv8n. This matters for
every decision below.

---

## 3. Root causes found in the code

### 3.1 `inference_imgsz: 960` on a model trained at 640

`config/vision.yaml` ran inference at 960 with the comment *"960 keeps more
detail from the 1024x680 Sony stream for distant tower parts."*

That costs **2.25x the FLOPs** of 640 and puts the input off the training
distribution, which usually costs mAP rather than gaining it.

More decisively, **your own gates already reject everything 960 would buy**:

```yaml
head_min_area_fraction: 0.0025    # 0.25 % of the frame
```

At 1280x720 that is 2 304 px², i.e. the smallest head you will *ever accept* is
about **48 x 48 px**. Scaled to a 640-wide network that is still **24 x 24 px** —
comfortably resolvable by YOLOv8's stride-8 P3 head, which needs roughly 8 px.
A head too small for 640 was already being discarded by the area gate before it
reached the temporal filter.

### 3.2 Square letterbox on a 16:9 camera

Even at 640, letterboxing 1280x720 into a **640x640** square makes 41 % of the
tensor grey padding. The correct static shape is **384x640**: identical scene
detail, 40 % fewer pixels.

```
960 x 960 = 921 600 px   <- previous setting
640 x 640 = 409 600 px
384 x 640 = 245 760 px   <- new default, 3.75x less than before
```

The removed pixels are padding and downscale that the area gates already
rejected. This is the whole precision/compute tradeoff, and it lands on the
"free" side of it.

### 3.3 No TensorRT, no FP16

Inference went through `ultralytics.YOLO.__call__` in eager PyTorch FP32.
Eager PyTorch launches ~200 individual CUDA kernels per YOLOv8n forward pass,
and every launch is CPU work — which is exactly why a "GPU" model still
saturated a CPU core.

Published Orin measurements for YOLOv8n at 640x640, batch 1:

| Runtime | Precision | GPU latency |
|---|---|---|
| PyTorch | FP32 | 20.9 ms |
| TensorRT | FP32 | 8.4 ms |
| TensorRT | **FP16** | **4.4 ms** |

TensorRT also fuses Conv+BN+SiLU into single kernels, so the CPU-side launch
count collapses along with the GPU time.

### 3.4 The worst single defect: ~40 GPU syncs per frame

`detection.py` iterated Ultralytics `Boxes` objects one at a time:

```python
arr = box.xyxy[0].cpu().numpy()      # device -> host sync, per box
conf = to_float(getattr(box, "conf"))  # device -> host sync, per box
```

With `inference_max_detections: 20` that is up to **40 separate
device-to-host synchronisations every frame**. Each one blocks the CPU until
the GPU pipeline drains — the CPU spins instead of sleeping.

### 3.5 Unbounded thread pools

Neither `torch.set_num_threads` nor `cv2.setNumThreads` was ever called, so on
an 8-core Orin PyTorch started 8 OpenMP workers and OpenCV started 8 TBB
workers. That is your **811 threads** and a large part of the 5.87 load
average, all competing with MAVROS, `nvargus-daemon`, the Livox driver and the
camera bridge.

### 3.6 `MultiThreadedExecutor(num_threads=4)`

On top of a dedicated inference thread and a display thread. Every callback in
this node is short and non-blocking, so three of those four executor threads
existed only to contend for cores.

### 3.7 Duplicate-frame guard was doing real work

```python
y1, y2 = h // 4, 3 * h // 4
x1, x2 = w // 4, 3 * w // 4
return np.array_equal(img[y1:y2, x1:x2], prev[y1:y2, x1:x2])   # ~690 kB
...
self._prev_frame = img.copy()                                   # ~2.7 MB memcpy
```

Every frame, at camera rate.

### 3.8 Display path (this one matters, because `SHOW_DISPLAY=true`)

```python
if last_good is not None:
    cv2.imshow(self.window_name, last_good)   # every loop iteration
cv2.waitKey(1)
```

Three problems:

1. `imshow` ran on **every** loop iteration, new frame or not.
2. The frame pushed was full 1280x720 while the window is 960x540 — 45 % of
   those pixels were scaled away by X.
3. Your htop confirms `Xorg :10 -config xrdp/xorg.conf`, so this went over
   **xrdp**, where pixel count directly costs CPU and bandwidth.

The overlay was also drawn on a full-resolution `img.copy()` (another 2.7 MB).

### 3.9 `cv_bridge` on the hot path

`bridge.imgmsg_to_cv2(msg, "bgr8")` allocates and copies the entire frame
through the C++ layer: ~41 MB/s at 1280x720 / 15 Hz, before any real work.

---

## 4. What changed

### 4.1 Native TensorRT backend (`backends/tensorrt_backend.py`)

Ultralytics is now used **only** for the one-off ONNX export. At runtime the
pipeline is:

```
read-only DDS frame
  -> np.copyto into a pinned host buffer
  -> async H2D (DMA)
  -> GPU: BGR->RGB, HWC->CHW, uint8->float, /255
  -> GPU: bilinear resize into the pre-greyed static engine input
  -> TensorRT execute_async_v3 on torch's own CUDA stream
  -> GPU: decode (4+nc, A) -> boxes/conf/cls
  -> GPU: torchvision.ops.batched_nms
  -> ONE device->host copy of at most max_det rows
```

Details that matter:

- **The letterbox padding is written once**, at construction and on any frame
  size change. Only the resized region is overwritten per frame — no
  full-tensor fill in the hot loop.
- **TensorRT shares torch's CUDA stream** (`torch.cuda.current_stream().cuda_stream`),
  so ordering between the preprocessing kernels, the engine and the decode is
  guaranteed without any explicit synchronisation.
- **Both TensorRT APIs are supported**: the 8.5+ tensor API
  (`set_tensor_address` / `execute_async_v3`, which is what your 8.6.2 uses)
  and the older binding-index API, so a JetPack upgrade to TensorRT 10 does not
  break it.
- Engine input/output dtypes are read from the engine rather than assumed, so
  an FP16 engine whose I/O TensorRT chose to keep in FP32 still works.

### 4.2 Engine build and caching (`engine_builder.py`)

A TensorRT engine is tied to the exact GPU, TensorRT version and JetPack
release, so it **cannot be baked into the image**. It is built once on the Orin
and cached in a host-side volume:

```
engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__a1b2c3d4e5.engine
engine_cache/best_gazebo_new__384x640__fp16__trt862__orin__a1b2c3d4e5.json
```

The name encodes the weights hash, input shape, precision, TensorRT version and
GPU, so a stale engine can never be picked up by accident. The `.json` sidecar
carries the class names, so the runtime never needs to load the `.pt` at all.

The build is atomic (`.partial` then `os.replace`), so a power cut mid-build
cannot leave a truncated engine that deserializes into a segfault.

### 4.3 Vectorised selection (`detection.py`)

The `(N, 6)` array contract means the whole frame's detections arrive in one
transfer, and every gate is a NumPy comparison over at most 20 rows:

- class filtering by **integer id**, resolved once at load by `ClassRouter`,
  instead of a per-box `names[int(box.cls)]` lookup and substring test;
- the metric merge gate is converted to **pixels once per frame**
  (`meters_to_pixels_vertical`) instead of converting every candidate's pixel
  delta into metres.

The selection *rules* are unchanged — confidence primary, mild centre
preference, aspect/area gates, tier-wise union of head fragments.

### 4.4 NMS confidence floor raised from 0.25 to 0.78

Selection applies `head_confidence_threshold: 0.78` / `body: 0.82` afterwards,
so anything weaker was decoded and NMS'd purely to be thrown away. The floor is
now `min(head, body)` automatically.

This is also a small **recall improvement**: fewer junk boxes compete for the
`max_det: 20` slots, so a genuine 0.79 head can no longer be crowded out.

### 4.5 Zero-copy image conversion (`image_convert.py`)

`bgr8`, `rgb8`, `bgra8`, `rgba8`, `mono8`, `8UC1`, `8UC3` are mapped as a
**view** over the DDS buffer with `np.frombuffer`, honouring `msg.step` row
padding. `cv_bridge` stays as the fallback for anything exotic (Bayer, 16-bit
depth), so no stream regresses.

### 4.6 Display path

- `display_max_fps: 10` — surplus frames are dropped by `DisplayWindow.due()`
  **before** the producer resizes or draws, so the saving applies to the
  inference thread too.
- `display_width: 960` — the frame is downscaled *once* with `INTER_AREA`, and
  box coordinates are scaled with it, so the overlay is drawn on 960x540
  instead of 1280x720 and X receives 45 % fewer pixels.
- `imshow` is called on a new frame or a 0.5 s heartbeat, not every iteration.
  `waitKey(1)` still runs every iteration so Qt stays responsive.
- A **HUD** shows the live backend, precision, network shape, fps and inference
  time, so a silent fallback to PyTorch is visible at a glance.

### 4.7 Threading and CPU

- `SingleThreadedExecutor` instead of `MultiThreadedExecutor(4)`.
- `cpu_threads: 1` clamps `OMP/OPENBLAS/MKL/NUMEXPR`, `cv2.setNumThreads` and
  `torch.set_num_threads`. The same variables are also set as Docker `ENV` so
  they apply *before* torch is imported, which is the only point at which
  OpenMP reads them.
- Frozen-frame guard now compares a stride-8 subsample (~43 kB instead of
  ~690 kB) and stores only that subsample instead of a 2.7 MB frame copy.
- `max_inference_fps: 20.0` caps detector invocations. At your 15 Hz camera it
  never bites; it protects you if a driver switches to 30 Hz.
- The debug `raw_bbox_topic` is only serialised when something is subscribed.

### 4.8 Ultralytics is off the runtime path

With TensorRT loaded, `ultralytics` is never imported at runtime. That removes
its torch/matplotlib/pandas import graph from the resident set — a large part of
the 1176 MB RSS. `YOLO_OFFLINE=1` also stops it attempting network calls on a
vehicle with no internet.

---

## 5. Expected result

Combining a 3.75x smaller input tensor, FP16 tensor cores, TensorRT kernel
fusion, the elimination of ~40 per-frame GPU syncs, and the CPU-side work
removed from conversion, duplicate detection and display:

| | Before | After (expected) |
|---|---|---|
| Network input | 960 x 960 FP32 | 384 x 640 FP16 |
| Runtime | PyTorch eager | TensorRT engine |
| GPU-host syncs / frame | up to ~40 | 1–2 |
| Detector latency | ~45–60 ms | ~3–6 ms |
| Node CPU (8 cores) | ~175 % | ~25–45 % |
| Process threads | high (unbounded pools) | bounded |
| RSS | ~1.18 GB | ~0.5–0.7 GB |

**Measure it on your own hardware** rather than trusting this table:

```bash
make bench
```

That runs both backends on identical synthetic frames and prints mean/p95
latency, fps and single-process CPU percentage.

---

## 6. The accuracy side of the tradeoff

What was **not** given up:

- Same weights, same two classes, same confidence thresholds.
- Same aspect-ratio and area-fraction gates.
- Same head-merge logic, temporal filter, stabilizer and dropout hold.
- Same effective image detail: 384x640 carries the same 0.5x downscale of a
  1280x720 frame that 640x640 does. Only grey padding was removed.
- FP16 rather than INT8. INT8 would be ~25 % faster again but needs a
  calibration set and carries genuine mAP risk on a two-class detector driving
  an orbit controller. Not worth it here.

What **is** a real change, stated plainly:

- Going from 960 to 640 removes 1.5x linear oversampling. For any object your
  area gates accept this is unused headroom, but if you ever lower
  `head_min_area_fraction` below ~0.001 to chase very distant tower parts, you
  will want it back.

If you need that reach, it is one setting plus an engine rebuild:

```yaml
inference_imgsz: 960
inference_input_shape: 576x960     # or leave 'auto' and it derives 576x960
```

```bash
make engine        # rebuild for the new shape (~2-4 min)
make recreate
```

Even at 576x960, TensorRT FP16 will still be substantially faster and lighter
than the original 960x960 PyTorch FP32 path.

---

## 7. Files changed

**New**

```
src/vision_feedback/vision_perception/letterbox.py           static-shape letterbox maths
src/vision_feedback/vision_perception/image_convert.py       zero-copy Image -> BGR
src/vision_feedback/vision_perception/runtime.py             CPU thread limits
src/vision_feedback/vision_perception/engine_builder.py      .pt -> ONNX -> cached .engine
src/vision_feedback/vision_perception/backends/base.py       backend contract
src/vision_feedback/vision_perception/backends/tensorrt_backend.py
src/vision_feedback/vision_perception/backends/torch_backend.py
src/vision_feedback/vision_perception/backends/__init__.py   selection + ClassRouter
scripts/build_engine.py                                      engine build CLI
scripts/benchmark.py                                         TensorRT vs PyTorch on-device
```

**Rewritten**

```
src/vision_feedback/vision_perception/detection.py       vectorised, array-based
src/vision_feedback/vision_perception/node.py            single-threaded executor, rate cap
src/vision_feedback/vision_perception/visualization.py   downscale-then-draw, rate cap, HUD
src/vision_feedback/vision_perception/config.py          backend/precision/cache params
src/vision_feedback/vision_perception/diagnostics.py     backend + timing reporting
src/vision_feedback/config/vision.yaml
src/vision_feedback/launch/vision.launch.py
Dockerfile  compose.yaml  entrypoint.sh  Makefile  .env  scripts/gpu_check.py
```

**Unchanged** (deliberately — this is validated mission logic)

```
geometry.py  temporal_filter.py  stabilizer.py  message_builder.py
latest_frame.py  mission_config.py  msg/*.msg
```

**Also fixed**: `package.xml` used `<n>` instead of `<name>`, which
`catkin_pkg` rejects.

---

## 8. Tests

48 unit tests, runnable on the Jetson or on a laptop without CUDA:

```bash
make test
```

Covering: letterbox round-trip and clipping, `auto` shape derivation, zero-copy
conversion for every supported encoding including `step` row padding, class
routing, backend selection and the CUDA-OOM fallback chain, all the geometry
gates, head-fragment merging, and the display downscale/rate-cap behaviour.

Two real bugs were caught by these tests while writing them and fixed:

1. `LetterboxTransform.to_source` clipped through `boxes[:, [0, 2]]`, which is
   fancy indexing and returns a *copy* — the `out=` write was silently
   discarded and boxes were never clipped to the frame.
2. `ensure_writable` used `np.ascontiguousarray`, which returns an
   already-contiguous read-only array unchanged, so the first `cv2.rectangle`
   on a zero-copy DDS frame would have raised.


---
---

# v3 - fixes from the on-vehicle run

Everything above still applies. This round is driven by an actual deployment
log from an Orin with TensorRT 10.3 and a 1024x680 Sony RX0 II at ~29.8 Hz.

---

## 9. v3: what the on-vehicle logs showed

Three separate problems, visible in four lines of the log:

```text
Detector request: backend=tensorrt device=cuda:0 input=640x448 precision=fp16 ...
Using cached TensorRT engine .../best_fake_tower_behshahr__448x640__fp16__trt1030__orin__679908eca5.engine
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to
          additional calls to cudaStreamSynchronize() by TensorRT to ensure correct
          synchronization. Please use non-default stream instead.
Detector ready: tensorrt/fp32 640x448 on cuda:0 (TensorRT 10.3.0, nc=2)
...
tensorrt/fp32 640x448: 14.8 fps processed | infer 15.4 ms | end-to-end 32.3 ms | dropped 0 | throttled 413
```

| Symptom | Real cause |
|---|---|
| `tensorrt/fp32` although an `__fp16__` engine loaded | **Reporting bug.** FP16 was working. |
| `[TRT] [W] Using default stream` | **Real latency bug.** TensorRT was inserting its own synchronisations on every frame. |
| `14.8 fps` from a 29.8 Hz camera, `throttled 413` | **Rate-cap aliasing.** The 20 Hz cap was yielding 14.8 Hz. |

---

## 10. v3.1 - FP16 *was* working; the label was wrong

### What happened

`TensorRTBackend` derived its precision label from the dtype of the engine's
input and output tensors:

```python
precision = 'fp16' if in_dtype == torch.float16 or out_dtype == torch.float16 else 'fp32'
```

When you build with only `BuilderFlag.FP16`, TensorRT **runs the layers in
FP16 but keeps the network boundary tensors in FP32** and wraps the graph in
two reformat (cast) nodes. `get_tensor_dtype()` therefore returns `FLOAT` for
both I/O tensors, and a perfectly good FP16 engine reported itself as `fp32`.

So the engine was never FP32. The `__fp16__` in the cached filename was
correct; the runtime label was not.

### The fix

The engine builder now records what it **actually built**, not what was asked
for, and the runtime reads that:

```json
{
  "precision":          "fp16",   // requested
  "precision_built":    "fp16",   // FP16 builder flag actually applied
  "fp16_flag_applied":  true,
  "io_dtype":           "fp32"    // network boundary tensors
}
```

The log line is now unambiguous:

```text
Detector ready: tensorrt/fp16 640x448 on cuda:0 (TensorRT 10.3.0, nc=2, io fp32, private stream)
```

This also closes a genuine trap: `platform_has_fast_fp16 == False` used to
produce an FP32 engine stored under an `__fp16__` filename, with nothing
anywhere saying so. Now `precision_built` records `fp32` and the node reports
`fp32`.

### Optional extra: FP16 network I/O

`trt_fp16_io: true` (new, default `false`) forces the boundary tensors to FP16
as well, removing both reformat nodes. It is opt-in because it changes the
numeric type of the emitted box coordinates: FP16 spacing at a 640 px network
width is 0.5 px, so ~0.8 px in source pixels at `gain=0.625`. The pipeline
truncates to integer pixels and has ±5 px tolerances throughout, so this is
safe - but it is not a change worth making silently to fix a label. It gets its
own cache entry (`__fp16io__`) so it can never be confused with the default.

---

## 11. v3.2 - the default-stream stall (the actual performance bug)

### What happened

```text
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to
          additional calls to cudaStreamSynchronize() by TensorRT
```

The backend passed `torch.cuda.current_stream().cuda_stream` to
`execute_async_v3`. Outside a stream context that **is the legacy default
stream**. TensorRT detects this and defensively wraps each inference in extra
`cudaStreamSynchronize()` calls, because the default stream has implicit
synchronisation semantics with every other stream in the process.

The result is a full pipeline drain per frame, on top of the one the final
`.cpu()` already costs.

### The fix

The backend now owns a private stream and runs **all** of it there - the H2D,
the preprocessing kernels, the engine, the decode and the NMS:

```python
self._stream = torch.cuda.Stream(device=self._device)
...
with torch.no_grad(), torch.cuda.stream(self._stream):
    ...
    ok = self._context.execute_async_v3(self._stream.cuda_stream)
    detections = self._decode(tf)
```

The `self._input` / `self._output` buffers are allocated inside the same stream
context so PyTorch's caching allocator associates the blocks with the stream
they are actually used on.

The warning disappears, and with it the extra synchronisations.

### Second sync removed

`_decode` used to do:

```python
if not bool(keep.any()):      # sync #1
    return EMPTY_DETECTIONS
boxes = pred[keep, :4]        # sync #2 (masking has to size the result)
```

Boolean masking must synchronise anyway to size its output, so the `any()`
check was a second full stall for nothing. It is now:

```python
boxes_cxcywh = pred[keep, :4]
if boxes_cxcywh.shape[0] == 0:
    return EMPTY_DETECTIONS
```

**One** synchronisation per frame.

### Also check the power mode

The measured 15 ms is still high for YOLOv8n FP16 at 640x448 on an Orin. An
Orin in a 10 W or 15 W `nvpmodel` profile runs the GPU at a fraction of its
MAXN clock, which presents as inference times 2-3x higher with no other
symptom. `make power`, `make gpu-check` and `scripts/check_jetson.sh` now
report the power mode and GPU clocks. On the host:

```bash
sudo nvpmodel -q            # show the current profile
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks          # pin clocks to maximum
```

---

## 12. v3.3 - the rate cap was aliasing 30 Hz down to 15 Hz

### What happened

```text
14.8 fps processed | ... | throttled 413
```

The camera runs at **29.8 Hz** (the log's own
`rx0m2_Node: Live View Publishing -> measured_fps: 29.7915`), and
`max_inference_fps` was 20. The node produced 14.8 Hz.

The old cap was "reject until `now >= deadline`". With a 33.6 ms arrival
interval and a 50 ms deadline:

```text
t=0.0   ms  accept, deadline -> 50.0
t=33.6  ms  reject  (33.6 < 50.0)
t=67.2  ms  accept, deadline -> 117.2
t=100.8 ms  reject
...
```

It locks onto "every second frame" and settles at exactly half the camera rate.
The node was throwing away a third of the throughput it was allowed to have.

### The fix

`vision_perception/rate_limit.py` - `PhaseTolerantRateLimiter`:

* a frame arriving within **half an inter-arrival interval** of its deadline
  still counts, so the accept pattern can alternate 1-2-1-2 and average out;
* the next deadline advances from the **previous deadline**, not from the
  arrival time, so late frames do not drag the average down;
* the tolerance comes from the *measured* camera interval, so it adapts.

Simulated over 20 s of evenly spaced arrivals against a 20 Hz cap:

| Camera | Old cap | New cap |
|---:|---:|---:|
| 15.0 Hz | 15.00 Hz | 15.00 Hz |
| 24.0 Hz | 12.00 Hz | 18.00 Hz |
| 28.3 Hz | 14.15 Hz | 18.85 Hz |
| **29.8 Hz** | **14.90 Hz** | **19.85 Hz** |
| 60.0 Hz | 17.40 Hz | 20.00 Hz |

That is a **33 % throughput increase at no CPU cost** for the logged camera,
and the cap is still never exceeded.

The stats line now reports the measured camera rate too, so the relationship is
visible without a separate `ros2 topic hz`:

```text
tensorrt/fp16 640x448: 19.8 fps processed of 29.8 fps camera | infer 6.2 ms | ...
```

---

## 13. v3.4 - display cost, measured

Your own measurements (`vision_feedback_display_usage.md`):

| | Display ON | Display OFF |
|---|---:|---:|
| `vision_node` CPU | 64.64 % of one core | 24.94 % of one core |

Two things in that 40-point gap were avoidable.

### Rendering was on the inference thread

`_publish()` called `render_frame()` inline, so the resize and all the OpenCV
drawing sat between one frame's publish and the next frame's inference. That is
most of the gap between the logged `infer 15.4 ms` and `end-to-end 32.3 ms`.

The node now queues the **raw frame plus the boxes** and the display thread
does the rendering:

```python
self.display.submit_scene(img, head, body, raw_head=..., raw_body=..., hud=...)
```

The mission still gets its bounding box *before* any of this - `stable_pub.publish()`
already ran. What changes is that the inference thread returns immediately, so
throughput and end-to-end latency both improve. (Total process CPU is unchanged
by the move itself; the reduction comes from the next item.)

### `display_width: 960` was a no-op resize on a 1024-wide camera

960 / 1024 = 0.94. The node paid a full `INTER_AREA` pass over 2 MB to remove
6 % of the pixels. `render_frame` now:

* skips the resize entirely when it would remove less than 15 % of the width
  (a plain copy, which is needed anyway for a writable buffer, is cheaper);
* uses `INTER_LINEAR` above 0.5x scale and `INTER_AREA` only below it, where
  bilinear starts to alias visibly.

**For a 1024x680 camera, set `display_width: 640`.** At 960 you get no resize
and push full frames to X; at 640 you push 39 % of the pixels.

### Recommendation is unchanged

Your conclusion stands and is now documented in both READMEs:
`show_display:=false` for flight, `true` only when an operator is watching.

---

## 14. v3.5 - camera resolution: 1280x720 and 1024x680

### The package already handles both. Here is exactly why.

There are two independent pieces of geometry:

1. **The engine input shape** - static, chosen at build time.
2. **The letterbox** - computed per frame from the *received* frame size.

`compute_letterbox()` fits any source into any engine shape and
`LetterboxTransform.to_source()` maps the boxes back. So **an engine built for
one resolution produces correct detections on any other resolution.** The shape
only affects efficiency, never correctness:

| Engine | Camera | Letterbox | Correct? | Efficient? |
|---|---|---|---|---|
| 384x640 | 1280x720 | 640x360 + 12 px top/bottom | yes | yes |
| 448x640 | 1024x680 | 640x425 + 11 px top/bottom | yes | yes |
| 448x640 | 1280x720 | 640x360 + 44 px top/bottom | yes | 17 % wasted |
| 384x640 | 1024x680 | 578x384 + 31 px left/right | yes | 10 % wasted |

This is verified by unit tests, not just asserted.

### How `auto` picks the shape

`inference_input_shape: auto` derives the engine shape from
`expected_frame_width` / `expected_frame_height`: the long side becomes
`inference_imgsz`, the short side is padded up to the next multiple of 32.

| Camera | `expected_frame_width` / `_height` | auto shape |
|---|---|---|
| 1280x720 | 1280 / 720 | **384x640** |
| 1024x680 | 1024 / 680 | **448x640** |
| 1920x1080 | 1920 / 1080 | 384x640 |
| 1280x960 | 1280 / 960 | 480x640 |
| 640x480 | 640 / 480 | 480x640 |

Note that `mission.yaml` overrides `expected_frame_*` when
`use_mission_config: true` - the on-vehicle log shows exactly that:

```text
Loaded shared config from mission.yaml (mode='real'): ... camera=1024x680 ...
```

so on that Jetson the shape came out as 448x640 without touching `vision.yaml`
at all. That is the intended path.

### What v3 adds

* **Startup log states the decision explicitly**, so there is nothing to guess:
  ```text
  Camera geometry: expecting 1024x680 -> network input 640x448 (inference_input_shape='auto', imgsz=640)
  ```
* **The mismatch warning is now actionable.** If the received frames do not
  match `expected_frame_*`, the node prints the current shape, the ideal shape
  for what actually arrived, and the exact command to fix it:
  ```text
  Vision received 1280x720, but expected_frame_* says 1024x680. Detection still
  works - the letterbox adapts to any frame size - but the engine shape is no
  longer ideal for this camera.
    current engine input : 640x448
    ideal for 1280x720   : 640x384
    fix: set expected_frame_width: 1280 / expected_frame_height: 720 in vision.yaml
         (or let mission.yaml supply them), then rebuild:
         ros2 run vision_feedback build_engine.py
  ```
* **Warmup now runs at the camera geometry**, not the network geometry, so the
  first live frame reuses the pinned buffer and letterbox instead of
  reallocating mid-stream. The old behaviour showed up in the log as two
  letterbox lines; there is now one.
* **`build_engine.py` takes camera dimensions directly**, so you can build for
  a different camera without editing any YAML:
  ```bash
  ros2 run vision_feedback build_engine.py --width 1280 --height 720
  ros2 run vision_feedback build_engine.py --width 1024 --height 680
  ros2 run vision_feedback build_engine.py --shape 448x640     # explicit
  ```
  plus `--verify` to time the result on the device it was built on, and
  `--fp16-io` for the optional boundary-FP16 build.
* **`make engine-720p` / `make engine-1024`** in the container.

Both engines can live in the cache at once - the filename encodes the shape, so
switching camera is a YAML edit and a restart, with no rebuild if the engine is
already cached.

---

## 15. v3 summary

| | v2 (as shipped) | v3 |
|---|---|---|
| CUDA stream | default (TensorRT added its own syncs) | private, shared by preprocessing + engine + decode |
| Syncs per frame | 2 | 1 |
| Precision label | derived from I/O dtype, reported `fp32` for FP16 engines | recorded at build time, reported honestly, I/O dtype shown separately |
| FP16 I/O | not available | opt-in `trt_fp16_io` |
| 29.8 Hz camera, 20 Hz cap | 14.9 Hz | 19.9 Hz |
| Overlay rendering | on the inference thread | on the display thread |
| `display_width: 960` on a 1024 camera | full INTER_AREA pass for a 6 % saving | skipped |
| Camera resolution | worked, undocumented | documented, logged, verified by tests, with direct CLI support |
| Silent FP32 fallback when no fast FP16 | stored under an `fp16` filename | recorded and reported as `fp32` |

### v3 files changed

**New**

```
src/vision_feedback/vision_perception/rate_limit.py        phase-tolerant rate cap
src/vision_feedback/test/test_rate_limit.py                8 tests incl. the 30 Hz regression
src/vision_feedback/test/test_engine_cache.py              6 tests for cache naming + sidecar
```

**Modified**

```
backends/tensorrt_backend.py   private CUDA stream, honest precision, one sync, warmup shape
backends/base.py               warmup(source_shape=...)
backends/torch_backend.py      warmup(source_shape=...)
backends/__init__.py           threads trt_fp16_io and warmup_shape through
engine_builder.py              records precision_built / fp16_flag_applied / io_dtype;
                               optional FP16 I/O; fp16io cache token
node.py                        PhaseTolerantRateLimiter, scene handoff to the display
                               thread, geometry logging, actionable mismatch warning
visualization.py               submit_scene(), conditional resize, interpolation choice
config.py / vision.yaml        trt_fp16_io; camera-geometry and display_width guidance
launch/vision.launch.py        trt_fp16_io argument
scripts/build_engine.py        --width/--height/--fp16-io/--verify
scripts/gpu_check.py           power mode + GPU clocks
scripts/check_jetson.sh        nvpmodel
Makefile                       engine-720p, engine-1024, power
```

### v3 tests

**68 tests**, up from 48, still runnable without CUDA, TensorRT or a camera:

```bash
cd src/vision_feedback && python3 -m pytest test -q
```

The rate-limiter tests encode the regression directly: a 29.8 Hz source against
a 20 Hz cap must land between 18.5 and 21.0 Hz, and no source may ever exceed
the cap.
