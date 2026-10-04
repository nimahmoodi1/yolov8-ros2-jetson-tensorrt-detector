#!/usr/bin/env python3
"""Prove the container can actually use the Orin GPU through both stacks.

Checks PyTorch/CUDA (the fallback path) *and* TensorRT (the production path).
A missing TensorRT binding is the difference between ~6 ms and ~25 ms per
frame, so it is reported as a failure rather than a note.
"""

import platform
import sys

print("=== Jetson vision GPU check ===")
print("architecture:", platform.machine())
print("python:", sys.version.split()[0])

import numpy as np
print("numpy:", np.__version__)

import cv2
print("opencv:", cv2.__version__)

import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available inside the container")

print("GPU 0:", torch.cuda.get_device_name(0))
x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA tensor test:", tuple(y.shape), y.device, float(y[0, 0]))

# FP16 tensor-core path, which is what the FP16 engine relies on.
h = x.half()
z = h @ h
torch.cuda.synchronize()
print("CUDA FP16 tensor test:", tuple(z.shape), z.dtype, float(z[0, 0]))

import torchvision
print("torchvision:", torchvision.__version__)

from torchvision.ops import batched_nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
idxs = torch.tensor([0, 0], device="cuda")
kept = batched_nms(boxes, scores, idxs, 0.5)
print("GPU NMS test: kept", int(kept.numel()), "of 2 overlapping boxes")

trt_ok = False
try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print("TensorRT fast FP16:", builder.platform_has_fast_fp16)
    print("TensorRT fast INT8:", builder.platform_has_fast_int8)
    trt_ok = True
except Exception as exc:
    print(f"tensorrt: UNAVAILABLE ({exc})")

import ultralytics
print("ultralytics:", ultralytics.__version__)

# Power mode. An Orin in a 10W/15W profile runs the GPU at a fraction of its
# MAXN clock, which shows up as inference times 2-3x higher than expected with
# no other symptom.
print()
print("=== Jetson power mode ===")
try:
    with open("/etc/nvpmodel.conf") as handle:
        default = [ln for ln in handle if ln.startswith("< PM_CONFIG")]
    print("nvpmodel default:", default[0].strip() if default else "unknown")
except Exception:
    print("nvpmodel default: /etc/nvpmodel.conf not visible from the container")
try:
    with open("/sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq") as handle:
        print("GPU current clock:", int(handle.read().strip()) // 1000000, "MHz")
    with open("/sys/devices/platform/gpu.0/devfreq/17000000.gpu/max_freq") as handle:
        print("GPU max clock:    ", int(handle.read().strip()) // 1000000, "MHz")
except Exception:
    print("GPU clocks: not exposed to the container "
          "(run 'sudo nvpmodel -q' and 'sudo jetson_clocks --show' on the host)")

if not trt_ok:
    raise SystemExit(
        "ERROR: TensorRT Python bindings are missing. The node would silently fall "
        "back to PyTorch and use several times more CPU. On JetPack these come "
        "from python3-libnvinfer in the l4t-jetpack base image."
    )

print("PASS: Jetson GPU stack (PyTorch CUDA + FP16 + TensorRT) is operational")
