#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /ws_vision_detection/install/setup.bash ]; then
  source /ws_vision_detection/install/setup.bash
fi

# --------------------------------------------------------------------------
# Startup diagnostic. Intentionally non-fatal so AUTOSTART=0 can still be used
# for debugging even if CUDA or TensorRT is misconfigured.
# --------------------------------------------------------------------------
python3 - <<'PY' || true
import platform
print(f"[entrypoint] arch={platform.machine()}")
try:
    import torch
    print(f"[entrypoint] torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"[entrypoint] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[entrypoint] GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[entrypoint] WARNING: PyTorch/CUDA check failed: {exc}")
try:
    import tensorrt
    print(f"[entrypoint] TensorRT: {tensorrt.__version__}")
except Exception as exc:
    print(f"[entrypoint] WARNING: TensorRT bindings unavailable ({exc});"
          " the node will fall back to PyTorch and use far more CPU.")
PY

if [ "${SHOW_DISPLAY:-true}" = "true" ]; then
  echo "[entrypoint] SHOW_DISPLAY=true, DISPLAY=${DISPLAY:-<unset>}"
  if command -v xdpyinfo >/dev/null 2>&1; then
    if xdpyinfo >/dev/null 2>&1; then
      echo "[entrypoint] X server reachable on ${DISPLAY}"
    else
      echo "[entrypoint] WARNING: cannot reach the X server on '${DISPLAY}'."
      echo "[entrypoint]          On the Jetson desktop session run: xhost +local:docker"
      echo "[entrypoint]          and confirm DISPLAY in .env matches the active session"
      echo "[entrypoint]          (this machine runs xrdp on :10)."
    fi
  fi
fi

# --------------------------------------------------------------------------
# Pre-build the TensorRT engine.
#
# A TensorRT engine is tied to the exact GPU, TensorRT version and JetPack
# release, so it cannot be baked into the image; it is built once here and
# cached in the /engine_cache volume. Skipping this would simply move the
# 1-4 minute build into the first launch instead.
# --------------------------------------------------------------------------
if [ "${PREBUILD_ENGINE:-1}" = "1" ] && [ "${AUTOSTART:-1}" = "1" ]; then
  echo "[entrypoint] Ensuring the TensorRT engine cache is populated..."
  ros2 run vision_feedback build_engine.py \
      --cache-dir "${ENGINE_CACHE_DIR:-/engine_cache}" \
    || echo "[entrypoint] WARNING: engine build failed; the node will fall back to PyTorch."
fi

if [ "${AUTOSTART:-1}" != "1" ]; then
  echo "[entrypoint] AUTOSTART=0, idling. Use 'make run', 'make engine' or 'make shell'."
  exec sleep infinity
fi

exec "$@"
