#!/usr/bin/env bash
set -uo pipefail

echo "=== /etc/nv_tegra_release (container, if exposed) ==="
cat /etc/nv_tegra_release 2>/dev/null || true

echo
echo "=== NVIDIA device nodes ==="
ls -l /dev/nvhost-gpu /dev/nvhost-ctrl-gpu /dev/nvmap 2>/dev/null || true

echo
echo "=== GPU stack (PyTorch CUDA + TensorRT) ==="
python3 /ws_vision_detection/scripts/gpu_check.py

echo
echo "=== TensorRT engine cache (${ENGINE_CACHE_DIR:-/engine_cache}) ==="
ls -lh "${ENGINE_CACHE_DIR:-/engine_cache}" 2>/dev/null || echo "cache directory missing"

echo
echo "=== Display ==="
echo "DISPLAY=${DISPLAY:-<unset>}  SHOW_DISPLAY=${SHOW_DISPLAY:-<unset>}"
xdpyinfo 2>/dev/null | head -4 || echo "X server not reachable (run 'xhost +local:docker' on the desktop)"

echo
echo "=== Power mode (low profiles cost 2-3x inference time) ==="
nvpmodel -q 2>/dev/null || echo "nvpmodel not reachable from the container; run 'sudo nvpmodel -q' on the host"

echo
echo "=== CPU thread limits ==="
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"

echo
echo "=== ROS camera topic ==="
source /opt/ros/humble/setup.bash
source /ws_vision_detection/install/setup.bash
ros2 topic info "${IMAGE_TOPIC:-/camera/out/live_view}" --verbose || true
