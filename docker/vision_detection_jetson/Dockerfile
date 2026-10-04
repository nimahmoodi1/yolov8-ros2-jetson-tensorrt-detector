# syntax=docker/dockerfile:1
#
# vision_detection for NVIDIA Jetson Orin NX 16 GB
# Target host: Jetson Linux R36.3 / JetPack 6.0 / CUDA 12.2 / TensorRT 8.6 / Ubuntu 22.04
#
# This deliberately uses NVIDIA's R36.3 JetPack image instead of the x86 ROS
# image. It gives the container the Jetson CUDA/cuDNN/TensorRT userspace that
# the NVIDIA PyTorch aarch64 wheel and the TensorRT engine builder expect.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.3.0
FROM ${BASE_IMAGE}

ARG HOST_UID=1000
ARG HOST_GID=1000

# NVIDIA's JetPack-6 / CUDA-12.2 PyTorch wheels for Python 3.10.
# These URLs are overrideable build args in case NVIDIA moves the files.
ARG TORCH_WHEEL_URL=https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
ARG TORCHVISION_WHEEL_URL=https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Required for the NVIDIA runtime on Jetson. Keep the same pattern as the
# working lr1_camera container.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ---------------------------------------------------------------------------
# ROS 2 Humble + build/runtime dependencies
#
# The x11 / xcb block is needed because SHOW_DISPLAY=true is a supported mode
# on this vehicle: the opencv-python wheel ships its own Qt but still links
# against the system xcb platform libraries. Without them cv2.namedWindow
# fails with "could not load the Qt platform plugin xcb".
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg2 \
      locales \
      lsb-release \
      software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libgl1 \
      libglib2.0-0 \
      libjpeg-dev \
      libomp-dev \
      libopenblas-dev \
      libopenmpi-dev \
      zlib1g-dev \
      python3-colcon-common-extensions \
      python3-dev \
      python3-pip \
      python3-yaml \
      ros-humble-ros-base \
      ros-humble-ament-cmake-python \
      ros-humble-cv-bridge \
      ros-humble-diagnostic-msgs \
      ros-humble-rmw-fastrtps-cpp \
      ros-humble-rosidl-default-generators \
      ros-humble-sensor-msgs \
      ros-humble-std-msgs \
      libdbus-1-3 \
      libfontconfig1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libxkbcommon-x11-0 \
      libxcb-icccm4 \
      libxcb-image0 \
      libxcb-keysyms1 \
      libxcb-randr0 \
      libxcb-render-util0 \
      libxcb-shape0 \
      libxcb-xfixes0 \
      libxcb-xinerama0 \
      libxcb-xkb1 \
      x11-utils \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# Jetson CUDA PyTorch stack + ONNX export toolchain
#
# Do NOT use a generic `pip install torch torchvision` here. On JetPack 6.0
# R36.3, use NVIDIA's aarch64 CUDA 12.2 builds: torch 2.3 + torchvision 0.18.
# The model files in this project were saved by Ultralytics 8.4.72, so pin that
# exact Ultralytics version. OpenCV 4.11 still accepts NumPy 1.x and has an
# aarch64 wheel; pinning it prevents newer OpenCV releases from forcing NumPy 2.
#
# onnx + onnxslim are only used by the one-off .pt -> .onnx -> .engine export.
# TensorRT itself comes from the JetPack base image (python3-libnvinfer); do
# NOT pip install tensorrt here, it would fetch an incompatible build.
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade "pip<26" \
    && python3 -m pip install --no-cache-dir \
         "setuptools==68.1.2" \
         "packaging==23.2" \
         wheel \
         "numpy==1.26.4" \
         "opencv-python==4.11.0.86" \
    && curl -fL "${TORCH_WHEEL_URL}" \
         -o /tmp/torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir \
         /tmp/torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    && curl -fL "${TORCHVISION_WHEEL_URL}" \
         -o /tmp/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir \
         /tmp/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir \
         "ultralytics==8.4.72" \
         "onnx==1.16.2" \
         "onnxslim==0.1.34" \
         "protobuf<5" \
         "numpy==1.26.4" \
         "opencv-python==4.11.0.86" \
    && rm -f /tmp/torch-*.whl /tmp/torchvision-*.whl

# Verify that the JetPack TensorRT Python package is present without importing
# its native extension during `docker build`. On Jetson, libraries such as
# libnvdla_compiler.so are injected by the NVIDIA container runtime when the
# container starts, so a full `import tensorrt` here can fail even though the
# runtime container is correctly configured. The entrypoint and `make gpu-check`
# perform the real TensorRT/CUDA import after startup with runtime: nvidia.
RUN python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt'); assert s is not None, 'TensorRT Python package missing'; print('TensorRT Python package present:', s.origin)"

# ---------------------------------------------------------------------------
# Separate workspace/home from lr1_camera. The containers can share ROS
# networking and host IPC without sharing their filesystem workspaces.
# ---------------------------------------------------------------------------
ENV VISION_WS=/ws_vision_detection
WORKDIR /ws_vision_detection

RUN set -eux; \
    if ! getent group "${HOST_GID}" >/dev/null; then groupadd -g "${HOST_GID}" vision_detection; fi; \
    if ! getent passwd "${HOST_UID}" >/dev/null; then \
      useradd -M -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash vision_detection; \
    fi; \
    mkdir -p /home/vision_detection/.ros/log /ws_vision_detection /engine_cache; \
    chown -R "${HOST_UID}:${HOST_GID}" /home/vision_detection /ws_vision_detection /engine_cache

COPY src /ws_vision_detection/src
COPY config /ws_vision_detection/config
COPY scripts /ws_vision_detection/scripts

RUN source /opt/ros/humble/setup.bash \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /ws_vision_detection/build /ws_vision_detection/log \
    && chmod +x /ws_vision_detection/scripts/*.sh /ws_vision_detection/scripts/*.py \
    && chown -R "${HOST_UID}:${HOST_GID}" /ws_vision_detection

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime environment
#
# The *_NUM_THREADS variables matter. Without them PyTorch's OpenMP pool and
# OpenBLAS each start one worker per core; on an 8-core Orin that is 16 threads
# competing with MAVROS, the camera bridge and the DDS receive threads for
# cores that GPU inference does not need. They must be set before torch is
# imported, which is why they live here and not only in the node.
# ---------------------------------------------------------------------------
ENV HOME=/home/vision_detection \
    ROS_HOME=/home/vision_detection/.ros \
    ROS_LOG_DIR=/home/vision_detection/.ros/log \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CUDA_MODULE_LOADING=LAZY \
    ENGINE_CACHE_DIR=/engine_cache \
    FASTRTPS_DEFAULT_PROFILES_FILE=/ws_vision_detection/config/fastdds_profile.xml \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

VOLUME ["/engine_cache"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "vision_feedback", "vision.launch.py", \
     "device:=cuda:0", "backend:=tensorrt", \
     "image_topic:=/camera/out/live_view", "show_display:=true"]
