SHELL := /bin/bash
CONTAINER := vision_detection_jetson
EXEC := docker exec -it $(CONTAINER) bash -lc
EXEC_Q := docker exec $(CONTAINER) bash -lc
SRC := source /opt/ros/humble/setup.bash && source /ws_vision_detection/install/setup.bash
XAUTH := $(if $(XAUTHORITY),$(XAUTHORITY),/tmp/.docker.xauth)

.PHONY: help env xauth build rebuild pull up up-idle down restart recreate logs shell run \
        engine engine-all engine-720p engine-1024 engine-clean bench power \
        gpu-check check topics camera-hz bbox diag test clean

help:
	@echo "vision_detection_jetson targets"
	@echo ""
	@echo "  SETUP"
	@echo "    make env         write .env with host UID/GID, DISPLAY and Jetson defaults"
	@echo "    make xauth       create the X cookie used by SHOW_DISPLAY=true"
	@echo "    make pull        pull the NVIDIA JetPack R36.3 base image"
	@echo "    make build       build the Jetson ARM64 image"
	@echo "    make rebuild     rebuild without Docker cache"
	@echo ""
	@echo "  RUN"
	@echo "    make up          start detector (TensorRT FP16 on cuda:0, display on)"
	@echo "    make up-idle     start container without launching the detector"
	@echo "    make down        stop/remove the detector container only"
	@echo "    make restart     restart the detector container"
	@echo "    make recreate    recreate after .env/compose changes"
	@echo "    make logs        follow detector logs"
	@echo "    make shell       shell with the ROS workspace sourced"
	@echo "    make run         launch the detector in the foreground (after up-idle)"
	@echo ""
	@echo "  TENSORRT"
	@echo "    make engine      build/refresh the cached engine for the active model"
	@echo "    make engine-all  build engines for every packaged model"
	@echo "    make engine-720p build an engine for a 1280x720 camera (384x640)"
	@echo "    make engine-1024 build an engine for a 1024x680 camera (448x640)"
	@echo "    make engine-clean  delete the engine cache (forces a rebuild)"
	@echo "    make bench       TensorRT vs PyTorch throughput and CPU on this device"
	@echo "    make power       show the Jetson power mode and GPU clocks"
	@echo ""
	@echo "  VERIFY"
	@echo "    make gpu-check   prove torch + TensorRT see the Orin GPU"
	@echo "    make check       GPU check + ROS camera topic info"
	@echo "    make topics      list ROS 2 topics"
	@echo "    make camera-hz   measure the camera topic receive rate"
	@echo "    make bbox        print one bounding-box message"
	@echo "    make diag        print one /diagnostics message"
	@echo "    make test        run the package unit tests inside the container"

env:
	@printf 'HOST_UID=%s\nHOST_GID=%s\n' "$$(id -u)" "$$(id -g)" > .env
	@printf 'ROS_DOMAIN_ID=0\nROS_LOCALHOST_ONLY=0\n' >> .env
	@printf 'VISION_BACKEND=tensorrt\nVISION_DEVICE=cuda:0\n' >> .env
	@printf 'VISION_PRECISION=fp16\nVISION_INPUT_SHAPE=auto\nVISION_FP16_IO=false\n' >> .env
	@printf 'IMAGE_TOPIC=/camera/out/live_view\n' >> .env
	@printf 'BBOX_TOPIC=/art/bounding_boxes\nRAW_BBOX_TOPIC=/art/bounding_boxes_raw\n' >> .env
	@printf 'SHOW_DISPLAY=true\nAUTOSTART=1\nPREBUILD_ENGINE=1\n' >> .env
	@printf 'DISPLAY=%s\n' "$${DISPLAY:-:10}" >> .env
	@printf 'XAUTHORITY=/tmp/.docker.xauth\n' >> .env
	@printf 'OMP_NUM_THREADS=1\nOPENBLAS_NUM_THREADS=1\n' >> .env
	@echo "wrote .env:"; cat .env

# The container runs as your UID, so it needs a cookie it can actually read.
# Non-fatal: a headless setup with SHOW_DISPLAY=false does not need this.
xauth:
	@touch $(XAUTH)
	@chmod 0644 $(XAUTH)
	@if command -v xauth >/dev/null 2>&1; then \
	  xauth nlist "$${DISPLAY:-:10}" 2>/dev/null \
	    | sed -e 's/^..../ffff/' \
	    | xauth -f $(XAUTH) nmerge - 2>/dev/null || true; \
	  echo "X cookie ready at $(XAUTH) for DISPLAY=$${DISPLAY:-:10}"; \
	else \
	  echo "xauth not installed; relying on 'xhost +local:docker'"; \
	fi
	@xhost +local:docker >/dev/null 2>&1 || true

pull:
	docker pull nvcr.io/nvidia/l4t-jetpack:r36.3.0

build:
	docker compose build

rebuild:
	docker compose build --no-cache

up: xauth
	docker compose up -d
	@sleep 2; docker compose ps

up-idle: xauth
	AUTOSTART=0 docker compose up -d --force-recreate
	@sleep 2; docker compose ps

down:
	docker compose down

restart:
	docker compose restart

recreate: xauth
	docker compose up -d --force-recreate
	@sleep 2; docker compose ps

logs:
	docker compose logs -f --tail=100

shell:
	$(EXEC) "$(SRC) && exec bash"

run:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 launch vision_feedback vision.launch.py \
	  backend:=$${VISION_BACKEND:-tensorrt} \
	  device:=$${VISION_DEVICE:-cuda:0} \
	  inference_precision:=$${VISION_PRECISION:-fp16} \
	  inference_input_shape:=$${VISION_INPUT_SHAPE:-auto} \
	  trt_fp16_io:=$${VISION_FP16_IO:-false} \
	  image_topic:=$${IMAGE_TOPIC:-/camera/out/live_view} \
	  bbox_topic:=$${BBOX_TOPIC:-/art/bounding_boxes} \
	  raw_bbox_topic:=$${RAW_BBOX_TOPIC:-/art/bounding_boxes_raw} \
	  show_display:=$${SHOW_DISPLAY:-true}'

engine:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 run vision_feedback build_engine.py'

engine-all:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 run vision_feedback build_engine.py --all'

# Camera-specific engines. The node picks whichever engine matches the shape
# `auto` derives from expected_frame_* in vision.yaml.
engine-720p:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 run vision_feedback build_engine.py --width 1280 --height 720 --verify'

engine-1024:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 run vision_feedback build_engine.py --width 1024 --height 680 --verify'

engine-clean:
	@rm -fv ./engine_cache/*.engine ./engine_cache/*.json 2>/dev/null || true
	@echo "engine cache cleared; the next launch rebuilds it"

bench:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 run vision_feedback benchmark.py'

gpu-check:
	$(EXEC_Q) "python3 /ws_vision_detection/scripts/gpu_check.py"

check:
	$(EXEC_Q) "/ws_vision_detection/scripts/check_jetson.sh"

topics:
	$(EXEC_Q) "$(SRC) && ros2 topic list"

camera-hz:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 topic hz $${IMAGE_TOPIC:-/camera/out/live_view}'

bbox:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 topic echo --once $${BBOX_TOPIC:-/art/bounding_boxes}'

diag:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && ros2 topic echo --once /diagnostics'

power:
	@echo "--- host nvpmodel ---"; sudo nvpmodel -q 2>/dev/null || echo "nvpmodel unavailable"
	@echo "--- host jetson_clocks ---"; sudo jetson_clocks --show 2>/dev/null | head -20 || echo "jetson_clocks unavailable"

test:
	docker exec -it $(CONTAINER) bash -lc '$(SRC) && cd /ws_vision_detection/src/vision_feedback && python3 -m pytest test -q'

clean:
	docker compose down --remove-orphans
