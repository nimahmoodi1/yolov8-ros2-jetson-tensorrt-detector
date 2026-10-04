ENGINE_CACHE_DIR="$HOME/.cache/vision_feedback_engines" \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
YOLO_OFFLINE=1 \
CUDA_MODULE_LOADING=LAZY \
ros2 launch vision_feedback vision.launch.py


or the below, both of them works:
ENGINE_CACHE_DIR="$HOME/workspace/.cache/vision_feedback_engines" \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
YOLO_OFFLINE=1 \
CUDA_MODULE_LOADING=LAZY \
ros2 launch vision_feedback vision.launch.py 




ENGINE_CACHE_DIR="$HOME/workspace/.cache/vision_feedback_engines" \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
YOLO_OFFLINE=1 \
CUDA_MODULE_LOADING=LAZY \
ros2 launch vision_feedback vision.launch.py show_display:=false