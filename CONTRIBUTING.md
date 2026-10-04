# Contributing

Contributions are welcome for bug fixes, documentation, portability,
performance improvements, and ROS 2 integration work.

## Before opening a pull request

Run:

```bash
bash scripts/check_release_tree.sh
python -m compileall -q \
  standalone/vision_feedback \
  docker/vision_detection_jetson
```

Check shell scripts with:

```bash
find . -type f -name '*.sh' -not -path './.git/*' -print0 | \
xargs -0 -r -n1 bash -n
```

When ROS 2 Humble is available, also build and run the relevant package tests.

Jetson, CUDA, or TensorRT changes should be validated on compatible NVIDIA
Jetson hardware before being described as hardware-tested.

## Generated artifacts

Do not commit:

- `.env`
- TensorRT `.engine` or `.plan` files
- generated ONNX files
- ROS `build/`, `install/`, or `log/` directories
- Python caches
- runtime logs or ROS bags

## Compatibility

Changes to ROS 2 message types, topic names, or public configuration parameters
must be documented because they can affect host-side and containerized
subscribers.
