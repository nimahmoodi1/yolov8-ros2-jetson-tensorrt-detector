# Maintainer release checklist

Before tagging a public release:

1. Confirm the `vision_feedback` package versions are correct.
2. Update `CHANGELOG.md`.
3. Verify software licensing and trained-model redistribution rights.
4. Run `scripts/check_release_tree.sh`.
5. Confirm GitHub static checks are green.
6. Build and test the standalone package on the target NVIDIA Jetson.
7. Build and test the Docker deployment on the target NVIDIA Jetson.
8. Verify TensorRT FP16 inference and GPU diagnostics.
9. Verify ROS 2 topics and message compatibility.
10. Confirm `git status` reports a clean working tree.

## Release 1.1.0

Create the annotated tag only after all checks above pass:

```bash
git tag -a v1.1.0 \
  -m "YOLOv8 ROS 2 Jetson TensorRT Detector v1.1.0"

git push origin v1.1.0
```

TensorRT `.engine` files must not be committed or attached to general releases.
They depend on the target GPU, TensorRT version, JetPack version, model weights,
input shape, and precision configuration.
