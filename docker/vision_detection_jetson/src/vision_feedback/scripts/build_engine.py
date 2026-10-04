#!/usr/bin/env python3
"""Build (or rebuild) the cached TensorRT engine for the vision node.

The node builds its engine automatically on first launch, so this script is
only needed when you want to:

  * pre-build before a flight so the first launch is instant,
  * build engines for several packaged models at once,
  * try a different input shape or precision,
  * rebuild after a JetPack/TensorRT upgrade.

Run it with the workspace sourced:

    ros2 run vision_feedback build_engine.py

or, in the container:

    make engine
"""

from __future__ import annotations

import argparse
import os
import sys
import time


class _StdoutLogger:
    """Minimal logger with the rclpy logger interface."""

    @staticmethod
    def info(message):
        print(f'[build_engine] {message}', flush=True)

    @staticmethod
    def warn(message):
        print(f'[build_engine] WARN: {message}', flush=True)

    warning = warn

    @staticmethod
    def error(message):
        print(f'[build_engine] ERROR: {message}', file=sys.stderr, flush=True)


def _resolve_defaults():
    from vision_perception import config

    defaults = config.yaml_defaults()
    return defaults, config


def _verify(engine_path, height, width, logger, frames=60):
    """Time the freshly built engine so the numbers come from this device."""
    import time as _time

    import numpy as np

    from vision_perception.backends.tensorrt_backend import TensorRTBackend

    backend = TensorRTBackend(engine_path, device='cuda:0', logger=logger)
    backend.warmup(10, source_shape=(height, width))
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    timings = []
    for _ in range(frames):
        started = _time.perf_counter()
        backend.infer(frame)
        timings.append((_time.perf_counter() - started) * 1000.0)
    backend.close()

    timings.sort()
    logger.info(
        f'   verify: {backend.info.describe()} | '
        f'mean {sum(timings) / len(timings):.1f} ms | '
        f'p50 {timings[len(timings) // 2]:.1f} ms | '
        f'p95 {timings[int(len(timings) * 0.95)]:.1f} ms')
    return 0


def main() -> int:
    defaults, config = _resolve_defaults()
    from vision_perception import engine_builder
    from vision_perception.letterbox import parse_input_shape

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', default='',
        help='Path or packaged filename of the .pt weights '
             '(default: the model vision.yaml/mission.yaml would use)')
    parser.add_argument('--all', action='store_true',
                        help='Build an engine for every packaged .pt model')
    parser.add_argument('--imgsz', type=int, default=int(defaults['inference_imgsz']),
                        help='Long side of the network input')
    parser.add_argument('--shape', default=str(defaults['inference_input_shape']),
                        help="'auto', 'square', or explicit 'HxW' (e.g. 448x640)")
    parser.add_argument('--width', type=int,
                        default=int(defaults['expected_frame_width']),
                        help='Camera width that --shape auto derives from')
    parser.add_argument('--height', type=int,
                        default=int(defaults['expected_frame_height']),
                        help='Camera height that --shape auto derives from')
    parser.add_argument('--precision', default=str(defaults['inference_precision']),
                        choices=('fp16', 'fp32'))
    parser.add_argument('--cache-dir',
                        default=os.environ.get(
                            'ENGINE_CACHE_DIR', str(defaults['engine_cache_dir'])))
    parser.add_argument('--workspace-mib', type=int,
                        default=int(defaults['trt_workspace_mib']))
    parser.add_argument('--opset', type=int, default=int(defaults['onnx_opset']))
    parser.add_argument('--fp16-io', action='store_true',
                        help='Also force the network input/output tensors to FP16')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even when a cached engine already exists')
    parser.add_argument('--verify', action='store_true',
                        help='Time the built engine on synthetic frames afterwards')
    args = parser.parse_args()

    logger = _StdoutLogger()

    try:
        import tensorrt as trt

        logger.info(f'TensorRT {trt.__version__}')
    except Exception as exc:
        logger.error(
            f'TensorRT Python bindings unavailable ({exc}). On JetPack these come '
            'from python3-libnvinfer, which ships in the l4t-jetpack base image.')
        return 2

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error('CUDA is not available in this container; cannot build an engine.')
            return 2
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    except Exception as exc:
        logger.error(f'PyTorch/CUDA unavailable: {exc}')
        return 2

    if args.all:
        share = os.path.dirname(config.default_model_path())
        models = sorted(
            os.path.join(share, name)
            for name in os.listdir(share) if name.endswith('.pt')
        )
    elif args.model:
        models = [args.model if os.path.exists(args.model)
                  else config.packaged_model_path(args.model)]
    else:
        models = [config.default_model_path()]

    net_h, net_w = parse_input_shape(
        args.shape, args.height, args.width, args.imgsz)
    logger.info(
        f'Camera {args.width}x{args.height} -> network input {net_w}x{net_h} '
        f"({args.precision}, shape='{args.shape}', io "
        f"{'fp16' if args.fp16_io else 'fp32'})")

    failures = 0
    for model_path in models:
        if not os.path.exists(model_path):
            logger.error(f'Missing model: {model_path}')
            failures += 1
            continue

        spec = engine_builder.EngineSpec(
            model_path, net_h, net_w, args.precision, bool(args.fp16_io))
        engine_path = engine_builder.engine_path_for(spec, args.cache_dir)
        if args.force and os.path.exists(engine_path):
            logger.info(f'--force: removing {engine_path}')
            os.remove(engine_path)

        started = time.time()
        try:
            result = engine_builder.ensure_engine(
                model_path,
                net_h,
                net_w,
                precision=args.precision,
                cache_dir=args.cache_dir,
                opset=args.opset,
                workspace_mib=args.workspace_mib,
                allow_build=True,
                fp16_io=bool(args.fp16_io),
                logger=logger,
            )
        except Exception as exc:
            logger.error(f'{os.path.basename(model_path)}: {exc}')
            failures += 1
            continue
        size_mb = os.path.getsize(result) / (1 << 20)
        logger.info(
            f'OK {os.path.basename(model_path)} -> {result} '
            f'({size_mb:.1f} MB, {time.time() - started:.0f} s)')

        meta = engine_builder.load_sidecar_summary(result)
        if meta:
            logger.info(
                f'   built precision={meta.get("precision_built")} '
                f'(fp16 flag {meta.get("fp16_flag_applied")}) '
                f'io={meta.get("io_dtype")} tensorrt={meta.get("tensorrt")} '
                f'gpu={meta.get("gpu")}')

        if args.verify:
            _verify(result, args.height, args.width, logger)

    if failures:
        logger.error(f'{failures} engine(s) failed to build')
        return 1
    logger.info('All engines ready')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
