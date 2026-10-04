#!/usr/bin/env python3
"""Measure detector throughput and CPU cost on this Jetson.

Runs the same synthetic frames through the TensorRT and PyTorch backends and
reports wall time per frame plus the process CPU time consumed, which is the
number that shows up in htop.

    ros2 run vision_feedback benchmark.py

or, in the container:

    make bench
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np


class _StdoutLogger:
    @staticmethod
    def info(message):
        print(f'[benchmark] {message}', flush=True)

    @staticmethod
    def warn(message):
        print(f'[benchmark] WARN: {message}', flush=True)

    warning = warn

    @staticmethod
    def error(message):
        print(f'[benchmark] ERROR: {message}', file=sys.stderr, flush=True)


def _synthetic_frames(count: int, height: int, width: int, seed: int = 7):
    """Frames with structure and per-frame noise, so nothing is trivially cached."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    frames = []
    for i in range(count):
        frame = base.copy()
        y0 = 80 + (i % 40)
        frame[y0:y0 + 180, 400:600] = 220
        frame[y0 + 180:y0 + 420, 430:570] = 90
        frames.append(frame)
    return frames


def _run(backend_name, params, frames, warmup, logger):
    from vision_perception.backends import build_detector

    detector, warnings = build_detector(
        model_path=params['model_path'],
        backend=backend_name,
        device=params['device'],
        net_shape=params['net_shape'],
        precision=params['precision'],
        conf_threshold=params['conf'],
        iou_threshold=params['iou'],
        max_detections=params['max_det'],
        engine_cache_dir=params['cache_dir'],
        auto_build_engine=True,
        warmup_frames=warmup,
        logger=logger,
    )
    if backend_name in ('tensorrt', 'trt') and detector.info.kind != 'tensorrt':
        detector.close()
        raise RuntimeError('TensorRT backend was requested but did not load')

    for frame in frames[:warmup]:
        detector.infer(frame)

    timings = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for frame in frames:
        started = time.perf_counter()
        detector.infer(frame)
        timings.append((time.perf_counter() - started) * 1000.0)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    info = detector.info
    detector.close()

    timings.sort()
    return {
        'backend': info.describe(),
        'mean_ms': statistics.mean(timings),
        'p50_ms': timings[len(timings) // 2],
        'p95_ms': timings[int(len(timings) * 0.95)],
        'fps': len(timings) / wall,
        'cpu_pct': 100.0 * cpu / wall,
        'warnings': warnings,
    }


def main() -> int:
    from vision_perception import config
    from vision_perception.letterbox import parse_input_shape

    defaults = config.yaml_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--model', default='')
    parser.add_argument('--imgsz', type=int, default=int(defaults['inference_imgsz']))
    parser.add_argument('--shape', default=str(defaults['inference_input_shape']))
    parser.add_argument('--precision', default=str(defaults['inference_precision']))
    parser.add_argument('--width', type=int, default=int(defaults['expected_frame_width']))
    parser.add_argument('--height', type=int, default=int(defaults['expected_frame_height']))
    parser.add_argument('--backends', default='tensorrt,torch')
    parser.add_argument('--cache-dir',
                        default=os.environ.get(
                            'ENGINE_CACHE_DIR', str(defaults['engine_cache_dir'])))
    args = parser.parse_args()

    logger = _StdoutLogger()
    model_path = args.model or config.default_model_path()
    if args.model and not os.path.exists(model_path):
        model_path = config.packaged_model_path(args.model)

    net_h, net_w = parse_input_shape(args.shape, args.height, args.width, args.imgsz)
    params = {
        'model_path': model_path,
        'device': 'cuda:0',
        'net_shape': (net_h, net_w),
        'precision': args.precision,
        'conf': float(min(defaults['head_confidence_threshold'],
                          defaults['body_confidence_threshold'])),
        'iou': float(defaults['inference_iou_threshold']),
        'max_det': int(defaults['inference_max_detections']),
        'cache_dir': args.cache_dir,
    }

    print('=' * 78)
    print(f'model  : {os.path.basename(model_path)}')
    print(f'frames : {args.frames} x {args.width}x{args.height} BGR')
    print(f'network: {net_w}x{net_h} ({args.precision})')
    print('=' * 78)

    frames = _synthetic_frames(args.frames, args.height, args.width)
    rows = []
    for name in [b.strip() for b in args.backends.split(',') if b.strip()]:
        try:
            rows.append(_run(name, params, frames, args.warmup, logger))
        except Exception as exc:
            logger.error(f'{name}: {exc}')

    if not rows:
        return 1

    print()
    header = f'{"backend":<52}{"mean":>8}{"p95":>8}{"fps":>8}{"cpu%":>8}'
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f'{row["backend"]:<52}{row["mean_ms"]:>7.1f}m{row["p95_ms"]:>7.1f}m'
              f'{row["fps"]:>8.1f}{row["cpu_pct"]:>8.0f}')
    if len(rows) > 1:
        speedup = rows[-1]['mean_ms'] / max(rows[0]['mean_ms'], 1e-9)
        cpu_ratio = rows[-1]['cpu_pct'] / max(rows[0]['cpu_pct'], 1e-9)
        print()
        print(f'{rows[0]["backend"].split()[0]} is {speedup:.1f}x faster and uses '
              f'{cpu_ratio:.1f}x less CPU than {rows[-1]["backend"].split()[0]}')
    print()
    print('cpu% is single-process CPU time / wall time: 100 % = one saturated core.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
