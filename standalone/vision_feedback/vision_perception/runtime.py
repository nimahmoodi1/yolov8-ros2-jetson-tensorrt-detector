"""Process-wide CPU/threading limits for the vision node.

By default PyTorch starts one OpenMP worker per core and OpenCV starts one TBB
worker per core. On an 8-core Orin NX that is 16 extra threads fighting the ROS
executor, the DDS receive threads and the flight-control processes for cores
that the GPU work does not need. It inflates the load average, adds context
switches and measurably increases end-to-end latency jitter.

Once inference runs on TensorRT the CPU only has to letterbox and publish, so
one or two threads is plenty.
"""

from __future__ import annotations

import os

_ENV_KEYS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)


def limit_cpu_threads(threads: int = 1, logger=None) -> int:
    """Clamp OpenMP/BLAS/OpenCV/torch thread pools to ``threads``.

    Returns the applied value. ``threads <= 0`` disables the limiting so the
    behaviour can be reverted from config without editing code.
    """
    threads = int(threads)
    if threads <= 0:
        return 0

    for key in _ENV_KEYS:
        os.environ.setdefault(key, str(threads))

    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception as exc:  # pragma: no cover - depends on the OpenCV build
        if logger is not None:
            logger.warn(f'Could not limit OpenCV threads: {exc}')

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        # set_num_interop_threads throws once the pool is already running.
        pass
    except Exception as exc:  # pragma: no cover - torch may be absent in tests
        if logger is not None:
            logger.warn(f'Could not limit torch threads: {exc}')

    if logger is not None:
        logger.info(f'CPU thread pools limited to {threads} thread(s)')
    return threads
