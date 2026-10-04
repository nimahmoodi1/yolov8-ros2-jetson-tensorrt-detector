"""PyTorch / Ultralytics detector backend.

This is the fallback used when TensorRT is unavailable or an engine could not
be built. It is several times slower than the TensorRT backend on Jetson, but
it keeps the node runnable on a laptop, in CI and during model bring-up.

Two things make it noticeably cheaper than the original implementation:

* ``half=True`` on CUDA, which roughly halves both GPU time and the number of
  kernels PyTorch has to launch.
* The whole result tensor is pulled across the PCIe/iGPU boundary **once** as
  a single ``(N, 6)`` array instead of one ``.cpu()`` call per box.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from vision_perception.backends.base import (
    EMPTY_DETECTIONS,
    BackendInfo,
    DetectorBackend,
)

# Keep Ultralytics from phoning home / downloading fonts on a vehicle that has
# no internet. Must be set before the package is imported.
os.environ.setdefault('YOLO_OFFLINE', '1')
os.environ.setdefault('YOLO_VERBOSE', 'False')
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')


class TorchBackend(DetectorBackend):
    """Ultralytics YOLO running under PyTorch."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = 'cuda:0',
        net_shape: Tuple[int, int] = (384, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        max_detections: int = 20,
        half: bool = True,
        logger=None,
    ):
        from ultralytics import YOLO

        logging.getLogger('ultralytics').setLevel(logging.WARNING)

        self._logger = logger
        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._max_det = int(max_detections)
        self._net_h, self._net_w = int(net_shape[0]), int(net_shape[1])
        self._last_infer_ms = 0.0

        self._device = str(device)
        self._half = bool(half) and self._device.lower().startswith('cuda')

        self._model = YOLO(model_path)
        self._model.to(self._device)

        names = getattr(self._model, 'names', {}) or {}
        names = {int(k): str(v) for k, v in dict(names).items()}

        self.info = BackendInfo(
            kind='torch',
            device=self._device,
            precision='fp16' if self._half else 'fp32',
            net_h=self._net_h,
            net_w=self._net_w,
            model_path=os.path.abspath(model_path),
            names=names,
            detail='ultralytics predictor',
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        try:
            results = self._model.predict(
                frame,
                imgsz=(self._net_h, self._net_w),
                conf=self._conf,
                iou=self._iou,
                max_det=self._max_det,
                device=self._device,
                half=self._half,
                verbose=False,
                stream=False,
                show=False,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            if self._logger is not None:
                self._logger.warn(f'YOLO inference error: {exc}')
            return EMPTY_DETECTIONS
        finally:
            self._last_infer_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            return EMPTY_DETECTIONS
        boxes = getattr(results[0], 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return EMPTY_DETECTIONS

        # boxes.data is already (N, 6) = x1 y1 x2 y2 conf cls in source pixels.
        data = boxes.data
        array = data.detach().to('cpu').numpy() if hasattr(data, 'detach') else np.asarray(data)
        return np.ascontiguousarray(array[:, :6], dtype=np.float32)

    def warmup(self, frames: int = 3, source_shape=None) -> None:
        if source_shape is None:
            height, width = self._net_h, self._net_w
        else:
            height, width = int(source_shape[0]), int(source_shape[1])
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(1, int(frames))):
            self.infer(dummy)

    def close(self) -> None:
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def read_model_names(model_path: str) -> Optional[Dict[int, str]]:
    """Read the class-name map out of a ``.pt`` checkpoint without loading CUDA."""
    try:
        import torch

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    model = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    names = getattr(model, 'names', None) if model is not None else None
    if not names:
        return None
    return {int(k): str(v) for k, v in dict(names).items()}
