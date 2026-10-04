"""Detector backend selection.

Resolution order, highest first:

1. **TensorRT** - requested (or ``auto``) and CUDA + the TensorRT bindings are
   present. Builds/loads a cached FP16 engine.
2. **PyTorch CUDA** - TensorRT unavailable, an engine could not be built, or
   ``backend: torch`` was requested explicitly.
3. **PyTorch CPU** - only after a genuine CUDA out-of-memory failure, matching
   the original node's safety behaviour.

Every step logs why it was taken. A silent downgrade from TensorRT to CPU is
exactly the failure mode that makes a drone miss its detection deadline, so it
is always visible in the node log and in ``/diagnostics``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from vision_perception.backends.base import (
    EMPTY_DETECTIONS,
    BackendInfo,
    DetectorBackend,
    is_cuda_oom,
)

__all__ = [
    'EMPTY_DETECTIONS',
    'BackendInfo',
    'DetectorBackend',
    'ClassRouter',
    'build_detector',
    'is_cuda_oom',
    'resolve_device',
]


class ClassRouter:
    """Maps model class ids onto the pipeline's head/body roles, once.

    The old code called ``names[int(box.cls)]`` and did a substring test for
    every box on every frame. The class map never changes, so the test is done
    here at load time and inference only compares integers.
    """

    def __init__(self, names: Dict[int, str]):
        self.names = dict(names)
        self.head_ids = tuple(sorted(
            i for i, n in self.names.items() if 'head' in str(n).lower()))
        self.body_ids = tuple(sorted(
            i for i, n in self.names.items() if 'body' in str(n).lower()))

    def describe(self) -> str:
        head = ', '.join(f'{i}:{self.names[i]}' for i in self.head_ids) or 'NONE'
        body = ', '.join(f'{i}:{self.names[i]}' for i in self.body_ids) or 'NONE'
        return f'head classes [{head}] | body classes [{body}]'

    def is_valid(self) -> bool:
        return bool(self.head_ids) or bool(self.body_ids)


def resolve_device(requested: str, cuda_available: bool, logger=None) -> str:
    """Map a requested device string to a concrete torch device."""
    requested = (requested or 'auto').strip()
    low = requested.lower()
    if low == 'auto':
        return 'cuda:0' if cuda_available else 'cpu'
    if low.startswith('cuda') and not cuda_available:
        if logger is not None:
            logger.warn(f"Requested device '{requested}' but CUDA is unavailable; using CPU.")
        return 'cpu'
    return requested


def _load_tensorrt(
    model_path: str,
    device: str,
    net_shape: Tuple[int, int],
    *,
    precision: str,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    cache_dir: str,
    allow_build: bool,
    workspace_mib: int,
    onnx_opset: int,
    fp16_io: bool,
    logger,
) -> DetectorBackend:
    from vision_perception import engine_builder
    from vision_perception.backends.tensorrt_backend import (
        TensorRTBackend,
        load_sidecar,
    )

    if model_path.endswith('.engine'):
        engine_path = model_path
    else:
        engine_path = engine_builder.ensure_engine(
            model_path,
            net_shape[0],
            net_shape[1],
            precision=precision,
            cache_dir=cache_dir,
            opset=onnx_opset,
            workspace_mib=workspace_mib,
            allow_build=allow_build,
            fp16_io=fp16_io,
            logger=logger,
        )

    meta = load_sidecar(engine_path)
    names = {int(k): str(v) for k, v in (meta.get('names') or {}).items()}
    if not names:
        from vision_perception.backends.torch_backend import read_model_names

        names = read_model_names(model_path) or {}

    return TensorRTBackend(
        engine_path,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        names=names,
        logger=logger,
    )


def _load_torch(
    model_path: str,
    device: str,
    net_shape: Tuple[int, int],
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    half: bool,
    logger,
) -> DetectorBackend:
    from vision_perception.backends.torch_backend import TorchBackend

    return TorchBackend(
        model_path,
        device=device,
        net_shape=net_shape,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        half=half,
        logger=logger,
    )


def build_detector(
    *,
    model_path: str,
    backend: str,
    device: str,
    net_shape: Tuple[int, int],
    precision: str = 'fp16',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    max_detections: int = 20,
    engine_cache_dir: str = '/engine_cache',
    auto_build_engine: bool = True,
    trt_workspace_mib: int = 2048,
    onnx_opset: int = 16,
    trt_fp16_io: bool = False,
    warmup_frames: int = 3,
    warmup_shape=None,
    logger=None,
) -> Tuple[DetectorBackend, List[str]]:
    """Construct the best available backend, returning it plus any warnings."""
    warnings: List[str] = []
    requested = str(backend or 'auto').strip().lower()
    on_cuda = str(device).lower().startswith('cuda')

    def _warn(message: str) -> None:
        warnings.append(message)
        if logger is not None:
            logger.warn(message)

    def _info(message: str) -> None:
        if logger is not None:
            logger.info(message)

    if requested in ('tensorrt', 'trt', 'auto') and on_cuda:
        from vision_perception.backends.tensorrt_backend import tensorrt_available

        if not tensorrt_available():
            message = (
                'TensorRT Python bindings not importable; falling back to PyTorch. '
                'On JetPack the bindings come from the python3-libnvinfer package.'
            )
            if requested == 'auto':
                _warn(message)
            else:
                _warn(message + " (backend was set to 'tensorrt')")
        else:
            try:
                detector = _load_tensorrt(
                    model_path,
                    device,
                    net_shape,
                    precision=precision,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                    cache_dir=engine_cache_dir,
                    allow_build=auto_build_engine,
                    workspace_mib=trt_workspace_mib,
                    onnx_opset=onnx_opset,
                    fp16_io=trt_fp16_io,
                    logger=logger,
                )
                detector.warmup(warmup_frames, source_shape=warmup_shape)
                _info(f'Detector ready: {detector.info.describe()}')
                return detector, warnings
            except Exception as exc:
                _warn(f'TensorRT backend unavailable ({exc}); falling back to PyTorch.')
    elif requested in ('tensorrt', 'trt') and not on_cuda:
        _warn(f"backend='tensorrt' needs a CUDA device but device is '{device}'.")

    half = on_cuda and str(precision).lower() == 'fp16'
    try:
        detector = _load_torch(
            model_path, device, net_shape,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            half=half,
            logger=logger,
        )
        detector.warmup(warmup_frames, source_shape=warmup_shape)
    except Exception as exc:
        if not on_cuda or not is_cuda_oom(exc):
            raise
        _warn('PyTorch CUDA load failed on GPU memory pressure; falling back to CPU.')
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        detector = _load_torch(
            model_path, 'cpu', net_shape,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            half=False,
            logger=logger,
        )
        detector.warmup(warmup_frames, source_shape=warmup_shape)

    _info(f'Detector ready: {detector.info.describe()}')
    return detector, warnings
