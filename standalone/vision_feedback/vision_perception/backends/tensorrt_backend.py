"""TensorRT detector backend.

This is the fast path. It deliberately does **not** go through Ultralytics at
inference time; Ultralytics' predictor rebuilds a ``Results`` object graph, runs
the letterbox in NumPy and re-reads model attributes for every frame, all of
which is pure CPU work on the critical path.

Everything here stays on the GPU between the frame upload and the final, single
device->host copy of at most ``max_det`` rows:

    pinned host copy -> H2D -> BGR2RGB + HWC2CHW + scale -> bilinear resize
    -> pad into the static engine input -> TensorRT -> decode -> NMS -> D2H

The letterbox padding is written once at construction time; only the resized
image region is overwritten per frame, so no full-tensor fill happens in the
hot loop.

Compatible with the TensorRT 8.5+ tensor API (``execute_async_v3``) and the
older binding-index API (``execute_async_v2``), which covers TensorRT 8.6 as
shipped in JetPack 6.0 as well as the 10.x series in newer JetPacks.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np

from vision_perception.backends.base import (
    EMPTY_DETECTIONS,
    BackendInfo,
    DetectorBackend,
)
from vision_perception.letterbox import LetterboxTransform, anchor_count, compute_letterbox

_GREY = 114.0 / 255.0


def _torch_dtype(trt_dtype, trt_module, torch_module):
    mapping = {
        trt_module.DataType.FLOAT: torch_module.float32,
        trt_module.DataType.HALF: torch_module.float16,
        trt_module.DataType.INT32: torch_module.int32,
        trt_module.DataType.INT8: torch_module.int8,
        trt_module.DataType.BOOL: torch_module.bool,
    }
    for name, torch_name in (('BF16', 'bfloat16'), ('INT64', 'int64')):
        member = getattr(trt_module.DataType, name, None)
        if member is not None:
            mapping[member] = getattr(torch_module, torch_name)
    if trt_dtype not in mapping:
        raise RuntimeError(f'Unsupported TensorRT tensor dtype: {trt_dtype}')
    return mapping[trt_dtype]


def sidecar_path(engine_path: str) -> str:
    """Metadata file written next to an engine by the engine builder."""
    return os.path.splitext(engine_path)[0] + '.json'


def load_sidecar(engine_path: str) -> Dict:
    path = sidecar_path(engine_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


class TensorRTBackend(DetectorBackend):
    """YOLOv8 detection through a prebuilt TensorRT engine."""

    def __init__(
        self,
        engine_path: str,
        *,
        device: str = 'cuda:0',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        max_detections: int = 20,
        names: Optional[Dict[int, str]] = None,
        logger=None,
    ):
        import tensorrt as trt
        import torch
        from torchvision.ops import batched_nms

        self._trt = trt
        self._torch = torch
        self._batched_nms = batched_nms
        self._logger = logger
        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._max_det = int(max_detections)
        self._last_infer_ms = 0.0
        self._transform: Optional[LetterboxTransform] = None
        self._host_staging = None
        self._staging_np = None
        self._engine_path = os.path.abspath(engine_path)

        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT backend requires CUDA, but torch reports no GPU')
        self._device = torch.device(device if str(device).startswith('cuda') else 'cuda:0')
        torch.cuda.set_device(self._device)

        # A dedicated, non-default CUDA stream.
        #
        # torch.cuda.current_stream() is the *legacy default* stream unless a
        # stream context is active. Handing that to enqueueV3 makes TensorRT
        # emit:
        #     "Using default stream in enqueueV3() may lead to performance
        #      issues due to additional calls to cudaStreamSynchronize()"
        # and it means exactly that: TensorRT inserts extra full-stream
        # synchronisations around every single inference to stay correct.
        # Running the preprocessing, the engine and the decode together on one
        # private stream removes both the warning and the stalls.
        self._stream = torch.cuda.Stream(device=self._device)

        meta = load_sidecar(engine_path)
        resolved_names = names or {
            int(k): str(v) for k, v in (meta.get('names') or {}).items()
        }
        self._meta = meta

        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, '')
        with open(engine_path, 'rb') as handle:
            serialized = handle.read()
        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(serialized)
        if self._engine is None:
            raise RuntimeError(
                f'Failed to deserialize {engine_path}. TensorRT engines are tied to the '
                'exact GPU, TensorRT version and JetPack release that built them; delete '
                'the cached engine and rebuild it on this device.'
            )
        self._context = self._engine.create_execution_context()

        self._use_v3 = hasattr(self._engine, 'num_io_tensors')
        self._bind_tensors()
        self._allocate(resolved_names)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _bind_tensors(self) -> None:
        trt = self._trt
        engine = self._engine
        if self._use_v3:
            all_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            self._input_names = [
                n for n in all_names
                if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            self._output_names = [
                n for n in all_names
                if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]
            self._input_dtype = engine.get_tensor_dtype(self._input_names[0])
            self._output_dtype = engine.get_tensor_dtype(self._output_names[0])
            in_shape = tuple(engine.get_tensor_shape(self._input_names[0]))
        else:  # pragma: no cover - TensorRT <= 8.4
            self._input_names = [
                engine.get_binding_name(i) for i in range(engine.num_bindings)
                if engine.binding_is_input(i)
            ]
            self._output_names = [
                engine.get_binding_name(i) for i in range(engine.num_bindings)
                if not engine.binding_is_input(i)
            ]
            self._input_dtype = engine.get_binding_dtype(0)
            self._output_dtype = engine.get_binding_dtype(1)
            in_shape = tuple(engine.get_binding_shape(0))

        if len(self._input_names) != 1 or len(self._output_names) != 1:
            raise RuntimeError(
                'Expected a single-input / single-output YOLO engine, got '
                f'inputs={self._input_names} outputs={self._output_names}'
            )
        if len(in_shape) != 4:
            raise RuntimeError(f'Engine input must be NCHW, got shape {in_shape}')
        if any(int(d) < 0 for d in in_shape):
            raise RuntimeError(
                'Dynamic-shape engines are not supported; rebuild with a static shape '
                '(scripts/build_engine.py does this by default).'
            )
        self._net_h, self._net_w = int(in_shape[2]), int(in_shape[3])
        self._input_shape = (int(in_shape[0]), int(in_shape[1]), self._net_h, self._net_w)

    def _allocate(self, names: Dict[int, str]) -> None:
        torch = self._torch
        engine = self._engine

        in_dtype = _torch_dtype(self._input_dtype, self._trt, torch)
        out_dtype = _torch_dtype(self._output_dtype, self._trt, torch)

        if self._use_v3:
            out_shape = tuple(int(d) for d in engine.get_tensor_shape(self._output_names[0]))
        else:  # pragma: no cover - TensorRT <= 8.4
            out_shape = tuple(int(d) for d in engine.get_binding_shape(1))

        # Allocate on the private stream so the caching allocator associates
        # the blocks with the stream they are actually used on.
        with torch.cuda.stream(self._stream):
            self._input = torch.full(
                self._input_shape, _GREY, dtype=in_dtype, device=self._device)
            self._output = torch.empty(out_shape, dtype=out_dtype, device=self._device)
        self._stream.synchronize()

        if len(out_shape) != 3:
            raise RuntimeError(f'Unexpected YOLO output rank: {out_shape}')
        anchors = anchor_count(self._net_h, self._net_w)
        if out_shape[2] == anchors:
            self._channels_first = True
            channels = out_shape[1]
        elif out_shape[1] == anchors:
            self._channels_first = False
            channels = out_shape[2]
        else:
            # Fall back to the smaller dimension being the channel axis.
            self._channels_first = out_shape[1] <= out_shape[2]
            channels = out_shape[1] if self._channels_first else out_shape[2]
        self._num_classes = int(channels) - 4
        if self._num_classes < 1:
            raise RuntimeError(
                f'Engine output {out_shape} does not look like a YOLOv8 detection head')

        if not names:
            names = {i: str(i) for i in range(self._num_classes)}
        elif len(names) != self._num_classes:
            if self._logger is not None:
                self._logger.warn(
                    f'Engine has {self._num_classes} classes but metadata lists '
                    f'{len(names)}; using the engine class count.'
                )

        if self._use_v3:
            self._context.set_tensor_address(self._input_names[0], self._input.data_ptr())
            self._context.set_tensor_address(self._output_names[0], self._output.data_ptr())
        else:  # pragma: no cover - TensorRT <= 8.4
            self._bindings = [self._input.data_ptr(), self._output.data_ptr()]

        # What precision the engine *was built with*, not what its I/O tensors
        # happen to be.
        #
        # With only BuilderFlag.FP16 set, TensorRT keeps the network input and
        # output in FP32 and inserts reformat nodes; the FP16 kernels are used
        # for everything in between. Deriving the label from the I/O dtype
        # therefore reported 'fp32' for a perfectly good FP16 engine, which is
        # what made FP16 look broken.
        io_dtype = 'fp16' if in_dtype == torch.float16 else 'fp32'
        precision = str(self._meta.get('precision_built')
                        or self._meta.get('precision')
                        or io_dtype).lower()
        detail = (
            f'TensorRT {self._trt.__version__}, nc={self._num_classes}, '
            f'io {io_dtype}, private stream'
        )
        if not self._meta:
            detail += ', precision from engine I/O (no sidecar)'
        self.info = BackendInfo(
            kind='tensorrt',
            device=str(self._device),
            precision=precision,
            net_h=self._net_h,
            net_w=self._net_w,
            model_path=self._engine_path,
            names=dict(names),
            detail=detail,
        )

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _refresh_transform(self, src_h: int, src_w: int) -> LetterboxTransform:
        """Recompute padding and re-grey the border when the frame size changes."""
        tf = compute_letterbox(src_h, src_w, self._net_h, self._net_w)
        self._input.fill_(_GREY)
        self._transform = tf
        self._host_staging = self._torch.empty(
            (src_h, src_w, 3), dtype=self._torch.uint8, pin_memory=True)
        # A NumPy view over the pinned buffer. Copying through it keeps the
        # zero-copy DDS frame read-only, which torch.from_numpy would warn
        # about on every single frame.
        self._staging_np = self._host_staging.numpy()
        if self._logger is not None:
            self._logger.info(
                f'TensorRT letterbox {src_w}x{src_h} -> {tf.resized_w}x{tf.resized_h} '
                f'padded into {self._net_w}x{self._net_h} (gain={tf.gain:.3f})'
            )
        return tf

    def infer(self, frame: np.ndarray) -> np.ndarray:
        torch = self._torch
        started = time.perf_counter()
        src_h, src_w = frame.shape[:2]

        tf = self._transform
        if tf is None or tf.src_h != src_h or tf.src_w != src_w:
            tf = self._refresh_transform(src_h, src_w)

        # no_grad rather than inference_mode: the output buffer is written by
        # TensorRT outside torch's bookkeeping, and inference tensors carry
        # extra lifetime rules that buy nothing for a pure forward path.
        #
        # Everything - the H2D, the preprocessing kernels, the engine and the
        # decode - runs on one private stream. Sharing a single non-default
        # stream is what keeps TensorRT from inserting its own
        # cudaStreamSynchronize() calls around enqueueV3.
        with torch.no_grad(), torch.cuda.stream(self._stream):
            # Host -> pinned -> device. The pinned staging buffer makes the H2D a
            # DMA instead of a synchronous pageable copy. The previous frame's
            # trailing .cpu() already synchronised this stream, so reusing the
            # buffer cannot race with an in-flight copy.
            np.copyto(self._staging_np, frame)
            gpu_frame = self._host_staging.to(self._device, non_blocking=True)

            # BGR->RGB, HWC->CHW, uint8->float, /255 - all on the GPU.
            # flip(0) rather than [[2, 1, 0]]: an index list would allocate a
            # CPU index tensor and copy it to the device on every frame.
            chw = gpu_frame.permute(2, 0, 1).flip(0).unsqueeze(0)
            chw = chw.to(self._input.dtype).div_(255.0)
            resized = torch.nn.functional.interpolate(
                chw,
                size=(tf.resized_h, tf.resized_w),
                mode='bilinear',
                align_corners=False,
            )
            self._input[
                :, :,
                tf.pad_top:tf.pad_top + tf.resized_h,
                tf.pad_left:tf.pad_left + tf.resized_w,
            ] = resized

            handle = self._stream.cuda_stream
            if self._use_v3:
                ok = self._context.execute_async_v3(handle)
            else:  # pragma: no cover - TensorRT <= 8.4
                ok = self._context.execute_async_v2(self._bindings, handle)
            if not ok:
                if self._logger is not None:
                    self._logger.warn('TensorRT execution returned failure')
                self._stream.synchronize()
                return EMPTY_DETECTIONS

            detections = self._decode(tf)

        self._last_infer_ms = (time.perf_counter() - started) * 1000.0
        return detections

    def _decode(self, tf: LetterboxTransform) -> np.ndarray:
        torch = self._torch
        batched_nms = self._batched_nms

        pred = self._output[0]
        if self._channels_first:
            pred = pred.transpose(0, 1)          # (anchors, 4 + nc)
        pred = pred.float()

        scores_all = pred[:, 4:4 + self._num_classes]
        conf, cls = scores_all.max(dim=1)
        keep = conf >= self._conf

        # Boolean masking already has to synchronise to size the result, so
        # checking keep.any() first would cost a second full-pipeline stall.
        boxes_cxcywh = pred[keep, :4]
        if boxes_cxcywh.shape[0] == 0:
            return EMPTY_DETECTIONS
        conf = conf[keep]
        cls = cls[keep]

        half_wh = boxes_cxcywh[:, 2:4] * 0.5
        boxes = torch.empty_like(boxes_cxcywh)
        boxes[:, 0:2] = boxes_cxcywh[:, 0:2] - half_wh
        boxes[:, 2:4] = boxes_cxcywh[:, 0:2] + half_wh

        order = batched_nms(boxes, conf, cls, self._iou)[: self._max_det]
        result = torch.cat(
            (boxes[order], conf[order].unsqueeze(1), cls[order].unsqueeze(1).float()),
            dim=1,
        )

        # One synchronising transfer per frame, at most max_det rows.
        out = result.cpu().numpy().astype(np.float32, copy=False)
        tf.to_source(out[:, :4])
        return out

    # ------------------------------------------------------------------ #
    def warmup(self, frames: int = 3, source_shape=None) -> None:
        """Pay first-call cost up front, at the geometry the camera will send.

        ``source_shape`` is ``(height, width)``. Warming up at the real camera
        size means the first live frame reuses the pinned buffer and letterbox
        computed here instead of reallocating mid-stream.
        """
        if source_shape is None:
            height, width = self._net_h, self._net_w
        else:
            height, width = int(source_shape[0]), int(source_shape[1])
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(1, int(frames))):
            self.infer(dummy)
        self._stream.synchronize()

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._input = None
        self._output = None
        self._host_staging = None
        self._staging_np = None
        self._stream = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def tensorrt_available() -> bool:
    """True when the TensorRT Python bindings can be imported."""
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False
