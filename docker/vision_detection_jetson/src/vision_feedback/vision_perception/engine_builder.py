"""Build and cache TensorRT engines from Ultralytics ``.pt`` checkpoints.

A TensorRT engine is not portable. It is tied to the GPU architecture, the
TensorRT version and (in practice) the JetPack release that produced it, so the
engine cannot be baked into the image on a build machine and shipped. It has to
be produced on the Orin itself, once, and then reused.

This module does that and caches the result under a content-addressed name::

    best_gazebo_new__384x640__fp16__trt8.6.2__orin.engine
    best_gazebo_new__384x640__fp16__trt8.6.2__orin.json   (class names, metadata)

Changing the weights, the input shape, the precision, the TensorRT version or
the GPU produces a different name, so a stale engine can never be picked up by
accident.

Build takes roughly 1-4 minutes for YOLOv8n on an Orin NX. It happens once; the
cache directory is a Docker volume so it survives container recreation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_DEFAULT_OPSET = 16
_DEFAULT_WORKSPACE_MIB = 2048


class EngineBuildError(RuntimeError):
    """Raised when an engine cannot be produced."""


@dataclass(frozen=True)
class EngineSpec:
    """Everything that makes one cached engine different from another."""

    model_path: str
    net_h: int
    net_w: int
    precision: str
    fp16_io: bool = False

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.model_path))[0]

    @property
    def precision_token(self) -> str:
        """Cache-name token. FP16 I/O produces a different engine binary."""
        return f'{self.precision}io' if self.fp16_io else self.precision


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower()) or 'gpu'


def _file_digest(path: str, length: int = 10) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:
        return None


def gpu_slug() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return _slug(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return 'nogpu'


def engine_path_for(spec: EngineSpec, cache_dir: str) -> str:
    """Deterministic cache path for ``spec``."""
    trt_ver = _slug(tensorrt_version() or 'notrt')
    name = (
        f'{spec.stem}__{spec.net_h}x{spec.net_w}__{spec.precision_token}'
        f'__trt{trt_ver}__{gpu_slug()}__{_file_digest(spec.model_path)}.engine'
    )
    return os.path.join(cache_dir, name)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(
    model_path: str,
    net_h: int,
    net_w: int,
    workdir: str,
    opset: int = _DEFAULT_OPSET,
    logger=None,
) -> Tuple[str, Dict[int, str]]:
    """Export ``model_path`` to ONNX at a static ``net_h x net_w`` input shape."""
    from ultralytics import YOLO

    # Ultralytics writes the export next to the weights; the packaged models
    # live in a share directory we would rather not touch.
    local_pt = os.path.join(workdir, os.path.basename(model_path))
    shutil.copy2(model_path, local_pt)

    model = YOLO(local_pt)
    names = {int(k): str(v) for k, v in dict(getattr(model, 'names', {}) or {}).items()}

    export_kwargs = dict(
        format='onnx',
        imgsz=(int(net_h), int(net_w)),
        opset=int(opset),
        dynamic=False,
        batch=1,
        half=False,
        device='cpu',
        verbose=False,
    )
    try:
        onnx_path = model.export(simplify=True, **export_kwargs)
    except Exception as exc:
        if logger is not None:
            logger.warn(f'ONNX simplification unavailable ({exc}); exporting unsimplified')
        onnx_path = model.export(simplify=False, **export_kwargs)

    onnx_path = str(onnx_path)
    if not os.path.exists(onnx_path):
        raise EngineBuildError(f'Ultralytics reported {onnx_path} but the file is missing')
    return onnx_path, names


# ---------------------------------------------------------------------------
# TensorRT build
# ---------------------------------------------------------------------------
def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    workspace_mib: int = _DEFAULT_WORKSPACE_MIB,
    fp16_io: bool = False,
    logger=None,
) -> Dict[str, object]:
    """Compile ``onnx_path`` into a serialized TensorRT engine.

    Returns a dict describing what was *actually* built, which is not always
    what was asked for: if the platform reports no fast FP16 the engine comes
    out FP32, and callers need to know that rather than assume.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, '')
    builder = trt.Builder(trt_logger)

    explicit = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    flags = (1 << int(explicit)) if explicit is not None else 0
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, 'rb') as handle:
        if not parser.parse(handle.read()):
            errors = '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise EngineBuildError(f'ONNX parse failed: {errors}')

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_mib) * (1 << 20)
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # pragma: no cover - TensorRT < 8.4
        config.max_workspace_size = workspace_bytes

    want_fp16 = str(precision).lower() == 'fp16'
    fp16_applied = False
    io_dtype = 'fp32'
    if want_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            fp16_applied = True
        elif logger is not None:
            logger.warn(
                'Platform reports no fast FP16; building an FP32 engine instead. '
                'The cached engine will be recorded as fp32.'
            )

    # Optional FP16 network I/O.
    #
    # With only the FP16 builder flag, TensorRT keeps the network input and
    # output in FP32 and wraps the graph in reformat (cast) nodes. Forcing the
    # boundary tensors to FP16 removes those two casts and halves the bytes
    # moved at the boundary. It is off by default because it changes the
    # numerical type of the emitted box coordinates (~0.5 px quantisation at a
    # 640 px network width), and that is not a change worth making silently.
    if fp16_applied and fp16_io:
        try:
            for index in range(network.num_inputs):
                network.get_input(index).dtype = trt.DataType.HALF
            for index in range(network.num_outputs):
                network.get_output(index).dtype = trt.DataType.HALF
            io_dtype = 'fp16'
        except Exception as exc:
            if logger is not None:
                logger.warn(f'Could not force FP16 network I/O ({exc}); keeping FP32 I/O')

    started = time.time()
    if logger is not None:
        logger.info(
            f'Building TensorRT engine (requested {precision}, '
            f'fp16 flag {"applied" if fp16_applied else "not applied"}, '
            f'io {io_dtype}); this takes a few minutes and only happens once '
            'per model/shape/device.'
        )
    if hasattr(builder, 'build_serialized_network'):
        serialized = builder.build_serialized_network(network, config)
    else:  # pragma: no cover - TensorRT < 8.0
        engine = builder.build_engine(network, config)
        serialized = engine.serialize() if engine is not None else None
    if serialized is None:
        raise EngineBuildError(
            'TensorRT returned no engine. The usual causes are an unsupported ONNX '
            'opset or not enough free GPU memory during the build.'
        )

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    tmp_path = f'{engine_path}.partial'
    with open(tmp_path, 'wb') as handle:
        handle.write(serialized if isinstance(serialized, bytes) else bytes(serialized))
    os.replace(tmp_path, engine_path)   # atomic: never leave a half-written engine

    if logger is not None:
        logger.info(f'Engine written to {engine_path} in {time.time() - started:.0f} s')
    return {
        'precision_built': 'fp16' if fp16_applied else 'fp32',
        'fp16_flag_applied': fp16_applied,
        'io_dtype': io_dtype,
    }


def write_sidecar(
    engine_path: str,
    spec: EngineSpec,
    names: Dict[int, str],
    build_info: Optional[Dict[str, object]] = None,
) -> str:
    """Store class names, provenance and the *built* precision next to the engine."""
    build_info = build_info or {}
    payload = {
        'source_model': os.path.abspath(spec.model_path),
        'source_sha256_10': _file_digest(spec.model_path),
        'net_h': int(spec.net_h),
        'net_w': int(spec.net_w),
        'precision': spec.precision,                       # requested
        'precision_built': build_info.get('precision_built', spec.precision),
        'fp16_flag_applied': build_info.get('fp16_flag_applied'),
        'io_dtype': build_info.get('io_dtype', 'fp32'),
        'tensorrt': tensorrt_version(),
        'gpu': gpu_slug(),
        'names': {str(k): v for k, v in names.items()},
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    path = os.path.splitext(engine_path)[0] + '.json'
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_sidecar_summary(engine_path: str) -> Dict:
    """Read the metadata written next to an engine (empty dict if absent)."""
    path = os.path.splitext(engine_path)[0] + '.json'
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def ensure_engine(
    model_path: str,
    net_h: int,
    net_w: int,
    precision: str = 'fp16',
    cache_dir: str = '/engine_cache',
    opset: int = _DEFAULT_OPSET,
    workspace_mib: int = _DEFAULT_WORKSPACE_MIB,
    allow_build: bool = True,
    fp16_io: bool = False,
    logger=None,
) -> str:
    """Return a usable engine path, building it if the cache misses."""
    if not os.path.exists(model_path):
        raise EngineBuildError(f'Model checkpoint not found: {model_path}')

    spec = EngineSpec(
        model_path, int(net_h), int(net_w), str(precision).lower(), bool(fp16_io))
    os.makedirs(cache_dir, exist_ok=True)
    engine_path = engine_path_for(spec, cache_dir)

    if os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
        if logger is not None:
            logger.info(f'Using cached TensorRT engine {engine_path}')
        return engine_path
    if not allow_build:
        raise EngineBuildError(
            f'No cached engine at {engine_path} and auto_build_engine is disabled. '
            'Run: make engine'
        )

    with tempfile.TemporaryDirectory(prefix='trt_export_', dir=cache_dir) as workdir:
        onnx_path, names = export_onnx(
            model_path, spec.net_h, spec.net_w, workdir, opset=opset, logger=logger)
        build_info = build_engine_from_onnx(
            onnx_path, engine_path, spec.precision, workspace_mib,
            fp16_io=spec.fp16_io, logger=logger)
    write_sidecar(engine_path, spec, names, build_info)
    return engine_path
