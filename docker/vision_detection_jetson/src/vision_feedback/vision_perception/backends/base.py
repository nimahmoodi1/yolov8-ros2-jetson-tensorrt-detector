"""Common contract for detector backends.

Every backend returns exactly the same thing so the rest of the pipeline never
knows (or cares) whether TensorRT or PyTorch produced the boxes:

    np.ndarray of shape (N, 6), float32, rows ``[x1, y1, x2, y2, conf, cls]``
    in **source image pixel coordinates**, sorted by descending confidence.

The single-array contract is deliberate. The previous implementation walked
Ultralytics ``Boxes`` objects and called ``.cpu().numpy()`` once per box, which
is one GPU->CPU synchronisation per detection (up to 40 per frame at
``max_det=20``). Each sync stalls the CPU until the GPU pipeline drains, so the
CPU spins instead of sleeping. Returning one array means exactly one transfer
per frame.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# Column indices into the (N, 6) detection array.
X1, Y1, X2, Y2, CONF, CLS = range(6)

EMPTY_DETECTIONS = np.zeros((0, 6), dtype=np.float32)


@dataclass
class BackendInfo:
    """What a loaded backend actually is, for logs and /diagnostics."""

    kind: str                       # 'tensorrt' | 'torch'
    device: str                     # 'cuda:0' | 'cpu'
    precision: str                  # 'fp16' | 'fp32'
    net_h: int
    net_w: int
    model_path: str
    names: Dict[int, str] = field(default_factory=dict)
    detail: str = ''

    def describe(self) -> str:
        return (
            f'{self.kind}/{self.precision} {self.net_w}x{self.net_h} on {self.device}'
            + (f' ({self.detail})' if self.detail else '')
        )


class DetectorBackend(abc.ABC):
    """A model that turns a BGR frame into a detection array."""

    info: BackendInfo

    @abc.abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run detection on one HWC uint8 BGR frame."""

    @abc.abstractmethod
    def warmup(self, frames: int = 3, source_shape=None) -> None:
        """Pay first-call allocation/autotuning cost before the camera starts.

        ``source_shape`` is an optional ``(height, width)`` for the frames the
        camera will actually send, so buffers sized during warmup are the ones
        the live stream reuses.
        """

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        """Release GPU resources. Safe to call more than once."""

    @property
    def names(self) -> Dict[int, str]:
        return self.info.names

    @property
    def last_infer_ms(self) -> float:
        """Wall time of the most recent :meth:`infer` call, in milliseconds."""
        return float(getattr(self, '_last_infer_ms', 0.0))


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when an exception is a CUDA out-of-memory failure."""
    text = f'{type(exc).__name__}: {exc}'.lower()
    return 'cuda' in text and (
        'outofmemoryerror' in text
        or 'out of memory' in text
        or 'cuda error: out of memory' in text
    )
