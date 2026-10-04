"""Backend resolution: TensorRT first, PyTorch fallback, CPU only on CUDA OOM."""

import pytest

from vision_perception import backends
from vision_perception.backends import ClassRouter, build_detector, is_cuda_oom
from vision_perception.backends import tensorrt_backend


class _Logger:
    def __init__(self):
        self.warns = []
        self.infos = []

    def info(self, msg):
        self.infos.append(str(msg))

    def warn(self, msg):
        self.warns.append(str(msg))

    warning = warn

    def error(self, msg):
        self.warns.append(str(msg))


class _FakeBackend:
    def __init__(self, kind, device):
        self.info = backends.BackendInfo(
            kind=kind, device=device, precision='fp16',
            net_h=384, net_w=640, model_path='x.pt',
            names={0: 'body', 1: 'head-YJtu'},
        )
        self.warmed = 0
        self.warmup_shape = None

    def warmup(self, frames=3, source_shape=None):
        self.warmed += 1
        self.warmup_shape = source_shape

    def infer(self, frame):
        return backends.EMPTY_DETECTIONS

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _pretend_tensorrt_is_installed(monkeypatch):
    """The host running these tests has no TensorRT; the selection logic still must be exercised."""
    monkeypatch.setattr(tensorrt_backend, 'tensorrt_available', lambda: True)


def _kwargs(**overrides):
    base = dict(
        model_path='model.pt',
        backend='auto',
        device='cuda:0',
        net_shape=(384, 640),
        logger=_Logger(),
    )
    base.update(overrides)
    return base


def test_tensorrt_is_preferred_when_available(monkeypatch):
    monkeypatch.setattr(
        backends, '_load_tensorrt',
        lambda *a, **k: _FakeBackend('tensorrt', 'cuda:0'))
    monkeypatch.setattr(
        backends, '_load_torch',
        lambda *a, **k: pytest.fail('torch backend must not be built'))

    detector, warnings = build_detector(**_kwargs())

    assert detector.info.kind == 'tensorrt'
    assert detector.warmed == 1
    assert warnings == []


def test_warmup_uses_the_real_camera_geometry(monkeypatch):
    """Warming up at the camera size stops the first live frame reallocating."""
    monkeypatch.setattr(
        backends, '_load_tensorrt',
        lambda *a, **k: _FakeBackend('tensorrt', 'cuda:0'))

    detector, _ = build_detector(**_kwargs(warmup_shape=(680, 1024)))

    assert detector.warmup_shape == (680, 1024)


def test_tensorrt_failure_falls_back_to_torch_with_a_visible_warning(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError('no engine and no builder')

    monkeypatch.setattr(backends, '_load_tensorrt', _boom)
    monkeypatch.setattr(
        backends, '_load_torch',
        lambda *a, **k: _FakeBackend('torch', 'cuda:0'))

    logger = _Logger()
    detector, warnings = build_detector(**_kwargs(logger=logger))

    assert detector.info.kind == 'torch'
    assert any('TensorRT backend unavailable' in w for w in warnings)
    assert any('TensorRT backend unavailable' in w for w in logger.warns)


def test_cuda_oom_on_torch_falls_back_to_cpu(monkeypatch):
    devices = []

    def _load(model_path, device, net_shape, **kwargs):
        devices.append(device)
        if str(device).startswith('cuda'):
            raise RuntimeError('CUDA out of memory. Tried to allocate 20.00 MiB.')
        return _FakeBackend('torch', device)

    monkeypatch.setattr(
        backends, '_load_tensorrt',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no trt')))
    monkeypatch.setattr(backends, '_load_torch', _load)

    detector, warnings = build_detector(**_kwargs())

    assert detector.info.device == 'cpu'
    assert devices == ['cuda:0', 'cpu']
    assert any('falling back to CPU' in w for w in warnings)


def test_non_oom_torch_failure_is_not_suppressed(monkeypatch):
    monkeypatch.setattr(
        backends, '_load_tensorrt',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no trt')))

    def _load(*_a, **_k):
        raise RuntimeError('model file missing')

    monkeypatch.setattr(backends, '_load_torch', _load)

    with pytest.raises(RuntimeError, match='model file missing'):
        build_detector(**_kwargs())


def test_is_cuda_oom_only_matches_memory_errors():
    assert is_cuda_oom(RuntimeError('CUDA out of memory. Tried to allocate 20 MiB'))
    assert not is_cuda_oom(RuntimeError('model file missing'))
    assert not is_cuda_oom(RuntimeError('CUDA error: invalid device ordinal'))


def test_class_router_maps_the_deployed_two_class_model():
    router = ClassRouter({0: 'body', 1: 'head-YJtu'})

    assert router.head_ids == (1,)
    assert router.body_ids == (0,)
    assert router.is_valid()


def test_class_router_reports_an_unusable_model():
    router = ClassRouter({0: 'person', 1: 'car'})

    assert router.head_ids == ()
    assert router.body_ids == ()
    assert not router.is_valid()
