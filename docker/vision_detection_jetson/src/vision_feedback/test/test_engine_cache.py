"""Engine cache naming and the precision actually recorded for an engine."""

import json

import pytest

from vision_perception import engine_builder
from vision_perception.engine_builder import EngineSpec


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / 'best_gazebo_new.pt'
    path.write_bytes(b'pretend-checkpoint')
    return str(path)


def test_cache_name_encodes_everything_that_changes_the_binary(weights, monkeypatch, tmp_path):
    monkeypatch.setattr(engine_builder, 'tensorrt_version', lambda: '10.3.0')
    monkeypatch.setattr(engine_builder, 'gpu_slug', lambda: 'orin')

    name = engine_builder.engine_path_for(
        EngineSpec(weights, 448, 640, 'fp16'), str(tmp_path))

    assert 'best_gazebo_new' in name
    assert '448x640' in name
    assert '__fp16__' in name
    assert 'trt1030' in name
    assert 'orin' in name


def test_shape_change_produces_a_different_cache_entry(weights, monkeypatch, tmp_path):
    monkeypatch.setattr(engine_builder, 'tensorrt_version', lambda: '10.3.0')
    monkeypatch.setattr(engine_builder, 'gpu_slug', lambda: 'orin')

    a = engine_builder.engine_path_for(EngineSpec(weights, 384, 640, 'fp16'), str(tmp_path))
    b = engine_builder.engine_path_for(EngineSpec(weights, 448, 640, 'fp16'), str(tmp_path))

    assert a != b


def test_fp16_io_produces_a_different_cache_entry(weights, monkeypatch, tmp_path):
    monkeypatch.setattr(engine_builder, 'tensorrt_version', lambda: '10.3.0')
    monkeypatch.setattr(engine_builder, 'gpu_slug', lambda: 'orin')

    plain = engine_builder.engine_path_for(
        EngineSpec(weights, 448, 640, 'fp16', False), str(tmp_path))
    io_fp16 = engine_builder.engine_path_for(
        EngineSpec(weights, 448, 640, 'fp16', True), str(tmp_path))

    assert plain != io_fp16
    assert '__fp16io__' in io_fp16


def test_changed_weights_produce_a_different_cache_entry(weights, monkeypatch, tmp_path):
    monkeypatch.setattr(engine_builder, 'tensorrt_version', lambda: '10.3.0')
    monkeypatch.setattr(engine_builder, 'gpu_slug', lambda: 'orin')
    spec = EngineSpec(weights, 448, 640, 'fp16')
    before = engine_builder.engine_path_for(spec, str(tmp_path))

    with open(weights, 'wb') as handle:
        handle.write(b'a-different-checkpoint')

    assert engine_builder.engine_path_for(spec, str(tmp_path)) != before


def test_sidecar_records_the_precision_that_was_actually_built(weights, tmp_path, monkeypatch):
    """A platform without fast FP16 must not leave an engine labelled fp16."""
    monkeypatch.setattr(engine_builder, 'tensorrt_version', lambda: '10.3.0')
    monkeypatch.setattr(engine_builder, 'gpu_slug', lambda: 'orin')
    engine = tmp_path / 'engine.engine'
    engine.write_bytes(b'x')

    spec = EngineSpec(weights, 448, 640, 'fp16')
    path = engine_builder.write_sidecar(
        str(engine), spec, {0: 'body', 1: 'head-YJtu'},
        {'precision_built': 'fp32', 'fp16_flag_applied': False, 'io_dtype': 'fp32'},
    )
    payload = json.loads(open(path).read())

    assert payload['precision'] == 'fp16'          # what was requested
    assert payload['precision_built'] == 'fp32'    # what TensorRT produced
    assert payload['fp16_flag_applied'] is False
    assert payload['names'] == {'0': 'body', '1': 'head-YJtu'}


def test_sidecar_summary_is_empty_for_a_missing_file(tmp_path):
    assert engine_builder.load_sidecar_summary(str(tmp_path / 'nope.engine')) == {}
