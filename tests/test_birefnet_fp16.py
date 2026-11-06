import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxconverter_common")

from onnx import TensorProto, helper, numpy_helper

from rembg.sessions import sessions
from rembg.sessions.birefnet_general import BiRefNetSessionGeneral
from rembg.sessions.birefnet_general_fp16 import BiRefNetSessionGeneralFP16


def _tiny_float32_model(path):
    """A one-Conv stand-in for the float32 BiRefNet, so no download is needed."""
    weight = numpy_helper.from_array(np.random.rand(1, 3, 3, 3).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8, 8])],
        [weight],
    )
    onnx.save(helper.make_model(graph), path)
    return str(path)


@pytest.fixture
def home(monkeypatch, tmp_path):
    for var in ("U2NET_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REMBG_HOME", str(tmp_path / "home"))

    source = _tiny_float32_model(tmp_path / "fp32.onnx")
    calls = []

    def fake_download(*args, **kwargs):
        calls.append(1)
        return source

    monkeypatch.setattr(
        BiRefNetSessionGeneral,
        "download_models",
        classmethod(lambda cls, *a, **k: fake_download()),
    )
    return calls


def test_registered():
    assert sessions["birefnet-general-fp16"] is BiRefNetSessionGeneralFP16


def test_converts_weights_but_keeps_float32_io(home):
    path = BiRefNetSessionGeneralFP16.download_models()

    assert path == BiRefNetSessionGeneralFP16.resolve_existing(
        "birefnet-general-fp16.onnx"
    )
    model = onnx.load(path)
    assert {t.data_type for t in model.graph.initializer} == {TensorProto.FLOAT16}
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_conversion_is_cached(home):
    first = BiRefNetSessionGeneralFP16.download_models()
    second = BiRefNetSessionGeneralFP16.download_models()

    assert first == second
    assert len(home) == 1
