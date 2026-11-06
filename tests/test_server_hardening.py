import importlib
import io
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

s_command = importlib.import_module("rembg.commands.s_command")


class FakeSession:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def predict(self, img, *args, **kwargs):
        return [Image.new("L", img.size, 255)]


@pytest.fixture
def png():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def built():
    sessions = []

    def fake_new_session(model_name, **kwargs):
        session = FakeSession(model_name, **kwargs)
        sessions.append(session)
        return session

    with mock.patch.object(s_command, "new_session", fake_new_session):
        yield sessions


def make_client(*cli_args):
    captured = {}
    with mock.patch.object(
        s_command.uvicorn, "run", lambda app, **kw: captured.update(app=app)
    ):
        s_command.s_command.main(["--no-ui", *cli_args], standalone_mode=False)
    return TestClient(captured["app"], raise_server_exceptions=False)


def post(client, png, model="u2net", **params):
    return client.post(
        "/api/remove", params=params, data={"model": model}, files={"file": png}
    )


@pytest.mark.parametrize("model", ["atlascloud", "withoutbg"])
def test_remote_models_are_refused_by_default(png, built, model):
    """Cloud models would bill the server's own API key to any caller."""
    assert post(make_client(), png, model=model).status_code == 422
    assert built == []


@pytest.mark.parametrize("model", ["atlascloud", "withoutbg"])
def test_remote_models_can_be_enabled(png, built, model):
    client = make_client("--allow-remote-models")
    assert post(client, png, model=model).status_code == 200


def test_model_name_must_match_exactly(png, built):
    assert post(make_client(), png, model="zzz-u2net-zzz").status_code == 422


def test_session_cache_is_bounded(png, built):
    client = make_client("--max-sessions", "2")

    for name in ["a", "b", "c"]:
        extras = json.dumps({"model_path": name})
        assert post(client, png, extras=extras).status_code == 200

    # "a" was evicted, so asking for it again builds a new session.
    post(client, png, extras=json.dumps({"model_path": "a"}))
    assert [s.kwargs["model_path"] for s in built] == ["a", "b", "c", "a"]


def test_predict_extras_reuse_the_cached_session(png, built):
    client = make_client()

    for i in range(5):
        extras = json.dumps({"sam_prompt": [{"type": "point", "data": [i, i]}]})
        assert post(client, png, extras=extras).status_code == 200

    assert len(built) == 1


@pytest.mark.parametrize(
    "extras",
    [
        '{"providers": ["CPUExecutionProvider"]}',
        '{"sess_opts": 1}',
        '{"session": 1}',
        '{"x": 1}',
        "not json",
        "[1, 2]",
    ],
)
def test_unsupported_extras_are_rejected(png, built, extras):
    assert post(make_client(), png, extras=extras).status_code == 400
    assert built == []


@pytest.mark.parametrize("bgc", ["abc", "1,2", "1,2,3,4,5", "0,0,0,256", "-1,0,0"])
def test_invalid_bgcolor_is_a_client_error(png, built, bgc):
    assert post(make_client(), png, bgc=bgc).status_code == 400


@pytest.mark.parametrize("bgc", ["255,255,255", "0,0,0,255"])
def test_valid_bgcolor_is_accepted(png, built, bgc):
    assert post(make_client(), png, bgc=bgc).status_code == 200


def test_erode_size_is_bounded(png, built):
    client = make_client()

    def remove(ae):
        data = {"model": "u2net", "a": "true", "ae": ae}
        return client.post("/api/remove", data=data, files={"file": png})

    assert remove("255").status_code == 200
    assert remove("100000").status_code == 422


@pytest.fixture
def small_upload_limit():
    with mock.patch.object(s_command, "_MAX_IMAGE_BYTES", 1024), mock.patch.object(
        s_command, "_MAX_FORM_OVERHEAD_BYTES", 0
    ):
        yield 1024


def test_upload_within_limit_is_accepted(png, built, small_upload_limit):
    assert post(make_client(), png).status_code == 200


def test_oversized_upload_is_rejected(built, small_upload_limit):
    big = b"\0" * (small_upload_limit * 2)
    r = make_client().post("/api/remove", data={"model": "u2net"}, files={"file": big})
    assert r.status_code == 413
    assert built == []


def test_oversized_chunked_upload_is_rejected(built, small_upload_limit):
    """A body without Content-Length is counted as it streams in."""

    def chunks():
        yield (
            b"--x\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
        )
        for _ in range(4):
            yield b"\0" * small_upload_limit
        yield b"\r\n--x--\r\n"

    r = make_client().post(
        "/api/remove",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413
    assert built == []
