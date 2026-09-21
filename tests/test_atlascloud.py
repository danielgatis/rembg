import io
import json
from unittest.mock import MagicMock, patch
from urllib import error

import pytest
from PIL import Image

from rembg.sessions.atlascloud import (
    DEFAULT_MODEL,
    MAX_UPLOAD_BYTES,
    AtlasCloudSession,
)


def _png_bytes(mode: str, size=(32, 24), color=0):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _json_resp(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _bytes_resp(raw):
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AtlasCloudSession("atlascloud", None)


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_from_env")
    session = AtlasCloudSession("atlascloud", None)
    assert session.api_key == "apikey_from_env"
    assert session.model == DEFAULT_MODEL


def test_api_key_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_from_env")
    session = AtlasCloudSession("atlascloud", None, api_key="apikey_kwarg")
    assert session.api_key == "apikey_kwarg"


def test_model_override(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    monkeypatch.setenv("ATLASCLOUD_MODEL", "youchuan/v8.1/remove-background")
    assert AtlasCloudSession("atlascloud", None).model == (
        "youchuan/v8.1/remove-background"
    )
    session = AtlasCloudSession("atlascloud", None, model="other/model")
    assert session.model == "other/model"


def test_name_and_download_models():
    assert AtlasCloudSession.name() == "atlascloud"
    assert AtlasCloudSession.download_models() == ""


def test_capability_flags():
    assert AtlasCloudSession.is_local() is False
    assert AtlasCloudSession.requires_credentials() is True
    assert AtlasCloudSession.has_usage_cost() is True


def test_predict_rejects_oversized_upload(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    session = AtlasCloudSession("atlascloud", None)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    with patch.object(
        img,
        "save",
        side_effect=lambda buf, *args, **kwargs: buf.write(
            b"x" * (MAX_UPLOAD_BYTES + 1)
        ),
    ):
        with patch("rembg.sessions.atlascloud.request.urlopen") as mock_urlopen:
            with pytest.raises(ValueError, match="20 MB limit"):
                session.predict(img)
            mock_urlopen.assert_not_called()


def test_predict_polls_then_returns_mask_matching_input_size(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    monkeypatch.setattr("rembg.sessions.atlascloud.POLL_INTERVAL", 0)
    session = AtlasCloudSession("atlascloud", None)

    img = Image.new("RGB", (40, 30), (255, 0, 0))
    # The API returns a cutout whose alpha channel is the mask, at its own size.
    cutout_bytes = _png_bytes("RGBA", size=(20, 15), color=(1, 2, 3, 200))

    responses = [
        _json_resp({"data": {"id": "req-1", "status": "processing"}}),
        _json_resp({"data": {"status": "processing", "outputs": None}}),
        _json_resp({"data": {"status": "completed", "outputs": ["https://cdn/x.png"]}}),
        _bytes_resp(cutout_bytes),
    ]

    with patch(
        "rembg.sessions.atlascloud.request.urlopen", side_effect=responses
    ) as mock_urlopen:
        masks = session.predict(img)

    assert len(masks) == 1
    assert masks[0].mode == "L"
    assert masks[0].size == img.size

    submit_req = mock_urlopen.call_args_list[0][0][0]
    assert submit_req.get_header("Authorization") == "Bearer apikey_test"
    body = json.loads(submit_req.data.decode("utf-8"))
    assert body["model"] == DEFAULT_MODEL
    assert body["image"].startswith("data:image/png;base64,")


def test_predict_raises_when_prediction_fails(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    monkeypatch.setattr("rembg.sessions.atlascloud.POLL_INTERVAL", 0)
    session = AtlasCloudSession("atlascloud", None)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    responses = [
        _json_resp({"data": {"id": "req-1", "status": "processing"}}),
        _json_resp({"data": {"status": "failed", "error": "upstream rejected"}}),
    ]

    with patch("rembg.sessions.atlascloud.request.urlopen", side_effect=responses):
        with pytest.raises(RuntimeError, match="upstream rejected"):
            session.predict(img)


def test_predict_times_out(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    monkeypatch.setattr("rembg.sessions.atlascloud.POLL_INTERVAL", 0)
    session = AtlasCloudSession("atlascloud", None, poll_timeout=0)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    responses = [
        _json_resp({"data": {"id": "req-1", "status": "processing"}}),
        _json_resp({"data": {"status": "processing"}}),
    ]

    with patch("rembg.sessions.atlascloud.request.urlopen", side_effect=responses):
        with pytest.raises(RuntimeError, match="did not finish within"):
            session.predict(img)


def test_predict_http_error_raises(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "apikey_test")
    session = AtlasCloudSession("atlascloud", None)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    http_error = error.HTTPError(
        url="https://api.atlascloud.ai/api/v1/model/generateImage",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(json.dumps({"message": "invalid api key"}).encode("utf-8")),
    )

    with patch("rembg.sessions.atlascloud.request.urlopen", side_effect=http_error):
        with pytest.raises(RuntimeError, match="invalid api key"):
            session.predict(img)


def test_session_registered():
    from rembg.sessions import sessions, sessions_names_downloadable

    assert "atlascloud" in sessions
    assert sessions["atlascloud"] is AtlasCloudSession
    # Remote backends must not appear as a download that does nothing.
    assert "atlascloud" not in sessions_names_downloadable
