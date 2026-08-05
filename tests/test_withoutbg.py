import io
import json
from unittest.mock import MagicMock, patch
from urllib import error

import pytest
from PIL import Image

from rembg.sessions.withoutbg import MAX_UPLOAD_BYTES, WithoutBgSession


def _png_bytes(mode: str, size=(32, 24), color=0):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("WITHOUTBG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        WithoutBgSession("withoutbg", None)


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_from_env")
    session = WithoutBgSession("withoutbg", None)
    assert session.api_key == "sk_from_env"


def test_api_key_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_from_env")
    session = WithoutBgSession("withoutbg", None, api_key="sk_kwarg")
    assert session.api_key == "sk_kwarg"


def test_name_and_download_models():
    assert WithoutBgSession.name() == "withoutbg"
    assert WithoutBgSession.download_models() == ""


def test_capability_flags():
    assert WithoutBgSession.is_local() is False
    assert WithoutBgSession.requires_credentials() is True
    assert WithoutBgSession.has_usage_cost() is True


def test_predict_rejects_oversized_upload(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_test")
    session = WithoutBgSession("withoutbg", None)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    with patch.object(
        img,
        "save",
        side_effect=lambda buf, *args, **kwargs: buf.write(
            b"x" * (MAX_UPLOAD_BYTES + 1)
        ),
    ):
        with patch("rembg.sessions.withoutbg.request.urlopen") as mock_urlopen:
            with pytest.raises(ValueError, match="20 MB limit"):
                session.predict(img)
            mock_urlopen.assert_not_called()


def test_predict_returns_mask_matching_input_size(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_test")
    session = WithoutBgSession("withoutbg", None)

    img = Image.new("RGB", (40, 30), (255, 0, 0))
    # API returns a differently sized mask; session should resize to input.
    mask_bytes = _png_bytes("L", size=(20, 15), color=200)

    mock_resp = MagicMock()
    mock_resp.read.return_value = mask_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch(
        "rembg.sessions.withoutbg.request.urlopen", return_value=mock_resp
    ) as mock_urlopen:
        masks = session.predict(img)

    assert len(masks) == 1
    assert masks[0].mode == "L"
    assert masks[0].size == img.size

    req = mock_urlopen.call_args[0][0]
    assert req.get_header("X-api-key") == "sk_test"
    assert "multipart/form-data" in req.get_header("Content-type")
    assert b'name="file"' in req.data


def test_predict_http_error_raises(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_test")
    session = WithoutBgSession("withoutbg", None)
    img = Image.new("RGB", (10, 10), (0, 0, 0))

    http_error = error.HTTPError(
        url="https://api.withoutbg.com/v1.0/alpha-channel",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(json.dumps({"detail": "Invalid API Key"}).encode("utf-8")),
    )

    with patch("rembg.sessions.withoutbg.request.urlopen", side_effect=http_error):
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            session.predict(img)


def test_session_registered():
    from rembg.sessions import sessions

    assert "withoutbg" in sessions
    assert sessions["withoutbg"] is WithoutBgSession
