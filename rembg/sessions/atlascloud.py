import base64
import io
import json
import os
import time
from typing import List
from urllib import error, request

from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession

API_BASE = "https://api.atlascloud.ai"
GENERATE_URL = f"{API_BASE}/api/v1/model/generateImage"
PREDICTION_URL = f"{API_BASE}/api/v1/model/prediction"
DEFAULT_MODEL = "youchuan/v8.2/remove-background"
DEFAULT_TIMEOUT = 60
POLL_INTERVAL = 2
POLL_TIMEOUT = 180
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        _VERSION = version("rembg")
    except PackageNotFoundError:
        _VERSION = "0.0.0"
except ImportError:
    _VERSION = "0.0.0"

USER_AGENT = f"rembg/{_VERSION}"


class AtlasCloudSession(BaseSession):
    """Session that removes backgrounds via the Atlas Cloud API."""

    def __init__(self, model_name: str, sess_opts, *args, **kwargs):
        """
        Initialize an AtlasCloudSession.

        Does not call BaseSession.__init__ because no local ONNX model is used.

        Parameters:
            model_name (str): The name of the model.
            sess_opts: Ignored; accepted for BaseSession compatibility.
            *args: Additional positional arguments.
            **kwargs: May include api_key, model and timeout. api_key falls back
                to ATLASCLOUD_API_KEY, model to ATLASCLOUD_MODEL.

        Raises:
            ValueError: If no API key is provided.
        """
        self.model_name = model_name
        api_key = kwargs.get("api_key") or os.getenv("ATLASCLOUD_API_KEY")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(
                "atlascloud requires an API key. Pass api_key=... to new_session() "
                "or set the ATLASCLOUD_API_KEY environment variable. "
                "Keys are created at https://www.atlascloud.ai"
            )
        self.api_key: str = api_key
        self.model: str = kwargs.get("model") or os.getenv(
            "ATLASCLOUD_MODEL", DEFAULT_MODEL
        )
        self.timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        self.poll_timeout = kwargs.get("poll_timeout", POLL_TIMEOUT)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
        }

    def _request_json(self, url: str, data: bytes = None) -> dict:
        headers = self._headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers)

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("message", detail) or detail
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(f"atlascloud API error ({e.code}): {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"atlascloud API request failed: {e.reason}") from e

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predict the alpha mask for the input image via the Atlas Cloud API.

        The endpoint is asynchronous: the image is submitted, the returned
        request id is polled until the prediction completes, and the resulting
        cutout's alpha channel is used as the mask.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: A single-item list with the L-mode alpha mask.

        Raises:
            ValueError: If the PNG-encoded image exceeds the 20 MB upload limit.
            RuntimeError: If the API errors, the prediction fails, or it does
                not finish within the polling timeout.
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"atlascloud upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit "
                f"({len(image_bytes)} bytes after PNG encode). "
                "Resize or compress the image and try again."
            )

        payload = json.dumps(
            {
                "model": self.model,
                "image": "data:image/png;base64,"
                + base64.b64encode(image_bytes).decode("ascii"),
                "enable_base64_output": False,
            }
        ).encode("utf-8")

        submitted = self._request_json(GENERATE_URL, payload).get("data") or {}
        request_id = submitted.get("id")
        if not request_id:
            raise RuntimeError(
                f"atlascloud response did not contain a request id: {submitted}"
            )

        outputs = self._poll(request_id)
        cutout_bytes = self._fetch(outputs[0])

        cutout = Image.open(io.BytesIO(cutout_bytes))
        if "A" not in cutout.getbands():
            cutout = cutout.convert("RGBA")
        mask = cutout.getchannel("A")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    def _poll(self, request_id: str) -> List[str]:
        """Poll a prediction until it produces outputs, or fails, or times out."""
        deadline = time.monotonic() + self.poll_timeout

        while True:
            result = (
                self._request_json(f"{PREDICTION_URL}/{request_id}").get("data") or {}
            )
            outputs = result.get("outputs")
            if outputs:
                return outputs

            status = (result.get("status") or "").lower()
            if status in ("failed", "error", "canceled", "cancelled"):
                raise RuntimeError(
                    f"atlascloud prediction {status}: "
                    f"{result.get('error') or 'no error detail returned'}"
                )

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"atlascloud prediction {request_id} did not finish within "
                    f"{self.poll_timeout}s (last status: {status or 'unknown'})"
                )

            time.sleep(POLL_INTERVAL)

    def _fetch(self, url: str) -> bytes:
        """Download a prediction output."""
        req = request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except error.HTTPError as e:
            raise RuntimeError(
                f"atlascloud output download failed ({e.code}): {url}"
            ) from e
        except error.URLError as e:
            raise RuntimeError(f"atlascloud output download failed: {e.reason}") from e

    @classmethod
    def is_local(cls, *args, **kwargs) -> bool:
        """Inference runs on Atlas Cloud's servers, not this machine."""
        return False

    @classmethod
    def requires_credentials(cls, *args, **kwargs) -> bool:
        """Construction needs an API key (api_key= or ATLASCLOUD_API_KEY)."""
        return True

    @classmethod
    def has_usage_cost(cls, *args, **kwargs) -> bool:
        """Each prediction bills against the Atlas Cloud API key."""
        return True

    @classmethod
    def download_models(cls, *args, **kwargs):
        """No local model to download for the Atlas Cloud API."""
        return ""

    @classmethod
    def name(cls, *args, **kwargs):
        """Return the session name."""
        return "atlascloud"
