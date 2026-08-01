import io
import json
import os
import uuid
from typing import List
from urllib import error, request

from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession

API_URL = "https://api.withoutbg.com/v1.0/alpha-channel"
DEFAULT_TIMEOUT = 60


class WithoutBgSession(BaseSession):
    """Session that removes backgrounds via the withoutBG cloud API."""

    def __init__(self, model_name: str, sess_opts, *args, **kwargs):
        """
        Initialize a WithoutBgSession.

        Does not call BaseSession.__init__ because no local ONNX model is used.

        Parameters:
            model_name (str): The name of the model.
            sess_opts: Ignored; accepted for BaseSession compatibility.
            *args: Additional positional arguments.
            **kwargs: May include api_key. Falls back to WITHOUTBG_API_KEY.

        Raises:
            ValueError: If no API key is provided.
        """
        self.model_name = model_name
        self.api_key = kwargs.get("api_key") or os.getenv("WITHOUTBG_API_KEY")
        if not self.api_key:
            raise ValueError(
                "withoutbg requires an API key. Pass api_key=... to new_session() "
                "or set the WITHOUTBG_API_KEY environment variable. "
                "Get 50 free credits at https://withoutbg.com/signup?ref=rembg"
            )
        self.timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predict the alpha mask for the input image via the withoutBG API.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: A single-item list with the L-mode alpha mask.

        Raises:
            RuntimeError: If the API returns a non-200 response.
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        boundary = f"----rembg{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                mask_bytes = resp.read()
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(
                f"withoutbg API error ({e.code}): {detail}"
            ) from e
        except error.URLError as e:
            raise RuntimeError(f"withoutbg API request failed: {e.reason}") from e

        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """No local model to download for the withoutBG cloud API."""
        return ""

    @classmethod
    def name(cls, *args, **kwargs):
        """Return the session name."""
        return "withoutbg"
