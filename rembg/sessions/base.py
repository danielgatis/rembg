import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


class BaseSession:
    """This is a base class for managing a session with a machine learning model."""

    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, *args, **kwargs):
        """Initialize an instance of the BaseSession class."""
        self.model_name = model_name

        if "providers" in kwargs and isinstance(kwargs["providers"], list):
            providers = kwargs.pop("providers")
        else:
            device_type = ort.get_device()
            if (
                device_type == "GPU"
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif (
                device_type[0:3] == "GPU"
                and "ROCMExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            elif "OpenVINOExecutionProvider" in ort.get_available_providers():
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.inner_session = ort.InferenceSession(
            str(self.__class__.download_models(*args, **kwargs)),
            sess_options=sess_opts,
            providers=providers,
        )

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / max(np.max(im_ary), 1e-6)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        raise NotImplementedError

    @classmethod
    def is_local(cls, *args, **kwargs) -> bool:
        """Whether inference runs on this machine.

        Return False for sessions that send the image to a remote service, so
        callers can tell the user their data leaves the machine.
        """
        return True

    @classmethod
    def requires_credentials(cls, *args, **kwargs) -> bool:
        """Whether constructing this session needs a credential.

        Sessions that return True cannot be instantiated from a bare model name,
        so interfaces that build sessions on the user's behalf should skip them
        rather than fail at construction time.
        """
        return False

    @classmethod
    def has_usage_cost(cls, *args, **kwargs) -> bool:
        """Whether each prediction bills the user.

        Batch entry points should warn before spending on a large run.
        """
        return False

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        return os.getenv("MODEL_CHECKSUM_DISABLED", None) is not None

    @classmethod
    def legacy_home(cls, *args, **kwargs):
        """The pre-0.0.0 flat model directory, kept for reading only.

        Models used to live directly in `~/.u2net`, whatever their architecture.
        Nothing is written here anymore, but existing downloads are still found
        so upgrading does not force a re-download.
        """
        return os.path.expanduser(
            os.getenv(
                "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
            )
        )

    @classmethod
    def rembg_home(cls, *args, **kwargs):
        """Root directory for rembg data.

        `U2NET_HOME` still wins when set, so anyone who already points it at a
        shared or pre-seeded directory keeps working.
        """
        if os.getenv("U2NET_HOME"):
            return cls.legacy_home(*args, **kwargs)

        xdg = os.getenv("XDG_DATA_HOME")
        default = os.path.join(xdg, "rembg") if xdg else os.path.join("~", ".rembg")

        return os.path.expanduser(os.getenv("REMBG_HOME", default))

    @classmethod
    def u2net_home(cls, *args, **kwargs):
        """Deprecated alias for `rembg_home`.

        Kept because third-party code calls it. New code should use
        `rembg_home` for the root or `model_dir` for a model's own directory.
        """
        return cls.rembg_home(*args, **kwargs)

    @classmethod
    def model_dir(cls, *args, **kwargs):
        """Directory holding this model's files: `<home>/models/<name>`."""
        return os.path.join(
            cls.rembg_home(*args, **kwargs), "models", cls.name(*args, **kwargs)
        )

    @classmethod
    def resolve_existing(cls, fname, *args, **kwargs):
        """Return an already-downloaded copy of `fname`, or None.

        Looks in the per-model directory first, then in the old flat layout, so
        a model downloaded by an earlier version is reused instead of fetched
        again.
        """
        candidates = [
            os.path.join(cls.model_dir(*args, **kwargs), fname),
            os.path.join(cls.legacy_home(*args, **kwargs), fname),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return None

    @classmethod
    def validate_model_path(cls, *args, **kwargs):
        """Resolve the caller's `model_path`, refusing anything outside the
        model directories.

        Both the current root and the legacy one are accepted, so a custom model
        that was placed under `~/.u2net` keeps loading after the move.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")

        abs_path = os.path.abspath(os.path.expanduser(model_path))

        allowed = [
            os.path.abspath(cls.rembg_home(*args, **kwargs)),
            os.path.abspath(cls.legacy_home(*args, **kwargs)),
        ]

        for root in allowed:
            if abs_path == root or abs_path.startswith(root + os.sep):
                return abs_path

        raise ValueError(
            f"model_path must be within the models directory: {allowed[0]}"
        )

    @classmethod
    def download_models(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def name(cls, *args, **kwargs):
        raise NotImplementedError
