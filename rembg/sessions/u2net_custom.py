import os
from typing import List

import numpy as np
import onnxruntime as ort
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class U2netCustomSession(BaseSession):
    def __init__(
        self,
        model_name: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")

        super().__init__(model_name, sess_opts, providers, *args, **kwargs)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        model_path = kwargs.get("model_path")
        if model_path is None:
            return

        return os.path.abspath(os.path.expanduser(model_path))

    @classmethod
    def name(cls, *args, **kwargs):
        return "u2net_custom"
