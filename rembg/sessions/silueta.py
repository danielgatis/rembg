import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class SiluetaSession(BaseSession):
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
        fname = f"{cls.name()}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
            "md5:55e59e0d8062d2f5d013f4725ee84782",
            fname=fname,
            path=cls.u2net_home(),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        return "silueta"
