import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from scipy.special import log_softmax

from .base import BaseSession

pallete1 = [
    0,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    0,
    0,
    0,
    0,
]

pallete2 = [
    0,
    0,
    0,
    0,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    0,
]

pallete3 = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    255,
    255,
    255,
]


class Unet2ClothSession(BaseSession):
    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (768, 768)
            ),
        )

        pred = ort_outs
        pred = log_softmax(pred[0], 1)
        pred = np.argmax(pred, axis=1, keepdims=True)
        pred = np.squeeze(pred, 0)
        pred = np.squeeze(pred, 0)

        mask = Image.fromarray(pred.astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        masks = []

        mask1 = mask.copy()
        mask1.putpalette(pallete1)
        mask1 = mask1.convert("RGB").convert("L")
        masks.append(mask1)

        mask2 = mask.copy()
        mask2.putpalette(pallete2)
        mask2 = mask2.convert("RGB").convert("L")
        masks.append(mask2)

        mask3 = mask.copy()
        mask3.putpalette(pallete3)
        mask3 = mask3.convert("RGB").convert("L")
        masks.append(mask3)

        return masks

    @classmethod
    def download_models(cls, *args, **kwargs):
        fname = f"{cls.name()}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx",
            "md5:2434d1f3cb744e0e49386c906e5a08bb",
            fname=fname,
            path=cls.u2net_home(),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        return "u2net_cloth_seg"
