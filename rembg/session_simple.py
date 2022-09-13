from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .session_base import BaseSession


class SimpleSession(BaseSession):
    def predict(self, img: PILImage) -> List[PILImage]:
        ort_outs = self.inner_session.run(None, self.normalize(img))

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]
