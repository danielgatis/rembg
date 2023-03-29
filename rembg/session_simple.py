from typing import List, Tuple

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .session_base import BaseSession


class SimpleSession(BaseSession):
    def predict(self, img: PILImage) -> List[PILImage]:
        if self.model_name == "isnet-general-use":
            mean = (0.5, 0.5, 0.5)
            std = (1., 1., 1.)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, mean, std
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
