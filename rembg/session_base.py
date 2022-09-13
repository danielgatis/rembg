from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


class BaseSession:
    def __init__(
        self,
        model_name: str,
        inner_session: ort.InferenceSession,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int]
    ):
        self.model_name = model_name
        self.inner_session = inner_session
        self.mean = mean
        self.std = std
        self.size = size

    def normalize(
        self, img: PILImage,
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(self.size, Image.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - self.mean[0]) / self.std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - self.mean[1]) / self.std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - self.mean[2]) / self.std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {self.inner_session.get_inputs()[0].name: np.expand_dims(tmpImg, 0).astype(np.float32)}

    def predict(self, img: PILImage) -> List[PILImage]:
        raise NotImplementedError
