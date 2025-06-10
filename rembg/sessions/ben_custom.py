import os
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class BenCustomSession(BaseSession):
    """This is a class representing a custom session for the Ben model."""

    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, *args, **kwargs):
        """
        Initialize a new BenCustomSession object.

        Parameters:
            model_name (str): The name of the model.
            sess_opts: The session options.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")

        super().__init__(model_name, sess_opts, *args, **kwargs)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the mask image for the input image.

        This method takes a PILImage object as input and returns a list of PILImage objects as output. It performs several image processing operations to generate the mask image.

        Parameters:
            img (PILImage): The input image.

        Returns:
            List[PILImage]: A list of PILImage objects representing the generated mask image.
        """

        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1024, 1024)),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Download the model files.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The absolute path to the model files.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")

        return os.path.abspath(os.path.expanduser(model_path))

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Get the name of the model.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model.
        """
        return "ben_custom"
