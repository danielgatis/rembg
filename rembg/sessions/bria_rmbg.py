import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class BriaRmBgSession(BaseSession):
    """
    This class represents a Bria-rmbg-2.0 session, which is a subclass of BaseSession.
    """

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the output masks for the input image using the inner session.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: The list of output masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024)
            ),
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
        Downloads the BRIA-RMBG 2.0 model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "sha256:5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958"
            ),
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the Bria-rmbg session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "bria-rmbg"
