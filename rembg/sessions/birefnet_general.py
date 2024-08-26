import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class BiRefNetSessionGeneral(BaseSession):
    """
    This class represents a BiRefNet-General session, which is a subclass of BaseSession.
    """

    base_url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"

    def sigmoid(self, mat):
        return 1 / (1 + np.exp(-mat))

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

        pred = self.sigmoid(ort_outs[0][:, 0, :, :])

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
        Downloads the BiRefNet-General model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        url = "".join([cls.base_url, cls.url_fname(*args, **kwargs)])
        path = cls.u2net_home(*args, **kwargs)
        pooch_instance = pooch.create(
            path=path,
            base_url=cls.base_url,
            registry={
                fname: (
                    None
                    if cls.checksum_disabled(*args, **kwargs)
                    else cls.model_hash(*args, **kwargs)
                )
            },
            urls={fname: url},
            retry_if_failed=2,
        )
        pooch_instance.fetch(fname, progressbar=True)

        return os.path.join(path, fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-general"

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-general-epoch_244.onnx"

    @classmethod
    def model_hash(cls, *args, **kwargs):
        """
        Returns the hash of the BiRefNet-General model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The hash of the model file.
        """
        return "md5:7a35a0141cbbc80de11d9c9a28f52697"
