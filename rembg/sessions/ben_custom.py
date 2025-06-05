import os
from typing import List
import onnxruntime as ort
import numpy as np

from PIL import Image
from PIL.Image import Image as PILImage
import torch

from .base import BaseSession

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


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

    def preprocess_image(self, image):
        original_size = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        img_tensor = transform(image)

        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.numpy(), image, original_size

    def postprocess_image(self, result_np: np.ndarray, im_size: list) -> np.ndarray:

        result = torch.from_numpy(result_np)


        if len(result.shape) == 3:
            result = result.unsqueeze(0)


        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)


        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)

        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the mask image for the input image.

        This method takes a PILImage object as input and returns a list of PILImage objects as output. It performs several image processing operations to generate the mask image.

        Parameters:
            img (PILImage): The input image.

        Returns:
            List[PILImage]: A list of PILImage objects representing the generated mask image.
        """

        input_data, original_image, (w, h) = self.preprocess_image(img)

        input_name = self.inner_session.get_inputs()[0].name

        outputs = self.inner_session.run(None, {input_name: input_data})


        alpha = self.postprocess_image(outputs[0], im_size=[w, h])

        mask = Image.fromarray(alpha, mode="L")
        mask = mask.resize((w, h), Image.Resampling.LANCZOS)

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
