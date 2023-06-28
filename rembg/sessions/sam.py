import os
from typing import List

import numpy as np
import onnxruntime as ort
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords: np.ndarray, original_size, target_length) -> np.ndarray:
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = coords.copy().astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def resize_longes_side(img: PILImage, size=1024):
    w, h = img.size
    if h > w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size

    return img.resize((new_w, new_h))


def pad_to_square(img: np.ndarray, size=1024):
    h, w = img.shape[:2]
    padh = size - h
    padw = size - w
    img = np.pad(img, ((0, padh), (0, padw), (0, 0)), mode="constant")
    img = img.astype(np.float32)
    return img


class SamSession(BaseSession):
    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, *args, **kwargs):
        self.model_name = model_name
        paths = self.__class__.download_models()
        self.encoder = ort.InferenceSession(
            str(paths[0]),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        )
        self.decoder = ort.InferenceSession(
            str(paths[1]),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: np.ndarray,
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        size=(1024, 1024),
        *args,
        **kwargs,
    ):
        pixel_mean = np.array([*mean]).reshape(1, 1, -1)
        pixel_std = np.array([*std]).reshape(1, 1, -1)
        x = (img - pixel_mean) / pixel_std
        return x

    def predict(
        self,
        img: PILImage,
        *args,
        **kwargs,
    ) -> List[PILImage]:
        # Preprocess image
        image = resize_longes_side(img)
        image = np.array(image)
        image = self.normalize(image)
        image = pad_to_square(image)

        input_labels = kwargs.get("input_labels")
        input_points = kwargs.get("input_points")

        if input_labels is None:
            raise ValueError("input_labels is required")
        if input_points is None:
            raise ValueError("input_points is required")

        # Transpose
        image = image.transpose(2, 0, 1)[None, :, :, :]
        # Run encoder (Image embedding)
        encoded = self.encoder.run(None, {"x": image})
        image_embedding = encoded[0]

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[
            None, :, :
        ]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = apply_coords(onnx_coord, img.size[::1], 1024).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(img.size[::-1], dtype=np.float32),
        }

        masks, _, low_res_logits = self.decoder.run(None, decoder_inputs)
        masks = masks > 0.0
        masks = [
            Image.fromarray((masks[i, 0] * 255).astype(np.uint8))
            for i in range(masks.shape[0])
        ]

        return masks

    @classmethod
    def download_models(cls, *args, **kwargs):
        fname_encoder = f"{cls.name(*args, **kwargs)}_encoder.onnx"
        fname_decoder = f"{cls.name(*args, **kwargs)}_decoder.onnx"

        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-encoder-quant.onnx",
            None
            if cls.checksum_disabled(*args, **kwargs)
            else "md5:13d97c5c79ab13ef86d67cbde5f1b250",
            fname=fname_encoder,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-decoder-quant.onnx",
            None
            if cls.checksum_disabled(*args, **kwargs)
            else "md5:fa3d1c36a3187d3de1c8deebf33dd127",
            fname=fname_decoder,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return (
            os.path.join(cls.u2net_home(*args, **kwargs), fname_encoder),
            os.path.join(cls.u2net_home(*args, **kwargs), fname_decoder),
        )

    @classmethod
    def name(cls, *args, **kwargs):
        return "sam"
