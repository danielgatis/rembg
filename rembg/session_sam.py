from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

from .session_base import BaseSession


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
    def __init__(
        self,
        model_name: str,
        encoder: ort.InferenceSession,
        decoder: ort.InferenceSession,
    ):
        super().__init__(model_name, encoder)
        self.decoder = decoder

    def normalize(
        self,
        img: np.ndarray,
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        size=(1024, 1024),
    ):
        pixel_mean = np.array([*mean]).reshape(1, 1, -1)
        pixel_std = np.array([*std]).reshape(1, 1, -1)
        x = (img - pixel_mean) / pixel_std
        return x

    def predict_sam(
        self,
        img: PILImage,
        input_point: np.ndarray,
        input_label: np.ndarray,
    ) -> List[PILImage]:
        # Preprocess image
        image = resize_longes_side(img)
        image = np.array(image)
        image = self.normalize(image)
        image = pad_to_square(image)

        # Transpose
        image = image.transpose(2, 0, 1)[None, :, :, :]
        # Run encoder (Image embedding)
        encoded = self.inner_session.run(None, {"x": image})
        image_embedding = encoded[0]

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[
            None, :, :
        ]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[
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
