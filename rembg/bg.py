import io
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion

from .detect import ort_session, predict


def alpha_matting_cutout(
    img: Image,
    mask: Image,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> Image:
    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout


def naive_cutout(img: Image, mask: Image) -> Image:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def remove(
    data: bytes,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    session: Optional[ort.InferenceSession] = None,
    only_mask: bool = False,
) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")

    if session is None:
        session = ort_session("u2net")

    mask = predict(session, np.array(img)).convert("L")
    mask = mask.resize(img.size, Image.LANCZOS)

    if only_mask:
        cutout = mask

    elif alpha_matting:
        try:
            cutout = alpha_matting_cutout(
                img,
                mask,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_size,
            )
        except Exception:
            cutout = naive_cutout(img, mask)
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()
