"""Learned edge refinement with ViTMatte.

Alpha matting turns a coarse mask into a real alpha channel. The `-a` path does
this with pymatting's closed-form solver, which is fast but can leave the fine
tips of hair behind and occasionally fails to converge. ViTMatte does the same
job with a network: it takes the image plus a trimap and predicts the alpha
directly, so it recovers more of the soft detail and cannot fail to converge.

ViTMatte only predicts coverage, not colour, so the caller still has to unmix
the foreground the way `decontaminate_cutout` does. Otherwise the recovered
strands keep whatever background colour was blended into them.

The model runs at a fixed 1024x1024. That is not a simplification: the VitDet
backbone builds its relative position tables from the input dimensions, so an
export with dynamic height and width silently returns wrong alphas at any other
size. Every rembg session already runs at a fixed resolution, so the image and
trimap are resized in and the alpha is resized back out.
"""

import os
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from scipy.ndimage import binary_erosion

from .sessions.base import BaseSession

# The resolution the checkpoints were exported at. Fixed on purpose; see above.
INPUT_SIZE = 1024

BASE_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"

# Distinctions-646 is trained on more varied real imagery than Composition-1k's
# synthetic composites, which leaves noticeably less background colour in the
# soft band on photographic input. The small variant is the default because the
# base one is 3.5x the download and 2.5x the runtime for a marginal gain.
DEFAULT_VARIANT = "small-distinctions-646"

VARIANTS: Dict[str, Dict[str, str]] = {
    "small-distinctions-646": {
        "fname": "vitmatte-small-distinctions-646.onnx",
        "sha256": "d232841ac9d9657df3e62f6c92ed425ee25df18d337245cd8b903fc1ba183631",
    },
    "small-composition-1k": {
        "fname": "vitmatte-small-composition-1k.onnx",
        "sha256": "659b9bb2870f80cffbe20dae4c7f18417fbdf68c0195861d24e9082c28373a24",
    },
    "base-distinctions-646": {
        "fname": "vitmatte-base-distinctions-646.onnx",
        "sha256": "d30190833269c1bec3e37319fe67af151028d3be32f6f60f6663a51b209cc8c1",
    },
    "base-composition-1k": {
        "fname": "vitmatte-base-composition-1k.onnx",
        "sha256": "87bb10979f816061497ba6867b338c65a05cebd7d1507f6dde92a86382dec1f2",
    },
}

_sessions: Dict[str, ort.InferenceSession] = {}


class _ViTMatteFiles(BaseSession):
    """Borrows BaseSession's download and path handling only.

    ViTMatte is not a segmentation model, so it is deliberately not registered
    as a session: it cannot answer `predict(img)` without a mask to refine.
    Subclassing keeps its weights in the same `~/.rembg/models/<name>` layout as
    everything else instead of inventing a second download path.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ViTMatte is a refiner, not a session")

    @classmethod
    def name(cls, *args, **kwargs) -> str:
        return "vitmatte"

    @classmethod
    def download_models(cls, *args, **kwargs) -> str:
        variant = kwargs.get("variant", DEFAULT_VARIANT)
        spec = VARIANTS[variant]
        fname = spec["fname"]

        existing = cls.resolve_existing(fname)
        if existing is not None:
            return existing

        pooch.retrieve(
            f"{BASE_URL}/{fname}",
            None if cls.checksum_disabled() else f"sha256:{spec['sha256']}",
            fname=fname,
            path=cls.model_dir(),
            progressbar=True,
        )

        return os.path.join(cls.model_dir(), fname)


def _resolve_variant(variant: Optional[str]) -> str:
    if variant is None:
        return DEFAULT_VARIANT

    if variant not in VARIANTS:
        raise ValueError(
            f"unknown vitmatte_model {variant!r}. "
            f"Available: {', '.join(sorted(VARIANTS))}"
        )

    return variant


def _get_session(variant: str) -> ort.InferenceSession:
    """Load the model once per variant; each is 110-380 MB."""
    if variant not in _sessions:
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        _sessions[variant] = ort.InferenceSession(
            _ViTMatteFiles.download_models(variant=variant),
            providers=providers,
        )

    return _sessions[variant]


def trimap_from_mask(
    mask: PILImage,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_size: int = 10,
) -> PILImage:
    """Build the three-valued map ViTMatte expects from a mask.

    255 is definite foreground, 0 definite background, and 128 the unknown band
    the network is asked to resolve. Both certain regions are eroded so the band
    covers the whole soft edge; anything left inside it is what gets refined.
    """
    mask_array = np.asarray(mask.convert("L"))

    is_foreground = mask_array > foreground_threshold
    is_background = mask_array < background_threshold

    structure = None
    if erode_size > 0:
        structure = np.ones((erode_size, erode_size), dtype=np.uint8)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    return Image.fromarray(trimap)


def vitmatte_alpha(
    img: PILImage,
    mask: PILImage,
    variant: Optional[str] = None,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_size: int = 10,
) -> PILImage:
    """Refine `mask` into an alpha matte for `img`.

    Returns the alpha alone. The caller is responsible for unmixing the
    foreground colour, which `decontaminate_cutout` already does.
    """
    variant = _resolve_variant(variant)
    session = _get_session(variant)

    trimap = trimap_from_mask(
        mask, foreground_threshold, background_threshold, erode_size
    )

    # Bilinear for the image, nearest for the trimap: interpolating the trimap
    # would invent values between the three levels the model expects.
    image_resized = np.asarray(
        img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    trimap_resized = np.asarray(
        trimap.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST),
        dtype=np.float32,
    )

    # ViTMatte takes the image and trimap concatenated on the channel axis, both
    # scaled to 0-1 with no mean/std shift.
    pixel_values = np.concatenate(
        [
            (image_resized / 255.0).transpose(2, 0, 1),
            (trimap_resized / 255.0)[None],
        ],
        axis=0,
    )[None].astype(np.float32)

    alphas = session.run(None, {"pixel_values": pixel_values})[0]
    alpha = np.clip(alphas[0, 0], 0.0, 1.0)

    return Image.fromarray((alpha * 255).astype(np.uint8)).resize(
        img.size, Image.Resampling.BILINEAR
    )
