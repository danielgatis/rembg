import numpy as np
import pytest
from PIL import Image

from rembg.matting import (
    DEFAULT_VARIANT,
    VARIANTS,
    _resolve_variant,
    trimap_from_mask,
)


def test_default_variant_is_a_known_one():
    assert DEFAULT_VARIANT in VARIANTS


def test_every_variant_declares_a_file_and_checksum():
    for name, spec in VARIANTS.items():
        assert spec["fname"].endswith(".onnx"), name
        assert len(spec["sha256"]) == 64, name


def test_resolve_variant_defaults_when_unset():
    assert _resolve_variant(None) == DEFAULT_VARIANT


def test_resolve_variant_accepts_known_names():
    for name in VARIANTS:
        assert _resolve_variant(name) == name


def test_resolve_variant_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown vitmatte_model"):
        _resolve_variant("no-such-model")


def test_resolve_variant_error_lists_the_options():
    with pytest.raises(ValueError) as excinfo:
        _resolve_variant("nope")

    for name in VARIANTS:
        assert name in str(excinfo.value)


def _mask_with_soft_edge():
    """A mask with a hard core and a ramp, so the trimap has all three levels."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[16:48, 16:48] = 255
    arr[12:16, 16:48] = 128
    return Image.fromarray(arr)


def test_trimap_has_only_the_three_expected_levels():
    trimap = trimap_from_mask(_mask_with_soft_edge())
    assert set(np.unique(np.asarray(trimap))) <= {0, 128, 255}


def test_trimap_marks_an_unknown_band():
    trimap = np.asarray(trimap_from_mask(_mask_with_soft_edge()))
    assert (trimap == 128).sum() > 0


def test_trimap_keeps_the_mask_size():
    mask = _mask_with_soft_edge()
    assert trimap_from_mask(mask).size == mask.size


def test_larger_erode_size_widens_the_unknown_band():
    mask = _mask_with_soft_edge()
    narrow = np.asarray(trimap_from_mask(mask, erode_size=2))
    wide = np.asarray(trimap_from_mask(mask, erode_size=10))
    assert (wide == 128).sum() > (narrow == 128).sum()


def test_trimap_accepts_a_non_grayscale_mask():
    mask = _mask_with_soft_edge().convert("RGB")
    assert set(np.unique(np.asarray(trimap_from_mask(mask)))) <= {0, 128, 255}
