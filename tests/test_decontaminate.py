import numpy as np
from PIL import Image

from rembg.bg import decontaminate_cutout, naive_cutout

# The foreground color the decontamination has to recover, and the background
# it was composited over. Building the fixture instead of shipping a photo keeps
# the ground truth exact: we know what every soft pixel should look like.
FG_COLOR = np.array([200, 90, 60], dtype=np.float64)
BG_COLOR = np.array([0, 220, 40], dtype=np.float64)


def make_fringed_image(size=256):
    """Composite thin strands over a saturated background.

    Returns the captured image, its mask, and the soft-edge selector. The
    strands are sub-pixel wide, so most of their pixels are partially
    transparent -- exactly where background color bleeds in.
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    alpha = np.zeros((size, size), dtype=np.float64)

    body = (
        ((xx - size / 2) / (size * 0.3)) ** 2
        + ((yy - size * 0.78) / (size * 0.25)) ** 2
    ) <= 1.0
    alpha[body] = 1.0

    for i in range(40):
        x0 = size / 2 + rng.normal(0, size * 0.11)
        length = rng.uniform(size * 0.35, size * 0.6)
        width = rng.uniform(0.7, 1.8)
        for t in np.linspace(0, 1, int(length * 2)):
            y = size * 0.72 - t * length
            x = x0 + np.sin(t * 5 + i) * 6
            if not (0 <= y < size and 0 <= x < size):
                continue
            yi, xi = int(y), int(x)
            lo_y, hi_y = max(0, yi - 3), min(size, yi + 4)
            lo_x, hi_x = max(0, xi - 3), min(size, xi + 4)
            d2 = (yy[lo_y:hi_y, lo_x:hi_x] - y) ** 2 + (
                xx[lo_y:hi_y, lo_x:hi_x] - x
            ) ** 2
            contrib = np.exp(-d2 / (2 * width**2)) * (1 - t * 0.5)
            alpha[lo_y:hi_y, lo_x:hi_x] = np.maximum(
                alpha[lo_y:hi_y, lo_x:hi_x], contrib
            )

    alpha = np.clip(alpha, 0, 1)
    a3 = alpha[:, :, None]
    composite = np.clip(a3 * FG_COLOR + (1 - a3) * BG_COLOR, 0, 255).astype(np.uint8)

    img = Image.fromarray(composite)
    mask = Image.fromarray((alpha * 255).astype(np.uint8))
    soft = (alpha > 0.05) & (alpha < 0.95)
    return img, mask, soft


def test_decontaminate_removes_background_color_fringing():
    img, mask, soft = make_fringed_image()

    naive = np.asarray(naive_cutout(img, mask).convert("RGBA")).astype(np.float64)
    decontaminated = np.asarray(decontaminate_cutout(img, mask).convert("RGBA")).astype(
        np.float64
    )

    naive_err = np.linalg.norm(naive[soft][:, :3] - FG_COLOR, axis=1).mean()
    decontaminate_err = np.linalg.norm(
        decontaminated[soft][:, :3] - FG_COLOR, axis=1
    ).mean()

    # The naive cutout leaves the blended color in place; decontamination
    # recovers something very close to the true foreground.
    assert decontaminate_err < naive_err / 10
    assert decontaminate_err < 10


def test_decontaminate_preserves_alpha():
    img, mask, _ = make_fringed_image()

    cutout = decontaminate_cutout(img, mask).convert("RGBA")
    alpha_out = np.asarray(cutout)[:, :, 3]

    # The mask is used as-is: decontamination only changes color, never coverage.
    assert np.array_equal(alpha_out, np.asarray(mask))


def test_decontaminate_leaves_opaque_pixels_alone():
    img, mask, _ = make_fringed_image()

    cutout = decontaminate_cutout(img, mask).convert("RGBA")
    arr = np.asarray(cutout).astype(np.float64)
    opaque = np.asarray(mask) == 255

    assert np.abs(arr[opaque][:, :3] - FG_COLOR).max() <= 2
