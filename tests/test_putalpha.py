import numpy as np
from PIL import Image

from rembg.bg import putalpha_cutout, remove


class ThreeMaskSession:
    """Returns one mask per vertical third, like multi-mask models such as u2net_cloth_seg."""

    def predict(self, img, *args, **kwargs):
        width, height = img.size
        masks = []
        for i in range(3):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[:, i * width // 3 : (i + 1) * width // 3] = 255
            masks.append(Image.fromarray(mask))
        return masks


def test_putalpha_keeps_every_mask():
    img = Image.new("RGB", (9, 4), (200, 100, 50))

    out = remove(img, session=ThreeMaskSession(), putalpha=True)
    alpha = np.asarray(out)[:, :, 3]

    # The cutouts are stacked vertically, one 4 pixel high block per mask.
    for i in range(3):
        block = alpha[i * 4 : (i + 1) * 4]
        assert np.flatnonzero(block[0]).tolist() == [3 * i, 3 * i + 1, 3 * i + 2]


def test_putalpha_cutout_does_not_modify_input():
    img = Image.new("RGB", (4, 4), (200, 100, 50))
    mask = Image.new("L", (4, 4), 128)

    cutout = putalpha_cutout(img, mask)

    assert img.mode == "RGB"
    assert np.asarray(cutout)[:, :, 3].tolist() == np.asarray(mask).tolist()
