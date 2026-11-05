import numpy as np
import pytest
from PIL import Image

from rembg.sessions.u2net_cloth_seg import Unet2ClothSession


class FakeInput:
    name = "input"


class FakeInferenceSession:
    """Predicts upper cloth (1) on the left and full cloth (3) on the right."""

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feeds):
        logits = np.zeros((1, 4, 768, 768), dtype=np.float32)
        logits[0, 1, :, :383] = 10
        logits[0, 3, :, 383:] = 10
        return [logits]


@pytest.mark.parametrize("size", [(1234, 900), (500, 400), (3000, 2000)])
def test_cloth_seg_does_not_mix_labels_when_resizing(size):
    session = Unet2ClothSession.__new__(Unet2ClothSession)
    session.inner_session = FakeInferenceSession()

    upper, lower, full = session.predict(Image.new("RGB", size))

    # No lower cloth was predicted, so its mask must stay empty instead of
    # picking up the interpolated boundary between classes 1 and 3.
    assert not np.asarray(lower).any()
    assert np.asarray(upper).any()
    assert np.asarray(full).any()
