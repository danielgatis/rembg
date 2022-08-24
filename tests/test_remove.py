from io import BytesIO
from pathlib import Path

from imagehash import average_hash
from PIL import Image

from rembg import remove

here = Path(__file__).parent.resolve()


def test_remove():
    image = Path(here / ".." / "examples" / "animal-1.jpg").read_bytes()
    expected = Path(here / ".." / "examples" / "animal-1.out.png").read_bytes()
    actual = remove(image)

    actual_hash = average_hash(Image.open(BytesIO(actual)))
    expected_hash = average_hash(Image.open(BytesIO(expected)))

    assert actual_hash == expected_hash
