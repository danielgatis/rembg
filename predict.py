from rembg import remove
from PIL import Image

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input image",
            default="",
        ),
    ) -> Path:

        image = Image.open(str(image))
        output = remove(image)
        output_path = f"/tmp/out.png"
        output.save(output_path)

        return Path(output_path)
