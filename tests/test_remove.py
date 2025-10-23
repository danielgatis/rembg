from io import BytesIO
from pathlib import Path

from imagehash import phash as hash_img
from PIL import Image

from rembg import new_session, remove

here = Path(__file__).parent.resolve()
failures_dir = here / "failures"
failures_dir.mkdir(exist_ok=True)

def test_remove():
    kwargs = {
        "sam": {
            "anime-girl-1" : {
                "sam_prompt" :[{"type": "point", "data": [400, 165], "label": 1}],
            },

            "car-1" : {
                "sam_prompt" :[{"type": "point", "data": [250, 200], "label": 1}],
            },

            "cloth-1" : {
                "sam_prompt" :[{"type": "point", "data": [370, 495], "label": 1}],
            },

            "plants-1" : {
                "sam_prompt" :[{"type": "point", "data": [724, 740], "label": 1}],
            },
        }
    }

    for model in [
        "u2net",
        "u2netp",
        "u2net_human_seg",
        "u2net_cloth_seg",
        "silueta",
        "isnet-general-use",
        "isnet-anime",
        "sam",
        "birefnet-general",
        "birefnet-general-lite",
        "birefnet-portrait",
        "birefnet-dis",
        "birefnet-hrsod",
        "birefnet-cod",
        "birefnet-massive"
    ]:
        for picture in ["anime-girl-1", "car-1", "cloth-1", "plants-1"]:
            image_path = Path(here / "fixtures" / f"{picture}.jpg")
            image = image_path.read_bytes()

            actual = remove(image, session=new_session(model), **kwargs.get(model, {}).get(picture, {}))
            actual_hash = hash_img(Image.open(BytesIO(actual)))

            expected_path = Path(here / "results" / f"{picture}.{model}.png")
            # Uncomment to update the expected results
            # f = open(expected_path, "wb")
            # f.write(actual)
            # f.close()

            expected = expected_path.read_bytes()
            expected_hash = hash_img(Image.open(BytesIO(expected)))

            print(f"image_path: {image_path}")
            print(f"expected_path: {expected_path}")
            print(f"actual_hash: {actual_hash}")
            print(f"expected_hash: {expected_hash}")
            print(f"actual_hash == expected_hash: {actual_hash == expected_hash}")
            print("---\n")

            if actual_hash != expected_hash:
                # Salva as imagens que falharam para comparação
                actual_failure_path = failures_dir / f"{picture}.{model}.actual.png"
                expected_failure_path = failures_dir / f"{picture}.{model}.expected.png"

                with open(actual_failure_path, "wb") as f:
                    f.write(actual)
                with open(expected_failure_path, "wb") as f:
                    f.write(expected)

                print(f"FAILURE: Saved comparison images to {failures_dir}")

            assert actual_hash == expected_hash
