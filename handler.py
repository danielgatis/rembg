#!/usr/bin/env python
import runpod
import requests
from rembg import remove
from PIL import Image
import io
import base64

def handler(job):
    job_input = job.get('input', {})
    image_url = job_input.get('image_url')

    if not image_url:
        return {"error": "image_url not provided"}

    try:
        response = requests.get(image_url)
        response.raise_for_status()

        input_image = Image.open(io.BytesIO(response.content))
        output_image = remove(input_image)

        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": img_str}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download image: {e}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

runpod.serverless.start({"handler": handler})