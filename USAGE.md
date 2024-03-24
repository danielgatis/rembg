# How to use the remove function

## Load the Image

```python
from PIL import Image
from rembg import new_session, remove

input_path = 'input.png'
output_path = 'output.png'

input = Image.open(input_path)
```

## Removing the background

### Without additional arguments

This defaults to the `u2net` model.

```python
output = remove(input)
output.save(output_path)
```

### With a specific model

You can use the `new_session` function to create a session with a specific model.

```python
model_name = "isnet-general-use"
session = new_session(model_name)
output = remove(input, session=session)
```

### For processing multiple image files

By default, `remove` initialises a new session every call. This can be a large bottleneck if you're having to process multiple images. Initialise a session and pass it in to the `remove` function for fast multi-image support

```python
model_name = "unet"
rembg_session = new_session(model_name)
for img in images:
    output = remove(img, session=rembg_session)
```

### With alpha matting

Alpha matting is a post processing step that can be used to improve the quality of the output.

```python
output = remove(input, alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)
```

### Only mask

If you only want the mask, you can use the `only_mask` argument.

```python
output = remove(input, only_mask=True)
```

### With post processing

You can use the `post_process_mask` argument to post process the mask to get better results.

```python
output = remove(input, post_process_mask=True)
```

### Replacing the background color

You can use the `bgcolor` argument to replace the background color.

```python
output = remove(input, bgcolor=(255, 255, 255, 255))
```

### Using input points

You can use the `input_points` and `input_labels` arguments to specify the points that should be used for the masks. This only works with the `sam` model.

```python
import numpy as np
# Define the points and labels
# The points are defined as [y, x]
input_points = np.array([[400, 350], [700, 400], [200, 400]])
input_labels = np.array([1, 1, 2])

image = remove(image,session=session, input_points=input_points, input_labels=input_labels)
```

## Save the image

```python
output.save(output_path)
```
