# Rembg

[![Downloads](https://pepy.tech/badge/rembg)](https://pepy.tech/project/rembg)
[![Downloads](https://pepy.tech/badge/rembg/month)](https://pepy.tech/project/rembg/month)
[![Downloads](https://pepy.tech/badge/rembg/week)](https://pepy.tech/project/rembg/week)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://img.shields.io/badge/License-MIT-blue.svg)

Rembg is a tool to remove images background. That is it.

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.out.png" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.jpg" width="100" />
  <img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.out.png" width="100" />
</p>

**If this project has helped you in any way, please consider making a [donation](https://www.buymeacoffee.com/danielgatis).**

### Installation

CPU support:
```bash
pip install rembg
```

GPU support:
```bash
pip install rembg[gpu]
```

### Usage as a cli

Remove the background from a remote image
```bash
curl -s http://input.png | rembg i > output.png
```

Remove the background from a local file
```bash
rembg i path/to/input.png path/to/output.png
```

Remove the background from all images in a folder
```bash
rembg p path/to/input path/to/output
```

### Usage as a server

Start the server
```bash
rembg s
```

Open your browser to
```
http://localhost:5000?url=http://image.png
```

Also you can send the file as a FormData (multipart/form-data):
```html
<form action="http://localhost:5000" method="post" enctype="multipart/form-data">
   <input type="file" name="file"/>
   <input type="submit" value="upload"/>
</form>
```

### Usage as a library

#### Example 1: Read from stdin and write to stdout

In `app.py`
```python
import sys
from rembg.bg import remove

sys.stdout.buffer.write(remove(sys.stdin.buffer.read()))
```

Then run
```
cat input.png | python app.py > out.png
```

#### Example 2: Using PIL

In `app.py`
```python
from rembg.bg import remove
import numpy as np
import io
from PIL import Image

input_path = 'input.png'
output_path = 'out.png'

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
# ImageFile.LOAD_TRUNCATED_IMAGES = True

f = np.fromfile(input_path)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")
img.save(output_path)
```

Then run
```
python app.py
```

### Usage as a docker

First compile with:

```
docker build . -t rembg
```

Then run with:

```
docker run --rm -i rembg i in.png out.png
```

### Advance usage

Sometimes it is possible to achieve better results by turning on alpha matting. Example:
```bash
curl -s http://input.png | rembg i -a -ae 15 > output.png
```

<table>
    <thead>
        <tr>
            <td>Original</td>
            <td>Without alpha matting</td>
            <td>With alpha matting (-a -ae 15)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/food-1.jpg"/></td>
            <td><img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/food-1.out.jpg"/></td>
            <td><img src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/food-1.out.alpha.jpg"/></td>
        </tr>
    </tbody>
</table>

### In the cloud

Please contact me at danielgatis@gmail.com if you need help to put it on the cloud.

### References

- https://arxiv.org/pdf/2005.09007.pdf
- https://github.com/NathanUA/U-2-Net
- https://github.com/pymatting/pymatting

### Buy me a coffee
Liked some of my work? Buy me a coffee (or more likely a beer)

<a href="https://www.buymeacoffee.com/danielgatis" target="_blank"><img src="https://bmc-cdn.nyc3.digitaloceanspaces.com/BMC-button-images/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;"></a>

### License

Copyright (c) 2020-present [Daniel Gatis](https://github.com/danielgatis)

Licensed under [MIT License](./LICENSE.txt)
