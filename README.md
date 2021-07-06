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

#### *** If you want to remove background from videos try this this fork: https://github.com/ecsplendid/rembg-greenscreen ***

### Requirements

* python 3.8 or newer

* torch and torchvision stable version (https://pytorch.org)

#### How to install torch/torchvision

Go to https://pytorch.org and scrool down to `INSTALL PYTORCH` section and follow the instructions.

For example:
```
PyTorch Build: Stable (1.7.1)
Your OS: Windows
Package: Pip
Language: Python
CUDA: None
```

The install cmd is:
```
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation

Install it from pypi

```bash
pip install rembg
```

### Usage as a cli

Remove the background from a remote image
```bash
curl -s http://input.png | rembg > output.png
```

Remove the background from a local file
```bash
rembg -o path/to/output.png path/to/input.png
```

Remove the background from all images in a folder
```bash
rembg -p path/to/input path/to/output
```

### Add a custom model

Copy the `custom-model.pth` file to `~/.u2net` and run:

```bash
curl -s http://input.png | rembg -m custom-model > output.png
```

### Usage as a server

Start the server
```bash
rembg-server
```

Open your browser to
```
http://localhost:5000?url=http://image.png
```

Also you can send the file as a FormData (multipart/form-data):
```
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

Just run

```
curl -s http://input.png | docker run -i -v ~/.u2net:/root/.u2net danielgatis/rembg:latest > output.png
```

### Advance usage

Sometimes it is possible to achieve better results by turning on alpha matting. Example:
```bash
curl -s http://input.png | rembg -a -ae 15 > output.png
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
