# Rembg

[![Downloads](https://img.shields.io/pypi/dm/rembg.svg)](https://img.shields.io/pypi/dm/rembg.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://img.shields.io/badge/License-MIT-blue.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KenjieDec/RemBG)
[![Streamlit App](https://img.shields.io/badge/🎈%20Streamlit%20Community-Cloud-blue)](https://bgremoval.streamlit.app/)

Rembg is a tool to remove images background.

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="example car-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.jpg" width="100" />
  <img alt="example car-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.out.png" width="100" />
  <img alt="example car-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.jpg" width="100" />
  <img alt="example car-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.out.png" width="100" />
  <img alt="example car-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.jpg" width="100" />
  <img alt="example car-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="example animal-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.jpg" width="100" />
  <img alt="example animal-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.out.png" width="100" />
  <img alt="example animal-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.jpg" width="100" />
  <img alt="example animal-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.out.png" width="100" />
  <img alt="example animal-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.jpg" width="100" />
  <img alt="example animal-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="example girl-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.jpg" width="100" />
  <img alt="example girl-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.out.png" width="100" />
  <img alt="example girl-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.jpg" width="100" />
  <img alt="example girl-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.out.png" width="100" />
  <img alt="example girl-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.jpg" width="100" />
  <img alt="example girl-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="example anime-girl-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-1.jpg" width="100" />
  <img alt="example anime-girl-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-1.out.png" width="100" />
  <img alt="example anime-girl-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-2.jpg" width="100" />
  <img alt="example anime-girl-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-2.out.png" width="100" />
  <img alt="example anime-girl-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-3.jpg" width="100" />
  <img alt="example anime-girl-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-3.out.png" width="100" />
</p>

**If this project has helped you, please consider making a [donation](https://www.buymeacoffee.com/danielgatis).**

## Sponsor

<table>
  <tr>
    <td align="center" vertical-align="center">
      <a href="https://photoroom.com/api/remove-background?utm_source=rembg&utm_medium=github_webpage&utm_campaign=sponsor" >
        <img src="https://font-cdn.photoroom.com/media/api-logo.png" width="120px;" alt="Unsplash" />
      </a>
    </td>
    <td align="center" vertical-align="center">
      <b>PhotoRoom Remove Background API</b>
      <br />
      <a href="https://photoroom.com/api/remove-background?utm_source=rembg&utm_medium=github_webpage&utm_campaign=sponsor">https://photoroom.com/api</a>
      <br />
      <p width="200px">
        Fast and accurate background remover API<br/>
      </p>
    </td>
  </tr>
</table>

## Requirements

```text
python: >3.7, <3.13
```

## Installation

CPU support:

```bash
pip install rembg # for library
pip install "rembg[cli]" # for library + cli
```

GPU support:

First of all, you need to check if your system supports the `onnxruntime-gpu`.

Go to <https://onnxruntime.ai> and check the installation matrix.

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="onnxruntime-installation-matrix" src="https://raw.githubusercontent.com/danielgatis/rembg/master/onnxruntime-installation-matrix.png" width="400" />
</p>

If yes, just run:

```bash
pip install "rembg[GPU]" # for library
pip install "rembg[gpu,cli]" # for library + cli
```

## Usage as a cli

After the installation step you can use rembg just typing `rembg` in your terminal window.

The `rembg` command has 4 subcommands, one for each input type:

- `i` for files
- `p` for folders
- `s` for http server
- `b` for RGB24 pixel binary stream

You can get help about the main command using:

```shell
rembg --help
```

As well, about all the subcommands using:

```shell
rembg <COMMAND> --help
```

### rembg `i`

Used when input and output are files.

Remove the background from a remote image

```shell
curl -s http://input.png | rembg i > output.png
```

Remove the background from a local file

```shell
rembg i path/to/input.png path/to/output.png
```

Remove the background specifying a model

```shell
rembg i -m u2netp path/to/input.png path/to/output.png
```

Remove the background returning only the mask

```shell
rembg i -om path/to/input.png path/to/output.png
```

Remove the background applying an alpha matting

```shell
rembg i -a path/to/input.png path/to/output.png
```

Passing extras parameters

```shell
SAM example

rembg i -m sam -x '{ "sam_prompt": [{"type": "point", "data": [724, 740], "label": 1}] }' examples/plants-1.jpg examples/plants-1.out.png
```

```shell
Custom model example

rembg i -m u2net_custom -x '{"model_path": "~/.u2net/u2net.onnx"}' path/to/input.png path/to/output.png
```

### rembg `p`

Used when input and output are folders.

Remove the background from all images in a folder

```shell
rembg p path/to/input path/to/output
```

Same as before, but watching for new/changed files to process

```shell
rembg p -w path/to/input path/to/output
```

### rembg `s`

Used to start http server.

```shell
rembg s --host 0.0.0.0 --port 7000 --log_level info
```

To see the complete endpoints documentation, go to: `http://localhost:7000/api`.

Remove the background from an image url

```shell
curl -s "http://localhost:7000/api/remove?url=http://input.png" -o output.png
```

Remove the background from an uploaded image

```shell
curl -s -F file=@/path/to/input.jpg "http://localhost:7000/api/remove"  -o output.png
```

### rembg `b`

Process a sequence of RGB24 images from stdin. This is intended to be used with another program, such as FFMPEG, that outputs RGB24 pixel data to stdout, which is piped into the stdin of this program, although nothing prevents you from manually typing in images at stdin.

```shell
rembg b image_width image_height -o output_specifier
```

Arguments:

- image_width : width of input image(s)
- image_height : height of input image(s)
- output_specifier: printf-style specifier for output filenames, for example if `output-%03u.png`, then output files will be named `output-000.png`, `output-001.png`, `output-002.png`, etc. Output files will be saved in PNG format regardless of the extension specified. You can omit it to write results to stdout.

Example usage with FFMPEG:

```shell
ffmpeg -i input.mp4 -ss 10 -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b 1280 720 -o folder/output-%03u.png
```

The width and height values must match the dimension of output images from FFMPEG. Note for FFMPEG, the "`-an -f rawvideo -pix_fmt rgb24 pipe:1`" part is required for the whole thing to work.

## Usage as a library

Input and output as bytes

```python
from rembg import remove

input_path = 'input.png'
output_path = 'output.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
```

Input and output as a PIL image

```python
from rembg import remove
from PIL import Image

input_path = 'input.png'
output_path = 'output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)
```

Input and output as a numpy array

```python
from rembg import remove
import cv2

input_path = 'input.png'
output_path = 'output.png'

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)
```

Force output as bytes

```python
from rembg import remove

input_path = 'input.png'
output_path = 'output.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input, force_return_bytes=True)
        o.write(output)
```

How to iterate over files in a performatic way

```python
from pathlib import Path
from rembg import remove, new_session

session = new_session()

for file in Path('path/to/folder').glob('*.png'):
    input_path = str(file)
    output_path = str(file.parent / (file.stem + ".out.png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)
```

To see a full list of examples on how to use rembg, go to the [examples](USAGE.md) page.

## Usage as a docker

Just replace the `rembg` command for `docker run danielgatis/rembg`.

Try this:

```shell
docker run -v path/to/input:/rembg danielgatis/rembg i input.png path/to/output/output.png
```

## Models

All models are downloaded and saved in the user home folder in the `.u2net` directory.

The available models are:

- u2net ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx), [source](https://github.com/xuebinqin/U-2-Net)): A pre-trained model for general use cases.
- u2netp ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx), [source](https://github.com/xuebinqin/U-2-Net)): A lightweight version of u2net model.
- u2net_human_seg ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx), [source](https://github.com/xuebinqin/U-2-Net)): A pre-trained model for human segmentation.
- u2net_cloth_seg ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx), [source](https://github.com/levindabhi/cloth-segmentation)): A pre-trained model for Cloths Parsing from human portrait. Here clothes are parsed into 3 category: Upper body, Lower body and Full body.
- silueta ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx), [source](https://github.com/xuebinqin/U-2-Net/issues/295)): Same as u2net but the size is reduced to 43Mb.
- isnet-general-use ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx), [source](https://github.com/xuebinqin/DIS)): A new pre-trained model for general use cases.
- isnet-anime ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx), [source](https://github.com/SkyTNT/anime-segmentation)): A high-accuracy segmentation for anime character.
- sam ([download encoder](https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-encoder-quant.onnx), [download decoder](https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-decoder-quant.onnx), [source](https://github.com/facebookresearch/segment-anything)): A pre-trained model for any use cases.
- birefnet-general ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model for general use cases.
- birefnet-general-lite ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A light pre-trained model for general use cases.
- birefnet-portrait ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model for human portraits.
- birefnet-dis ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-DIS-epoch_590.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model for dichotomous image segmentation (DIS).
- birefnet-hrsod ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-HRSOD_DHU-epoch_115.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model for high-resolution salient object detection (HRSOD).
- birefnet-cod ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-COD-epoch_125.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model for concealed object detection (COD).
- birefnet-massive ([download](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx), [source](https://github.com/ZhengPeng7/BiRefNet)): A pre-trained model with massive dataset.

### How to train your own model

If You need more fine tuned models try this:
<https://github.com/danielgatis/rembg/issues/193#issuecomment-1055534289>

## Some video tutorials

- <https://www.youtube.com/watch?v=3xqwpXjxyMQ>
- <https://www.youtube.com/watch?v=dFKRGXdkGJU>
- <https://www.youtube.com/watch?v=Ai-BS_T7yjE>
- <https://www.youtube.com/watch?v=D7W-C0urVcQ>

## References

- <https://arxiv.org/pdf/2005.09007.pdf>
- <https://github.com/NathanUA/U-2-Net>
- <https://github.com/pymatting/pymatting>

## FAQ

### When will this library provide support for Python version 3.xx?

This library directly depends on the [onnxruntime](https://pypi.org/project/onnxruntime) library. Therefore, we can only update the Python version when [onnxruntime](https://pypi.org/project/onnxruntime) provides support for that specific version.

## Buy me a coffee

Liked some of my work? Buy me a coffee (or more likely a beer)

<a href="https://www.buymeacoffee.com/danielgatis" target="_blank"><img src="https://bmc-cdn.nyc3.digitaloceanspaces.com/BMC-button-images/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;"></a> <!-- markdownlint-disable MD033 -->

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=danielgatis/rembg&type=Date)](https://star-history.com/#danielgatis/rembg&Date)

## License

Copyright (c) 2020-present [Daniel Gatis](https://github.com/danielgatis)

Licensed under [MIT License](./LICENSE.txt)
