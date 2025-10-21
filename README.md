# Rembg

[![RepoMapr](https://img.shields.io/badge/RepoMapr-View_Interactive_Diagram-blue?style=flat&logo=github)](https://repomapr.com/danielgatis/rembg)

[![Downloads](https://img.shields.io/pypi/dm/rembg.svg)](https://img.shields.io/pypi/dm/rembg.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://img.shields.io/badge/License-MIT-blue.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KenjieDec/RemBG)
[![Streamlit App](https://img.shields.io/badge/🎈%20Streamlit%20Community-Cloud-blue)](https://bgremoval.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielgatis/rembg/blob/main/rembg.ipynb)


Rembg 是一个用于移除图像背景的工具。

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="示例 car-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.jpg" width="100" />
  <img alt="示例 car-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-1.out.png" width="100" />
  <img alt="示例 car-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.jpg" width="100" />
  <img alt="示例 car-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-2.out.png" width="100" />
  <img alt="示例 car-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.jpg" width="100" />
  <img alt="示例 car-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/car-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="示例 animal-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.jpg" width="100" />
  <img alt="示例 animal-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-1.out.png" width="100" />
  <img alt="示例 animal-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.jpg" width="100" />
  <img alt="示例 animal-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.out.png" width="100" />
  <img alt="示例 animal-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.jpg" width="100" />
  <img alt="示例 animal-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="示例 girl-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.jpg" width="100" />
  <img alt="示例 girl-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-1.out.png" width="100" />
  <img alt="示例 girl-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.jpg" width="100" />
  <img alt="示例 girl-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-2.out.png" width="100" />
  <img alt="示例 girl-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.jpg" width="100" />
  <img alt="示例 girl-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/girl-3.out.png" width="100" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="示例 anime-girl-1" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-1.jpg" width="100" />
  <img alt="示例 anime-girl-1.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-1.out.png" width="100" />
  <img alt="示例 anime-girl-2" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-2.jpg" width="100" />
  <img alt="示例 anime-girl-2.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-2.out.png" width="100" />
  <img alt="示例 anime-girl-3" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-3.jpg" width="100" />
  <img alt="示例 anime-girl-3.out" src="https://raw.githubusercontent.com/danielgatis/rembg/master/examples/anime-girl-3.out.png" width="100" />
</p>

**如果这个项目对你有所帮助，请考虑[捐赠](https://www.buymeacoffee.com/danielgatis)。**

## 赞助商

<table>
 <tr>
    <td align="center" vertical-align="center">
      <a href="https://photoroom.com/api/remove-background?utm_source=rembg&utm_medium=github_webpage&utm_campaign=sponsor" >
        <img src="https://font-cdn.photoroom.com/media/api-logo.png" width="120px;" alt="Unsplash" />
      </a>
    </td>
    <td align="center" vertical-align="center">
      <b>PhotoRoom 背景移除 API</b>
      <br />
      <a href="https://photoroom.com/api/remove-background?utm_source=rembg&utm_medium=github_webpage&utm_campaign=sponsor">https://photoroom.com/api</a>
      <br />
      <p width="200px">
        快速且精准的背景移除 API<br/>
      </p>
    </td>
  </tr>
</table>

## 系统要求

```text
python: >=3.10, <3.14
```

## 安装

如果你已经安装了 `onnxruntime`，只需安装 `rembg`：

```bash
pip install rembg # 仅安装库
pip install "rembg[cli]" # 安装库 + 命令行工具
```

如果没有，请按照所需的平台支持（CPU/GPU）安装 `rembg`。

### CPU 支持：

```bash
pip install rembg[cpu] # 仅安装库
pip install "rembg[cpu,cli]" # 安装库 + 命令行工具
```

### GPU 支持（NVIDIA/CUDA）：

首先，你需要确认系统是否支持 `onnxruntime-gpu`。

访问 [onnxruntime.ai](<https://onnxruntime.ai/getting-started>) 并查看安装矩阵。

<p style="display: flex;align-items: center;justify-content: center;">
  <img alt="onnxruntime-installation-matrix" src="https://raw.githubusercontent.com/danielgatis/rembg/master/onnxruntime-installation-matrix.png" width="400" />
</p>

如果支持，直接运行：

```bash
pip install "rembg[gpu]" # 仅安装库
pip install "rembg[gpu,cli]" # 安装库 + 命令行工具
```

Nvidia GPU 可能需要 `onnxruntime-gpu`、CUDA 和 `cudnn-devel`。详见 [#668](https://github.com/danielgatis/rembg/issues/668#issuecomment-2689830314)。如果 `rembg[gpu]` 无法正常工作且无法安装 CUDA 或 `cudnn-devel`，请改用 `rembg[cpu]` 和 `onnxruntime`。

### GPU 支持（AMD/ROCM）：

ROCM 支持需要安装 `onnxruntime-rocm`。请按照[AMD 官方文档](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-onnx.html)进行安装。

如果 `onnxruntime-rocm` 已安装并可用，再安装 `rembg[rocm]` 版本：

```bash
pip install "rembg[rocm]" # 仅安装库
pip install "rembg[rocm,cli]" # 安装库 + 命令行工具
```

## 作为命令行工具使用

安装完成后，你可以在终端中直接输入 `rembg` 来使用。

`rembg` 命令包含 4 个子命令，对应不同的输入类型：

- `i`：处理文件
- `p`：处理文件夹
- `s`：启动 HTTP 服务器
- `b`：处理 RGB24 像素二进制流

可以通过以下命令查看主命令的帮助信息：

```shell
rembg --help
```

同样可以针对任意子命令查看帮助：

```shell
rembg <COMMAND> --help
```

### rembg `i`

用于输入和输出均为文件的情况。

移除远程图像的背景：

```shell
curl -s http://input.png | rembg i > output.png
```

移除本地图像的背景：

```shell
rembg i path/to/input.png path/to/output.png
```

指定模型来移除背景：

```shell
rembg i -m u2netp path/to/input.png path/to/output.png
```

仅返回遮罩：

```shell
rembg i -om path/to/input.png path/to/output.png
```

应用透明度抠图以移除背景：

```shell
rembg i -a path/to/input.png path/to/output.png
```

传递额外参数：

```shell
SAM 示例：

rembg i -m sam -x '{ "sam_prompt": [{"type": "point", "data": [724, 740], "label": 1}] }' examples/plants-1.jpg examples/plants-1.out.png
```

```shell
自定义模型示例：

rembg i -m u2net_custom -x '{"model_path": "~/.u2net/u2net.onnx"}' path/to/input.png path/to/output.png
```

### rembg `p`

用于输入和输出均为文件夹的情况。

批量移除文件夹内所有图像的背景：

```shell
rembg p path/to/input path/to/output
```

与上述命令相同，但会监听新增或修改的文件并自动处理：

```shell
rembg p -w path/to/input path/to/output
```

### rembg `s`

用于启动 HTTP 服务器。

```shell
rembg s --host 0.0.0.0 --port 7000 --log_level info
```

完整的接口文档请访问：`http://localhost:7000/api`。

通过图像 URL 移除背景：

```shell
curl -s "http://localhost:7000/api/remove?url=http://input.png" -o output.png
```

上传图像并移除背景：

```shell
curl -s -F file=@/path/to/input.jpg "http://localhost:7000/api/remove"  -o output.png
```

### rembg `b`

从标准输入读取一系列 RGB24 图像进行处理。该模式通常与 FFMPEG 等程序配合使用：这些程序将 RGB24 像素数据输出到标准输出，再通过管道传递给本程序的标准输入。当然，你也可以手动在标准输入中输入图像数据。

```shell
rembg b image_width image_height -o output_specifier
```

参数说明：

- image_width：输入图像的宽度
- image_height：输入图像的高度
- output_specifier：输出文件名的 printf 风格格式化字符串。例如使用 `output-%03u.png` 时，输出文件将依次命名为 `output-000.png`、`output-001.png`、`output-002.png` 等。无论指定的扩展名为何，输出均为 PNG 格式。若省略该参数，将把结果写入标准输出。

与 FFMPEG 配合使用的示例：

```shell
ffmpeg -i input.mp4 -ss 10 -an -f rawvideo -pix_fmt rgb24 pipe:1 | rembg b 1280 720 -o folder/output-%03u.png
```

宽度和高度必须与 FFMPEG 输出图像的尺寸匹配。注意，FFMPEG 命令中的 `-an -f rawvideo -pix_fmt rgb24 pipe:1` 参数是整个流程正常运行的必要条件。

## 作为库使用

以字节方式读写：

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

以 PIL 图像方式读写：

```python
from rembg import remove
from PIL import Image

input_path = 'input.png'
output_path = 'output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)
```

以 NumPy 数组方式读写：

```python
from rembg import remove
import cv2

input_path = 'input.png'
output_path = 'output.png'

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)
```

强制输出为字节：

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

如何高效地遍历文件：

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

更多使用示例请访问 [examples](USAGE.md) 页面。

## 作为 Docker 使用

### 仅使用 CPU

只需将命令中的 `rembg` 替换为 `docker run danielgatis/rembg`。

例如：

```shell
docker run -v path/to/input:/rembg danielgatis/rembg i input.png path/to/output/output.png
```

### Nvidia CUDA 硬件加速

注意：在 Docker 中使用 CUDA，宿主机必须安装 **NVIDIA Container Toolkit**。参考 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

要启用 **Nvidia CUDA 硬件加速**，需要 `cudnn-devel`，因此需要自行构建镜像。参见 [#668](https://github.com/danielgatis/rembg/issues/668#issuecomment-2689914205)。

以下示例展示如何构建镜像并命名为 *rembg-nvidia-cuda-cudnn-gpu*：
```shell
docker build -t rembg-nvidia-cuda-cudnn-gpu -f Dockerfile_nvidia_cuda_cudnn_gpu .
```
请注意：该镜像将占用约 11GB 磁盘空间（CPU 版本约为 1.6GB），且不包含模型文件。

构建完成后，可以如下方式运行命令行：
```shell
sudo docker run --rm -it --gpus all -v /dev/dri:/dev/dri -v $PWD:/rembg rembg-nvidia-cuda-cudnn-gpu i -m birefnet-general input.png output.png
```

- 小技巧 1：你也可以自制一个包含 NVIDIA CUDA + cuDNN 的镜像，再在其中安装 `rembg[gpu, cli]`。
- 小技巧 2：使用参数 `-v /somewhereYouStoresModelFiles/:/root/.u2net`，即可在镜像外部下载或存储模型文件。你甚至可以在构建镜像时注释掉 `RUN rembg d u2net` 这一行，这样镜像就不会默认下载模型，方便按需获取。

## 模型

所有模型会下载到用户主目录下的 `.u2net` 目录。

可用模型包括：

- u2net（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx)，[源码](https://github.com/xuebinqin/U-2-Net)）：通用场景预训练模型。
- u2netp（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx)，[源码](https://github.com/xuebinqin/U-2-Net)）：u2net 的轻量版本。
- u2net_human_seg（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx)，[源码](https://github.com/xuebinqin/U-2-Net)）：针对人体分割的预训练模型。
- u2net_cloth_seg（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx)，[源码](https://github.com/levindabhi/cloth-segmentation)）：用于人物服装解析的预训练模型，将衣物划分为上身、下身和整身三类。
- silueta（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx)，[源码](https://github.com/xuebinqin/U-2-Net/issues/295)）：与 u2net 相同但体积缩小至 43MB。
- isnet-general-use（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx)，[源码](https://github.com/xuebinqin/DIS)）：新的通用场景预训练模型。
- isnet-anime（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx)，[源码](https://github.com/SkyTNT/anime-segmentation)）：面向二次元角色的高精度分割模型。
- sam（[下载编码器](https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-encoder-quant.onnx)，[下载解码器](https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-decoder-quant.onnx)，[源码](https://github.com/facebookresearch/segment-anything)）：适用于多种场景的预训练模型。
- birefnet-general（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：通用场景预训练模型。
- birefnet-general-lite（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：通用场景的轻量模型。
- birefnet-portrait（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：针对人物肖像的预训练模型。
- birefnet-dis（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-DIS-epoch_590.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：二分图像分割（DIS）模型。
- birefnet-hrsod（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-HRSOD_DHU-epoch_115.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：高分辨率显著目标检测（HRSOD）模型。
- birefnet-cod（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-COD-epoch_125.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：隐蔽目标检测（COD）模型。
- birefnet-massive（[下载](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx)，[源码](https://github.com/ZhengPeng7/BiRefNet)）：使用大规模数据集训练的模型。

### 如何训练自定义模型

如果需要更精细的模型，请参考：
<https://github.com/danielgatis/rembg/issues/193#issuecomment-1055534289>

## 视频教程

- <https://www.youtube.com/watch?v=3xqwpXjxyMQ>
- <https://www.youtube.com/watch?v=dFKRGXdkGJU>
- <https://www.youtube.com/watch?v=Ai-BS_T7yjE>
- <https://www.youtube.com/watch?v=D7W-C0urVcQ>

## 参考资料

- <https://arxiv.org/pdf/2005.09007.pdf>
- <https://github.com/NathanUA/U-2-Net>
- <https://github.com/pymatting/pymatting>

## 常见问题

### 该库何时支持 Python 3.xx 版本？

该项目直接依赖 [onnxruntime](https://pypi.org/project/onnxruntime) 库。因此，只有当 [onnxruntime](https://pypi.org/project/onnxruntime) 支持对应 Python 版本时，我们才能更新兼容性。

## 请我喝杯咖啡

喜欢我的工作？请我喝杯咖啡（或一杯啤酒）。

<a href="https://www.buymeacoffee.com/danielgatis" target="_blank"><img src="https://bmc-cdn.nyc3.digitaloceanspaces.com/BMC-button-images/custom_images/orange_img.png" alt="请我喝杯咖啡" style="height: auto !important;width: auto !important;"></a>

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=danielgatis/rembg&type=Date)](https://star-history.com/#danielgatis/rembg&Date)

## 许可证

版权所有 (c) 2020 至今 [Daniel Gatis](https://github.com/danielgatis)

基于 [MIT 许可证](./LICENSE.txt) 授权
