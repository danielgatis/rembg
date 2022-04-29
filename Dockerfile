FROM nvidia/cuda:11.6.0-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list || true
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y curl software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9 python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

WORKDIR /rembg

COPY . .
RUN python3.9 -m pip install .[gpu]

RUN mkdir -p ~/.u2net
RUN gdown https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR -O ~/.u2net/u2netp.onnx
RUN gdown https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab -O ~/.u2net/u2net.onnx
RUN gdown https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j -O ~/.u2net/u2net_human_seg.onnx
RUN gdown https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz -O ~/.u2net/u2net_cloth_seg.onnx

EXPOSE 5000
ENTRYPOINT ["rembg"]
CMD ["--help"]
