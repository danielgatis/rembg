FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y curl python3.9 python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

WORKDIR /rembg

COPY . .
RUN python3.9 -m pip install .[gpu]

RUN mkdir -p ~/.u2net
RUN gdown https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR -O ~/.u2net/u2netp.onnx
RUN gdown https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab -O ~/.u2net/u2net.onnx
RUN gdown https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j -O ~/.u2net/u2net_human_seg.onnx
RUN gdown https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz -O ~/.u2net/u2net_cloth_seg.onnx

ENTRYPOINT ["rembg"]
CMD ["--help"]
