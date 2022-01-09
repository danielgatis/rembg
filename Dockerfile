FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev llvm llvm-dev
RUN pip3 install rembg

ENTRYPOINT ["rembg"]
CMD []
