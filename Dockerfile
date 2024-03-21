FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /rembg

COPY . .

RUN apt-get update
RUN apt-get install -y python-is-python3 python3 python3-pip
RUN python -m pip install ".[gpu,cli]"
RUN python -c 'from rembg.bg import download_models; download_models()'

EXPOSE 7000
ENTRYPOINT ["rembg"]
CMD ["s", "-t 8"]
