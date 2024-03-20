FROM nvcr.io/nvidia/tensorrt:24.02-py3

WORKDIR /rembg

COPY . .

RUN python -m pip install ".[gpu,cli]"
RUN python -c 'from rembg.bg import download_models; download_models()'

EXPOSE 7000
ENTRYPOINT ["rembg"]
CMD ["s", "-t 8"]
