FROM python:3.10-slim

WORKDIR /rembg

COPY . .

RUN python -m pip install ".[gpu,cli]"
RUN python -c 'from rembg.bg import download_models; download_models()'

EXPOSE 7000
ENTRYPOINT ["rembg"]
CMD ["s", "-t 8"]
