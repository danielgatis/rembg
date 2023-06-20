FROM python:3.10-slim

WORKDIR /rembg

COPY . .
RUN pip install --upgrade pip
RUN python -m pip install ".[cli]"
RUN python -c 'from rembg.bg import download_models; download_models()'

EXPOSE 5000
ENTRYPOINT ["rembg"]
CMD ["--help"]
