FROM python:3.10-slim

WORKDIR /rembg

RUN pip install --upgrade pip

COPY . .

RUN python -m pip install ".[cli]"
RUN rembg d

EXPOSE 7000
ENTRYPOINT ["rembg"]
CMD ["--help"]
