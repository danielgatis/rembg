FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

RUN rembg d u2net

CMD ["python", "handler.py"]