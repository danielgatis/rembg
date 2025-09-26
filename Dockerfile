# Stage 1: Build the rembg package wheel
FROM python:3.10-slim as builder

WORKDIR /src

# Copy all the source code
COPY . .

# Install build dependencies and build the wheel
RUN pip install wheel
RUN python setup.py bdist_wheel

# Stage 2: Create the final, clean image
FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

# Copy and install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the built wheel from the builder stage and install it
COPY --from=builder /src/dist/*.whl .
RUN pip install *.whl && rm *.whl

# Now that rembg is installed, copy the handler
COPY handler.py .

# Download the necessary model files using the 'rembg' command
RUN rembg d u2net

# Set the command to run our handler
CMD ["python", "handler.py"]