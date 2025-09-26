#!/bin/bash
set -e

# Replace with your container registry
REGISTRY="your-registry"
IMAGE_NAME="rembg-serverless"
TAG="latest"

docker build -t $REGISTRY/$IMAGE_NAME:$TAG .
docker push $REGISTRY/$IMAGE_NAME:$TAG