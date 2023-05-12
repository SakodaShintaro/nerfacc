#!/bin/bash

set -eux

# Image and container names as script arguments
IMAGE_NAME=$1
CONTAINER_NAME=$2

# Build the Docker image
docker build -t $IMAGE_NAME .

# Create the Docker container from the image
docker run -d --name $CONTAINER_NAME \
    --gpus all \
    -v $HOME/data:/root/data \
    --net=host \
    -it $IMAGE_NAME bash
