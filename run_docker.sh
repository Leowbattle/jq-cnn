#!/bin/bash

# Build and run the jq-cnn Docker container

set -e

echo "Building jq-cnn Docker image..."
docker build -t jq-cnn .

echo "Running jq-cnn container..."
docker run --rm -it jq-cnn

echo "Done!"