#!/bin/bash

IMAGE_NAME="thesis-rag-chunking"
TAG="${1:-latest}"
DOCKERHUB_USERNAME="jsstudentsffhs"

echo "🐳 Building Docker container..."
docker build -t $IMAGE_NAME:$TAG .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker build successful!"


echo "🏷️ Tagging image for Docker Hub..."
docker tag $IMAGE_NAME:$TAG $DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG

echo "🔐 Logging into Docker Hub..."
docker login

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed!"
    exit 1
fi

if [ "$TAG" = "dev" ]; then
    echo "🚫 Tag is 'dev', skipping upload to Docker Hub."
else
    echo "📤 Pushing image to Docker Hub..."
    docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG

    if [ $? -ne 0 ]; then
        echo "❌ Docker push failed!"
        exit 1
    fi
fi

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed!"
    exit 1
fi

echo "✅ Successfully uploaded to Docker Hub!"
echo "🎯 RunPod image name: $DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG"
echo ""
echo "To use in RunPod, specify this image:"
echo "$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG"
