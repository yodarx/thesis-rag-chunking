# Docker Configuration

This directory contains all Docker-related files for the thesis RAG chunking project.

## Files

- **Dockerfile**: Container definition for running experiments
- **.dockerignore**: Files to exclude from Docker build context
- **build-and-upload.sh**: Script to build and push Docker image to Docker Hub
- **run_and_stop.sh**: Script that runs inside the container to execute experiments and stop RunPod pods

## Building the Docker Image

### Local Build

From the **project root directory**, run:

```bash
docker build -f docker/Dockerfile -t thesis-rag-chunking .
```

### Build and Upload to Docker Hub

From the **project root directory**, run:

```bash
bash docker/build-and-upload.sh [TAG]
```

Arguments:

- `TAG` (optional): Image tag to use (default: `latest`)
    - Use `dev` to build locally without uploading to Docker Hub

Examples:

```bash
# Build and upload with 'latest' tag
bash docker/build-and-upload.sh

# Build and upload with custom tag
bash docker/build-and-upload.sh v1.2.3

# Build locally without uploading
bash docker/build-and-upload.sh dev
```

## Running the Container

### Local Run

```bash
docker run -v $(pwd)/results:/workspace/results thesis-rag-chunking
```

### RunPod.io Deployment

1. Build and upload the image using `build-and-upload.sh`
2. In RunPod.io, create a new pod with the image:
   ```
   jsstudentsffhs/thesis-rag-chunking:latest
   ```
3. Set environment variables:
    - `RUNPOD_API_KEY`: Your RunPod API key (for auto-shutdown)
4. Mount `/workspace/results` as a network volume to persist results

The container will automatically:

- Run all experiments defined in the configs
- Save results to `/workspace/results`
- Stop the RunPod pod after completion (if `RUNPOD_API_KEY` is set)

## Image Details

- **Base Image**: python:3.12-slim
- **Working Directory**: `/app`
- **Volume Mount**: `/workspace` (for results and data persistence)
- **Default Command**: Runs experiments and stops the pod

## Notes

- All Docker commands should be run from the **project root directory**, not from the `docker/` directory
- The build script must be run from the project root to ensure proper context
- Results are saved to `/workspace/results` for RunPod compatibility
- The container automatically downloads required NLTK resources during build

