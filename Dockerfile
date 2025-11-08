FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in same layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy requirements for faster Docker builds
COPY requirements.txt .

# Upgrade pip and install all Python packages from requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* \
    && find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY run.py .

# Set Python path
ENV PYTHONPATH=/app

# Create workspace directory for RunPod.io compatibility
RUN mkdir -p /workspace/results

# Download NLTK resources
RUN python -m nltk.downloader punkt punkt_tab

# Set workspace as volume mount point
VOLUME /workspace

# Default command - run all experiments with output to workspace
CMD python run.py --run-all-configs && echo "Process exited with code $? at path: $(pwd)"