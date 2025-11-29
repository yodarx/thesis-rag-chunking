#!/bin/bash
set -e

export PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Install dependencies
sudo apt-get update
sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip

# Create venv
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Run build_indices.py for all configs
for config in configs/*.json; do
    echo "Running build_indices.py for $config"
    python build_indices.py --config "$config"
done