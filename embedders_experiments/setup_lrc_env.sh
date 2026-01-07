#!/bin/bash
set -e

# Base dir on GPU cluster
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR

# Create venv
python3 -m venv master-thesis-env
source master-thesis-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch GPU build
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers chromadb sentence-transformers huggingface-hub accelerate tqdm

# (optional) login if private models
huggingface-cli login

# Download model locally
python - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-multilingual-gemma2", local_dir="models/bge-multilingual-gemma2")
EOF

echo "✅ master-thesis-env setup complete."

