#!/bin/bash
set -e  # Exit on any error

# Step 1: Install PyTorch 2.6 with CUDA 12.6. This is best combination for installing FlashAttention within seconds but also matching all the CUDA, NCCL and CUDNN versions.
echo "Installing PyTorch 2.6 with CUDA 12.6..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Step 2: Install Python packages
echo "Installing base Python packages..."
pip install nltk wandb pandas pyarrow multiprocess xxhash
pip install transformers -v --no-deps
pip install datasets tokenizers sentencepiece protobuf tqdm nvtx regex safetensors pyyaml wheel pybind11 packaging ninja huggingface_hub --no-deps

# Step 3: Install transformer_engine
echo "Installing transformer_engine..."
pip install --no-build-isolation "transformer_engine[pytorch]==2.3.0"

# Step 4: Clone and install NVIDIA Apex
echo "Installing NVIDIA Apex..."
git clone https://github.com/NVIDIA/apex
cd apex
pip install . --no-build-isolation
cd ..
rm -rf apex

# Step 5: Install FlashAttention
echo "Installing FlashAttention..."
pip install --no-build-isolation flash-attn
echo "✅ Environment setup complete!"