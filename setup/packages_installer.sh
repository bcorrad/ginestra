#!/bin/bash

# === Setup conda ===
CONDA_BASE="/repo/$USER/anaconda3"  # Or adjust to /repo/$USER/anaconda3 if needed
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ Cannot find conda at $CONDA_BASE"
    exit 1
fi

# === Create and activate the environment ===
conda create -y -n ginestra python=3.11
conda activate ginestra

# === Detect CUDA version ===
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")

# === Match compatible PyTorch packages ===
case $CUDA_VERSION in
  "11.8")
    TORCH_URL="https://download.pytorch.org/whl/torch_stable.html"
    PYG_URL="https://data.pyg.org/whl/torch-2.2.0+cu118.html"
    TORCH="torch==2.2.0+cu118 torchvision==0.17.0+cu118"
    ;;
  "12.1")
    TORCH_URL="https://download.pytorch.org/whl/torch_stable.html"
    PYG_URL="https://data.pyg.org/whl/torch-2.2.0+cu121.html"
    TORCH="torch==2.2.0+cu121 torchvision==0.17.0+cu121"
    ;;
  *)
    echo "⚠️  Unsupported or missing CUDA version: '$CUDA_VERSION'"
    echo "    You may need to manually specify compatible torch/torchvision versions."
    exit 1
    ;;
esac

# === Install PyTorch + PyG with CUDA support ===
pip install $TORCH torchmetrics==1.6.2 pytorch-lightning==2.5.0.post0 -f "$TORCH_URL"
pip install torch-geometric==2.6.1 -f "$PYG_URL"

# === Install NLP libraries ===
pip install transformers==4.45.2 datasets==2.19.1 tokenizers==0.20.1 huggingface_hub==0.29.2

# === Install scientific libraries ===
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 matplotlib==3.8.4 tqdm==4.66.5 optuna==4.3.0 scikit-learn==1.2.2 scikit-image==0.22.0 seaborn==0.12.2 py3Dmol==1.8.0

# === Install additional libraries ===
pip install rdkit-pypi

# === Install Wandb === 
pip install wandb
pip install optuna-integration[wandb]

echo "✅ Environment 'ginestra' successfully created with CUDA $CUDA_VERSION!"
