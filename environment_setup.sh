#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # Initialize conda for bash
    conda init bash
    source ~/.bashrc

    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    
    # Install CUDA toolkit from conda-forge instead of nvidia channel
    conda install -c conda-forge cudatoolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# for fast attn
pip install -U xformers==0.0.27

# Install nunchaku explicitly with its dependencies
pip install "numpy<2.0.0,>=1.20.3" 
pip install nunchaku

# install sana
pip install -e .
