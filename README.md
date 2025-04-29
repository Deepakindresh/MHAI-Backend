# MHAI-Backend

This repository contains the backend implementation for the MHAI (Mental Health AI) application using FastAPI and the Unsloth-optimized DeepSeek model.

## System Requirements

- Linux (WSL or native)
- NVIDIA GPU with CUDA support
- Python 3.11

## Installation Guide

Follow these steps to set up the development environment:

### 1. Clone the Repository

```bash
git clone https://github.com/Deepakindresh/MHAI-Backend.git
cd MHAI-Backend
```

### 2. Install CUDA Toolkit

First, check your NVIDIA driver version and make sure it supports the CUDA version that we need:

```bash
nvidia-smi
```

Then install CUDA 12.4:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

### 3. Set up CUDA Environment Variables

Add CUDA to your path:

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

To make these changes permanent, add the above lines to your `~/.bashrc` file and reload:

```bash
source ~/.bashrc
```

Verify installation:

```bash
nvcc --version
```

### 4. Install cuDNN

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

### 5. Install Anaconda

```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

Follow the prompts during installation. After installation is complete, activate the conda environment:

```bash
source ~/.bashrc
```

### 6. Create and Set Up Conda Environment

Create a new conda environment with Python 3.11 and PyTorch with CUDA support:

```bash
conda create --name unsloth_env python=3.11 pytorch-cuda=12.4 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate unsloth_env
```

### 7. Install Required Python Packages

```bash
pip install uvicorn fastapi pydantic unsloth supabase python-dotenv numpy
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
```

## Running the Application

Activate the conda environment:

```bash
conda activate unsloth_env
```

Start the FastAPI server:

```bash
uvicorn deepSeek:app --host 0.0.0.0 --port 5000 --reload
```

The API will be available at `http://localhost:5000`

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   - Ensure your NVIDIA drivers are compatible with CUDA 12.4
   - Check for version compatibility with `nvidia-smi` and `nvcc --version`

2. **PyTorch CUDA Issues**
   - Verify PyTorch can see your GPU:
     ```python
     import torch
     print(torch.cuda.is_available())
     print(torch.cuda.device_count())
     print(torch.cuda.get_device_name(0))
     ```

3. **Import Errors**
   - Make sure all dependencies are installed in the active conda environment
   - Try reinstalling problematic packages

### If You Encounter "ModuleNotFoundError"

Double-check that you've activated the correct conda environment:

```bash
conda activate unsloth_env
```

## Project Structure

```
MHAI-Backend/
├── deepSeek.py        # Main FastAPI application file
├── requirements.txt   # Generated package requirements
└── ...                # Other project files
```

## License

Copyright © 2025 Dandelion Fly. All rights reserved.

This is proprietary software. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited without express written permission from Dandelion Fly.

## Contributors

Deepakindresh Narayana Gandhi


## ENV file

Request me privately to access the Env variables, this should soon be changed to the company's profile.
