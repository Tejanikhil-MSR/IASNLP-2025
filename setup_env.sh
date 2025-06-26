#!/bin/bash

# Clone the NeMo repo from AI4Bharat
git clone https://github.com/AI4Bharat/NeMo.git

# Install PyTorch and related packages
pip3 install torch torchvision torchaudio

# Install packaging
pip install packaging

# Clone and install NVIDIA Apex with custom build options
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    --config-settings "--build-option=--fast_layer_norm" \
    --config-settings "--build-option=--distributed_adam" \
    --config-settings "--build-option=--deprecated_fused_adam" ./
cd ..

# Reinstall NeMo (AI4Bharat version)
cd NeMo
chmod +x ./reinstall.sh
./reinstall.sh
cd ..

# Install OpenSMILE Python wrapper
pip install opensmile