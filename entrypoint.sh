#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda create -n textgen python=3.10.9 -y
conda activate textgen

git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip3 install -r requirements.txt
cd ..

conda install -c conda-forge cudatoolkit-dev -y

pip3 install torch torchvision torchaudio

cd repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip3 install -r requirements.txt

CUDA_HOME=/usr/local/cuda-11.7 python3 setup_cuda.py install

cd ../../
# runs with less then 4GB of vram
#python3 server.py --model llama-7b-hf --gptq-bits 4 --gptq-pre-layer 20 --listen

python3 server.py --model llama-7b-hf --gptq-bits 4 --gptq-pre-layer 30 --listen
