# Docker Instructions

## Setup

1. Move start_docker.sh and kill_docker.sh to /text-generation-webui
2. Copy .env.example into .env and edit the values within
3. Download a model like so:

  ```python
  pip install requests tqdm
  python download_model.py TheBloke/wizardLM-7B-GPTQ
  ```

4. To start container execute:

  ```sh
  sh start_docker.sh
  ```

5. To kill container (because container does not wanna die gracefully)

  ```sh
  sh stop_docker.sh
  ```

## Changelog

5/25/23: Added shell script to execute Docker. added variable to define QUANT_MODEL_URL. added pip install git+<https://github.com/johnsmith0031/alpaca_lora_4bit@winglian-setup_pip> just in case it works for anyone. Added installation files for WSL with their sources.

## Works On

RTX 3060
WIN 11
WSL 2
Docker v23.0.5
Docker Compose version v2.17.3

## Contributing

[EHGP](https://github.com/ehgp)

## Disclaimer

Not responsible if your GPU bricks or WSL takes your GPU hostage.

## Sources

Docker <https://gist.github.com/r7l/99d4c880d5dc9bfa57a8a988eae78696>
CUDA on WSL <https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl>
NVIDIA container runtime <https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup>
