GPTQ is a clever quantization algorithm that lightly reoptimizes the weights during quantization so that the accuracy loss is compensated relative to a round-to-nearest quantization. See the paper for more details: https://arxiv.org/abs/2210.17323

4-bit GPTQ models reduce VRAM usage by about 75%. So LLaMA-7B fits into a 6GB GPU, and LLaMA-30B fits into a 24GB GPU.

## Overview

There are two ways of loading GPTQ models in the web UI at the moment:

* Using AutoGPTQ:
  * supports more models
  * standardized (no need to guess any parameter)
  * is a proper Python library
  * ~no wheels are presently available so it requires manual compilation~
  * supports loading both triton and cuda models

* Using GPTQ-for-LLaMa directly:
  * faster CPU offloading
  * faster multi-GPU inference
  * supports loading LoRAs using a monkey patch
  * requires you to manually figure out the wbits/groupsize/model_type parameters for the model to be able to load it
  * supports either only cuda or only triton depending on the branch

For creating new quantizations, I recommend using AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ

## AutoGPTQ

### Installation

No additional steps are necessary as AutoGPTQ is already in the `requirements.txt` for the webui. If you still want or need to install it manually for whatever reason, these are the commands:

```
conda activate textgen
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .
```

The last command requires `nvcc` to be installed (see the [instructions above](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#step-1-install-nvcc)).

### Usage

When you quantize a model using AutoGPTQ, a folder containing a filed called `quantize_config.json` will be generated. Place that folder inside your `models/` folder and load it with the `--autogptq` flag:

```
python server.py --autogptq --model model_name
```

Alternatively, check the `autogptq` box in the "Model" tab of the UI before loading the model.

### Offloading

In order to do CPU offloading or multi-gpu inference with AutoGPTQ, use the `--gpu-memory` flag. It is currently somewhat slower than offloading with the `--pre_layer` option in GPTQ-for-LLaMA.

For CPU offloading:

```
python server.py --autogptq --gpu-memory 3000MiB --model model_name
```

For multi-GPU inference:

```
python server.py --autogptq --gpu-memory 3000MiB 6000MiB --model model_name
```

### Using LoRAs with AutoGPTQ

Works fine for a single LoRA.

## GPTQ-for-LLaMa

GPTQ-for-LLaMa is the original adaptation of GPTQ for the LLaMA model. It was made possible by [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa): https://github.com/qwopqwop200/GPTQ-for-LLaMa

A Python package containing both major CUDA versions of GPTQ-for-LLaMa is used to simplify installation and compatibility: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA

### Precompiled wheels

Kindly provided by our friend jllllll: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA/releases

Wheels are included in requirements.txt and are installed with the webui on supported systems.

### Manual installation

#### Step 1: install nvcc

```
conda activate textgen
conda install cuda -c nvidia/label/cuda-11.7.1
```

The command above takes some 10 minutes to run and shows no progress bar or updates along the way.

You are also going to need to have a C++ compiler installed. On Linux, `sudo apt install build-essential` or equivalent is enough. On Windows, Visual Studio or Visual Studio Build Tools is required.

If you're using an older version of CUDA toolkit (e.g. 11.7) but the latest version of `gcc` and `g++` (12.0+) on Linux, you should downgrade with: `conda install -c conda-forge gxx==11.3.0`. Kernel compilation will fail otherwise.

#### Step 2: compile the CUDA extensions

```
python -m pip install git+https://github.com/jllllll/GPTQ-for-LLaMa-CUDA -v
```

### Getting pre-converted LLaMA weights

* Direct download (recommended):

https://huggingface.co/Neko-Institute-of-Science/LLaMA-7B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-13B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-30B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-65B-4bit-128g

These models were converted with `desc_act=True`. They work just fine with ExLlama. For AutoGPTQ, they will only work on Linux with the `triton` option checked.

* Torrent:

https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617

https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105

These models were converted with `desc_act=False`. As such, they are less accurate, but they work with AutoGPTQ on Windows. The `128g` versions are better from 13b upwards, and worse for 7b. The tokenizer files in the torrents are outdated, in particular the files called `tokenizer_config.json` and `special_tokens_map.json`. Here you can find those files: https://huggingface.co/oobabooga/llama-tokenizer

### Starting the web UI:

Use the `--gptq-for-llama` flag.

For the models converted without `group-size`:

```
python server.py --model llama-7b-4bit --gptq-for-llama 
```

For the models converted with `group-size`:

```
python server.py --model llama-13b-4bit-128g  --gptq-for-llama --wbits 4 --groupsize 128
```

The command-line flags `--wbits` and `--groupsize` are automatically detected based on the folder names in many cases.

### CPU offloading

It is possible to offload part of the layers of the 4-bit model to the CPU with the `--pre_layer` flag. The higher the number after `--pre_layer`, the more layers will be allocated to the GPU.

With this command, I can run llama-7b with 4GB VRAM:

```
python server.py --model llama-7b-4bit --pre_layer 20
```

This is the performance:

```
Output generated in 123.79 seconds (1.61 tokens/s, 199 tokens)
```

You can also use multiple GPUs with `pre_layer` if using the oobabooga fork of GPTQ, eg `--pre_layer 30 60` will load a LLaMA-30B model half onto your first GPU and half onto your second, or `--pre_layer 20 40` will load 20 layers onto GPU-0, 20 layers onto GPU-1, and 20 layers offloaded to CPU.

### Using LoRAs with GPTQ-for-LLaMa

This requires using a monkey patch that is supported by this web UI: https://github.com/johnsmith0031/alpaca_lora_4bit

To use it:

1. Install alpaca_lora_4bit using pip

```
git clone https://github.com/johnsmith0031/alpaca_lora_4bit.git
cd alpaca_lora_4bit
git fetch origin winglian-setup_pip
git checkout winglian-setup_pip
pip install .
```

2. Start the UI with the `--monkey-patch` flag:

```
python server.py --model llama-7b-4bit-128g --listen --lora tloen_alpaca-lora-7b --monkey-patch
```


