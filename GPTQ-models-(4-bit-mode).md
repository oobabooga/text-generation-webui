In 4-bit mode, models are loaded with just 25% of their regular VRAM usage. So LLaMA-7B fits into a 6GB GPU, and LLaMA-30B fits into a 24GB GPU.

This is possible thanks to [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa)'s adaptation of the GPTQ algorithm for LLaMA: https://github.com/qwopqwop200/GPTQ-for-LLaMa

GPTQ is a clever quantization algorithm that lightly reoptimizes the weights during quantization so that the accuracy loss is compensated relative to a round-to-nearest quantization. See the paper for more details: https://arxiv.org/abs/2210.17323

## Installation

### Step 0: install nvcc

```
conda activate textgen
conda install -c conda-forge cudatoolkit-dev
```

The command above takes some 10 minutes to run and shows no progress bar or updates along the way.

See this issue for more details: https://github.com/oobabooga/text-generation-webui/issues/416#issuecomment-1475078571

### Step 1: install GPTQ-for-LLaMa

Clone the GPTQ-for-LLaMa repository into the `text-generation-webui/repositories` subfolder and install it:

```
mkdir repositories
cd repositories
git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

You are going to need to have a C++ compiler installed into your system for the last command. On Linux, `sudo apt install build-essential` or equivalent is enough.

https://github.com/oobabooga/GPTQ-for-LLaMa corresponds to commit `a6f363e3f93b9fb5c26064b5ac7ed58d22e3f773` in the `cuda` branch of the original repository and is recommended by default for stability. Some models might require you to use the up-to-date CUDA or triton branches:

```
cd repositories
rm -r GPTQ-for-LLaMa
pip uninstall -y quant-cuda
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git -b cuda
...
```

```
cd repositories
rm -r GPTQ-for-LLaMa
pip uninstall -y quant-cuda
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git -b triton
...
```


https://github.com/qwopqwop200/GPTQ-for-LLaMa

### Step 2: get the pre-converted weights

* Converted without `group-size` (better for the 7b model): https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617
* Converted with `group-size` (better from 13b upwards): https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105 

Note: the tokenizer files in those torrents are not up to date.

### Step 3: Start the web UI:

For the models converted without `group-size`:

```
python server.py --model llama-7b-4bit 
```

For the models converted with `group-size`:

```
python server.py --model llama-13b-4bit-128g 
```

The command-line flags `--wbits` and `--groupsize` are automatically detected based on the folder names, but you can also specify them manually like 

```
python server.py --model llama-13b-4bit-128g --wbits 4 --groupsize 128
```

## CPU offloading

It is possible to offload part of the layers of the 4-bit model to the CPU with the `--pre_layer` flag. The higher the number after `--pre_layer`, the more layers will be allocated to the GPU.

With this command, I can run llama-7b with 4GB VRAM:

```
python server.py --model llama-7b-4bit --pre_layer 20
```

This is the performance:

```
Output generated in 123.79 seconds (1.61 tokens/s, 199 tokens)
```

## Using LoRAs in 4-bit mode

At the moment, this feature is not officially supported by the relevant libraries, but a patch exists and is supported by this web UI: https://github.com/johnsmith0031/alpaca_lora_4bit

In order to use it:

1. Make sure that your requirements are up to date:

```
cd text-generation-webui
pip install -r requirements.txt --upgrade
```

2. Clone `johnsmith0031/alpaca_lora_4bit` into the repositories folder:

```
cd text-generation-webui/repositories
git clone https://github.com/johnsmith0031/alpaca_lora_4bit
```

3. Install https://github.com/sterlind/GPTQ-for-LLaMa with this command:

```
pip install git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit
```

4. Start the UI with the `--monkey-patch` flag:

```
python server.py --model llama-7b-4bit-128g --listen --lora tloen_alpaca-lora-7b --monkey-patch
```
