LLaMA is a Large Language Model developed by Meta AI. 

It was trained on more tokens than previous models. The result is that the smallest version with 7 billion parameters has similar performance to GPT-3 with 175 billion parameters.

## 4-bit mode

In 4-bit mode, the LLaMA models are loaded with just 25% of their regular VRAM usage. So LLaMA-7B fits into a 6GB GPU, and LLaMA-30B fits into a 24GB GPU.

This is possible thanks to [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa)'s adaptation of the GPTQ algorithm for LLaMA: https://github.com/qwopqwop200/GPTQ-for-LLaMa

GPTQ is a clever quantization algorithm that lightly reoptimizes the weights during quantization so that the accuracy loss is compensated relative to a round-to-nearest quantization. See the paper for more details: https://arxiv.org/abs/2210.17323

### Installation

#### Step 0: install nvcc

```
conda activate textgen
conda install -c conda-forge cudatoolkit-dev
```

The command above takes some 10 minutes to run and shows no progress bar or updates along the way.

See this issue for more details: https://github.com/oobabooga/text-generation-webui/issues/416#issuecomment-1475078571

#### Step 1: install GPTQ-for-LLaMa

Clone [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) into the `text-generation-webui/repositories` subfolder and install it:

```
mkdir repositories
cd repositories
git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

You are going to need to have a C++ compiler installed into your system for the last command. On Linux, `sudo apt install build-essential` or equivalent is enough.

**Note**: I am using my own fork of GPTQ-for-LLaMa until qwopqwop200's branch becomes more stable. It corresponds to commit `a6f363e3f93b9fb5c26064b5ac7ed58d22e3f773` in the `cuda` branch of his repository.

#### Step 2: get the pre-converted weights

* Converted without `group-size` (better for the 7b model): https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617
* Converted with `group-size` (better from 13b upwards): https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105 

#### Step 3: Start the web UI:

For the models converted without `group-size`:

```
python server.py --model llama-7b-4bit --wbits 4 
```

For the models converted with `group-size`:

```
python server.py --model llama-13b-4bit-128g --wbits 4 --groupsize 128 
```

### CPU offloading

It is possible to offload part of the layers of the 4-bit model to the CPU with the `--pre_layer` flag. The higher the number after `--pre_layer`, the more layers will be allocated to the GPU.

With this command, I can run llama-7b with 4GB VRAM:

```
python server.py --model llama-7b-4bit --wbits 4 --pre_layer 20
```

This is the performance:

```
Output generated in 123.79 seconds (1.61 tokens/s, 199 tokens)
```
## Hugging Face format weights

For regular 16-bit/8-bit inference.

### Getting the weights

#### Option 1: pre-converted weights

* Torrent: https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789
* Direct download: https://huggingface.co/Neko-Institute-of-Science

Both are from the same author. 

**Note**: the tokenizer files in those sources seem to be incorrect for now: https://github.com/oobabooga/text-generation-webui/issues/931#issuecomment-1501259027

#### Option 2: convert the weights yourself

1. Use the script below to convert the model in `.pth` format that you, a fellow academic, downloaded using Meta's official link:

### [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```
python convert_llama_weights_to_hf.py --input_dir /path/to/LLaMA --model_size 7B --output_dir /tmp/llama-7b-converted
```

3. Move the `llama-7b` folder inside your `text-generation-webui/models` folder.

### Starting the web UI

```python
python server.py --model llama-7b
```