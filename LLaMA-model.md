LLaMA is a Large Language Model developed by Meta AI. 

It was trained on more tokens than previous models. The result is that the smallest version with 7 billion parameters has similar performance to GPT-3 with 175 billion parameters.

## Installation

1. Uninstall your existing `transformers` (if any) and install this patched version:

```
pip uninstall transformers
pip install git+https://github.com/zphang/transformers@llama_push
```


2. Use this script to convert the model in `.pth` format that you, a fellow academic, downloaded using Meta's official link:

### [convert_llama_weights_to_hf.py](https://github.com/zphang/transformers/blob/llama_push/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```
python convert_llama_weights_to_hf.py --input_dir /path/to/LLaMA --model_size 7B --output_dir /path/to/outputs
```

Two folders will be created at the end:

```
/path/to/outputs/llama-7b
/path/to/outputs/tokenizer
```

3. Move the files inside `/path/to/outputs/tokenizer` to `/path/to/outputs/llama-7b`:

```
mv /path/to/outputs/tokenizer/* /path/to/outputs/llama-7b
```

4. Move the `llama-7b` folder inside your `text-generation-webui/models` folder.

5. Launch the web UI:

```
python server.py --model llama-7b
```

## 4-bit mode

In 4-bit mode, it is possible to load LLaMA models with just 25% of the regular VRAM usage. So LLaMA-7B fits into a 6GB GPU, and LLaMA-30B fits into a 24GB GPU.

This is possible thanks to [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa)'s adaptation of the GPTQ algorithm for LLaMA: https://github.com/qwopqwop200/GPTQ-for-LLaMa

GPTQ is a clever quantization algorithm that lightly reoptimizes the weights during quantization so that the accuracy loss is compensated relative to a naive round-to-nearest quantization. See the paper for more details: https://arxiv.org/abs/2210.17323

#### Installation

1. Clone [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) into the `text-generation-webui/repositories` subfolder and install it:

```
mkdir repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

You are going to need to have a C++ compiler installed into your system for the last command. On Linux, `sudo apt install build-essential` or equivalent is enough.

2. Install the newest LLaMA-patched transformers:

```
pip uninstall transformers
pip install git+https://github.com/zphang/transformers@llama_push
```

3. Place the LLaMA model that you [converted](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model/_edit#convert_llama_weights_to_hfpy) using `convert_llama_weights_to_hf.py` into the `models` folder. For instance, `models/llama-7b`.

4. Place the corresponding 4-bit model directly into your `models` folder. For instance, `models/llama-7b-4bit.pt`. You can find pre-converted models here: https://huggingface.co/decapoda-research

5. Start the web UI with `--load-in-4bit`:

```
python server.py --model llama-7b --load-in-4bit
```

For more information, check the comments in the PR for this: https://github.com/oobabooga/text-generation-webui/pull/206.