LLaMA is a Large Language Model developed by Meta AI. 

It was trained on more tokens than previous models. The result is that the smallest version with 7 billion parameters has similar performance to GPT-3 with 175 billion parameters.

This guide will cover usage through the official `transformers` implementation. For 4-bit mode, head over to [GPTQ models (4 bit mode)
](GPTQ-models-(4-bit-mode).md).

## Getting the weights

### Option 1: pre-converted weights

* Torrent: https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789
* Direct download: https://huggingface.co/Neko-Institute-of-Science

⚠️ The tokenizers for the Torrent source above and also for many LLaMA fine-tunes available on Hugging Face may be outdated, in particular the files called `tokenizer_config.json` and `special_tokens_map.json`. Here you can find those files: https://huggingface.co/oobabooga/llama-tokenizer

### Option 2: convert the weights yourself

1. Install the `protobuf` library:

```
pip install protobuf==3.20.1
```

2. Use the script below to convert the model in `.pth` format that you, a fellow academic, downloaded using Meta's official link.

If you have `transformers` installed in place:

```
python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir /path/to/LLaMA --model_size 7B --output_dir /tmp/outputs/llama-7b
```

Otherwise download [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) first and run:

```
python convert_llama_weights_to_hf.py --input_dir /path/to/LLaMA --model_size 7B --output_dir /tmp/outputs/llama-7b
```

3. Move the `llama-7b` folder inside your `text-generation-webui/models` folder.

## Starting the web UI

```python
python server.py --model llama-7b
```
