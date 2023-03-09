LLaMA is a Large Language Model developed Meta AI. 

It was trained on more tokens than previous models. The result is that the smallest version with 7 billion parameters has similar performance to GPT-3 with 175 billion parameters.

## Installation

1. Uninstall your existing `transformers` (if any) and install this patched version:

```
pip uninstall transformers
pip install git+https://github.com/zphang/transformers@llama_push
```


2. Convert the model that you, a fellow academic, downloaded using Meta's official link using this script:

### [convert_llama_weights_to_hf.py](https://github.com/zphang/transformers/blob/llama_push/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```
python convert_llama_weights_to_hf.py --input_dir /path/to/LLaMA --model_size 7B --output_dir /path/to/outputs
```

The script will create two new folders:

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

## 4-bit installation

TODO
