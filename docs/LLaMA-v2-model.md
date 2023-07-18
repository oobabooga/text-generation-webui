# LLaMA-v2

To convert LLaMA-v2 from the `.pth` format provided by Meta to transformers format, follow the steps below:

1) `cd` into your `llama` folder (the one containing `download.sh` and the models that you downloaded):

```
cd llama
```

2) Clone the transformers library:

```
git clone 'https://github.com/huggingface/transformers'

```

3) Create symbolic links from the downloaded folders to names that the conversion script can recognize:

```
ln -s llama-2-7b 7B
ln -s llama-2-13b 13B
```

4) Do the conversions:

```
mkdir llama-2-7b-hf llama-2-13b-hf
python ./transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir . --model_size 7B --output_dir llama-2-7b-hf --safe_serialization true
python ./transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir . --model_size 13B --output_dir llama-2-13b-hf --safe_serialization true
```

5) Move the output folders inside `text-generation-webui/models`

6) Have fun
