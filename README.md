# Text generation web UI

A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

[[Try it on Google Colab]](https://colab.research.google.com/github/oobabooga/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb)

|![Image1](https://github.com/oobabooga/screenshots/raw/main/qa.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/cai3.png) |
|:---:|:---:|
|![Image3](https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png) | ![Image4](https://github.com/oobabooga/screenshots/raw/main/galactica.png) |

## Features

* Switch between different models using a dropdown menu.
* Notebook mode that resembles OpenAI's playground.
* Chat mode for conversation and role playing.
* Generate nice HTML output for GPT-4chan.
* Generate Markdown output for [GALACTICA](https://github.com/paperswithcode/galai), including LaTeX support.
* Support for [Pygmalion](https://huggingface.co/models?search=pygmalionai/pygmalion) and custom characters in JSON or TavernAI Character Card formats ([FAQ](https://github.com/oobabooga/text-generation-webui/wiki/Pygmalion-chat-model-FAQ)).
* Advanced chat features (send images, get audio responses with TTS).
* Stream the text output in real time very efficiently.
* Load parameter presets from text files.
* Load large models in 8-bit mode.
* Split large models across your GPU(s), CPU, and disk.
* CPU mode.
* [FlexGen offload](https://github.com/oobabooga/text-generation-webui/wiki/FlexGen).
* [DeepSpeed ZeRO-3 offload](https://github.com/oobabooga/text-generation-webui/wiki/DeepSpeed).
* Get responses via API, [with](https://github.com/oobabooga/text-generation-webui/blob/main/api-example-streaming.py) or [without](https://github.com/oobabooga/text-generation-webui/blob/main/api-example.py) streaming.
* [LLaMA model, including 4-bit mode](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model).
* [RWKV model](https://github.com/oobabooga/text-generation-webui/wiki/RWKV-model).
* [Supports LoRAs](https://github.com/oobabooga/text-generation-webui/wiki/Using-LoRAs).
* Supports softprompts.
* [Supports extensions](https://github.com/oobabooga/text-generation-webui/wiki/Extensions).
* [Works on Google Colab](https://github.com/oobabooga/text-generation-webui/wiki/Running-on-Colab).

## Installation

The recommended installation methods are the following:

* Linux and MacOS: using conda natively.
* Windows: using conda on WSL ([WSL installation guide](https://github.com/oobabooga/text-generation-webui/wiki/Windows-Subsystem-for-Linux-(Ubuntu)-Installation-Guide)).

Conda can be downloaded here: https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands:

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

Source: https://educe-ubc.github.io/conda.html

#### 1. Create a new conda environment

```
conda create -n textgen python=3.10.9
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch torchvision torchaudio` |
| Linux | AMD | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2` |
| MacOS + MPS (untested) | Any | `pip3 install torch torchvision torchaudio` |

The up to date commands can be found here: https://pytorch.org/get-started/locally/. 

MacOS users, refer to the comments here: https://github.com/oobabooga/text-generation-webui/pull/393


#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

> **Note**
> 
> For bitsandbytes and `--load-in-8bit` to work on Linux/WSL, this dirty fix is currently necessary: https://github.com/oobabooga/text-generation-webui/issues/400#issuecomment-1474876859

### Alternative: one-click installers

[oobabooga-windows.zip](https://github.com/oobabooga/one-click-installers/archive/refs/heads/oobabooga-windows.zip)

[oobabooga-linux.zip](https://github.com/oobabooga/one-click-installers/archive/refs/heads/oobabooga-linux.zip)

Just download the zip above, extract it, and double click on "install". The web UI and all its dependencies will be installed in the same folder.

* To download a model, double click on "download-model"
* To start the web UI, double click on "start-webui" 

Source codes: https://github.com/oobabooga/one-click-installers

> **Note**
> 
> To get 8-bit and 4-bit models working in your 1-click Windows installation, you can use the [one-click-bandaid](https://github.com/ClayShoaf/oobabooga-one-click-bandaid).

### Alternative: native Windows installation

As an alternative to the recommended WSL method, you can install the web UI natively on Windows using this guide. It will be a lot harder and the performance may be slower: [Installation instructions for human beings](https://github.com/oobabooga/text-generation-webui/wiki/Installation-instructions-for-human-beings).

### Alternative: Docker

https://github.com/oobabooga/text-generation-webui/issues/174, https://github.com/oobabooga/text-generation-webui/issues/87

## Downloading models

Models should be placed inside the `models` folder.

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some noteworthy examples:

* [Pythia](https://huggingface.co/models?search=eleutherai/pythia)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)
* [GPT-Neo](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads&search=eleutherai+%2F+gpt-neo)
* [\*-Erebus](https://huggingface.co/models?search=erebus) (NSFW)
* [Pygmalion](https://huggingface.co/models?search=pygmalion) (NSFW)

You can automatically download a model from HF using the script `download-model.py`:

    python download-model.py organization/model

For instance:

    python download-model.py facebook/opt-1.3b

If you want to download a model manually, note that all you need are the json, txt, and pytorch\*.bin (or model*.safetensors) files. The remaining files are not necessary.

### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

The 32-bit version is only relevant if you intend to run the model in CPU mode. Otherwise, you should use the 16-bit version.

After downloading the model, follow these steps:

1. Place the files under `models/gpt4chan_model_float16` or `models/gpt4chan_model`.
2. Place GPT-J 6B's config.json file in that same folder: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. Download GPT-J 6B's tokenizer files (they will be automatically detected when you attempt to load GPT-4chan):

```
python download-model.py EleutherAI/gpt-j-6B --text-only
```

## Starting the web UI

    conda activate textgen
    cd text-generation-webui
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`



Optionally, you can use the following command-line flags:

| Flag             | Description |
|------------------|-------------|
| `-h`, `--help`   | show this help message and exit |
| `--model MODEL`  | Name of the model to load by default. |
| `--lora LORA`    | Name of the LoRA to apply to the model by default. |
| `--notebook`     | Launch the web UI in notebook mode, where the output is written to the same text box as the input. |
| `--chat`         | Launch the web UI in chat mode.|
| `--cai-chat`     | Launch the web UI in chat mode with a style similar to Character.AI's. If the file `img_bot.png` or `img_bot.jpg` exists in the same folder as server.py, this image will be used as the bot's profile picture. Similarly, `img_me.png` or `img_me.jpg` will be used as your profile picture. |
| `--cpu`          | Use the CPU to generate text.|
| `--load-in-8bit` | Load the model with 8-bit precision.|
| `--load-in-4bit` | DEPRECATED: use `--gptq-bits 4` instead. |
| `--gptq-bits GPTQ_BITS` |  GPTQ: Load a pre-quantized model with specified precision. 2, 3, 4 and 8 (bit) are supported. Currently only works with LLaMA and OPT. |
| `--gptq-model-type MODEL_TYPE` |  GPTQ: Model type of pre-quantized model. Currently only LLaMa and OPT are supported. |
| `--gptq-pre-layer GPTQ_PRE_LAYER` |  GPTQ: The number of layers to preload. |
| `--bf16`         | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--auto-devices` | Automatically split the model across the available GPU(s) and CPU.|
| `--disk`         | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR` | Directory to save the disk cache to. Defaults to `cache/`. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` |  Maxmimum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY` | Maximum CPU memory in GiB to allocate for offloaded weights. Must be an integer number. Defaults to 99.|
| `--no-cache`     | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost. |
| `--flexgen`      | Enable the use of FlexGen offloading. |
|  `--percent PERCENT [PERCENT ...]` |  FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0). |
|  `--compress-weight` |  FlexGen: Whether to compress weight (default: False).|
|  `--pin-weight [PIN_WEIGHT]` |       FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%). |
| `--deepspeed`    | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK` | DeepSpeed: Optional argument for distributed setups. |
|  `--rwkv-strategy RWKV_STRATEGY` |    RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8". |
|  `--rwkv-cuda-on` |   RWKV: Compile the CUDA kernel for better performance. |
| `--no-stream`    | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE` | Load the default interface settings from this json file. See `settings-template.json` for an example. If you create a file called `settings.json`, this file will be loaded by default without the need to use the `--settings` flag.|
|  `--extensions EXTENSIONS [EXTENSIONS ...]` |  The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--listen`       | Make the web UI reachable from your local network.|
|  `--listen-port LISTEN_PORT` | The listening port that the server will use. |
| `--share`        | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`  | Open the web UI in the default browser upon launch. |
| `--verbose`      | Print the prompts to the terminal. |

Out of memory errors? [Check the low VRAM guide](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

By default, 10 presets by NovelAI and KoboldAI are included. These were selected out of a sample of 43 presets after applying a K-Means clustering algorithm and selecting the elements closest to the average of each cluster.

## System requirements

Check the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) for some examples of VRAM and RAM usage in both GPU and CPU mode.

## Contributing

Pull requests, suggestions, and issue reports are welcome.

Before reporting a bug, make sure that you have:

1. Created a conda environment and installed the dependencies exactly as in the *Installation* section above.
2. [Searched](https://github.com/oobabooga/text-generation-webui/issues) to see if an issue already exists for the issue you encountered.

## Credits

- Gradio dropdown menu refresh button, code for reloading the interface: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Verbose preset: Anonymous 4chan user.
- NovelAI and KoboldAI presets: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Pygmalion preset, code for early stopping in chat mode, code for some of the sliders, --chat mode colors: https://github.com/PygmalionAI/gradio-ui/
