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
* Stream the text output in real time.
* Load parameter presets from text files.
* Load large models in 8-bit mode (see [here](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134), [here](https://github.com/oobabooga/text-generation-webui/issues/20#issuecomment-1411650652) and [here](https://www.reddit.com/r/PygmalionAI/comments/1115gom/running_pygmalion_6b_with_8gb_of_vram/) if you are on Windows).
* Split large models across your GPU(s), CPU, and disk.
* CPU mode.
* [FlexGen offload](https://github.com/oobabooga/text-generation-webui/wiki/FlexGen).
* [DeepSpeed ZeRO-3 offload](https://github.com/oobabooga/text-generation-webui/wiki/DeepSpeed).
* Get responses via API, [with](https://github.com/oobabooga/text-generation-webui/blob/main/api-example-streaming.py) or [without](https://github.com/oobabooga/text-generation-webui/blob/main/api-example.py) streaming.
* [Supports the LLaMA model, including 4-bit mode](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model).
* [Supports the RWKV model](https://github.com/oobabooga/text-generation-webui/wiki/RWKV-model).
* Supports softprompts.
* [Supports extensions](https://github.com/oobabooga/text-generation-webui/wiki/Extensions).
* [Works on Google Colab](https://github.com/oobabooga/text-generation-webui/wiki/Running-on-Colab).

## Installation option 1: conda

Open a terminal and copy and paste these commands one at a time ([install conda](https://docs.conda.io/en/latest/miniconda.html) first if you don't have it already):

```
conda create -n textgen
conda activate textgen
conda install torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

The third line assumes that you have an NVIDIA GPU. 

* If you have an AMD GPU, replace the third command with this one:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
```
  	  
* If you are running it in CPU mode, replace the third command with this one:

```
conda install pytorch torchvision torchaudio git -c pytorch
```

> **Note**
> 1. If you are on Windows, it may be easier to run the commands above in a WSL environment. The performance may also be better.
> 2. For a more detailed, user-contributed guide, see: [Installation instructions for human beings](https://github.com/oobabooga/text-generation-webui/wiki/Installation-instructions-for-human-beings).

## Installation option 2: one-click installers

[oobabooga-windows.zip](https://github.com/oobabooga/one-click-installers/archive/refs/heads/oobabooga-windows.zip)

[oobabooga-linux.zip](https://github.com/oobabooga/one-click-installers/archive/refs/heads/oobabooga-linux.zip)

Just download the zip above, extract it, and double click on "install". The web UI and all its dependencies will be installed in the same folder.

* To download a model, double click on "download-model"
* To start the web UI, double click on "start-webui" 

## Downloading models

Models should be placed under `models/model-name`. For instance, `models/gpt-j-6B` for [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main).

#### Hugging Face

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some noteworthy examples:

* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)
* [GPT-Neo](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads&search=eleutherai+%2F+gpt-neo)
* [Pythia](https://huggingface.co/models?search=eleutherai/pythia)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [\*-Erebus](https://huggingface.co/models?search=erebus) (NSFW)
* [Pygmalion](https://huggingface.co/models?search=pygmalion) (NSFW)

You can automatically download a model from HF using the script `download-model.py`:

    python download-model.py organization/model

For instance:

    python download-model.py facebook/opt-1.3b

If you want to download a model manually, note that all you need are the json, txt, and pytorch\*.bin (or model*.safetensors) files. The remaining files are not necessary.

#### GPT-4chan

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
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`



Optionally, you can use the following command-line flags:

| Flag        | Description |
|-------------|-------------|
| `-h`, `--help`  | show this help message and exit |
| `--model MODEL`    | Name of the model to load by default. |
| `--notebook`  | Launch the web UI in notebook mode, where the output is written to the same text box as the input. |
| `--chat`      | Launch the web UI in chat mode.|
| `--cai-chat`  | Launch the web UI in chat mode with a style similar to Character.AI's. If the file `img_bot.png` or `img_bot.jpg` exists in the same folder as server.py, this image will be used as the bot's profile picture. Similarly, `img_me.png` or `img_me.jpg` will be used as your profile picture. |
| `--cpu`       | Use the CPU to generate text.|
| `--load-in-8bit`  | Load the model with 8-bit precision.|
| `--load-in-4bit`  | DEPRECATED: use `--gptq-bits 4` instead. |
| `--gptq-bits GPTQ_BITS`  |  Load a pre-quantized model with specified precision. 2, 3, 4 and 8 (bit) are supported. Currently only works with LLaMA and OPT. |
| `--gptq-model-type MODEL_TYPE`  |  Model type of pre-quantized model. Currently only LLaMa and OPT are supported. |
| `--bf16`  | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--auto-devices` | Automatically split the model across the available GPU(s) and CPU.|
| `--disk` | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR` | Directory to save the disk cache to. Defaults to `cache/`. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` |  Maxmimum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. |
| `--cpu-memory CPU_MEMORY`    | Maximum CPU memory in GiB to allocate for offloaded weights. Must be an integer number. Defaults to 99.|
| `--flexgen`                   |         Enable the use of FlexGen offloading. |
|  `--percent PERCENT [PERCENT ...]`    |  FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0). |
|  `--compress-weight`                  |  FlexGen: Whether to compress weight (default: False).|
|  `--pin-weight [PIN_WEIGHT]`          |       FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%). |
| `--deepspeed`    | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR`    | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK`    | DeepSpeed: Optional argument for distributed setups. |
|  `--rwkv-strategy RWKV_STRATEGY`         |    RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8". |
|  `--rwkv-cuda-on`                        |   RWKV: Compile the CUDA kernel for better performance. |
| `--no-stream`   | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE` | Load the default interface settings from this json file. See `settings-template.json` for an example. If you create a file called `settings.json`, this file will be loaded by default without the need to use the `--settings` flag.|
|  `--extensions EXTENSIONS [EXTENSIONS ...]` |  The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--listen`   | Make the web UI reachable from your local network.|
|  `--listen-port LISTEN_PORT` | The listening port that the server will use. |
| `--share`   | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch` | Open the web UI in the default browser upon launch. |
| `--verbose`   | Print the prompts to the terminal. |

Out of memory errors? [Check this guide](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

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

- Gradio dropdown menu refresh button: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Verbose preset: Anonymous 4chan user.
- NovelAI and KoboldAI presets: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Pygmalion preset, code for early stopping in chat mode, code for some of the sliders, --chat mode colors: https://github.com/PygmalionAI/gradio-ui/
