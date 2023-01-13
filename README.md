# text-generation-webui

A gradio webui for running large language models locally. Supports gpt-j-6B, gpt-neox-20b, opt, galactica, and many others. 

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

![webui screenshot](https://github.com/oobabooga/text-generation-webui/raw/main/webui.png)

## Features

* Switch between different models using a dropdown menu.
* Generate nice HTML output for GPT-4chan.
* Generate Markdown output for [GALACTICA](https://github.com/paperswithcode/galai), including LaTeX support.
* Notebook mode that resembles OpenAI's playground.
* Chat mode for conversation and role playing.
* Load parameter presets from text files.
* Load large models in 8-bit mode.
* Split large models across your GPU(s) and CPU.
* CPU mode.
* Get responses via API.

## Installation

1. You need to have the conda environment manager installed into your system. If you don't have it already, [get miniconda here](https://docs.conda.io/en/latest/miniconda.html).

2. Open a terminal window and create a conda environment:

```
conda create -n textgen
conda activate textgen
```

3. Install the appropriate pytorch. For NVIDIA GPUs, this should work:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

For AMD GPUs, you need the ROCm version of pytorch. If you don't have any GPU and want to run in CPU mode, you just need the stock pytorch and this should work:

```
conda install pytorch torchvision torchaudio -c pytorch
```

4. Clone or download this repository, and then `cd` into its directory from your terminal window.

5. Install the required Python libraries:

```
pip install -r requirements.txt
```

After these steps, you should be able to start the webui, but first you need to download some model to load.

## Downloading models

Models should be placed under `models/model-name`. For instance, `models/gpt-j-6B` for [gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main).

#### Hugging Face

Hugging Face is the main place to download models. These are some of my favorite:

* [gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)
* [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [\*-Erebus](https://huggingface.co/models?search=erebus)

The files that you need to download are the json, txt, and pytorch\*.bin files. The remaining files are not necessary.

For your convenience, you can automatically download a model from HF using the script `download-model.py`. Its usage is very simple:

    python download-model.py organization/model

For instance:

    python download-model.py facebook/opt-1.3b

#### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

The 32-bit version is only relevant if you intend to run the model in CPU mode. Otherwise, I recommend using the 16-bit version.

After downloading the model, follow these steps:

1. Place the files under `models/gpt4chan_model_float16` or `models/gpt4chan_model`.
2. Place GPT-J-6B's config.json file in that same folder: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. Download GPT-J-6B under `models/gpt-j-6B`:

```
python download-model.py EleutherAI/gpt-j-6B
```

You don't really need all of GPT-J's files, just the tokenizer files, but you might as well download the whole thing. Those files will be automatically detected when you attempt to load GPT-4chan.

#### Converting to pytorch (optional)

The script `convert-to-torch.py` allows you to convert models to .pt format, which is about 10x faster to load to the GPU:

    python convert-to-torch.py models/model-name

The output model will be saved to `torch-dumps/model-name.pt`. When you load a new model, the webui first looks for this .pt file; if it is not found, it loads the model as usual from `models/model-name`. 

## Starting the webui

    conda activate textgen
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

```
-h, --help      show this help message and exit
--model MODEL   Name of the model to load by default.
--notebook      Launch the webui in notebook mode, where the output is written to the same text
                box as the input.
--chat          Launch the webui in chat mode.
--cpu           Use the CPU to generate text.
--auto-devices  Automatically split the model across the available GPU(s) and CPU.
--load-in-8bit  Load the model with 8-bit precision.
--no-listen     Make the webui unreachable from your local network.
```

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

## System requirements

Check the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) for some examples of VRAM and RAM usage in both GPU and CPU mode.

## Contributing

Pull requests, suggestions and issue reports are welcome.

## Other projects

Make sure to also check out the great work by [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client). I have borrowed some of the presets listed on their [wiki](https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets) after performing a k-means clustering analysis to select the most relevant subsample.
