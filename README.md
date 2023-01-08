# text-generation-webui

A gradio webui for running large language models locally. Supports gpt-j-6B, gpt-neox-20b, opt, galactica, and many others. 

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

![webui screenshot](https://github.com/oobabooga/text-generation-webui/raw/main/webui.png)

## Features

* Switch between different models using a dropdown menu.
* Generate nice HTML output for gpt4chan.
* Generate Markdown output for [GALACTICA](https://github.com/paperswithcode/galai), including LaTeX support.
* Notebook mode that resembles OpenAI's playground.
* Chat mode for conversation and role playing.
* Load 13b/20b models in 8-bit mode.
* Load parameter presets from text files.

## Installation

Create a conda environment:

    conda create -n textgen
    conda activate textgen

Install the appropriate pytorch for your GPU. For NVIDIA GPUs, this should work:

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Install the requirements:

    pip install -r requirements.txt

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

#### gpt4chan

[gpt4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

## Converting to pytorch

The script `convert-to-torch.py` allows you to convert models to .pt format, which is about 10x faster to load:

    python convert-to-torch.py models/model-name/

The output model will be saved to `torch-dumps/model-name.pt`. When you load a new model, the webui first looks for this .pt file; if it is not found, it loads the model as usual from `models/model-name/`. 

## Starting the webui

    conda activate textgen
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

`--model model-name`: Load this model by default.

`--notebook`: Launch the webui in notebook mode, where the output is written to the same text box as the input.

`--chat`: Launch the webui in chat mode.

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

## Contributing

Pull requests are welcome.
