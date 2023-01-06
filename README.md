# text-generation-webui

A gradio webui for running large language models locally. Supports gpt-j-6B, gpt-neox-20b, opt, galactica, and many others. 

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

![webui screenshot](https://github.com/oobabooga/text-generation-webui/raw/main/webui.png)

## Installation

Create a conda environment:

    conda create -n textgen
    conda activate textgen

Install the appropriate pytorch for your GPU. For NVIDIA GPUs, this should work:

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Install the requirements:

    pip install -r requirements.txt

## Downloading models

Models should be placed under `models/model-name`.

#### Hugging Face

Hugging Face is the main place to download models. For instance, [here](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main) you can find the files for the model gpt-j-6B.

The files that you need to download and put under `models/gpt-j-6B` are the json, txt, and pytorch*.bin files. The remaining files are not necessary.

#### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

## Converting to pytorch

This webui allows you to switch between different models on the fly, so it must be fast to load the models from disk.

One way to make this process about 10x faster is to convert the models to pytorch format using the script `convert-to-torch.py`. Create a folder called `torch-dumps` and then make the conversion with:

    python convert-to-torch.py models/model-name/

The output model will be saved to `torch-dumps/model-name.pt`. This is the default way to load all models except for `gpt-neox-20b`, `opt-13b`, `OPT-13B-Erebus`, `gpt-j-6B`, and `flan-t5`. I don't remember why these models are exceptions.

If I get enough ⭐s on this repository, I will make the process of loading models saner and more customizable.

## Starting the webui

    conda activate textgen
    python server.py

Then browse to `http://localhost:7860/?__theme=dark`

## Contributing

Pull requests are welcome.
