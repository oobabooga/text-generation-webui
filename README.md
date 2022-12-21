# text-generation-webui
A gradio webui for running large language models locally. Supports gpt-j-6B, gpt-neox-20b, opt, galactica, and many others.

![webui screenshot](https://github.com/oobabooga/text-generation-webui/raw/main/webui.png)

## Installation

    conda env create -f environment.yml

## Downloading models

Models should be placed under `models/model-name`.

### Hugging Face

Hugging Face is the main place to download models. For instance, [here](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main) you can find the files for the model gpt-j-6B.

The files that you need to download and put under `models/gpt-j-6B` are the json, txt, and pytorch*.bin files. The remaining files are not necessary.

### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

## Starting the webui

    conda activate textgen
    python server.py
    


Then browse to `http://localhost:7860/?__theme=dark`
