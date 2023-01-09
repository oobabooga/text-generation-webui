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
* Load 13b/20b models in 8-bit mode.
* Load parameter presets from text files.
* CPU mode.

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

#### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

You also need to put GPT-J-6B's config.json file in the same folder: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json)

## Converting to pytorch

The script `convert-to-torch.py` allows you to convert models to .pt format, which is about 10x faster to load:

    python convert-to-torch.py models/model-name

The output model will be saved to `torch-dumps/model-name.pt`. When you load a new model, the webui first looks for this .pt file; if it is not found, it loads the model as usual from `models/model-name`. 

## Starting the webui

    conda activate textgen
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

`--model model-name`: Load this model by default.

`--notebook`: Launch the webui in notebook mode, where the output is written to the same text box as the input.

`--chat`: Launch the webui in chat mode.

`--cpu`: Use the CPU to generate text instead of the GPU.

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

## System requirements

These are the VRAM and RAM requirements to run some model examples.

#### GPU mode (default)

| model                  |   VRAM (GPU) |     RAM |
|:-----------------------|-------------:|--------:|
| OPT-350M-Erebus        |         0.62 | 1939.3  |
| arxiv_ai_gpt2          |         1.48 | 6350.68 |
| blenderbot-1B-distill  |         2.38 | 2705.9  |
| opt-1.3b               |         2.45 | 2868.12 |
| gpt-neo-1.3b           |         2.54 | 4047.04 |
| gpt4chan_model_float16 |        11.38 | 1909.79 |
| gpt-j-6b-float16       |        11.38 | 2847.75 |
| gpt-j-6B               |        11.38 | 3959.55 |
| galactica-6.7b         |        12.4  | 1933.19 |
| opt-6.7b               |        12.4  | 1944.21 |
| bloomz-7b1-p3          |        13.17 | 1845.58 |

#### GPU mode with 8-bit precision

Allows you to load models that would not normally fit into your GPU. Enabled by default for 13b and 20b models in this webui.

| model          |   VRAM (GPU) |     RAM |
|:---------------|-------------:|--------:|
| OPT-13B-Erebus |        12.23 |  749.08 |
| opt-13b        |        12.23 | 1258.95 |
| gpt-neox-20b   |        19.91 | 2104.04 |

#### CPU mode

A lot slower, but does not require a GPU.

| model                  |      RAM |
|:-----------------------|---------:|
| OPT-350M-Erebus        |  2622.17 |
| arxiv_ai_gpt2          |  3764.81 |
| gpt-neo-1.3b           |  5937.81 |
| opt-1.3b               |  7346.08 |
| blenderbot-1B-distill  |  7565.36 |
| bloomz-7b1-p3          | 23613.9  |
| gpt-j-6B               | 23975.5  |
| gpt4chan_model         | 23999.5  |
| gpt-j-6b-float16       | 24999.1  |
| galactica-6.7b         | 26248    |
| opt-6.7b               | 27334.2  |

## Contributing

Pull requests, suggestions and issue reports are welcome.

## Other projects

Make sure to also check out the great work by [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client). I have borrowed some of the presets listed on their [wiki](https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets) after performing a k-means clustering analysis to select the most relevant subsample.
