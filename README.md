# text-generation-webui
A gradio webui for running large language models locally. Supports gpt-j-6B, gpt-neox-20b, opt, galactica, and many others.

## Installation

    conda env create -f environment.yml

## Starting the webui

    conda activate textgen
    python server.py

Then browse to `http://localhost:7860/?__theme=dark`
