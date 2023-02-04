#### Pygmalion chat notebook

This is the official Pygmalion notebook for this web UI: https://colab.research.google.com/github/oobabooga/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb

Just execute all cells and a gradio URL will automatically appear at the bottom in 5 minutes.

It is based on the [original notebook](https://colab.research.google.com/github/81300/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb) by [@81300](https://github.com/81300/AI-Notebooks).

See also [this discussion](https://github.com/oobabooga/text-generation-webui/issues/14).

#### Basic commands

    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !pip install -r requirements.txt
    !python download-model.py PygmalionAI/pygmalion-1.3b
    !python server.py --cai-chat --share 