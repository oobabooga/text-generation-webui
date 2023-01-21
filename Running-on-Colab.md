#### Notebook

You can use [this notebook](https://colab.research.google.com/github/81300/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb), kindly provided by a very clever Anonymous 4chan poster.

Just execute every cell and in about 10 minutes a private gradio URL will appear.

[See here a discussion about this](https://github.com/oobabooga/text-generation-webui/issues/14).

#### Basic commands

For debugging purposes, you can use these commands, replacing `PygmalionAI/pygmalion-6b` with whatever model you want to load:

    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !python download-model.py PygmalionAI/pygmalion-6b
    !pip install -r requirements.txt
    !python server.py --cai-chat --share --load-in-8bit

