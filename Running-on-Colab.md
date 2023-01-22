#### Pygmalion chat notebook

[This Colab notebook](https://colab.research.google.com/github/81300/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb) can be used to chat with the `pygmalion-6b` conversational model (NSFW). It was kindly provided by [@81300](https://github.com/81300).

Just execute every cell and a private gradio URL will appear.

[See here a discussion about this](https://github.com/oobabooga/text-generation-webui/issues/14).

#### Basic commands

    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !python download-model.py PygmalionAI/pygmalion-1.3b
    !pip install -r requirements.txt
    !python server.py --cai-chat --share 