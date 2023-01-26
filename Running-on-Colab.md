#### Pygmalion chat notebook

* [Original notebook](https://colab.research.google.com/github/81300/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb): can be used to chat with the `pygmalion-6b` conversational model (NSFW). It was kindly provided by [@81300](https://github.com/81300), and it supports persistent storage of characters and models on Google Drive.
* [Simplified notebook](https://colab.research.google.com/github/oobabooga/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb) (**use this one for now**): this is a variation of the notebook above for casual users. Just execute all cells and a gradio URL will automatically appear in 12 minutes.

See also [this discussion](https://github.com/oobabooga/text-generation-webui/issues/14).

#### Basic commands

    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !pip install -r requirements.txt
    !python download-model.py PygmalionAI/pygmalion-1.3b
    !python server.py --cai-chat --share 