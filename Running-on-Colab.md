Use [this notebook](https://colab.research.google.com/github/81300/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb), kindly provided by a very clever Anonymous 4chan poster.

#### Old method (doesn't work with 6b models)

    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !python download-model.py PygmalionAI/pygmalion-1.3b
    !pip install -r requirements.txt
    !python server.py --cai-chat --share 

Replace `PygmalionAI/pygmalion-1.3b` with whatever model you want to use.