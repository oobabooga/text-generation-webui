    !git clone https://github.com/oobabooga/text-generation-webui
    %cd text-generation-webui
    !python download-model.py PygmalionAI/pygmalion-6b
    !pip install -r requirements.txt
    !python server.py --cai-chat --share --auto-devices

Replace `PygmalionAI/pygmalion-6b` with whatever model you want to use.