## Audio notification

If your computer takes a long time to generate each response for the model that you are using, you can enable an audio notification for when the response is completed. This feature was kindly contributed by HappyWorldGames in [#1277](https://github.com/oobabooga/text-generation-webui/pull/1277).

### Installation

Simply place a file called "notification.mp3" in the same folder as `server.py`. Here you can find some examples:

* https://pixabay.com/sound-effects/search/ding/?duration=0-30
* https://pixabay.com/sound-effects/search/notification/?duration=0-30

Source: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/1126

This file will be automatically detected the next time you start the web UI.

## DeepSpeed

`DeepSpeed ZeRO-3` is an alternative offloading strategy for full-precision (16-bit) transformers models.

With this, I have been able to load a 6b model (GPT-J 6B) with less than 6GB of VRAM. The speed of text generation is very decent and much better than what would be accomplished with `--auto-devices --gpu-memory 6`.

As far as I know, DeepSpeed is only available for Linux at the moment.

### How to use it

1. Install DeepSpeed: 

```
conda install -c conda-forge mpi4py mpich
pip install -U deepspeed
```

2. Start the web UI replacing `python` with `deepspeed --num_gpus=1` and adding the `--deepspeed` flag. Example:

```
deepspeed --num_gpus=1 server.py --deepspeed --chat --model gpt-j-6B
```

## Miscellaneous info

### You can train LoRAs in CPU mode

Load the web UI with

```
python server.py --cpu
```

and start training the LoRA from the training tab as usual.

### You can check the sha256sum of downloaded models with the download script

```
python download-model.py facebook/galactica-125m --check
```

### The download script continues interrupted downloads by default

It doesn't start over.

