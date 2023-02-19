An alternative way of reducing the GPU memory usage of models is to use `DeepSpeed ZeRO-3` optimization.

With this, I have been able to load a 6b model (pygmalion-6b) with less than 6GB of VRAM. The speed of text generation is very decent and much better than what would be accomplished with `--auto-devices --gpu-memory 6`.

As far as I know, DeepSpeed is only available for Linux at the moment.

### How to use it

1. Install deepspeed: 

```
pip install deepspeed
```

2. Start the web UI replacing `python` with `deepspeed --num_gpus=1` and adding the `--deepspeed` flag. Example:

```
deepspeed --num_gpus=1 server.py --deepspeed --cai-chat --model pygmalion-6b
```

### Learn more

For more information, check out [this comment](https://github.com/oobabooga/text-generation-webui/issues/40#issuecomment-1412038622) by 81300, who came up with the DeepSpeed support in this web UI.