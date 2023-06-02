An alternative way of reducing the GPU memory usage of models is to use the `DeepSpeed ZeRO-3` optimization.

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

### Learn more

For more information, check out [this comment](https://github.com/oobabooga/text-generation-webui/issues/40#issuecomment-1412038622) by 81300, who came up with the DeepSpeed support in this web UI.