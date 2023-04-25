>FlexGen is a high-throughput generation engine for running large language models with limited GPU memory (e.g., a 16GB T4 GPU or a 24GB RTX3090 gaming card!).

https://github.com/FMInference/FlexGen

## Installation

No additional installation steps are necessary. FlexGen is in the `requirements.txt` file for this project.

## Converting a model

FlexGen only works with the OPT model, and it needs to be converted to numpy format before starting the web UI:

```
python convert-to-flexgen.py models/opt-1.3b/
```

The output will be saved to `models/opt-1.3b-np/`.

## Usage

The basic command is the following:

```
python server.py --model opt-1.3b  --flexgen
```

For large models, the RAM usage may be too high and your computer may freeze. If that happens, you can try this:

```
python server.py --model opt-1.3b  --flexgen --compress-weight
```

With this second command, I was able to run both OPT-6.7b and OPT-13B with **2GB VRAM**, and the speed was good in both cases.

You can also manually set the offload strategy with

```
python server.py --model opt-1.3b  --flexgen --percent 0 100 100 0 100 0
```

where the six numbers after `--percent` are:

```
the percentage of weight on GPU
the percentage of weight on CPU
the percentage of attention cache on GPU
the percentage of attention cache on CPU
the percentage of activations on GPU
the percentage of activations on CPU
```

You should typically only change the first two numbers. If their sum is less than 100, the remaining layers will be offloaded to the disk, by default into the `text-generation-webui/cache` folder.

## Performance

In my experiments with OPT-30B using a RTX 3090 on Linux, I have obtained these results:

* `--flexgen --compress-weight --percent 0 100 100 0 100 0`: 0.99 seconds per token.
* `--flexgen --compress-weight --percent 100 0 100 0 100 0`: 0.765 seconds per token.

## Limitations

* Only works with the OPT models.
* Only two generation parameters are available: `temperature` and `do_sample`.