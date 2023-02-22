>FlexGen is a high-throughput generation engine for running large language models with limited GPU memory (e.g., a 16GB T4 GPU or a 24GB RTX3090 gaming card!).

https://github.com/FMInference/FlexGen

## Installation

To use FlexGen with this web UI, first install it with these commands:

```
conda activate textgen
git clone https://github.com/FMInference/FlexGen
python setup.py build
python setup.py install
```

## Converting a model

FlexGen only works with the OPT model, and it needs to be converted to numpy format before starting the web UI:

```
python convert-to-flexgen.py models/opt-1.3b/
```

The output will be saved to `models/opt-1.3b-np/`.

## Usage

The basic command is the following one:

```
python server.py --model opt-1.3b  --flexgen
```

For large models, the CPU memory usage may be too high and your computer may freeze. If that happens, you can try this:

```
python server.py --model opt-1.3b  --flexgen --compress-weight
```

With this second command, I was seemingly able to run both OPT-6.7b and OPT-13B with **2GB VRAM**, and the speed was good. I have to double check this later, because it's a miracle.

You can also manually set the offload strategy with

```
python server.py --model opt-1.3b  --flexgen --percent 0 0 100 0 100 0
```

Those six numbers are:

```
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")

```


## Limitations

* This only works with OPT models.
* Only two generation parameters are used: `temperature` and `do_sample`.