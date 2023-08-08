# AutoGPTQ

## Background

GPTQ is a clever quantization algorithm that lightly reoptimizes the weights during quantization so that the accuracy loss is compensated relative to a round-to-nearest quantization. See the paper for more details: https://arxiv.org/abs/2210.17323

The first adaptation of GPTQ for the LLaMA model was GPTQ-for-LLaMa by [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa): https://github.com/qwopqwop200/GPTQ-for-LLaMa

This repository evolved into AutoGPTQ, which is now the recommended way to create new GPTQ quantizations: https://github.com/PanQiWei/AutoGPTQ

### Installation

No additional steps are necessary as AutoGPTQ is already in the `requirements.txt` for the webui. If you need to install it manually, these are the commands:

```
conda activate textgen
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .
```

The last command requires `nvcc` to be installed. 

**Installing nvcc**

```
conda activate textgen
conda install -c conda-forge cudatoolkit-dev
```

The command above takes some 10 minutes to run and shows no progress bar or updates along the way.

You are also going to need to have a C++ compiler installed. On Linux, `sudo apt install build-essential` or equivalent is enough.

If you're using an older version of CUDA toolkit (e.g. 11.7) but the latest version of `gcc` and `g++` (12.0+), you should downgrade with: `conda install -c conda-forge gxx==11.3.0`. Kernel compilation will fail otherwise.


### CPU offloading

In order to do CPU offloading or multi-gpu inference with AutoGPTQ, use the `--gpu-memory` flag. 

For CPU offloading:

```
python server.py --loader autogptq --gpu-memory 3000MiB --model model_name
```

For multi-GPU inference:

```
python server.py --loader autogptq --gpu-memory 3000MiB 6000MiB --model model_name
```
