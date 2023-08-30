# llama.cpp

llama.cpp is the best backend in two important scenarios:

1) You don't have a GPU.
2) You want to run a model that doesn't fit into your GPU.

## Setting up the models

#### Pre-converted

Download the GGUF or GGML models directly into your `text-generation-webui/models` folder. It will be a single file.

* For GGUF models, make sure its name contains `.gguf`.
* For GGML models, make sure its name contains `ggml` and ends in `.bin`.

`q4_K_M` quantization is recommended.

#### Convert Llama yourself

Follow the instructions in the llama.cpp README to generate a ggml: https://github.com/ggerganov/llama.cpp#prepare-data--run

## GPU acceleration

Enabled with the `--n-gpu-layers` parameter. 

* If you have enough VRAM, use a high number like `--n-gpu-layers 1000` to offload all layers to the GPU. 
* Otherwise, start with a low number like `--n-gpu-layers 10` and then gradually increase it until you run out of memory.

This feature works out of the box for NVIDIA GPUs on Linux (amd64) or Windows. For other GPUs, you need to uninstall `llama-cpp-python` with

```
pip uninstall -y llama-cpp-python
```

and then recompile it using the commands here: https://pypi.org/project/llama-cpp-python/

#### macOS

For macOS, these are the commands:

```
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```
