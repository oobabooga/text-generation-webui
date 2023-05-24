# Using llama.cpp in the web UI

## Setting up the models

#### Pre-converted

Place the model in the `models` folder, making sure that its name contains `ggml` somewhere and ends in `.bin`.

#### Convert LLaMA yourself

Follow the instructions in the llama.cpp README to generate the `ggml-model.bin` file: https://github.com/ggerganov/llama.cpp#usage

## GPU offloading

Enabled with the `--n-gpu-layers` parameter. If you have enough VRAM, use a high number like `--n-gpu-layers 200000` to offload all layers to the GPU.

Note that you need to manually install `llama-cpp-python` with GPU support. To do that:

#### Linux

```
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

#### Windows

```
pip uninstall -y llama-cpp-python
set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```

Here you can find the different compilation options for OpenBLAS / cuBLAS / CLBlast: https://pypi.org/project/llama-cpp-python/

## Performance

This was the performance of llama-7b int4 on my i5-12400F (cpu only):

> Output generated in 33.07 seconds (6.05 tokens/s, 200 tokens, context 17)

You can change the number of threads with `--threads N`.
