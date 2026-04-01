This is where you load models, apply LoRAs to a loaded model, and download new models.

## Model loaders

### llama.cpp

Loads: GGUF models. Note: GGML models have been deprecated and do not work anymore.

Example: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF

* **gpu_layers**: The number of layers to allocate to the GPU. If set to 0, only the CPU will be used. If you want to offload all layers, you can simply set this to the maximum value.
* **ctx_size**: Context length of the model. In llama.cpp, the cache is preallocated, so the higher this value, the higher the VRAM. It is automatically set to the maximum sequence length for the model based on the metadata inside the GGUF file, but you may need to lower this value to fit the model into your GPU. Set to 0 for automatic context size based on available memory. After loading the model, the "Truncate the prompt up to this length" parameter under "Parameters" > "Generation" is automatically set to your chosen "ctx_size" so that you don't have to set the same thing twice.
* **cache_type**: KV cache quantization type. Valid options: `fp16`, `q8_0`, `q4_0`. Lower quantization saves VRAM at the cost of some quality.
* **tensor_split**: For multi-gpu only. Sets the amount of memory to allocate per GPU as proportions. Not to be confused with other loaders where this is set in GB; here you can set something like `30,70` for 30%/70%.
* **batch_size**: Maximum number of prompt tokens to batch together when calling llama_eval.
* **ubatch_size**: Physical maximum batch size for prompt processing.
* **threads**: Number of threads. Recommended value: your number of physical cores.
* **threads_batch**: Number of threads for batch processing. Recommended value: your total number of cores (physical + virtual).
* **cpu_moe**: Force MoE expert layers to run on the CPU, keeping the rest on the GPU.
* **extra_flags**: Extra flags to pass to llama-server. Format: `flag1=value1,flag2,flag3=value3`. Example: `override-tensor=exps=CPU`.
* **mmproj**: Path to the mmproj file for multimodal (vision) models. This enables image understanding capabilities.
* **streaming_llm**: Experimental feature to avoid re-evaluating the entire prompt when part of it is removed, for instance, when you hit the context length for the model in chat mode and an old message is removed.
* **cpu**: Force a version of llama.cpp compiled without GPU acceleration to be used. Can usually be ignored. Only set this if you want to use CPU only and llama.cpp doesn't work otherwise.
* **row_split**: Split the model by rows across GPUs. This may improve multi-gpu performance.
* **no_kv_offload**: Do not offload the KV cache to the GPU. This saves VRAM but reduces performance.
* **no_mmap**: Loads the model into memory at once, possibly preventing I/O operations later on at the cost of a longer load time.
* **mlock**: Force the system to keep the model in RAM rather than swapping or compressing.
* **numa**: May improve performance on certain multi-cpu systems.

### Transformers

Loads: full precision (16-bit or 32-bit) models, as well as bitsandbytes-quantized models. The repository usually has a clean name without GGUF or EXL3 in its name, and the model files are named `model.safetensors` or split into parts like `model-00001-of-00004.safetensors`.

Example: [https://huggingface.co/lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5).

Full precision models use a ton of VRAM, so you will usually want to select the "load_in_4bit" and "use_double_quant" options to load the model in 4-bit precision using bitsandbytes.

Options:

* **gpu_split**: When using multiple GPUs, sets the amount of VRAM in GB to allocate per GPU. Example: `20,7,7`.
* **cpu_memory**: Maximum CPU memory in GiB to use for CPU offloading via the accelerate library. Whatever doesn't fit in the GPU or CPU will go to a disk cache if the "disk" checkbox is enabled.
* **compute_dtype**: Used when "load_in_4bit" is checked. I recommend leaving the default value.
* **quant_type**: Used when "load_in_4bit" is checked. I recommend leaving the default value.
* **attn_implementation**: Choose the attention implementation. Valid options: `sdpa`, `eager`, `flash_attention_2`. The default (`sdpa`) works well in most cases; `flash_attention_2` may be useful for training.
* **cpu**: Loads the model in CPU mode using Pytorch. The model will be loaded in 32-bit precision, so a lot of RAM will be used. CPU inference with transformers is older than llama.cpp and it works, but it's a lot slower. Note: this parameter has a different interpretation in the llama.cpp loader (see above).
* **load_in_8bit**: Load the model in 8-bit precision using bitsandbytes. The 8-bit kernel in that library has been optimized for training and not inference, so load_in_8bit is slower than load_in_4bit (but more accurate).
* **bf16**: Use bfloat16 precision instead of float16 (the default). Only applies when quantization is not used.
* **disk**: Enable disk offloading for layers that don't fit into the GPU and CPU combined.
* **load_in_4bit**: Load the model in 4-bit precision using bitsandbytes.
* **use_double_quant**: Use double quantization with 4-bit loading for reduced memory usage.
* **trust-remote-code**: Some models use custom Python code to load the model or the tokenizer. For such models, this option needs to be set. It doesn't download any remote content: all it does is execute the .py files that get downloaded with the model. Those files can potentially include malicious code; I have never seen it happen, but it is in principle possible.
* **no_use_fast**: Do not use the "fast" version of the tokenizer. Can usually be ignored; only check this if you can't load the tokenizer for your model otherwise.

### ExLlamav3_HF

Loads: EXL3 models. These models usually have "EXL3" or "exl3" in the model name.

Uses the ExLlamaV3 backend with Transformers samplers.

* **ctx_size**: Context length of the model. The cache is preallocated, so the higher this value, the higher the VRAM. It is automatically set to the maximum sequence length for the model based on its metadata, but you may need to lower this value to fit the model into your GPU. After loading the model, the "Truncate the prompt up to this length" parameter under "Parameters" > "Generation" is automatically set to your chosen "ctx_size" so that you don't have to set the same thing twice.
* **cache_type**: KV cache quantization type. Valid options: `fp16`, `q2` to `q8`. You can also specify key and value bits separately, e.g. `q4_q8`. Lower quantization saves VRAM at the cost of some quality.
* **gpu_split**: Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: `20,7,7`.
* **cfg_cache**: Creates a second cache to hold the CFG negative prompts. You need to set this if and only if you intend to use CFG in the "Parameters" > "Generation" tab. Checking this parameter doubles the cache VRAM usage.
* **no_use_fast**: Do not use the "fast" version of the tokenizer.
* **enable_tp**: Enable Tensor Parallelism (TP) to split the model across GPUs.
* **tp_backend**: The backend for tensor parallelism. Valid options: `native`, `nccl`. Default: `native`.

### ExLlamav3

The same as ExLlamav3_HF but using the internal samplers of ExLlamaV3 instead of the ones in the Transformers library. Supports speculative decoding with a draft model. Also supports multimodal (vision) models natively.

* **ctx_size**: Same as ExLlamav3_HF.
* **cache_type**: Same as ExLlamav3_HF.
* **gpu_split**: Same as ExLlamav3_HF.
* **enable_tp**: Enable Tensor Parallelism (TP) to split the model across GPUs.
* **tp_backend**: The backend for tensor parallelism. Valid options: `native`, `nccl`. Default: `native`.

### TensorRT-LLM

Loads: TensorRT-LLM engine models. These are highly optimized models compiled specifically for NVIDIA GPUs.

* **ctx_size**: Context length of the model.
* **cpp_runner**: Use the ModelRunnerCpp runner, which is faster than the default ModelRunner but doesn't support streaming yet.

## Model dropdown

Here you can select a model to be loaded, refresh the list of available models, load/unload/reload the selected model, and save the settings for the model. The "settings" are the values in the input fields (checkboxes, sliders, dropdowns) below this dropdown.

After saving, those settings will get restored whenever you select that model again in the dropdown menu.

If the **Autoload the model** checkbox is selected, the model will be loaded as soon as it is selected in this menu. Otherwise, you will have to click on the "Load" button.

## LoRA dropdown

Used to apply LoRAs to the model. Note that LoRA support is not implemented for all loaders. Check the [What Works](https://github.com/oobabooga/text-generation-webui/wiki/What-Works) page for details.

## Download model or LoRA

Here you can download a model or LoRA directly from the https://huggingface.co/ website.

* Models will be saved to `user_data/models`.
* LoRAs will be saved to `user_data/loras`.

In the input field, you can enter either the Hugging Face username/model path (like `facebook/galactica-125m`) or the full model URL (like `https://huggingface.co/facebook/galactica-125m`). To specify a branch, add it at the end after a ":" character like this: `facebook/galactica-125m:main`.

To download a single file, as necessary for models in GGUF format, you can click on "Get file list" after entering the model path in the input field, and then copy and paste the desired file name in the "File name" field before clicking on "Download".
