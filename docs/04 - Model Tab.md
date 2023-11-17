This is where you load models, apply LoRAs to a loaded model, and download new models.

## Model loaders

### Transformers

Loads: full precision (16-bit or 32-bit) models. The repository usually has a clean name without GGUF, EXL2, GPTQ, or AWQ in its name, and the model files are named `pytorch_model.bin` or `model.safetensors`. 

Example: [https://huggingface.co/lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5). 

Full precision models use a ton of VRAM, so you will usually want to select the "load_in_4bit" and "use_double_quant" options to load the model in 4-bit precision using bitsandbytes.

This loader can also load GPTQ models and train LoRAs with them. For that, make sure to check the "auto-devices" and "disable_exllama" options before loading the model.

Options:

* **gpu-memory**: When set to greater than 0, activates CPU offloading using the accelerate library, where part of the layers go to the CPU. The performance is very bad. Note that accelerate doesn't treat this parameter very literally, so if you want the VRAM usage to be at most 10 GiB, you may need to set this parameter to 9 GiB or 8 GiB. It can be used in conjunction with "load_in_8bit" but not with "load-in-4bit" as far as I'm aware.
* **cpu-memory**: Similarly to the parameter above, you can also set a limit on the amount of CPU memory used. Whatever doesn't fit either in the GPU or the CPU will go to a disk cache, so to use this option you should also check the "disk" checkbox.
* **compute_dtype**: Used when "load-in-4bit" is checked. I recommend leaving the default value.
* **quant_type**: Used when "load-in-4bit" is checked. I recommend leaving the default value.
* **alpha_value**: Used to extend the context length of a model with a minor loss in quality. I have measured 1.75 to be optimal for 1.5x context, and 2.5 for 2x context. That is, with alpha = 2.5 you can make a model with 4096 context length go to 8192 context length.
* **rope_freq_base**: Originally another way to write "alpha_value", it ended up becoming a necessary parameter for some models like CodeLlama, which was fine-tuned with this set to 1000000 and hence needs to be loaded with it set to 1000000 as well.
* **compress_pos_emb**: The first and original context-length extension method, discovered by [kaiokendev](https://kaiokendev.github.io/til). When set to 2, the context length is doubled, 3 and it's tripled, etc. It should only be used for models that have been fine-tuned with this parameter set to different than 1. For models that have not been tuned to have greater context length, alpha_value will lead to a smaller accuracy loss.
* **cpu**: Loads the model in CPU mode using Pytorch. The model will be loaded in 32-bit precision, so a lot of RAM will be used. CPU inference with transformers is older than llama.cpp and it works, but it's a lot slower. Note: this parameter has a different interpretation in the llama.cpp loader (see below).
* **load-in-8bit**: Load the model in 8-bit precision using bitsandbytes. The 8-bit kernel in that library has been optimized for training and not inference, so load-in-8bit is slower than load-in-4bit (but more accurate).
* **bf16**: Use bfloat16 precision instead of float16 (the default). Only applies when quantization is not used.
* **auto-devices**: When checked, the backend will try to guess a reasonable value for "gpu-memory" to allow you to load a model with CPU offloading. I recommend just setting "gpu-memory" manually instead. This parameter is also needed for loading GPTQ models, in which case it needs to be checked before loading the model.
* **disk**: Enable disk offloading for layers that don't fit into the GPU and CPU combined.
* **load-in-4bit**: Load the model in 4-bit precision using bitsandbytes.
* **trust-remote-code**: Some models use custom Python code to load the model or the tokenizer. For such models, this option needs to be set. It doesn't download any remote content: all it does is execute the .py files that get downloaded with the model. Those files can potentially include malicious code; I have never seen it happen, but it is in principle possible.
* **no_use_fast**: Do not use the "fast" version of the tokenizer. Can usually be ignored; only check this if you can't load the tokenizer for your model otherwise.
* **use_flash_attention_2**: Set use_flash_attention_2=True while loading the model. Possibly useful for training.
* **disable_exllama**: Only applies when you are loading a GPTQ model through the transformers loader. It needs to be checked if you intend to train LoRAs with the model.

### ExLlama_HF

Loads: GPTQ models. They usually have GPTQ in the model name, or alternatively something like "-4bit-128g" in the name.

Example: https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ

ExLlama_HF is the v1 of ExLlama (https://github.com/turboderp/exllama) connected to the transformers library for sampling, tokenizing, and detokenizing. It is very fast and memory-efficient.

* **gpu-split**: If you have multiple GPUs, the amount of memory to allocate per GPU should be set in this field. Make sure to set a lower value for the first GPU, as that's where the cache is allocated.
* **max_seq_len**: The maximum sequence length for the model. In ExLlama, the cache is preallocated, so the higher this value, the higher the VRAM. It is automatically set to the maximum sequence length for the model based on its metadata, but you may need to lower this value be able to fit the model into your GPU. After loading the model, the "Truncate the prompt up to this length" parameter under "Parameters" > "Generation" is automatically set to your chosen "max_seq_len" so that you don't have to set the same thing twice.
* **cfg-cache**: Creates a second cache to hold the CFG negative prompts. You need to set this if and only if you intend to use CFG in the "Parameters" > "Generation" tab. Checking this parameter doubles the cache VRAM usage.
* **no_flash_attn**: Disables flash attention. Otherwise, it is automatically used as long as the library is installed.
* **cache_8bit**: Create a 8-bit precision cache instead of a 16-bit one. This saves VRAM but increases perplexity (I don't know by how much).

### ExLlamav2_HF

Loads: GPTQ and EXL2 models. EXL2 models usually have "EXL2" in the model name.

Example: https://huggingface.co/turboderp/Llama2-70B-exl2

The parameters are the same as in ExLlama_HF.

### ExLlama

The same as ExLlama_HF but using the internal samplers of ExLlama instead of the ones in the Transformers library.

### ExLlamav2

The same as ExLlamav2_HF but using the internal samplers of ExLlamav2 instead of the ones in the Transformers library.

### AutoGPTQ

Loads: GPTQ models.

* **wbits**: For ancient models without proper metadata, sets the model precision in bits manually. Can usually be ignored.
* **groupsize**: For ancient models without proper metadata, sets the model group size manually. Can usually be ignored.
* **triton**: Only available on Linux. Necessary to use models with both act-order and groupsize simultaneously. Note that ExLlama can load these same models on Windows without triton.
* **no_inject_fused_attention**: Improves performance while increasing the VRAM usage.
* **no_inject_fused_mlp**: Similar to the previous parameter but for Triton only.
* **no_use_cuda_fp16**: On some systems, the performance can be very bad with this unset. Can usually be ignored.
* **desc_act**: For ancient models without proper metadata, sets the model "act-order" parameter manually. Can usually be ignored.

### GPTQ-for-LLaMa

Loads: GPTQ models.

Ancient loader, the first one to implement 4-bit quantization. It works on older GPUs for which ExLlama and AutoGPTQ do not work, and it doesn't work with "act-order", so you should use it with simple 4-bit-128g models.

* **pre_layer**: Used for CPU offloading. The higher the number, the more layers will be sent to the GPU. GPTQ-for-LLaMa CPU offloading was faster than the one implemented in AutoGPTQ the last time I checked.

### llama.cpp

Loads: GGUF models. Note: GGML models have been deprecated and do not work anymore.

Example: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF

* **n-gpu-layers**: The number of layers to allocate to the GPU. If set to 0, only the CPU will be used. If you want to offload all layers, you can simply set this to the maximum value.
* **n_ctx**: Context length of the model. In llama.cpp, the cache is preallocated, so the higher this value, the higher the VRAM. It is automatically set to the maximum sequence length for the model based on the metadata inside the GGUF file, but you may need to lower this value be able to fit the model into your GPU. After loading the model, the "Truncate the prompt up to this length" parameter under "Parameters" > "Generation" is automatically set to your chosen "n_ctx" so that you don't have to set the same thing twice.
* **threads**: Number of threads. Recommended value: your number of physical cores. 
* **threads_batch**: Number of threads for batch processing. Recommended value: your total number of cores (physical + virtual).
* **n_batch**: Batch size for prompt processing. Higher values are supposed to make generation faster, but I have never obtained any benefit from changing this value.
* **no_mul_mat_q**: Disable the mul_mat_q kernel. This kernel usually improves generation speed significantly. This option to disable it is included in case it doesn't work on some system.
* **no-mmap**: Loads the model into memory at once, possibly preventing I/O operations later on at the cost of a longer load time.
* **mlock**: Force the system to keep the model in RAM rather than swapping or compressing (no idea what this means, never used it).
* **numa**: May improve performance on certain multi-cpu systems.
* **cpu**: Force a version of llama.cpp compiled without GPU acceleration to be used. Can usually be ignored. Only set this if you want to use CPU only and llama.cpp doesn't work otherwise. 
* **tensor_split**: For multi-gpu only. Sets the amount of memory to allocate per GPU.
* **Seed**: The seed for the llama.cpp random number generator. Not very useful as it can only be set once (that I'm aware).

### llamacpp_HF

The same as llama.cpp but with transformers samplers, and using the transformers tokenizer instead of the internal llama.cpp tokenizer.

To use it, you need to download a tokenizer. There are two options:

1) Download `oobabooga/llama-tokenizer` under "Download model or LoRA". That's a default Llama tokenizer.
2) Place your .gguf in a subfolder of `models/` along with these 3 files: `tokenizer.model`, `tokenizer_config.json`, and `special_tokens_map.json`. This takes precedence over Option 1.

It has an additional parameter:

* **logits_all**: Needs to be checked if you want to evaluate the perplexity of the llama.cpp model using the "Training" > "Perplexity evaluation" tab. Otherwise, leave it unchecked, as it makes prompt processing slower.

### ctransformers

Loads: GGUF/GGML models.

Similar to llama.cpp but it works for certain GGUF/GGML models not originally supported by llama.cpp like Falcon, StarCoder, StarChat, and GPT-J.

### AutoAWQ

Loads: AWQ models.

Example: https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-AWQ

The parameters are overall similar to AutoGPTQ.

## Model dropdown

Here you can select a model to be loaded, refresh the list of available models (🔄), load/unload/reload the selected model, and save the settings for the model. The "settings" are the values in the input fields (checkboxes, sliders, dropdowns) below this dropdown. 

After saving, those settings will get restored whenever you select that model again in the dropdown menu.

If the **Autoload the model** checkbox is selected, the model will be loaded as soon as it is selected in this menu. Otherwise, you will have to click on the "Load" button.

## LoRA dropdown

Used to apply LoRAs to the model. Note that LoRA support is not implemented for all loaders. Check this [page](https://github.com/oobabooga/text-generation-webui/wiki) for details.

## Download model or LoRA

Here you can download a model or LoRA directly from the https://huggingface.co/ website.

* Models will be saved to `text-generation-webui/models`.
* LoRAs will be saved to `text-generation-webui/loras`.

In the input field, you can enter either the Hugging Face username/model path (like `facebook/galactica-125m`) or the full model URL (like `https://huggingface.co/facebook/galactica-125m`). To specify a branch, add it at the end after a ":" character like this: `facebook/galactica-125m:main`. 

To download a single file, as necessary for models in GGUF format, you can click on "Get file list" after entering the model path in the input field, and then copy and paste the desired file name in the "File name" field before clicking on "Download".
