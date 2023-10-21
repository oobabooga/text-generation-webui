This is where you load models, apply LoRAs to a loaded model, and download new models.

## Model loaders

### Transformers

Loads full precision (16-bit or 32-bit) pytorch models. The repository usually has a clean name without GGUF, EXL2, GPTQ, or AWQ in its name, and the model files are named `pytorch_model.bin` or `model.safetensors`. Example: [https://huggingface.co/lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5). 

Full precision models use a ton of VRAM, so usually you will want to select the "load_in_4bit" and "use_double_quant" options to load the model in 4-bit precision using bitsandbytes.

This loader can also load GPTQ models and train LoRAs with them. For that, make sure to check the "auto-devices" and "disable_exllama" options before loading the model.

Options:

* **gpu-memory**: when set to greater than 0, activates CPU offloading using the accelerate library, where part of the layers go to the CPU. The performance is very bad. Note that accelerate doesn't treat this parameter very literally, so if you want the VRAM usage to be at most 10 GiB, you may need to set this parameter to 9 GiB or 8 GiB. It can be used in conjunction with "load_in_8bit" but not with "load-in-4bit" as far as I'm aware.
* **cpu-memory**: similarly to the parameter above, you can also set a limit on the amount of CPU memory used. Whatever doesn't fit either in the GPU or the CPU will go to a disk cache, so to use this option you should also check the "disk" one.
* **compute_dtype": used when "load-in-4bit" is checked. I recommend leaving the default value.
* **quant_type": used when "load-in-4bit" is checked. I recommend leaving the default value.
* **alpha_value": used to extend the context length of a model with a minor loss in quality. I have measured 1.75 to be optimal for 1.5x context, and 2.5 for 2x context. That is, with alpha = 2.5 you can make a model with 4096 context length go to 8192 context length.
* **rope_freq_base**: originally another way to write "alpha_value", it ended up becoming a necessary parameter for some models like CodeLlama, which was fine-tuned with this set to 1000000 and hence needs to be loaded with it set to 1000000 as well.
* **compress_pos_emb**: the first context-length extension method, discovered by [kaiokendev](https://kaiokendev.github.io/til). When set to 2, the context length is doubled, 3 and its tripled, etc. It should only be used for models that have been fine-tuned with this parameter set to different than 1. For models that have not been tuned to have greater context length, alpha_value will lead to a smaller accuracy loss.
* **cpu**: loads the model in CPU mode using Pytorch. The model will be loaded in 32-bit precision, so a lot of RAM will be used. CPU inference with transformers is older than llama.cpp and it works, but it's a lot slower. Note: this parameter has a different interpretation in the llama.cpp loader (see below).
* **load-in-8bit**: load the model in 8-bit precision using bitsandbytes. The 8-bit kernel in that library has been optimized for training and not inference, so load-in-8bit is slower than load-in-4bit (but more accurate).
* **bf16**: use bfloat16 precision instead of float16 (the default). Only applies when quantization is not used.
* **auto-devices**: when checked, the backend will try to guess a reasonable value for "gpu-memory" to allow you to load a model with CPU offloading. I recommend just setting "gpu-memory" manually instead. This parameter is also needed for loading GPTQ models, in which case it needs to be checked before loading the model.
* **disk**: enable disk offloading for layers that don't fit into the GPU and CPU combined.
* **load-in-4bit**: load the model in 4-bit precision using bitsandbytes.
* **trust-remote-code**: some models use custom Python code to load the model or the tokenizer. For such models, this option needs to be set. It doesn't actually download any remote content: all it does is execute the .py files that get downloaded with the model. Those files can potentially include malicious code; I have never seen it happen, but it is in principle possible.
* **use_fast**: use the "fast" version of the tokenizer. Especially useful for Llama models, which originally had a "slow" tokenizer that received an update. If your local files are in the old "slow" format, checking this option may trigger a conversion that takes several minutes. The fast tokenizer is mostly useful if you are generating 50+ tokens/second using ExLlama or if you are tokenizing a huge dataset for training.
* **disable_exllama**: only applies when you are loading a GPTQ model though the transformers loader. It needs to be checked if you intend to train LoRAs with the model.

### Transformers
