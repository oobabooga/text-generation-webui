## What Works

| Loader         | Loading 1 LoRA | Loading 2 or more LoRAs | Training LoRAs | Multimodal extension | Perplexity evaluation |
|----------------|----------------|-------------------------|----------------|----------------------|-----------------------|
| Transformers   |       ✅       |           ✅            |       ✅*       |          ✅          |           ✅          |
| ExLlama_HF     |       ✅       |           ❌            |       ❌       |          ❌          |           ✅          |
| ExLlamav2_HF   |       ✅       |           ✅            |       ❌       |          ❌          |           ✅          |
| ExLlama        |       ✅       |           ❌            |       ❌       |          ❌          |           use ExLlama_HF      |
| ExLlamav2      |       ✅       |           ✅            |       ❌       |          ❌          |           use ExLlamav2_HF    |
| AutoGPTQ       |       ✅       |           ❌            |       ❌       |          ✅          |           ✅          |
| GPTQ-for-LLaMa |       ✅**       |           ✅            |       ✅       |          ✅          |           ✅          |
| llama.cpp      |       ❌       |           ❌            |       ❌       |          ❌          |           use llamacpp_HF    |
| llamacpp_HF    |       ❌       |           ❌            |       ❌       |          ❌          |           ✅          |
| ctransformers  |       ❌       |           ❌            |       ❌       |          ❌          |           ❌          |
| AutoAWQ        |       ?        |           ❌            |       ?       |          ?          |           ✅          |

❌ = not implemented

✅ = implemented

\* Training LoRAs with GPTQ models also works with the Transformers loader. Make sure to check "auto-devices" and "disable_exllama" before loading the model.

\*\* Requires the monkey-patch. The instructions can be found [here](https://github.com/oobabooga/text-generation-webui/wiki/08-%E2%80%90-Additional-Tips#using-loras-with-gptq-for-llama).
