## What Works

| Loader         | Loading 1 LoRA | Loading 2 or more LoRAs | Training LoRAs | Multimodal extension | Perplexity evaluation |
|----------------|----------------|-------------------------|----------------|----------------------|-----------------------|
| Transformers   |       ✅       |           ❌            |       ✅*       |          ✅          |           ✅          |
| ExLlama_HF     |       ✅       |           ❌            |       ❌       |          ❌          |           ✅          |
| ExLlamav2_HF   |       ✅       |           ✅            |       ❌       |          ❌          |           ✅          |
| ExLlama        |       ✅       |           ❌            |       ❌       |          ❌          |           use ExLlama_HF      |
| ExLlamav2      |       ✅       |           ✅            |       ❌       |          ❌          |           use ExLlamav2_HF    |
| AutoGPTQ       |       ✅       |           ❌            |       ❌       |          ✅          |           ✅          |
| GPTQ-for-LLaMa |       ✅       |           ❌            |       ✅       |          ✅          |           ✅          |
| llama.cpp      |       ❌       |           ❌            |       ❌       |          ❌          |           use llamacpp_HF    |
| llamacpp_HF    |       ❌       |           ❌            |       ❌       |          ❌          |           ✅          |
| ctransformers  |       ❌       |           ❌            |       ❌       |          ❌          |           ❌          |

❌ = not implemented

✅ = implemented

\* Training LoRAs with GPTQ models also works with the Transformers loader. Make sure to check "auto-devices" and "disable_exllama" before loading the model.
