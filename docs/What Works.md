## What Works

| Loader         | Loading 1 LoRA | Loading 2 or more LoRAs | Training LoRAs | Multimodal extension | Perplexity evaluation |
|----------------|----------------|-------------------------|----------------|----------------------|-----------------------|
| Transformers   |       ✅       |           ✅\*\*        |       ✅\*     |          ✅          |           ✅          |
| llama.cpp      |       ❌       |           ❌            |       ❌       |          ❌          |    use llamacpp_HF    |
| llamacpp_HF    |       ❌       |           ❌            |       ❌       |          ❌          |           ✅          |
| ExLlamav2_HF   |       ✅       |           ✅            |       ❌       |          ❌          |           ✅          |
| ExLlamav2      |       ✅       |           ✅            |       ❌       |          ❌          |   use ExLlamav2_HF    |
| AutoGPTQ       |       ✅       |           ❌            |       ❌       |          ✅          |           ✅          |
| AutoAWQ        |       ?        |           ❌            |       ?        |          ?           |           ✅          |
| HQQ            |       ?        |           ?             |       ?        |          ?           |           ✅          |

❌ = not implemented

✅ = implemented

\* Training LoRAs with GPTQ models also works with the Transformers loader. Make sure to check "auto-devices" and "disable_exllama" before loading the model.

\*\* Multi-LoRA in PEFT is tricky and the current implementation does not work reliably in all cases.
