| Loader         | Loading 1 LoRA | Loading 2 or more LoRAs | Training LoRAs | Multimodal extension | Perplexity evaluation | Classifier-Free Guidance (CFG) |
|----------------|----------------|-------------------------|----------------|----------------------|-----------------------|--------------------------------|
| Transformers   |       ✅       |           ❌            |       ✅       |          ✅          |           ✅          |               ✅               |
| ExLlama_HF     |       ❌       |           ❌            |       ❌       |          ❌          |           ✅          |               ✅               |
| ExLlamav2_HF   |       ✅       |           ❌            |       ❌       |          ❌          |           ✅          |               ✅               |
| ExLlama        |       ❌       |           ❌            |       ❌       |          ❌          |           ❌          |               ✅               |
| ExLlamav2      |       ✅       |           ❌            |       ❌       |          ❌          |           ❌          |               ❌               |
| AutoGPTQ       |       ✅       |           ❌            |       ❌       |          ✅          |           ✅          |               ✅               |
| GPTQ-for-LLaMa |       ✅       |           ❌            |       ✅       |          ✅          |           ✅          |               ✅               |
| llama.cpp      |       ❌       |           ❌            |       ❌       |          ❌          |           ❌          |               ❌               |
| llamacpp_HF    |       ❌       |           ❌            |       ❌       |          ❌          |           ✅          |               ✅               |
| ctransformers  |       ❌       |           ❌            |       ❌       |          ❌          |           ❌          |               ❌               |

❌ = not implemented (feel free to contribute and submit a PR)
✅ = implemented
