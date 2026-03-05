## What Works

| Loader         | Loading LoRAs | Training LoRAs | Multimodal | Perplexity evaluation |
|----------------|---------------|----------------|------------|-----------------------|
| llama.cpp      |      ❌       |       ❌       |    ✅\*    |           ❌          |
| Transformers   |      ✅       |       ✅       |    ✅\*\*  |           ✅          |
| ExLlamav3_HF   |      ❌       |       ❌       |    ❌      |           ✅          |
| ExLlamav3      |      ❌       |       ❌       |    ✅      |           ❌          |
| ExLlamav2_HF   |      ✅       |       ❌       |    ❌      |           ✅          |
| ExLlamav2      |      ✅       |       ❌       |    ❌      |           ❌          |
| TensorRT-LLM   |      ❌       |       ❌       |    ❌      |           ❌          |

❌ = not supported

✅ = supported

\* Via the `mmproj` parameter (multimodal projector file).

\*\* Via the `send_pictures` extension.
