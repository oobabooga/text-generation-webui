# LoRA

LoRA (Low-Rank Adaptation) is an extremely powerful method for customizing a base model by training only a small number of parameters. They can be attached to models at runtime.

For instance, a 50mb LoRA can teach LLaMA an entire new language, a given writing style, or give it instruction-following or chat abilities.

This is the current state of LoRA integration in the web UI:

|Loader | Status |
|--------|------|
| Transformers | Full support in 16-bit, `--load-in-8bit`, `--load-in-4bit`, and CPU modes. |
| ExLlama | Single LoRA support. Fast to remove the LoRA afterwards. |
| AutoGPTQ | Single LoRA support. Removing the LoRA requires reloading the entire model.|
| GPTQ-for-LLaMa | Full support with the [monkey patch](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#using-loras-with-gptq-for-llama). |

## Downloading a LoRA

The download script can be used. For instance:

```
python download-model.py tloen/alpaca-lora-7b
```

The files will be saved to `loras/tloen_alpaca-lora-7b`.

## Using the LoRA

The `--lora` command-line flag can be used. Examples:

```
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --load-in-8bit
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --load-in-4bit
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --cpu
```

Instead of using the `--lora` command-line flag, you can also select the LoRA in the "Parameters" tab of the interface.

## Prompt
For the Alpaca LoRA in particular, the prompt must be formatted like this:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a Python script that generates text using the transformers library.
### Response:
```

Sample output:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a Python script that generates text using the transformers library.
### Response:

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
texts = ["Hello world", "How are you"]
for sentence in texts:
sentence = tokenizer(sentence)
print(f"Generated {len(sentence)} tokens from '{sentence}'")
output = model(sentences=sentence).predict()
print(f"Predicted {len(output)} tokens for '{sentence}':\n{output}")
```

## Training a LoRA

You can train your own LoRAs from the `Training` tab. See [Training LoRAs](Training-LoRAs.md) for details.
